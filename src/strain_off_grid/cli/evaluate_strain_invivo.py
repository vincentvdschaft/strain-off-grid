import argparse
import re
import warnings
from pathlib import Path

import h5py
import numpy as np
from imagelib import Image, Limits, LimitsND, stack
from plotlib import STYLE_DARK, use_style
from scipy.spatial import KDTree

try:
    from storepari import (
        AxesSettings,
        DimsSettings,
        GridSettings,
        NapariImage,
        NapariPoints,
        ViewerSettings,
        ViewerState,
    )

    _STOREPARI_IMPORT_ERROR = None
except Exception as exc:  # napari/vispy pull in Qt/OpenGL, which may be unavailable
    # in headless environments (e.g. Docker without a display/GL stack).
    _STOREPARI_IMPORT_ERROR = exc

from strain_off_grid import console
from strain_off_grid.phantoms.block_matching import get_baseline_tracking_velocities
from strain_off_grid.phantoms.chained_boxes import _points_in_any_box
from strain_off_grid.strain.evaluate import _filter_solver_velocities_harmonic
from strain_off_grid.strain.strain_rate_map import compute_strain_rate_map
from strain_off_grid.velocities import Velocities2D, Velocities3D

TX_PATTERN = re.compile(r"tx(\d+)-(\d+)")
FRAME_PATTERN = re.compile(r"frame-(\d+)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path",
        nargs="?",
        type=str,
        default="out/sweep/in_vivo_00",
        help="Sweep directory containing frame-*/tx*.hdf5 solver solutions.",
    )
    parser.add_argument(
        "--source-file",
        type=str,
        default="source/20251222_s3_a4ch_line_dw_0000.hdf5",
        help="Raw acquisition file used to look up the real transmit timing "
        "(scan/time_to_next_transmit). Falls back to dt=1 (arbitrary units) "
        "if not found.",
    )
    parser.add_argument(
        "--direction",
        type=float,
        nargs=2,
        default=(0.0, 1.0),
        metavar=("DX", "DZ"),
        help="Direction to project the strain rate onto. There is no phantom "
        "to derive a principal axis from for in-vivo data, so a fixed "
        "direction vector is used instead.",
    )
    parser.add_argument(
        "--average-transmits",
        action="store_true",
        help="Average the strain rate maps over the transmits within a frame. "
        "If not set, every transmit burst is kept as its own map.",
    )
    parser.add_argument(
        "--map-pixel-size",
        type=float,
        default=2e-4,
    )
    parser.add_argument(
        "--max-distance",
        type=float,
        default=5e-3,
        help="Neighbourhood radius used to fit the local strain rate.",
    )
    parser.add_argument(
        "--kernel-size",
        type=float,
        default=5e-3,
        help="Gaussian kernel bandwidth used to interpolate strain rates onto the map grid.",
    )
    parser.add_argument(
        "--mask-distance",
        type=float,
        default=10e-3,
        help="Mask out map pixels further than this distance from any solver point.",
    )
    parser.add_argument(
        "--min-intensity",
        type=float,
        default=0.5,
        help="Minimum intensity, relative to the max intensity within "
        "--neighborhood-radius, required to keep a solver point.",
    )
    parser.add_argument(
        "--neighborhood-radius",
        type=float,
        default=2e-3,
        help="Radius used to find the local max intensity that --min-intensity "
        "is relative to. Harmonic imaging intensity varies strongly over the "
        "field of view, so a local rather than global reference is used.",
    )
    parser.add_argument("--max-velocity", type=float, default=0.3)
    parser.add_argument("--block-size", type=int, nargs=2, default=(21, 21))
    parser.add_argument(
        "--no-show", action="store_true", help="Do not show the results in napari"
    )
    parser.add_argument(
        "--out",
        type=str,
        default="out/",
    )
    parser.add_argument(
        "--max-n-frames",
        type=int,
        default=None,
        help="Maximum number of frames to process",
    )
    parser.add_argument(
        "--specific-frame",
        type=int,
        default=None,
        help="Process only the specified frame",
    )
    parser.add_argument(
        "--specific-transmit",
        type=int,
        default=None,
        help="Process only the specified transmit",
    )
    args = parser.parse_args()

    use_style(STYLE_DARK)

    sweep_dir = Path(args.path)
    frame_dirs = sorted(
        (p for p in sweep_dir.iterdir() if p.is_dir() and FRAME_PATTERN.match(p.name)),
        key=lambda p: int(FRAME_PATTERN.match(p.name).group(1)),
    )

    if args.specific_frame is not None:
        frame_dirs = [
            frame_dir
            for frame_dir in frame_dirs
            if int(FRAME_PATTERN.match(frame_dir.name).group(1)) == args.specific_frame
        ]
    elif args.max_n_frames is not None:
        frame_dirs = frame_dirs[: args.max_n_frames]
    if not frame_dirs:
        raise FileNotFoundError(f"No frame-* directories found in {sweep_dir}")

    time_to_next_transmit = _load_time_to_next_transmit(args.source_file)

    first_image = Image.load(next(frame_dirs[0].glob("tx*.hdf5")))
    map_limits_yx = first_image.limits[1:]
    map_limits_yx = LimitsND((5e-3, 90e-3, -40e-3, 40e-3))

    direction = np.array(args.direction)

    maps_solver = []
    maps_baseline = []
    maps_das = []
    positions_solver = []
    positions_baseline = []
    frame_indices = []
    transmit_starts = []

    for frame_dir in frame_dirs:
        frame_index = int(FRAME_PATTERN.match(frame_dir.name).group(1))
        tx_paths = sorted(frame_dir.glob("tx*.hdf5"))
        if args.specific_transmit is not None:
            tx_paths = [tx_paths[args.specific_transmit]]

        frame_maps_solver = []
        frame_maps_baseline = []
        frame_maps_das = []
        frame_positions_solver = []
        frame_positions_baseline = []
        for tx_path in tx_paths:
            transmits = _parse_transmits(tx_path.stem)
            dt = _burst_dt(time_to_next_transmit, frame_index, transmits)

            console.log(
                f"Processing frame {frame_index}, transmits {transmits[0]}-{transmits[-1]} (dt={dt:.3e}s)"
            )
            (
                map_solver,
                map_baseline,
                map_das,
                tx_positions_solver,
                tx_positions_baseline,
            ) = _compute_invivo_strain_maps(
                tx_path,
                dt=dt,
                map_limits_yx=map_limits_yx,
                map_pixel_size=args.map_pixel_size,
                direction=direction,
                max_distance=args.max_distance,
                kernel_size=args.kernel_size,
                mask_distance=args.mask_distance,
                min_intensity=args.min_intensity,
                max_velocity_magnitude=args.max_velocity,
                neighborhood_radius=args.neighborhood_radius,
                block_size=tuple(args.block_size),
            )
            frame_maps_solver.append(map_solver)
            frame_maps_baseline.append(map_baseline)
            frame_maps_das.append(map_das)
            frame_positions_solver.append(tx_positions_solver)
            frame_positions_baseline.append(tx_positions_baseline)

        if args.average_transmits:
            maps_solver.append(_nanmean_maps(frame_maps_solver))
            maps_baseline.append(_nanmean_maps(frame_maps_baseline))
            maps_das.append(frame_maps_das[len(frame_maps_das) // 2])
            positions_solver.append(np.concatenate(frame_positions_solver))
            positions_baseline.append(np.concatenate(frame_positions_baseline))
            frame_indices.append(frame_index)
            transmit_starts.append(_parse_transmits(tx_paths[0].stem)[0])
        else:
            maps_solver.extend(frame_maps_solver)
            maps_baseline.extend(frame_maps_baseline)
            maps_das.extend(frame_maps_das)
            positions_solver.extend(frame_positions_solver)
            positions_baseline.extend(frame_positions_baseline)
            frame_indices.extend([frame_index] * len(tx_paths))
            transmit_starts.extend(_parse_transmits(p.stem)[0] for p in tx_paths)

    map_solver = stack(maps_solver, limits=Limits(0, len(maps_solver) - 1))
    map_baseline = stack(maps_baseline, limits=Limits(0, len(maps_baseline) - 1))
    map_das = stack(maps_das, limits=Limits(0, len(maps_das) - 1))
    map_solver = map_solver.add_metadata("frame_indices", np.array(frame_indices))
    map_solver = map_solver.add_metadata("transmit_starts", np.array(transmit_starts))

    out_dir = Path(args.out)
    output_path = out_dir / "strain_maps_invivo.hdf5"
    map_solver.save(output_path, group="/solver")
    map_baseline.save(output_path, group="/baseline", append=True)
    map_das.save(output_path, group="/das", append=True)
    console.log(f"Saved strain maps to {output_path}")

    if _STOREPARI_IMPORT_ERROR is not None:
        console.log(
            f"[yellow]Skipping napari viewer state (storepari/napari unavailable: "
            f"{_STOREPARI_IMPORT_ERROR})"
        )
        return

    # ======================================================================================
    # Show maps in a napari grid
    # ======================================================================================

    def nan_safe_contrast_limits(*maps, fraction_vmax=1.0) -> tuple[float, float]:
        vmax = 0.0
        for map in maps:
            vmax = max(vmax, np.nanmax(np.abs(map.array)))

        vmax *= fraction_vmax
        return (-vmax, vmax)

    contrast_limits = (-3, 3)

    groups = [
        ("Solver", map_solver, positions_solver, "white"),
        ("Baseline", map_baseline, positions_baseline, "orange"),
    ]

    layers = []
    n_layers_per_group = None

    for group_name, strain_image, group_positions, point_color in groups:
        layers.append(
            NapariImage(
                data=map_das.array,
                name=f"DAS ({group_name})",
                scale=map_das.scale,
                translate=map_das.translate,
                contrast_limits=(-60.0, 0.0),
                colormap="gray",
            )
        )
        layers.append(
            NapariImage(
                data=strain_image.array,
                name=f"Strain Rate ({group_name})",
                scale=strain_image.scale,
                translate=strain_image.translate,
                contrast_limits=contrast_limits,
                colormap="coolwarm",
                colorbar=True,
                opacity=0.6,
            )
        )
        layers.append(
            NapariPoints(
                data=_positions_to_napari_points(group_positions),
                name=f"Velocity Positions ({group_name})",
                face_color=point_color,
                border_color=point_color,
                size=map_das.scale[-1] * 4,
            )
        )
        if n_layers_per_group is None:
            n_layers_per_group = len(layers)

    viewer_state = ViewerState(
        layers=layers,
        settings=ViewerSettings(
            dims=DimsSettings(
                ndisplay=2, order=(0, 1), axis_labels=["frame", "z", "x"]
            ),
            axes=AxesSettings(visible=True),
            grid=GridSettings(enabled=True, shape=(1, 2), stride=n_layers_per_group),
        ),
    ).to_hdf5(out_dir / "viewer_state.hdf5")
    if not args.no_show:
        viewer_state.run()


def _positions_to_napari_points(positions_per_map: list[np.ndarray]) -> np.ndarray:
    """Stacks per-map (x, z) positions into napari (stack, z, x) point coordinates."""
    point_rows = [
        _prepend_stack_index(positions, stack_index)
        for stack_index, positions in enumerate(positions_per_map)
    ]
    return np.concatenate(point_rows, axis=0)


def _prepend_stack_index(positions_xz: np.ndarray, stack_index: int) -> np.ndarray:
    """Turns (N, 2) (x, z) positions into (N, 3) (stack, z, x) napari coordinates."""
    stack_column = np.full(len(positions_xz), stack_index)
    return np.column_stack([stack_column, positions_xz[:, 1], positions_xz[:, 0]])


def _compute_invivo_strain_maps(
    path: str | Path,
    dt: float,
    map_limits_yx: LimitsND,
    map_pixel_size: float,
    direction: np.ndarray,
    max_distance: float,
    kernel_size: float,
    mask_distance: float,
    min_intensity: float,
    max_velocity_magnitude: float,
    neighborhood_radius: float,
    block_size: tuple[int, int],
) -> tuple[Image, Image, Image, np.ndarray, np.ndarray]:
    image = Image.load(path).abs().normalize().log_compress().clip(-60, 0)
    map_limits_yx = LimitsND(map_limits_yx).fitted_to_pixel_sizes(map_pixel_size)
    target_grid = map_limits_yx.make_grid(pixel_sizes=map_pixel_size)[..., ::-1]

    velocities_2d_solver = _get_solver_velocities_invivo(
        image,
        dt=dt,
        min_intensity=min_intensity,
        max_velocity_magnitude=max_velocity_magnitude,
        neighborhood_radius=neighborhood_radius,
    )
    velocities_2d_solver = _filter_velocities_inside_mask(velocities_2d_solver)
    velocities_2d_baseline = get_baseline_tracking_velocities(
        image0=image[0],
        image1=image[-1],
        positions0=velocities_2d_solver.positions,
        dt=dt,
        block_size=block_size,
    )
    print(velocities_2d_baseline)
    velocities_2d_baseline = _filter_baseline_velocities(
        velocities_2d_baseline, max_velocity_magnitude=0.4
    )
    strain_rate_map_solver = compute_strain_rate_map(
        velocities=velocities_2d_solver,
        directions=direction,
        target_grid=target_grid,
        max_distance=max_distance,
        kernel_size=kernel_size,
    )
    strain_rate_map_baseline = compute_strain_rate_map(
        velocities=velocities_2d_baseline,
        directions=direction,
        target_grid=target_grid,
        max_distance=max_distance,
        kernel_size=kernel_size,
    )

    mask = _define_mask(target_grid)
    strain_rate_map_solver[mask] = np.nan
    strain_rate_map_baseline[mask] = np.nan

    return (
        Image(strain_rate_map_solver, limits=map_limits_yx, metadata={"dt": dt}),
        Image(strain_rate_map_baseline, limits=map_limits_yx, metadata={"dt": dt}),
        _middle_das_image(image),
        velocities_2d_solver.positions,
        velocities_2d_baseline.positions,
    )


def _filter_velocities_inside_mask(velocities: Velocities2D) -> Velocities2D:
    """Keeps only velocities whose positions fall inside the chained boxes."""
    box_endpoints = _get_mask_box_endpoints()
    inside = _points_in_any_box(velocities.positions, box_endpoints)
    return velocities[inside]


def _filter_baseline_velocities(
    velocities: Velocities2D, max_velocity_magnitude: float
) -> Velocities2D:
    """Filter out baseline velocities that are too large in magnitude.

    Baseline velocities are computed from block matching, which can produce
    spurious large velocities in regions of low SNR. This filter removes those
    points to avoid them dominating the strain rate map interpolation.

    Args:
        velocities: Velocities to filter.
        max_velocity_magnitude: Maximum allowed velocity magnitude. Points with
            larger magnitudes will be removed.

    Returns:
        Filtered velocities.
    """
    magnitudes = np.linalg.norm(velocities.velocities, axis=-1)
    mask = magnitudes <= max_velocity_magnitude
    print(mask)
    return velocities[mask]


def _middle_das_image(image: Image) -> Image:
    """Returns the middle transmit of the DAS image stack."""
    return image[image.array.shape[0] // 2]


def mask_far_from_points(
    target_grid: np.ndarray, positions: np.ndarray, max_distance: float
) -> np.ndarray:
    """Boolean mask that is True for grid points further than `max_distance` from every point.

    Used to hide the interpolated strain rate far away from any tracked scatterer,
    since there is no phantom geometry to mask against for in-vivo data.
    """
    grid_shape = target_grid.shape[:-1]
    flat_grid = target_grid.reshape(-1, target_grid.shape[-1])
    tree = KDTree(positions)
    distances, _ = tree.query(flat_grid)
    return distances.reshape(grid_shape) > max_distance


def _get_solver_velocities_invivo(
    image: Image,
    dt: float,
    min_intensity: float,
    max_velocity_magnitude: float,
    neighborhood_radius: float,
) -> Velocities2D:
    """Computes velocities from the first to the last transmit of a solved burst."""
    positions_all_transmits = image.metadata["scat_pos"]
    positions_first_transmit = positions_all_transmits[:, 0]
    velocities_2d_solver = Velocities3D(
        positions=positions_first_transmit,
        velocities=np.diff(positions_all_transmits[:, [0, -1]], axis=1)[:, 0] / dt,
    ).to_2d()

    intensities = image.metadata["scat_amp"][:, 0]
    intensities = intensities / np.max(intensities)
    velocities_2d_solver = _filter_solver_velocities_harmonic(
        velocities_2d_solver,
        intensities=intensities,
        neighborhood_radius=neighborhood_radius,
        min_relative_intensity=min_intensity,
        max_velocity_magnitude=max_velocity_magnitude,
    )
    console.log("Filtered solver velocities: %d points" % len(velocities_2d_solver))

    return velocities_2d_solver


def _nanmean_maps(maps: list[Image]) -> Image:
    stacked = np.stack([m.array for m in maps], axis=0)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Mean of empty slice")
        mean_array = np.nanmedian(stacked, axis=0)
    return Image(mean_array, limits=maps[0].limits, metadata=maps[0].metadata)


def _parse_transmits(stem: str) -> list[int]:
    match = TX_PATTERN.match(stem)
    if match is None:
        raise ValueError(f"Could not parse transmit range from '{stem}'")
    start, end = int(match.group(1)), int(match.group(2))
    return list(range(start, end + 1))


def _load_time_to_next_transmit(source_file: str | Path) -> np.ndarray | None:
    """Loads the (n_frames, n_tx) transmit timing table from the raw acquisition.

    This gives the real, physical time between transmits so that solver and
    baseline velocities (and therefore strain rates) come out in correct units.
    Only this small dataset is read, not the multi-gigabyte raw RF data.
    """
    source_file = Path(source_file)
    if not source_file.exists():
        console.log(
            f"[yellow]Source file {source_file} not found; "
            "velocities will be computed with dt=1 per transmit (arbitrary units)."
        )
        return None
    with h5py.File(source_file, "r") as f:
        return np.asarray(f["scan/time_to_next_transmit"])


def _get_mask_box_endpoints():
    box_endpoints = np.array(
        [
            [[0.089290179, 0.00915494], [0.089845218, -0.00049206428]],
            [[0.08003193, 0.00740516], [0.08263265, -0.00236527]],
            [[0.06836364, 0.00410149], [0.07026145, -0.00841028]],
            [[0.05571129, 0.0017116], [0.05627362, -0.01072988]],
            [[0.04594086, 0.00143044], [0.04312927, -0.00953494]],
            [[0.03792771, 0.00375003], [0.03272618, -0.0051769]],
            [[0.03195298, 0.00852981], [0.02309636, 0.00585876]],
            [[0.02991455, 0.01619152], [0.02267461, 0.01752704]],
            [[0.03581898, 0.02300973], [0.02893047, 0.03024967]],
            [[0.04376185, 0.02800038], [0.04024728, 0.03861428]],
            [[0.05585187, 0.02989823], [0.05402432, 0.04311289]],
            [[0.06906655, 0.03172579], [0.06927742, 0.04437815]],
            [[0.07939931, 0.03207725], [0.07904785, 0.04409698]],
            [[0.08741247, 0.03263958], [0.08818565, 0.04423752]],
        ],
        dtype=np.float32,
    )[..., [1, 0]]
    return box_endpoints


# def _get_mask_box_endpoints():
#     # Frame 13
#     box_endpoints = np.array(
#         [
#             # [[0.08519252, 0.01008674], [0.08751212, -0.00144096]],
#             [[0.07584383, 0.00678307], [0.07753094, -0.00298736]],
#             [[0.06522992, 0.00411202], [0.06628429, -0.00572871]],
#             [[0.05412396, 0.00305765], [0.0537725, -0.00734539]],
#             [[0.04554847, 0.00291707], [0.04273683, -0.0072751]],
#             [[0.03746502, 0.00383085], [0.03345844, -0.00383085]],
#             [[0.03036565, 0.00861063], [0.02340687, 0.00600987]],
#             [[0.03001419, 0.01760786], [0.02474239, 0.02048978]],
#             [[0.03627008, 0.02351229], [0.03170114, 0.0289247]],
#             [[0.04540789, 0.027308], [0.04266655, 0.03630518]],
#             [[0.05735734, 0.02906527], [0.05517832, 0.03897625]],
#             [[0.068393, 0.0311037], [0.06719806, 0.04157701]],
#             [[0.07661703, 0.03257981], [0.07563296, 0.04305312]],
#             # [[0.07978012, 0.03272039], [0.07907723, 0.04284224]],
#         ],
#         dtype=np.float32,
#     )[..., [1, 0]]
#     return box_endpoints


# def _get_mask_box_endpoints():
#     # Frame 6
#     box_endpoints = np.array(
#         [
#             [[0.08902149, 0.00525475], [0.09066789, -0.00592347]],
#             [[0.07896975, 0.00083546], [0.08000958, -0.00912962]],
#             [[0.06744493, -0.00055099], [0.06787818, -0.01068937]],
#             [[0.05678662, -0.00358384], [0.05721989, -0.01328897]],
#             [[0.043962, -0.00497029], [0.04136241, -0.01311565]],
#             [[0.0364232, 0.00083546], [0.02879776, -0.00540355]],
#             [[0.02914436, 0.00785434], [0.02160554, 0.00603463]],
#             [[0.02871109, 0.01911921], [0.02238545, 0.02189209]],
#             [[0.03668316, 0.02579148], [0.03148399, 0.0322038]],
#             [[0.04656158, 0.03177053], [0.04352872, 0.03896272]],
#             [[0.0590396, 0.03471673], [0.05704658, 0.04208222]],
#             [[0.06848476, 0.0367964], [0.06779155, 0.04502841]],
#             [[0.07663013, 0.03662309], [0.07723671, 0.04459514]],
#         ],
#         dtype=np.float32,
#     )[..., [1, 0]]
#     return box_endpoints


def _get_mask_box_endpoints():
    # Frame 18
    box_endpoints = np.array(
        [
            [[0.07862435, 0.01382602], [0.07932728, 0.00328239]],
            [[0.0689945, 0.01206875], [0.06885392, 0.00011931]],
            [[0.05718563, 0.01094409], [0.05655302, -0.00227058]],
            [[0.04579852, 0.00988973], [0.04481445, -0.00255175]],
            [[0.0375042, 0.00778101], [0.03124832, -0.00044302]],
            [[0.03110773, 0.0139666], [0.0220402, 0.00686722]],
            [[0.03033453, 0.02001162], [0.0221105, 0.02289355]],
            [[0.03771507, 0.02690012], [0.031178, 0.03442124]],
            [[0.04347892, 0.0301335], [0.04059698, 0.03997424]],
            [[0.05205441, 0.03224222], [0.04966451, 0.04208295]],
            [[0.06077047, 0.03456182], [0.05845086, 0.04454311]],
        ],
        dtype=np.float32,
    )[..., [1, 0]]
    return box_endpoints


def _get_mask_box_endpoints():
    # Frame 34
    box_endpoints = np.array(
        [
            [[0.08699417, 0.00886707], [0.08870339, 0.0008553]],
            [[0.07908922, 0.00566237], [0.08005065, -0.00160164]],
            [[0.06947511, 0.00406001], [0.07065019, -0.0041654]],
            [[0.06124969, 0.00267131], [0.06189063, -0.00608823]],
            [[0.05238334, 0.00181672], [0.05163557, -0.00758376]],
            [[0.04255557, 0.00267131], [0.04041911, -0.00598141]],
            [[0.0345438, 0.00352589], [0.03069815, -0.00331081]],
            [[0.02824121, 0.00886707], [0.0222591, 0.0044873]],
            [[0.02802756, 0.01474237], [0.02012262, 0.01463555]],
            [[0.02845486, 0.02104496], [0.02247274, 0.02371556]],
            [[0.03347556, 0.02692026], [0.02866852, 0.03247507]],
            [[0.0418078, 0.0330092], [0.03838946, 0.03845717]],
            [[0.05078098, 0.03589343], [0.04875133, 0.04219601]],
            [[0.06178381, 0.03728214], [0.0608224, 0.0444393]],
            [[0.07257299, 0.03717532], [0.07214569, 0.04475977]],
            [[0.08389629, 0.03717532], [0.08378946, 0.0444393]],
        ],
        dtype=np.float32,
    )[..., [1, 0]]
    return box_endpoints


def _define_mask(grid: np.ndarray) -> np.ndarray:
    """Boolean mask that is True for grid points outside the chained boxes."""
    box_endpoints = _get_mask_box_endpoints()
    flat_grid = grid.reshape(-1, grid.shape[-1])
    inside = _points_in_any_box(flat_grid, box_endpoints)
    return ~inside.reshape(grid.shape[:-1])


def _burst_dt(
    time_to_next_transmit: np.ndarray | None, frame_index: int, transmits: list[int]
) -> float:
    if time_to_next_transmit is None:
        return float(len(transmits) - 1)
    gaps = time_to_next_transmit[frame_index, transmits[:-1]]
    return float(gaps.sum())


if __name__ == "__main__":
    main()
