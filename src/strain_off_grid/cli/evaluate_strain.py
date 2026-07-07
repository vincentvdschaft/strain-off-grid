import argparse
from pathlib import Path

import numpy as np
from imagelib import Image
from plotlib import STYLE_DARK, use_style
from storepari import (
    AxesSettings,
    DimsSettings,
    GridSettings,
    NapariImage,
    ViewerSettings,
    ViewerState,
)

from strain_off_grid import console
from strain_off_grid.phantoms import DynamicPhantom, load_dataclass
from strain_off_grid.strain.evaluate import compute_strain_maps_multi
from strain_off_grid.velocities import Velocities3D


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "paths",
        nargs="*",
        type=str,
        default="out/latest_solution.hdf5",
    )
    parser.add_argument(
        "--direction",
        type=int,
        default=0,
        help="Direction along which to compute strain (0=principal axis, 1=orthogonal axis)",
    )
    parser.add_argument(
        "--no-show", action="store_true", help="Do not show the results in napari"
    )
    parser.add_argument(
        "--out",
        type=str,
        default="out/",
    )
    args = parser.parse_args()

    use_style(STYLE_DARK)
    paths = [Path(p) for p in args.paths]
    images = [
        Image.load(path).abs().normalize().log_compress().clip(-60, 0).to_pixels()
        for path in paths
    ]

    phantom: DynamicPhantom = load_dataclass(str(paths[0]), group="/custom/phantom")
    frame = 0
    image = images[0]
    timestamps = image.metadata["timestamps"].ravel()
    dt = timestamps[1] - timestamps[0]
    print(timestamps)
    positions_all = image.metadata["scat_pos"]
    positions = positions_all[:, frame]
    velocities_3d_solver = Velocities3D(
        positions=positions, velocities=np.diff(positions_all, axis=1)[:, frame] / dt
    )

    if args.direction == 2:
        direction = np.array([0.0, 1.0])
    else:
        direction = args.direction

    first_image = Image.load(paths[0])

    (map_solver, map_ground_truth, map_baseline) = compute_strain_maps_multi(
        paths,
        map_pixel_size=2e-4,
        map_limits_yx=first_image.limits[1:],
        direction=direction,
        kernel_size=5e-3,
    )
    out_dir = Path(args.out)

    if args.direction == 0:
        output_path = out_dir / "strain_maps_principal.hdf5"
    elif args.direction == 1:
        output_path = out_dir / "strain_maps_orthogonal.hdf5"
    elif args.direction == 2:
        output_path = out_dir / "strain_maps_vertical.hdf5"
    map_solver.save(output_path, group="/solver")
    map_ground_truth.save(output_path, group="/ground_truth", append=True)
    map_baseline.save(output_path, group="/baseline", append=True)
    phantom.to_hdf5(output_path, group="/phantom")
    console.log(f"Saved strain maps to {output_path}")

    # ======================================================================================
    # Show maps, points, and vectors in a napari grid
    # ======================================================================================

    def nan_safe_contrast_limits(*maps, fraction_vmax=1.0) -> tuple[float, float]:
        vmax = 0.0
        for map in maps:
            vmax = max(vmax, np.nanmax(np.abs(map.array)))

        vmax *= fraction_vmax
        return (-vmax, vmax)

    contrast_limits = nan_safe_contrast_limits(map_ground_truth, fraction_vmax=0.8)

    groups = [
        ("Ground Truth", map_ground_truth, "green"),
        ("Solver", map_solver, "white"),
        ("Baseline", map_baseline, "orange"),
    ]

    layers = []
    n_layers_per_group = None

    for group_name, strain_image, vector_color in groups:
        layers.append(
            NapariImage(
                data=strain_image.array,
                name=f"Strain Rate ({group_name})",
                scale=strain_image.scale,
                translate=strain_image.translate,
                contrast_limits=contrast_limits,
                colormap="coolwarm",
                colorbar=True,
            )
        )
        # layers.append(
        #     NapariPoints(
        #         name=f"Sampled Points ({group_name})",
        #         face_color="red",
        #         size=0.2e-3,
        #         data=group_velocities.positions[:, [1, 0]],
        #     )
        # )
        if n_layers_per_group is None:
            n_layers_per_group = len(layers)
        strain_image.save(f"out/strain_map_{group_name.lower().replace(' ', '_')}.hdf5")

    viewer_state = ViewerState(
        layers=layers,
        settings=ViewerSettings(
            dims=DimsSettings(
                ndisplay=2, order=(0, 1), axis_labels=["frame", "z", "x"]
            ),
            axes=AxesSettings(visible=True),
            grid=GridSettings(enabled=True, shape=(1, 3), stride=n_layers_per_group),
        ),
    ).to_hdf5(out_dir / "viewer_state.hdf5")
    if not args.no_show:
        viewer_state.run()
