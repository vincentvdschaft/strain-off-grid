from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from imagelib import Image
from plotlib import (
    STYLE_PAPER,
    DimensionsSingleBesidesGrid,
    IntShape,
    flip_ylims,
    mm_formatter_ax,
    quickfig_single_besides_grid,
    remove_internal_labels,
    remove_internal_ticks,
    use_style,
)

from strain_off_grid.phantoms import load_dataclass
from strain_off_grid.strain import compute_rate_strain_curve
from strain_off_grid.utils import register_table_result


def smoothed_strain_curve(image, phantom, x, y):
    """Compute a smoothed strain-rate-versus-time curve at one location."""
    curve = compute_rate_strain_curve(
        strain_rate_map=image, phantom=phantom, position_at_frame_0=np.array([x, y])
    )
    return np.convolve(curve, np.ones(3) / 3, mode="same")


def register_curve_error(location, method, curve, ground_truth_curve):
    """Save the error of a method's curve against ground truth to the table json."""
    difference = curve - ground_truth_curve
    register_table_result(
        phantom="Ring",
        location=location,
        method=method,
        mean=float(np.nanmean(np.abs(difference))),
        std=float(np.nanstd(np.abs(difference))),
    )


def main():
    use_style(STYLE_PAPER)

    path_principal = "out/strain_maps_principal.hdf5"
    path_orthogonal = "out/strain_maps_orthogonal.hdf5"
    image_true = Image.load(path_principal, group="/ground_truth")
    image_solver = Image.load(path_principal, group="/solver")
    image_baseline = Image.load(path_principal, group="/baseline")
    image_true_orthogonal = Image.load(path_orthogonal, group="/ground_truth")
    image_solver_orthogonal = Image.load(path_orthogonal, group="/solver")
    image_baseline_orthogonal = Image.load(path_orthogonal, group="/baseline")
    phantom = load_dataclass(
        path_principal,
        group="/phantom",
    )
    window_half_size = 30e-3
    image_das = (
        Image.load("out/sweep/ring_00/simulated_phantom_ring-000.hdf5")
        .abs()
        .normalize()
        .log_compress()
        .clip(-60, 0)[0]
        .get_window(
            (
                (40e-3 - window_half_size, 40e-3 + window_half_size),
                (-window_half_size, window_half_size),
            )
        )
    )

    angles = [np.pi, np.pi / 2, 0]
    locations = [
        (np.cos(angle) * 20e-3, np.sin(angle) * 20e-3 + 40e-3) for angle in angles
    ]
    print(locations)
    fig, axes = plt.subplots(
        len(locations), 1, figsize=(6, 2 * len(locations)), sharex=True
    )

    dims = DimensionsSingleBesidesGrid.from_solve(
        grid_shape=IntShape(width=2, height=len(locations)),
        fig_width=5,
        margins_left=0.5,
        margins_right=0.4,
        margins_top=0.6,
        margins_bottom=0.4,
        single_axis_aspect=image_das.limits.aspect,
        single_axis_width=1.5,
        # grid_axis_width=4,
        # grid_axis_height=1.7,
        grid_vertical_spacing=0.1,
        middle_spacing=0.1,
    )
    print(dims)
    fig, ax_single, axes = quickfig_single_besides_grid(
        dimensions=dims, grid_on_right=False
    )

    directions = [
        (
            "circumferential",
            (image_solver, image_baseline, image_true),
        ),
        (
            "radial",
            (image_solver_orthogonal, image_baseline_orthogonal, image_true_orthogonal),
        ),
    ]

    curves = []

    for row, (x, y) in enumerate(locations):
        location_letter = chr(ord("A") + row)
        for col, (direction_label, direction_images) in enumerate(directions):
            ax = axes[row, col]
            image_solver_d, image_baseline_d, image_true_d = direction_images
            ground_truth_curve = smoothed_strain_curve(image_true_d, phantom, x, y)
            location = f"{location_letter} {direction_label}"
            for image, label, line_style in [
                (image_solver_d, "CDT (proposed)", "-x"),
                (image_baseline_d, "Speckle Tracking", "-o"),
                (image_true_d, "Ground truth", "--"),
            ]:
                curve_smoothed = smoothed_strain_curve(image, phantom, x, y)
                if label != "Ground truth":
                    register_curve_error(
                        location, label, curve_smoothed, ground_truth_curve
                    )
                t = np.linspace(
                    image_true.limits[0].min,
                    image_true.limits[0].max,
                    len(curve_smoothed),
                )
                (new_curve,) = ax.plot(
                    t,
                    curve_smoothed,
                    line_style,
                    label=label,
                    linewidth=0.7,
                    markersize=1,
                )
                curves.append(new_curve)

    for col, (direction_label, _) in enumerate(directions):
        axes[0, col].set_title(direction_label, fontsize=6)
        axes[-1, col].set_xlabel("Time [s]")
    axes[len(locations) // 2, 0].set_ylabel("Strain rate [1/s]", labelpad=10)

    for col in range(axes.shape[1]):
        for row in range(axes.shape[0]):
            ax = axes[row, col]
            bbox = fig.get_ax_bbox(ax)
            fig.add_text(
                bbox.x0 + 0.05,
                bbox.y0 + 0.05,
                f"{chr(ord('A') + row)}",
                fontsize=10,
                fontweight="bold",
                va="top",
                ha="left",
            )

    ax_single.imshow(
        image_das.array, extent=image_das.extent_imshow, cmap="gray", origin="lower"
    )
    flip_ylims(ax_single)
    mm_formatter_ax(ax_single)
    ax_single.set_xlabel("x [mm]")
    ax_single.set_ylabel("z [mm]")
    # Remove yticks and labels from the single axis
    # ax_single.set_yticks([])
    ax_single.yaxis.tick_right()
    ax_single.yaxis.set_label_position("right")

    for n, loc in enumerate(locations):
        ax_single.plot(loc[0], loc[1], "C3o", markersize=3)
        ax_single.text(
            loc[0] + 2e-3,
            loc[1] - 3e-3,
            f"{chr(ord('A') + n)}",
            color="white",
            fontsize=10,
            fontweight="bold",
            va="center",
            ha="center",
        )

    bbox = fig.get_ax_bbox(axes[0, 0])
    fig.add_legend(
        x=bbox.x0,
        y=0.1,
        width=1.0,
        height=0.05,
        labels=[c.get_label() for c in curves[:3]],
        handles=curves[:3],
        fontsize=7,
    )

    miccai_figures_dir = Path.home() / "1-projects/papers/miccai/figures"

    remove_internal_ticks(axes)
    remove_internal_labels(axes)
    for output_path in (
        "out/strainrate_curves.png",
        f"{miccai_figures_dir}/ring_curves.pdf",
    ):
        plt.savefig(output_path, dpi=300)
        print(f"Saved strain rate curves plot to {output_path}")
