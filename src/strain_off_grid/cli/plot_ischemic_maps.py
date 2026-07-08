import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from imagelib import Image, LimitsND
from plotlib import (
    STYLE_PAPER,
    MPLFigure,
    flip_ylims,
    mm_formatter_ax,
    remove_internal_labels,
    remove_internal_ticks,
    use_style,
)

from strain_off_grid.phantoms import load_dataclass
from strain_off_grid.strain import compute_rate_strain_curve
from strain_off_grid.utils import register_table_result

DAS_PATH = "out/sweep/ischemic_00/simulated_phantom_ischemic-000.hdf5"
STRAIN_CMAP = "coolwarm"
CONTRAST_FRACTION = 0.8
METHODS = [
    ("CDT (proposed)", "solver", "C0", "-x"),
    ("Speckle Tracking", "baseline", "C1", "-o"),
    ("Ground truth", "ground_truth", "C2", "--"),
]
LIMITS = LimitsND((20e-3, 120e-3, -45e-3, 45e-3))

POINT_A_BOX_INDEX = 2
POINT_B_BOX_INDEX = 1


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot ischemic maps")
    parser.add_argument("--source-dir", default="out/ischemic", nargs="?")
    parser.add_argument("--das-path", default=DAS_PATH, nargs="?")
    return parser.parse_args()


def load_strain_images(path: Path) -> dict[str, Image]:
    """Load the strain-rate map of every method from an hdf5 file."""
    return {
        label: Image.load(path, group=f"/{group}") for label, group, _, _ in METHODS
    }


def load_das_image(path: str, plane_limits) -> Image:
    """Load and log-compress the delay-and-sum image, cropped to the map plane."""
    return (
        Image.load(path)
        .abs()
        .normalize()
        .log_compress()
        .clip(-60, 0)[0]
        .get_window(LIMITS)
    )


def peak_strain_frame(image: Image) -> int:
    """Return the frame index with the largest median absolute strain rate."""
    per_frame_strength = np.nanmedian(np.abs(image.array), axis=(1, 2))
    return int(np.nanargmax(per_frame_strength))


def contrast_limit(image: Image) -> float:
    """Return a robust symmetric colour limit for the strain-rate colormap."""
    return (
        round(CONTRAST_FRACTION * float(np.nanpercentile(np.abs(image.array), 99)) * 10)
        / 10
    )


@dataclass
class Layout:
    """Positions and sizes (in inches) of every axis in the figure."""

    figure_size: tuple[float, float]
    margin_left: float
    margin_top: float
    top_width: float
    top_height: float
    top_spacing: float
    colorbar_gap: float
    colorbar_width: float
    middle_gap: float
    curve_height: float
    curve_spacing: float

    @property
    def grid_width(self) -> float:
        return 4 * self.top_width + 3 * self.top_spacing

    def top_x(self, column: int) -> float:
        return self.margin_left + column * (self.top_width + self.top_spacing)

    @property
    def colorbar_x(self) -> float:
        return self.margin_left + self.grid_width + self.colorbar_gap

    @property
    def curve_y(self) -> float:
        return self.margin_top + self.top_height + self.middle_gap

    def curve_row_y(self, row: int) -> float:
        return self.curve_y + row * (self.curve_height + self.curve_spacing)


def compute_layout(
    figure_width: float, image_aspect: float, lineplot_height=0.5
) -> Layout:
    """Derive all axis sizes from the figure width and the image aspect ratio."""
    margin_left, margin_right = 0.6, 0.45
    margin_top, margin_bottom = 0.5, 0.5
    colorbar_width, colorbar_gap = 0.1, 0.05
    grid_spacing = 0.1
    spacing_between_curves = 0.1
    spacing_grid_curves = 0.2
    image_width = (
        figure_width
        - margin_left
        - margin_right
        - 3 * grid_spacing
        - colorbar_width
        - colorbar_gap
    ) / 4
    image_height = image_width / image_aspect

    figure_height = (
        image_height
        + spacing_grid_curves
        + 2 * lineplot_height
        + spacing_between_curves
        + margin_top
        + margin_bottom
    )

    return Layout(
        figure_size=(figure_width, figure_height),
        margin_left=margin_left,
        margin_top=margin_top,
        top_width=image_width,
        top_height=image_height,
        top_spacing=grid_spacing,
        colorbar_gap=colorbar_gap,
        colorbar_width=colorbar_width,
        middle_gap=spacing_grid_curves,
        curve_height=lineplot_height,
        curve_spacing=spacing_between_curves,
    )


def add_top_axes(figure: MPLFigure, layout: Layout) -> list:
    """Create the four square image axes across the top of the figure."""
    axes = []
    for column in range(4):
        ax = figure.add_ax(
            x=layout.top_x(column),
            y=layout.margin_top,
            width=layout.top_width,
            height=layout.top_height,
        )
        axes.append(ax)
    return axes


def add_curve_axes(figure: MPLFigure, layout: Layout) -> list:
    """Create the two full-width strain-rate curve axes at the bottom."""
    axes = []
    for row in range(2):
        ax = figure.add_ax(
            x=layout.margin_left,
            y=layout.curve_row_y(row),
            width=layout.grid_width + layout.colorbar_gap + layout.colorbar_width,
            height=layout.curve_height,
        )
        axes.append(ax)
    return axes


def show_grayscale(ax, image: Image) -> None:
    """Draw a grayscale image on an axis with millimeter axes."""
    ax.imshow(image.array, extent=image.extent_imshow, cmap="gray", origin="lower")
    print(image.extent_imshow)
    flip_ylims(ax)
    mm_formatter_ax(ax)


def show_strain_map(ax, image: Image, vmax: float) -> None:
    """Draw a single strain-rate map frame with the shared color limits."""
    image = image.get_window(LIMITS)
    ax.imshow(
        image.array,
        extent=image.extent_imshow,
        cmap=STRAIN_CMAP,
        origin="lower",
        vmin=-vmax,
        vmax=vmax,
    )
    print(image.extent_imshow)
    flip_ylims(ax)
    mm_formatter_ax(ax)


def box_corners(endpoints: np.ndarray, box: int) -> np.ndarray:
    """Return the four corners of a chained box, closed for plotting."""
    inner_start, inner_end = endpoints[box, 0], endpoints[box + 1, 0]
    outer_end, outer_start = endpoints[box + 1, 1], endpoints[box, 1]
    return np.array([inner_start, inner_end, outer_end, outer_start, inner_start])


def box_center(phantom, box: int) -> np.ndarray:
    """Return the (x, z) center of a chained box at time 0."""
    endpoints = np.asarray(phantom.box_endpoints)[0]
    return box_corners(endpoints, box)[:4].mean(axis=0)


def draw_phantom_boxes(ax, phantom, color: str = "C0") -> None:
    """Overlay every quadrilateral of the chained-boxes phantom on an axis."""
    endpoints = np.asarray(phantom.box_endpoints)[0]
    for box in range(len(endpoints) - 1):
        corners = box_corners(endpoints, box)
        ax.plot(corners[:, 0], corners[:, 1], color=color, linewidth=0.6)


def mark_location(ax, location, letter: str) -> None:
    """Mark a curve sampling location with a labelled point."""
    ax.plot(location[0], location[1], "C3o", markersize=1)
    ax.text(
        location[0] + 2e-3,
        location[1],
        letter,
        color="white",
        fontsize=9,
        fontweight="bold",
        va="center",
    )


def strain_rate_curve(image: Image, phantom, location) -> np.ndarray:
    """Compute a smoothed strain-rate-versus-time curve at one location."""
    curve = compute_rate_strain_curve(
        strain_rate_map=image,
        phantom=phantom,
        position_at_frame_0=np.array([location[0], location[1]]),
    )
    return np.convolve(curve, np.ones(3) / 3, mode="same")


def plot_curves_at_location(
    ax, images: dict, phantom, location, location_letter
) -> list:
    """Plot every method's strain-rate curve at a location on one axis."""
    reference = next(iter(images.values()))
    time = np.linspace(
        reference.limits[0].min, reference.limits[0].max, reference.shape[0]
    )
    lines = []
    ground_truth_curve = strain_rate_curve(images["Ground truth"], phantom, location)
    for label, _, color, line_style in METHODS:
        curve = strain_rate_curve(images[label], phantom, location)
        if label != "Ground truth":
            register_table_result(
                phantom="ischemic",
                location=location_letter,
                method=label,
                mean=float(np.nanmean(np.abs(curve - ground_truth_curve))),
                std=float(np.nanstd(np.abs(curve - ground_truth_curve))),
            )
        (line,) = ax.plot(
            time,
            curve,
            line_style,
            label=label,
            color=color,
            linewidth=0.7,
            markersize=1,
        )
        lines.append(line)
    return lines


def draw_top_row(figure, layout, das_image, strain_frames, phantom, vmax) -> list:
    """Fill the top row with the DAS image and the three strain-rate maps."""
    top_axes = add_top_axes(figure, layout)
    show_grayscale(top_axes[0], das_image)
    draw_phantom_boxes(top_axes[0], phantom)
    top_axes[0].set_title("B-mode", fontsize=7)

    for ax, (label, _, color, _) in zip(
        np.array(top_axes[1:])[np.array((1, 2, 0))], METHODS
    ):
        show_strain_map(ax, strain_frames[label], vmax)
        ax.set_title(label, fontsize=7)
    return top_axes


def draw_curve_rows(
    figure: MPLFigure, layout, curve_images, phantom, locations
) -> list:
    """Fill the bottom rows with strain-rate curves at the two locations."""
    curve_axes = add_curve_axes(figure, layout)
    lines = []
    for ax, location, letter in zip(curve_axes, locations, "AB"):
        lines = plot_curves_at_location(ax, curve_images, phantom, location, letter)
        ax.set_ylabel("Strain\nrate [1/s]", fontsize=6)
        x, y = figure.get_ax_position(ax)
        bbox = figure.get_ax_bbox(ax)
        figure.add_text(
            x + 0.05,
            y + bbox.height - 0.05,
            f"Location {letter}",
            fontsize=7,
            color="k",
            fontweight="bold",
            va="bottom",
        )
    curve_axes[0].set_xticklabels([])
    curve_axes[-1].set_xlabel("Time [s]")
    curve_axes[0].legend(handles=lines, fontsize=6, loc="lower right")
    return lines


def add_strain_colorbar(figure, layout, vmax) -> None:
    """Add a shared colorbar for the strain-rate maps to the right of the grid."""
    figure.add_colorbar(
        x=layout.colorbar_x,
        y=layout.margin_top,
        width=layout.colorbar_width,
        height=layout.top_height,
        cmap=STRAIN_CMAP,
        vmin=-vmax,
        vmax=vmax,
        ticks=[-vmax, 0, vmax],
    )
    ax = figure.cbar_axes[0]
    ax.set_ylabel("Strain rate [1/s]")
    ax.yaxis.set_label_position("right")
    ax.set_yticks([-vmax, 0, vmax])


def main() -> None:
    arguments = parse_arguments()
    use_style(STYLE_PAPER)

    source_dir = Path(arguments.source_dir)
    path_principal = source_dir / "strain_maps_principal.hdf5"
    strain_images = load_strain_images(path_principal)
    phantom = load_dataclass(path_principal, group="/phantom")

    frame = peak_strain_frame(strain_images["Ground truth"])
    strain_frames = {label: image[frame] for label, image in strain_images.items()}
    plane_limits = strain_frames["Ground truth"].limits
    vmax = contrast_limit(strain_images["Ground truth"])

    das_image = load_das_image(arguments.das_path, plane_limits).get_window(
        plane_limits
    )

    curve_locations = [
        box_center(phantom, POINT_A_BOX_INDEX),
        box_center(phantom, POINT_B_BOX_INDEX),
    ]

    layout = compute_layout(figure_width=5, image_aspect=das_image.limits.aspect)
    figure = MPLFigure(figsize=layout.figure_size)

    top_axes = draw_top_row(figure, layout, das_image, strain_frames, phantom, vmax)
    add_strain_colorbar(figure, layout, vmax)
    for letter, location in zip("AB", curve_locations):
        mark_location(top_axes[0], location, letter)

    draw_curve_rows(figure, layout, strain_images, phantom, curve_locations)
    remove_internal_ticks(top_axes)
    remove_internal_labels(top_axes)
    miccai_figures_dir = Path.home() / "1-projects/papers/miccai/figures"

    for output_path in (
        source_dir / "ischemic_maps.png",
        source_dir / "ischemic_maps.pdf",
        miccai_figures_dir / "ischemic_maps.pdf",
    ):
        figure.savefig(output_path, dpi=300)
        print(f"Saved ischemic maps figure to {output_path}")


if __name__ == "__main__":
    main()
