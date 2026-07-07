from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from imagelib import Image, Limits, LimitsND
from plotlib import (
    STYLE_PAPER,
    MPLFigure,
    box_fit_tick_labels,
    flip_ylims,
    mm_formatter_ax,
    remove_internal_labels,
    remove_internal_ticks,
    use_style,
)


def main():
    use_style(STYLE_PAPER)

    path = Path("out/strain_maps_invivo.hdf5")
    limits = LimitsND((Limits(10e-3, 100e-3), Limits(-30e-3, 50e-3)))

    strain_map_solver = Image.load(path, group="solver")[0].get_window(limits=limits)
    strain_map_baseline = Image.load(path, group="baseline")[0].get_window(
        limits=limits
    )
    im_das = (
        Image.load("source/beamformed_all.hdf5")[34]
        .abs()
        .normalize()
        .log_compress()
        .clip(-60, 0)
        .get_window(limits=limits)
    )

    print(f"Solver strain map: {strain_map_solver.limits}")
    print(f"Baseline strain map: {strain_map_baseline.limits}")
    print(f"DAS image: {im_das.limits}")

    fig_width = 5
    spacing = 0.05
    colorbar_width = 0.1

    margins_left = 0.5
    margins_right = 0.5
    margins_top = 0.5
    margins_bottom = 0.5

    ax_width = (
        fig_width - margins_left - margins_right - colorbar_width - spacing * 2
    ) / 2
    ax_height = ax_width / strain_map_solver.limits.aspect
    colorbar_x = margins_left + ax_width * 2 + spacing * 2

    fig_height = margins_top + ax_height + margins_bottom

    fig = MPLFigure(figsize=(fig_width, fig_height))
    axes = fig.add_axes_grid(
        n_rows=1,
        n_cols=2,
        x=margins_left,
        y=margins_top,
        width=ax_width,
        height=ax_height,
        spacing=spacing,
    ).ravel()

    vmax = 2
    fig.add_colorbar(
        x=colorbar_x,
        y=margins_top,
        width=colorbar_width,
        height=ax_height,
        cmap="coolwarm",
        vmin=-vmax,
        vmax=vmax,
        ticks=[-vmax, 0, vmax],
    )
    ax_cbar = fig.cbar_axes[0]

    ax_cbar.set_ylabel("Strain rate [1/s]")
    ax_cbar.yaxis.set_label_position("right")

    shared_kwargs = dict(
        origin="lower",
        aspect="equal",
        interpolation="nearest",
    )

    alpha = 0.6

    print(
        f"Solver strain map mean: {strain_map_solver.array[~np.isnan(strain_map_solver.array)].mean():.3f}"
    )

    axes[0].imshow(
        im_das.array,
        extent=im_das.extent_imshow,
        cmap="gray",
        vmin=-60,
        vmax=0,
        **shared_kwargs,
    )
    axes[0].imshow(
        strain_map_solver.array,
        extent=strain_map_solver.extent_imshow,
        cmap="coolwarm",
        alpha=alpha,
        vmin=-vmax,
        vmax=vmax,
        **shared_kwargs,
    )

    axes[1].imshow(
        im_das.array,
        extent=im_das.extent_imshow,
        cmap="gray",
        vmin=-60,
        vmax=0,
        **shared_kwargs,
    )
    axes[1].imshow(
        strain_map_baseline.array,
        cmap="coolwarm",
        vmin=-vmax,
        vmax=vmax,
        alpha=alpha,
        extent=strain_map_baseline.extent_imshow,
        **shared_kwargs,
    )
    for ax in axes:
        flip_ylims(ax)
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("z [mm]")

    axes[0].set_title("CDT (proposed)")
    axes[1].set_title("Speckle Tracking")

    mm_formatter_ax(axes)
    remove_internal_ticks(axes)
    remove_internal_labels(axes)
    box_fit_tick_labels(axes)
    plt.savefig("out/strain_rate_maps_in_vivo.png", dpi=300)
    miccai_figures_dir = Path.home() / "1-projects/papers/miccai/figures"
    plt.savefig(
        f"{miccai_figures_dir}/strain_rate_maps_in_vivo.pdf",
        dpi=300,
    )
    plt.show()


if __name__ == "__main__":
    main()
