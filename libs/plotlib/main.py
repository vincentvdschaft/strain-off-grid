import matplotlib.pyplot as plt

from plotlib import (
    DimensionsSingleBesidesGrid,
    quickfig_single_besides_grid,
    remove_ticks,
)

fig, ax_single, axes_grid = quickfig_single_besides_grid(
    DimensionsSingleBesidesGrid.from_solve(
        grid_shape=(1, 3),
        fig_width=12,
        fig_height=None,
        margins_left=0.5,
        margins_right=0.5,
        margins_top=0.5,
        margins_bottom=0.5,
        single_axis_aspect=1,
        grid_axis_aspect=0.5,
        grid_vertical_spacing=0.1,
        # grid_spacings_equal=True,
        all_spacings_equal=True,
        margin_left_right_equal=False,
    )
)
ax_single.set_xlim(0, 12)
ax_single.set_ylim(0, 12)
remove_ticks(axes_grid)

# add_ruler(
#     ax=ax_single,
#     start=(5.5, 5.5),
#     end=(11.5, 0.5),
#     formatter=lambda d: f"{d:.1f} mm",
#     color="white",
#     linewidth=4.0,
#     label_side="above",
#     label_offset=0.4,
#     fontsize=12,
# )
plt.show()
