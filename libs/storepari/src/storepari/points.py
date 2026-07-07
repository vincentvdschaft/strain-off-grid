from dataclasses import dataclass

import numpy as np
from napari import Viewer

from storepari.layer import Layer


@dataclass
class NapariPoints(Layer):
    """
    Points data stored in tzyx format
    - tzyx
    """

    data: np.ndarray | None = None
    ndim: int | None = None
    affine: np.ndarray | None = None
    antialiasing: float = 1
    axis_labels: tuple | None = None
    blending: str = "translucent"
    border_color: str | None = "dimgray"
    border_color_cycle: list | None = None
    border_colormap: str = "viridis"
    border_contrast_limits: tuple | None = None
    border_width: float = 0.05
    border_width_is_relative: bool = True
    cache: bool = True
    canvas_size_limits: tuple = (2, 10000)
    experimental_clipping_planes: list | None = None
    face_color: str | None = "white"
    face_color_cycle: list | None = None
    face_colormap: str = "viridis"
    face_contrast_limits: tuple | None = None
    feature_defaults: dict | None = None
    features: dict | None = None
    metadata: dict | None = None
    n_dimensional: bool | None = None
    opacity: float = 1.0
    out_of_slice_display: bool = False
    projection_mode: str = "all"
    properties: dict | None = None
    property_choices: dict | None = None
    rotate: np.ndarray | None = None
    scale: np.ndarray | None = None
    shading: str = "none"
    shear: np.ndarray | None = None
    shown: bool = True
    size: np.ndarray | float = 10
    symbol: str = "o"
    text: str | None = None
    translate: np.ndarray | None = None
    units: tuple | None = None
    visible: bool = True

    def add_to_viewer(self, viewer: Viewer, **kwargs) -> None:
        viewer.add_points(
            self.data,
            name=self.name,
            ndim=self.ndim,
            affine=self.affine,
            antialiasing=self.antialiasing,
            axis_labels=self.axis_labels,
            blending=self.blending,
            border_color=self.border_color,
            border_color_cycle=self.border_color_cycle,
            border_colormap=self.border_colormap,
            border_contrast_limits=self.border_contrast_limits,
            border_width=self.border_width,
            border_width_is_relative=self.border_width_is_relative,
            cache=self.cache,
            canvas_size_limits=self.canvas_size_limits,
            experimental_clipping_planes=self.experimental_clipping_planes,
            face_color=self.face_color,
            face_color_cycle=self.face_color_cycle,
            face_colormap=self.face_colormap,
            face_contrast_limits=self.face_contrast_limits,
            feature_defaults=self.feature_defaults,
            features=self.features,
            metadata=self.metadata,
            n_dimensional=self.n_dimensional,
            opacity=self.opacity,
            out_of_slice_display=self.out_of_slice_display,
            projection_mode=self.projection_mode,
            properties=self.properties,
            property_choices=self.property_choices,
            rotate=self.rotate,
            scale=self.scale,
            shading=self.shading,
            shear=self.shear,
            shown=self.shown,
            size=self.size,
            symbol=self.symbol,
            text=self.text,
            translate=self.translate,
            units=self.units,
            visible=self.visible,
            **kwargs,
        )

    def __post_init__(self):
        self.type = "points"
