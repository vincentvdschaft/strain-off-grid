from dataclasses import dataclass

import numpy as np
from napari import Viewer

from storepari.layer import Layer


@dataclass
class NapariVectors(Layer):
    """
    Vector data stored in (N, 2, D) format
    - N vectors, each with a start point and a projection, in tzyx dims
    """

    data: np.ndarray | None = None
    affine: np.ndarray | None = None
    axis_labels: tuple | None = None
    blending: str = "translucent"
    cache: bool = True
    edge_color: str = "red"
    edge_color_cycle: list | None = None
    edge_colormap: str = "viridis"
    edge_contrast_limits: tuple | None = None
    edge_width: float = 1
    experimental_clipping_planes: list | None = None
    feature_defaults: dict | None = None
    features: dict | None = None
    length: float = 1
    metadata: dict | None = None
    ndim: int | None = None
    opacity: float = 0.7
    out_of_slice_display: bool = False
    projection_mode: str = "all"
    properties: dict | None = None
    property_choices: dict | None = None
    rotate: np.ndarray | None = None
    scale: np.ndarray | None = None
    shear: np.ndarray | None = None
    translate: np.ndarray | None = None
    units: tuple | None = None
    vector_style: str = "triangle"
    visible: bool = True

    def add_to_viewer(self, viewer: Viewer, **kwargs) -> None:
        viewer.add_vectors(
            self.data,
            name=self.name,
            affine=self.affine,
            axis_labels=self.axis_labels,
            blending=self.blending,
            cache=self.cache,
            edge_color=self.edge_color,
            edge_color_cycle=self.edge_color_cycle,
            edge_colormap=self.edge_colormap,
            edge_contrast_limits=self.edge_contrast_limits,
            edge_width=self.edge_width,
            experimental_clipping_planes=self.experimental_clipping_planes,
            feature_defaults=self.feature_defaults,
            features=self.features,
            length=self.length,
            metadata=self.metadata,
            ndim=self.ndim,
            opacity=self.opacity,
            out_of_slice_display=self.out_of_slice_display,
            projection_mode=self.projection_mode,
            properties=self.properties,
            property_choices=self.property_choices,
            rotate=self.rotate,
            scale=self.scale,
            shear=self.shear,
            translate=self.translate,
            units=self.units,
            vector_style=self.vector_style,
            visible=self.visible,
            **kwargs,
        )

    def __post_init__(self):
        self.type = "vectors"
