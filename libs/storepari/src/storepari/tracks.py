from dataclasses import dataclass

import numpy as np
from napari import Viewer

from storepari.layer import Layer


@dataclass
class NapariTracks(Layer):
    """
    Track data stored in id,t,z,y,x format
    - id,t,z,y,x
    """

    data: np.ndarray | None = None
    affine: np.ndarray | None = None
    axis_labels: tuple | None = None
    blending: str = "additive"
    cache: bool = True
    color_by: str = "track_id"
    colormap: str = "turbo"
    colormaps_dict: dict | None = None
    experimental_clipping_planes: list | None = None
    features: dict | None = None
    graph: dict | None = None
    head_length: int = 0
    hide_completed_tracks: bool = False
    metadata: dict | None = None
    opacity: float = 1.0
    projection_mode: str = "none"
    properties: dict | None = None
    rotate: np.ndarray | None = None
    scale: np.ndarray | None = None
    shear: np.ndarray | None = None
    tail_length: int = 30
    tail_width: int = 2
    translate: np.ndarray | None = None
    units: tuple | None = None
    visible: bool = True

    def add_to_viewer(self, viewer: Viewer, **kwargs) -> None:
        viewer.add_tracks(
            self.data,
            name=self.name,
            affine=self.affine,
            axis_labels=self.axis_labels,
            blending=self.blending,
            cache=self.cache,
            color_by=self.color_by,
            colormap=self.colormap,
            colormaps_dict=self.colormaps_dict,
            experimental_clipping_planes=self.experimental_clipping_planes,
            features=self.features,
            graph=self.graph,
            head_length=self.head_length,
            hide_completed_tracks=self.hide_completed_tracks,
            metadata=self.metadata,
            opacity=self.opacity,
            projection_mode=self.projection_mode,
            properties=self.properties,
            rotate=self.rotate,
            scale=self.scale,
            shear=self.shear,
            tail_length=self.tail_length,
            tail_width=self.tail_width,
            translate=self.translate,
            units=self.units,
            visible=self.visible,
            **kwargs,
        )

    def __post_init__(self):
        self.type = "tracks"
