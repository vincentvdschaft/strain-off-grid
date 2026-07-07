from dataclasses import dataclass

import numpy as np
from napari import Viewer

from storepari.layer import Layer


@dataclass
class NapariImage(Layer):
    """
    Image data stored in tzyx format
    - tzyx
    """

    data: np.ndarray | None = None
    channel_axis: int | None = None
    affine: np.ndarray | None = None
    axis_labels: tuple | None = None
    attenuation: float = 0.05
    blending: str | None = None
    cache: bool = True
    colorbar: bool = False
    colormap: str | None = None
    contrast_limits: tuple | None = None
    custom_interpolation_kernel_2d: np.ndarray | None = None
    depiction: str = "volume"
    experimental_clipping_planes: list | None = None
    gamma: float = 1.0
    interpolation2d: str = "nearest"
    interpolation3d: str = "linear"
    iso_threshold: float | None = None
    metadata: dict | None = None
    multiscale: bool | None = None
    opacity: float = 1.0
    plane: dict | None = None
    projection_mode: str = "mean"
    rendering: str = "mip"
    rgb: bool | None = None
    rotate: np.ndarray | None = None
    scale: np.ndarray | None = None
    shear: np.ndarray | None = None
    translate: np.ndarray | None = None
    units: tuple | None = None
    visible: bool = True

    def add_to_viewer(self, viewer: Viewer, **kwargs) -> None:
        layer = viewer.add_image(
            self.data,
            name=self.name,
            channel_axis=self.channel_axis,
            affine=self.affine,
            axis_labels=self.axis_labels,
            attenuation=self.attenuation,
            blending=self.blending,
            cache=self.cache,
            colormap=self.colormap,
            contrast_limits=self.contrast_limits,
            custom_interpolation_kernel_2d=self.custom_interpolation_kernel_2d,
            depiction=self.depiction,
            experimental_clipping_planes=self.experimental_clipping_planes,
            gamma=self.gamma,
            interpolation2d=self.interpolation2d,
            interpolation3d=self.interpolation3d,
            iso_threshold=self.iso_threshold,
            metadata=self.metadata,
            multiscale=self.multiscale,
            opacity=self.opacity,
            plane=self.plane,
            projection_mode=self.projection_mode,
            rendering=self.rendering,
            rgb=self.rgb,
            rotate=self.rotate,
            scale=self.scale,
            shear=self.shear,
            translate=self.translate,
            units=self.units,
            visible=self.visible,
            **kwargs,
        )

        for image_layer in layer if isinstance(layer, list) else [layer]:
            image_layer.colorbar.visible = self.colorbar

    def __post_init__(self):
        self.type = "image"
