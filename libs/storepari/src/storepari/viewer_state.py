from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import napari
import pint
from napari import Viewer
from napari.components.dims import RangeTuple

from storepari import SaveableDataclass, load_dataclass


@dataclass
class ViewerState(SaveableDataclass):
    """
    Represents the state of a napari viewer, including its layers and settings.
    """

    layers: list | None = None
    settings: ViewerSettings | None = None

    def apply(self, viewer: Viewer) -> None:
        """
        Apply the state to a napari viewer.
        """
        if self.layers is not None:
            for layer in self.layers:
                layer.add_to_viewer(viewer)

        if self.settings is not None:
            self.settings.apply(viewer)

    @classmethod
    def load(cls, path: str, group: str = "/") -> ViewerState:
        """
        Load a ViewerState from an HDF5 file.
        """
        return load_dataclass(path, group)

    def run(self) -> None:
        """
        Create a napari viewer, apply the state, and run the event loop.
        """
        viewer = Viewer()
        self.apply(viewer)
        napari.run()


@dataclass
class ViewerSettings(SaveableDataclass):
    """
    Represents the settings of a napari viewer, such as contrast limits, colormaps, etc.
    """

    dims: DimsSettings | None = None
    axes: AxesSettings | None = None
    grid: GridSettings | None = None
    camera: CameraSettings | None = None

    def apply(self, viewer: Viewer) -> None:
        """
        Apply the settings to a napari viewer.
        """
        if self.dims is not None:
            self.dims.apply(viewer)

        if self.axes is not None:
            self.axes.apply(viewer)

        if self.grid is not None:
            self.grid.apply(viewer)

        if self.camera is not None:
            self.camera.apply(viewer)


@dataclass
class DimsSettings(SaveableDataclass):
    """
    Represents the settings of a napari viewer, such as contrast limits, colormaps, etc.
    """

    ndim: int | None = None
    ndisplay: Literal[2, 3] | None = None

    order: tuple[int, ...] | None = None
    axis_labels: tuple[str, ...] | None = None
    rollable: tuple[bool, ...] | None = None

    range: tuple[RangeTuple, ...] | None = None
    margin_left: tuple[float, ...] | None = None
    margin_right: tuple[float, ...] | None = None
    point: tuple[float, ...] | None = None
    units: tuple[pint.Unit, ...] | None = None

    def apply(self, viewer: Viewer) -> None:
        """
        Apply the settings to a napari viewer.
        """
        self.set_if_not_none("ndim", self.ndim, viewer)
        self.set_if_not_none("ndisplay", self.ndisplay, viewer)

        self.set_if_not_none("order", self.order, viewer)
        self.set_if_not_none("axis_labels", self.axis_labels, viewer)
        self.set_if_not_none("rollable", self.rollable, viewer)

        self.set_if_not_none("range", self.range, viewer)
        self.set_if_not_none("margin_left", self.margin_left, viewer)
        self.set_if_not_none("margin_right", self.margin_right, viewer)
        self.set_if_not_none("point", self.point, viewer)

    @staticmethod
    def set_if_not_none(attr: str, value: Any, viewer: Viewer) -> None:
        """
        Set an attribute of the viewer if the value is not None.
        """
        if value is not None:
            setattr(viewer.dims, attr, value)


@dataclass
class AxesSettings(SaveableDataclass):
    visible: bool | None = None
    opacity: float | None = None
    order: int | None = None
    blending: Literal["translucent", "additive", "opaque"] | None = None
    labels: bool | None = None
    colored: bool | None = None
    dashed: bool | None = None

    def apply(self, viewer: Viewer) -> None:
        """
        Apply the axes settings to a napari viewer.
        """
        self.set_if_not_none("visible", self.visible, viewer)
        self.set_if_not_none("opacity", self.opacity, viewer)
        self.set_if_not_none("order", self.order, viewer)
        self.set_if_not_none("blending", self.blending, viewer)
        self.set_if_not_none("labels", self.labels, viewer)
        self.set_if_not_none("colored", self.colored, viewer)
        self.set_if_not_none("dashed", self.dashed, viewer)

    def set_if_not_none(self, attr: str, value: Any, viewer: Viewer) -> None:
        """
        Set an attribute of the viewer's axes if the value is not None.
        """
        if value is not None:
            setattr(viewer.axes, attr, value)


@dataclass
class GridSettings(SaveableDataclass):
    """
    Represents the grid view settings of a napari viewer.
    """

    enabled: bool | None = None
    stride: int | None = None
    shape: tuple[int, int] | None = None
    spacing: float | None = None

    def apply(self, viewer: Viewer) -> None:
        """
        Apply the grid settings to a napari viewer.
        """
        self.set_if_not_none("stride", self.stride, viewer)
        self.set_if_not_none("shape", self.shape, viewer)
        self.set_if_not_none("spacing", self.spacing, viewer)
        self.set_if_not_none("enabled", self.enabled, viewer)

    @staticmethod
    def set_if_not_none(attr: str, value: Any, viewer: Viewer) -> None:
        """
        Set an attribute of the viewer's grid if the value is not None.
        """
        if value is not None:
            setattr(viewer.grid, attr, value)


@dataclass
class CameraSettings(SaveableDataclass):
    """
    Represents the camera settings of a napari viewer.
    """

    center: tuple[float, float, float] | tuple[float, float] | None = None
    zoom: float | None = None
    angles: tuple[float, float, float] | None = None
    perspective: float | None = None
    mouse_pan: bool | None = None
    mouse_zoom: bool | None = None
    orientation: tuple[str, str, str] | None = None

    def apply(self, viewer: Viewer) -> None:
        """
        Apply the camera settings to a napari viewer.
        """
        self.set_if_not_none("center", self.center, viewer)
        self.set_if_not_none("zoom", self.zoom, viewer)
        self.set_if_not_none("angles", self.angles, viewer)
        self.set_if_not_none("perspective", self.perspective, viewer)
        self.set_if_not_none("mouse_pan", self.mouse_pan, viewer)
        self.set_if_not_none("mouse_zoom", self.mouse_zoom, viewer)
        self.set_if_not_none("orientation", self.orientation, viewer)

    @staticmethod
    def set_if_not_none(attr: str, value: Any, viewer: Viewer) -> None:
        """
        Set an attribute of the viewer's camera if the value is not None.
        """
        if value is not None:
            setattr(viewer.camera, attr, value)
