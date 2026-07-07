from abc import ABC, abstractmethod
from dataclasses import dataclass

from napari import Viewer

from .saveable_dataclass import SaveableDataclass


@dataclass
class Layer(ABC, SaveableDataclass):
    name: str = ""
    type: str = ""
    description: str = ""

    @abstractmethod
    def add_to_viewer(self, viewer: Viewer, **kwargs) -> None:
        pass
