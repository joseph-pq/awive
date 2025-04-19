"""Class ImageVelocimetry."""

import abc

from awive.loader import Loader


class ImageVelocimetry(abc.ABC):
    """Abstract class for image velocimetry algorithms."""

    @abc.abstractmethod
    def run(self, loader: Loader) -> None:
        """Run the algorithm."""
        raise NotImplementedError
