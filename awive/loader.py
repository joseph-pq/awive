"""Loader of videos of frames."""

import abc
import argparse
import os
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from numpy.typing import NDArray

from awive.config import Dataset as DatasetConfig, Config

FOLDER_PATH = "/home/joseph/Documents/Thesis/Dataset/config"


class Loader(metaclass=abc.ABCMeta):
    """Abstract class of loader.

    Args:
        config: Configuration for the dataset.
    """

    def __init__(self, config: DatasetConfig) -> None:
        self._offset: int = config.image_number_offset
        self._index: int = 0
        self.config = config
        self.current_image: NDArray | None = None
        self.fps: int = 1
        self.total_frames = 0

    @property
    @abc.abstractmethod
    def width(self) -> int: ...

    @property
    @abc.abstractmethod
    def height(self) -> int: ...

    @property
    def index(self) -> int:
        """Get the current index.

        Returns:
            The current index as an integer.
        """
        return self._index

    @abc.abstractmethod
    def has_images(self) -> bool:
        """Check if the source contains more frames.

        Returns:
            True if there are more frames, False otherwise.
        """

    @abc.abstractmethod
    def read(self) -> np.ndarray | None:
        """Read a new image from the source.

        Returns:
            The next image as a numpy array, or None if no image is available.
        """

    @abc.abstractmethod
    def end(self) -> None:
        """Free all resources."""


class ImageLoader(Loader):
    """Loader that loads images from a directory."""

    def __init__(self, config: DatasetConfig) -> None:
        """Initialize loader."""
        super().__init__(config)
        if config.image_dataset_dp is None:
            raise ValueError("Image dataset path not provided")
        self._image_dataset: Path = config.image_dataset_dp
        self._prefix = config.image_path_prefix
        self._digits = config.image_path_digits
        self._image_number = len(os.listdir(self._image_dataset))
        # Read first image
        img = self.read()
        if img is None:
            raise FileNotFoundError(
                f"Image not found: {self._path(self._index)}"
            )
        self.set_index(0)  # Reset index
        self._width = img.shape[1]
        self._height = img.shape[0]

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    def has_images(self) -> bool:
        """Check if the source contains one more frame."""
        return self._index < self._image_number

    def _path(self, i: int) -> str:
        i += self._offset
        if self._digits == 5:
            return f"{self._image_dataset}/{self._prefix}{i:05}.jpg"
        if self._digits == 3:
            return f"{self._image_dataset}/{self._prefix}{i:03}.jpg"
        return f"{self._image_dataset}/{self._prefix}{i:04}.jpg"

    def set_index(self, index: int) -> None:
        """Set the index of the loader to read any image from the folder.

        Args:
            index: The index to set.
        """
        self._index = index

    def read(self) -> np.ndarray | None:
        """Read a new image from the source."""
        self._index += 1
        path: str = self._path(self._index)
        if not Path(path).exists():
            raise FileNotFoundError(f"Image not found: {path}")
        return cv2.imread(self._path(self._index))

    def read_iter(self) -> Iterable[np.ndarray]:
        """Read a new image from the source."""
        self._index += 1
        path: str = self._path(self._index)
        if Path(path).exists():
            yield cv2.imread(self._path(self._index))

    def end(self) -> None:
        """Free all resources."""
        pass


class VideoLoader(Loader):
    """Loader that loads from a video."""

    def __init__(self, config: DatasetConfig) -> None:
        """Initialize loader."""
        super().__init__(config)

        # check if config.video_path exists
        if config.video_fp is None:
            raise ValueError("Video path not provided")
        if not config.video_fp.exists():
            raise FileNotFoundError(f"Video not found: {config.video_fp}")

        self._cap: cv2.VideoCapture = cv2.VideoCapture(
            str(self.config.video_fp)
        )  # type: ignore[call-arg]
        self._image_read: bool = False  # Check if the current images was read

        # Get number of frames
        cap: cv2.VideoCapture = cv2.VideoCapture(str(self.config.video_fp))  # type: ignore[call-arg]
        property_id: int = int(cv2.CAP_PROP_FRAME_COUNT)
        self.total_frames = int(cv2.VideoCapture.get(cap, property_id)) + 1
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        self._width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Skip offset
        for _ in range(self._offset + 1):
            if self.has_images():
                self.read()

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    def has_images(self):
        """Check if the source contains one more frame."""
        if not self._cap.isOpened():
            return False
        ret, self.current_image = self._cap.read()
        self._image_read = False
        return ret

    def read(self) -> np.ndarray | None:
        """Read a new image from the source."""
        self._index += 1
        if self._image_read:
            ret, self.current_image = self._cap.read()
            if not ret:
                print("error at reading")
        self._image_read = True
        return self.current_image

    def end(self) -> None:
        """Free all resources."""
        self._cap.release()


def make_loader(config: DatasetConfig):
    """Make a loader based on config."""
    # check if the image_folder_path contains any jpg or png file
    if config.image_dataset_dp is not None:
        for file in config.image_dataset_dp.iterdir():
            if file.suffix in (".jpg", ".png"):
                return ImageLoader(config)

    return VideoLoader(config)


def get_loader(config_fp: Path) -> Loader:
    """Return a ImageLoader or VideoLoader class.

    Args:
        config_path: Path to the config file.
        video_identifier: Identifier of the video in the config file.
    """
    return make_loader(Config.from_fp(config_fp).dataset)


def main(config_path: Path, video_identifier: str, save_image: bool):
    """Execute a basic example of loader."""
    loader = get_loader(config_path)
    image = loader.read()
    if image is None:
        print("No image found")
        return
    if save_image:
        cv2.imwrite("tmp.jpg", image)
    else:
        cv2.imshow("image", cv2.resize(image, (1000, 1000)))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    loader.end()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "statio_name", help="Name of the station to be analyzed"
    )
    parser.add_argument(
        "video_identifier", help="Index of the video of the json config file"
    )
    parser.add_argument(
        "-s",
        "--save",
        help="Save images instead of showing",
        action="store_true",
    )
    parser.add_argument(
        "-p",
        "--path",
        help="Path to the config folder",
        type=str,
        default=FOLDER_PATH,
    )

    args = parser.parse_args()
    CONFIG_PATH = Path(f"{args.path}/{args.statio_name}.json")
    main(
        config_path=CONFIG_PATH,
        video_identifier=args.video_identifier,
        save_image=args.save,
    )
