"""Loader of videos of frames."""

import abc
import argparse
import json
import os
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from numpy.typing import NDArray

from awive.config import ConfigDataset

FOLDER_PATH = '/home/joseph/Documents/Thesis/Dataset/config'


class Loader(metaclass=abc.ABCMeta):
    """Abstract class of loader."""

    def __init__(self, config: ConfigDataset) -> None:
        """Initialize loader."""
        self._offset: int = config.image_number_offset
        self._index: int = 0
        self.config = config
        self.current_image: NDArray | None = None
        self.fps: int = 1
        self.total_frames = 0

    @property
    def image_shape(self):
        """Return the shape of the images."""
        return (self.config.width, self.config.height)

    @property
    def index(self) -> int:
        """Index getter."""
        return self._index

    @abc.abstractmethod
    def has_images(self) -> bool:
        """Check if the source contains one more frame."""

    @abc.abstractmethod
    def read(self) -> cv2.typing.MatLike | None:
        """Read a new image from the source."""

    @abc.abstractmethod
    def end(self) -> None:
        """Free all resources."""


class ImageLoader(Loader):
    """Loader that loads images from a directory."""

    def __init__(self, config: ConfigDataset) -> None:
        """Initialize loader."""
        super().__init__(config)
        self._image_dataset = config.image_dataset
        self._prefix = config.image_path_prefix
        self._digits = config.image_path_digits
        self._image_number = len(os.listdir(self._image_dataset))

    def has_images(self) -> bool:
        """Check if the source contains one more frame."""
        return self._index < self._image_number

    def _path(self, i: int) -> str:
        i += self._offset
        if self._digits == 5:
            return f'{self._image_dataset}/{self._prefix}{i:05}.jpg'
        if self._digits == 3:
            return f'{self._image_dataset}/{self._prefix}{i:03}.jpg'
        return f'{self._image_dataset}/{self._prefix}{i:04}.jpg'

    def set_index(self, index: int) -> None:
        """Set index of the loader to read any image from the folder."""
        self._index = index

    def read(self) -> cv2.typing.MatLike | None:
        """Read a new image from the source."""
        self._index += 1
        path: str = self._path(self._index)
        if not Path(path).exists():
            raise FileNotFoundError(f'Image not found: {path}')
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

    def __init__(self, config: ConfigDataset) -> None:
        """Initialize loader."""
        super().__init__(config)

        # check if config.video_path exists
        if not Path(config.video_path).exists():
            raise FileNotFoundError(f'Video not found: {config.video_path}')

        self._cap: cv2.VideoCapture = cv2.VideoCapture(self.config.video_path)
        self._image_read: bool = False  # Check if the current images was read

        # Get number of frames
        cap: cv2.VideoCapture = cv2.VideoCapture(self.config.video_path)
        property_id: int = int(cv2.CAP_PROP_FRAME_COUNT)
        self.total_frames = int(cv2.VideoCapture.get(cap, property_id)) + 1
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Skip offset
        for _ in range(self._offset+1):
            if self.has_images():
                self.read()

    def has_images(self):
        """Check if the source contains one more frame."""
        if not self._cap.isOpened():
            return False
        ret, self.current_image = self._cap.read()
        self._image_read = False
        return ret

    def read(self) -> cv2.typing.MatLike | None:
        """Read a new image from the source."""
        self._index += 1
        if self._image_read:
            ret, self.current_image = self._cap.read()
            if not ret:
                print('error at reading')
        self._image_read = True
        return self.current_image

    def end(self) -> None:
        """Free all resources."""
        self._cap.release()


def make_loader(config: ConfigDataset):
    """Make a loader based on config."""
    # check if the image_folder_path contains any jpg or png file
    for file in Path(config.image_dataset).iterdir():
        if file.suffix in ('.jpg', '.png'):
            return ImageLoader(config)

    return VideoLoader(config)


def get_loader(config_path: str, video_identifier: str) -> Loader:
    """Return a ImageLoader or VideoLoader class.

    :return image_loader: if the image_dataset has any image jpg or png
    :return video_loader: if the previous assumption is not true
    """
    # check if in image folder there are located the extracted images
    config = ConfigDataset(
        **json.loads(
            Path(config_path).read_text()
        )[video_identifier]['dataset']
    )
    return make_loader(config)


def main(config_path: str, video_identifier: str, save_image: bool):
    """Execute a basic example of loader."""
    loader = get_loader(config_path, video_identifier)
    image = loader.read()
    if save_image:
        cv2.imwrite('tmp.jpg', image)
    else:
        cv2.imshow('image', cv2.resize(image, (1000, 1000)))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    loader.end()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "statio_name",
        help="Name of the station to be analyzed")
    parser.add_argument(
        "video_identifier",
        help="Index of the video of the json config file")
    parser.add_argument(
        '-s',
        '--save',
        help='Save images instead of showing',
        action='store_true')
    parser.add_argument(
        '-p',
        '--path',
        help='Path to the config folder',
        type=str,
        default=FOLDER_PATH)

    args = parser.parse_args()
    CONFIG_PATH = f'{args.path}/{args.statio_name}.json'
    main(
        config_path=CONFIG_PATH,
        video_identifier=args.video_identifier,
        save_image=args.save
    )
