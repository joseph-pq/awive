"""Correct distortion of videos.

This module contains classes and functions needed to correct distortion of
videos, either intrinsic or extrinsic to the camera. It also saves the corrected
frames in a defined directory path.

"""

from pathlib import Path
import logging
import time
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

import awive.preprocess.imageprep as ip
from awive.config import Config
from awive.exceptions import VideoSourceError
from awive.loader import make_loader
from awive.tools import imshow

LOG = logging.getLogger(__name__)


class Formatter:
    """Format frames in order to be used by image processing methods."""

    def __init__(self, config: Config) -> None:
        """Initialize Formatter object.

        Args:
            config: Configuration object containing settings for processing.

        Raises:
            VideoSourceError: If no sample image is found.
        """
        # read configuration file
        self._config: Config = config
        self.resolution = config.preprocessing.resolution
        self.ppm = config.preprocessing.ppm
        sample_image = self._get_sample_image(self._config)
        if sample_image is None:
            raise VideoSourceError("No sample image found")
        self._shape = (sample_image.shape[0], sample_image.shape[1])
        if self._config.dataset.gcp.apply:
            self._or_params: tuple[Any, np.ndarray] | None = (
                self._get_orthorectification_params(sample_image)
            )
        else:
            self._or_params = None

        self._rotation_angle = self._config.preprocessing.rotate_image
        self._rotation_matrix = self._get_rotation_matrix()
        self._slice = tuple(
            map(
                lambda x: slice(x[0], x[1]),
                zip(*self._config.preprocessing.roi),
            )
        )
        self._pre_slice = tuple(
            map(
                lambda x: slice(x[0], x[1]),
                zip(*self._config.preprocessing.pre_roi),
            )
        )

    def _get_orthorectification_params(
        self, sample_image: NDArray, reduce: NDArray | None = None
    ) -> tuple[NDArray, NDArray]:
        pixels_coordinates = self._config.dataset.gcp.pixels_coordinates
        meters_coordinates = self._config.dataset.gcp.meters_coordinates
        if reduce is not None:
            pixels_coordinates = pixels_coordinates - reduce
        if self._config.preprocessing.image_correction.apply:
            corr_img = ip.apply_lens_correction(
                sample_image,
                k1=self._config.preprocessing.image_correction.k1,
                c=self._config.preprocessing.image_correction.c,
                f=self._config.preprocessing.image_correction.f,
            )
        else:
            corr_img = sample_image
        m, c = ip.build_orthorect_params(
            corr_img,
            pixels_coordinates,
            meters_coordinates,
            ppm=self._config.preprocessing.ppm,
            lonlat=False,
        )
        return m, c

    def _get_rotation_matrix(self):
        """Calculate the rotation matrix for image rotation.

        Based on:
        https://stackoverflow.com/questions/43892506/opencv-python-rotate-image-without-cropping-sides

        Returns:
            The rotation matrix for the given angle and image center.
        """
        a = 1.0  # TODO: idk why is 1.0
        height, width = self._shape
        image_center = (width / 2, height / 2)
        # getRotationMatrix2D needs coordinates in reverse
        # order (width, height) compared to shape
        rot_mat = cv2.getRotationMatrix2D(
            image_center, self._rotation_angle, a
        )
        # rotation calculates the cos and sin, taking absolutes of those.
        abs_cos = abs(rot_mat[0, 0])
        abs_sin = abs(rot_mat[0, 1])
        # find the new width and height bounds
        bound_w = int(height * abs_sin + width * abs_cos)
        bound_h = int(height * abs_cos + width * abs_sin)
        # subtract old image center (bringing image back to origo) and adding
        # the new image center coordinates
        rot_mat[0, 2] += bound_w / 2 - image_center[0]
        rot_mat[1, 2] += bound_h / 2 - image_center[1]
        self._bound = (bound_w, bound_h)
        return rot_mat

    @staticmethod
    def _get_sample_image(config: Config) -> np.ndarray | None:
        loader = make_loader(config.dataset)
        image: np.ndarray | None = loader.read()
        loader.end()
        return image

    @staticmethod
    def _gray(image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 2:
            return image
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def show_entire_image(self):
        """Set slice to cover the entire image."""
        w_slice = slice(0, 6000)
        h_slice = slice(0, 6000)
        self._slice = (w_slice, h_slice)

    def _rotate(self, image: np.ndarray) -> np.ndarray:
        if self._rotation_angle != 0:
            # rotate image with the new bounds and translated rotation matrix
            rotated_mat = cv2.warpAffine(
                image, self._rotation_matrix, self._bound
            )
            return rotated_mat
        return image

    def _pre_crop(self, image: np.ndarray) -> np.ndarray:
        new_image = image[self._pre_slice[0], self._pre_slice[1]]
        self._shape = (new_image.shape[0], new_image.shape[1])
        # TODO: this shouldn't be done here. Find a better way
        self._rotation_matrix = self._get_rotation_matrix()
        return new_image

    def _crop(self, image: np.ndarray) -> np.ndarray:
        new_image = image[self._slice[0], self._slice[1]]
        self._shape = (new_image.shape[0], new_image.shape[1])
        # TODO: this shouldn't be done here. Find a better way
        self._rotation_matrix = self._get_rotation_matrix()
        return new_image

    def apply_roi_extraction(self, image: NDArray, gray=True) -> NDArray:
        """Apply image rotation, cropping, and convert to grayscale.

        Args:
            image: The input image to process.
            gray: Whether to convert the image to grayscale.
            resize_factor: Factor by which to resize the image.

        Returns:
            The processed image.
        """
        # it must be in this order in order to calibrate easier
        image = self._pre_crop(image)
        image = self._rotate(image)
        image = self._crop(image)
        if gray:
            image = self._gray(image)
        return image

    def apply_image_enhancement(self, image: np.ndarray) -> np.ndarray:
        """Apply contrast and gamma correction.

        Args:
            image: The input image to enhance.

        Returns:
            The enhanced image.
        """
        # img_grey = ip.color_corr(
        #     img_orth,
        #     alpha=self.enhance_alpha,
        #     beta=self.enhance_beta,
        #     gamma=self.enhance_gamma)
        return image

    def _crop_using_refs(self, image: np.ndarray) -> np.ndarray:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        x_min, y_min = np.min(
            self._config.dataset.gcp.pixels_coordinates, axis=0
        )
        x_max, y_max = np.max(
            self._config.dataset.gcp.pixels_coordinates, axis=0
        )
        image = image[y_min:y_max, x_min:x_max]
        self._shape = (image.shape[0], image.shape[1])
        self._or_params = self._get_orthorectification_params(
            image, reduce=np.array([y_min, x_min])
        )
        self._rotation_matrix = self._get_rotation_matrix()
        return image

    def apply_resolution(self, image: np.ndarray) -> np.ndarray:
        """Apply resolution reduction to the image.

        Args:
            image: The input image to reduce resolution.

        Returns:
            The image with reduced resolution.
        """
        if self._config.preprocessing.resolution >= 1:
            return image
        return cv2.resize(
            image,
            (0, 0),
            fx=self._config.preprocessing.resolution,
            fy=self._config.preprocessing.resolution,
        )

    def apply_distortion_correction(self, image: np.ndarray) -> np.ndarray:
        """Undistort image using Ground Control Points (GCP).

        Args:
            image: The input image to correct.

        Returns:
            The undistorted image.
        """
        if not self._config.dataset.gcp.apply:
            return image
        if self._or_params is None:
            LOG.error("No orthorectification parameters found")
            return image

        image = self._crop_using_refs(image)
        # apply lens distortion correction
        if self._config.preprocessing.image_correction.apply:
            image = ip.apply_lens_correction(
                image,
                k1=self._config.preprocessing.image_correction.k1,
                c=self._config.preprocessing.image_correction.c,
                f=self._config.preprocessing.image_correction.f,
            )

        # apply orthorectification
        image = ip.apply_orthorec(
            image, self._or_params[0], self._or_params[1]
        )
        self._shape = (image.shape[0], image.shape[1])
        # update rotation matrix such as the shape of the image changed
        self._rotation_matrix = self._get_rotation_matrix()
        return image


def main(config_fp: Path, save_image: bool = False) -> None:
    """Demonstrate basic example of video correction.

    Args:
        config_path: Path to the configuration file.
        save_image: Whether to save the corrected image or display it.
    """
    config: Config = Config.from_fp(config_fp)
    t0 = time.process_time()
    loader = make_loader(config.dataset)
    t1 = time.process_time()
    formatter = Formatter(config)
    t2 = time.process_time()
    image = loader.read()
    if image is None:
        print("No image found")
        return
    t3 = time.process_time()
    image = formatter.apply_distortion_correction(image)
    t4 = time.process_time()
    image = formatter.apply_roi_extraction(image)
    t5 = time.process_time()
    loader.end()
    t6 = time.process_time()
    print("- get_loader:", t1 - t0)
    print("- Formatter:", t2 - t1)
    print("- loader.read:", t3 - t2)
    print("- formatter.apply_distortion_correction:", t4 - t3)
    print("- formatter.apply_roi_extraction:", t5 - t4)
    print("- loader.end:", t6 - t5)

    if save_image:
        cv2.imwrite("tmp.jpg", image)
    else:
        imshow(image)


if __name__ == "__main__":
    import typer

    typer.run(main)
