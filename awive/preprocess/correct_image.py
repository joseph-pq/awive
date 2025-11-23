"""Correct distortion of videos.

This module contains classes and functions needed to correct distortion of
videos, either intrinsic or extrinsic to the camera. It also saves the
corrected frames in a defined directory path.

"""

import logging
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

import awive.preprocess.imageprep as ip
from awive.config import (
    Config,
)
from awive.config import (
    Dataset as DatasetConfig,
)
from awive.config import (
    PreProcessing as PreProcessingConfig,
)
from awive.exceptions import VideoSourceError
from awive.loader import make_loader
from awive.tools import imshow

LOG = logging.getLogger(__name__)


class Formatter:
    """Format frames in order to be used by image processing methods."""

    def __init__(
        self,
        dataset_config: DatasetConfig,
        preprocessing_config: PreProcessingConfig,
    ) -> None:
        """Initialize Formatter object.

        Args:
            dataset_config: Configuration object containing settings for
                processing.
            preprocessing_config: Configuration object containing settings for
                preprocessing.

        Raises:
            VideoSourceError: If no sample image is found.
        """
        # read configuration file
        self.dataset = dataset_config
        self.preprocessing = preprocessing_config
        self.resolution = self.preprocessing.resolution
        self.ppm = self.preprocessing.ppm
        sample_image = self._get_sample_image(self.dataset)
        if sample_image is None:
            raise VideoSourceError("No sample image found")
        self._shape = (sample_image.shape[0], sample_image.shape[1])
        if self.dataset.gcp.apply:
            self._or_params: tuple[Any, np.ndarray] | None = (
                self._get_orthorectification_params(sample_image)
            )
        else:
            self._or_params = None

        self._rotation_angle = self.preprocessing.rotate_image
        self._rotation_matrix = self._get_rotation_matrix()
        self._slice = tuple(
            slice(x[0], x[1]) for x in zip(*self.preprocessing.roi)
        )
        self._pre_slice = tuple(
            slice(x[0], x[1]) for x in zip(*self.preprocessing.pre_roi)
        )
        self._lens_params = None

    def _get_orthorectification_params(
        self, sample_image: NDArray, reduce: NDArray | None = None
    ) -> tuple[NDArray, NDArray]:
        pixels_coordinates = self.dataset.gcp.pixels_coordinates
        meters_coordinates = self.dataset.gcp.meters_coordinates
        if reduce is not None:
            pixels_coordinates = pixels_coordinates - reduce
        m, c = ip.build_orthorect_params(
            sample_image,
            pixels_coordinates,
            meters_coordinates,
            ppm=self.preprocessing.ppm,
            lonlat=False,
        )
        return m, c

    def _get_rotation_matrix(self) -> NDArray:
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
    def _get_sample_image(config: DatasetConfig) -> np.ndarray | None:
        loader = make_loader(config)
        image: np.ndarray | None = loader.read()
        loader.end()
        return image

    @staticmethod
    def _gray(image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 2:
            return image
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def show_entire_image(self) -> None:
        """Set slice to cover the entire image."""
        w_slice = slice(0, 6000)
        h_slice = slice(0, 6000)
        self._slice = (w_slice, h_slice)

    def _rotate(self, image: np.ndarray) -> np.ndarray:
        if self._rotation_angle != 0:
            # rotate image with the new bounds and translated rotation matrix
            return cv2.warpAffine(image, self._rotation_matrix, self._bound)
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
        # self._rotation_matrix = self._get_rotation_matrix()
        return new_image

    def apply_roi_extraction(
        self, image: NDArray, gray: bool = True
    ) -> NDArray:
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
        x_min, y_min = np.min(self.dataset.gcp.pixels_coordinates, axis=0)
        x_max, y_max = np.max(self.dataset.gcp.pixels_coordinates, axis=0)
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
        if self.preprocessing.resolution >= 1:
            return image
        return cv2.resize(
            image,
            (0, 0),
            fx=self.preprocessing.resolution,
            fy=self.preprocessing.resolution,
        )

    def apply_lens_correction(self, image: np.ndarray) -> np.ndarray:
        """Lens correction using calibration parameters.

        Updates:
        - self._shape
        - self._rotation_matrix
        - self._lens_params
        """
        # apply lens distortion correction
        if not self.preprocessing.image_correction.apply:
            return image
        # check if we have calibration parameters directly
        has_calibration = (
            self.preprocessing.image_correction.camera_matrix is not None
            and self.preprocessing.image_correction.dist_coeffs is not None
        )
        # compute undistortion maps once
        if not self._lens_params and has_calibration:
            self._lens_params = ip.compute_undistort_maps(
                image.shape,
                camera_matrix=self.preprocessing.image_correction.lens_camera_matrix,
                dist_coeffs=self.preprocessing.image_correction.lens_dist_coeffs,
            )

        image = ip.apply_lens_correction(
            image,
            k1=self.preprocessing.image_correction.k1,
            c=self.preprocessing.image_correction.c,
            f=self.preprocessing.image_correction.f,
            lens_params=self._lens_params,
        )

        self._shape = (image.shape[0], image.shape[1])
        # update rotation matrix such as the shape of the image changed
        self._rotation_matrix = self._get_rotation_matrix()
        self._or_params = self._get_orthorectification_params(image)

        return image

    def apply_distortion_correction(self, image: np.ndarray) -> np.ndarray:
        """Undistort image using Ground Control Points (GCP).

        Updates:
        - self._shape
        - self._or_params
        - self._rotation_matrix

        Args:
            image: The input image to correct.

        Returns:
            The undistorted image.
        """
        if not self.dataset.gcp.apply:
            return image
        if self._or_params is None:
            LOG.error("No orthorectification parameters found")
            return image
        # image = self._crop_using_refs(image)
        # apply orthorectification
        image = ip.apply_orthorec(
            image, self._or_params[0], self._or_params[1]
        )
        self._shape = (image.shape[0], image.shape[1])
        # update rotation matrix such as the shape of the image changed
        self._rotation_matrix = self._get_rotation_matrix()
        return image

    def apply(self, image: NDArray) -> tuple[NDArray, NDArray]:
        """Apply all preprocessing steps to the image.

        Args:
            image: The input image to process.
            positions: Positions to be transformed of shape (N, 2).

        Returns:
            The processed image and transformed positions.
        """
        image = self.apply_lens_correction(image)
        image = self.apply_distortion_correction(image)
        image = self.apply_resolution(image)

        return self.apply_image_enhancement(image)


def main(config_fp: Path, save_image: bool = False) -> None:
    """Demonstrate basic example of video correction.

    Args:
        config_fp: Path to the configuration file.
        save_image: Whether to save the corrected image or display it.
    """
    config: Config = Config.from_fp(config_fp)
    t0 = time.process_time()
    loader = make_loader(config.dataset)
    t1 = time.process_time()
    formatter = Formatter(config.dataset, config.preprocessing)
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
