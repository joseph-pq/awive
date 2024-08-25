"""Correct distortion of videos.

This module contains classes and functions needed to correct distortion of
videos, either intrinsic or extring to the camera. Also it saves the corrected
frames in defined directory path.

"""

import argparse
import time
import logging
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

import awive.imageprep as ip
from awive.loader import make_loader, get_loader
from awive.config import Config
from awive.exceptions import VideoSourceError


FOLDER_PATH = '/home/joseph/Documents/Thesis/Dataset/config'
LOG = logging.getLogger(__name__)


class Formatter:
    """Format frames in order to be used by image processing methods."""

    def __init__(self, config: Config) -> None:
        """Initialize Formatter object."""
        # read configuration file
        self._config: Config = config
        sample_image = self._get_sample_image(self._config)
        if sample_image is None:
            raise VideoSourceError('No sample image found')
        self._shape = (sample_image.shape[0], sample_image.shape[1])
        if self._config.dataset.gcp.apply:
            self._or_params = self._get_orthorectification_params(sample_image)
        else:
            self._or_params = None

        self._rotation_angle = self._config.preprocessing.rotate_image
        self._rotation_matrix = self._get_rotation_matrix()

        w_slice = slice(self._config.preprocessing.roi.w1,
                        self._config.preprocessing.roi.w2)
        h_slice = slice(self._config.preprocessing.roi.h1,
                        self._config.preprocessing.roi.h2)
        self._slice = (w_slice, h_slice)
        w_slice = slice(self._config.preprocessing.pre_roi.w1,
                        self._config.preprocessing.pre_roi.w2)
        h_slice = slice(self._config.preprocessing.pre_roi.h1,
                        self._config.preprocessing.pre_roi.h2)
        self._pre_slice = (w_slice, h_slice)

    def _get_orthorectification_params(
        self,
        sample_image: np.ndarray,
        reduce=None
    ) -> tuple[Any, NDArray]:
        # FIX: This is a temporary fix. I shouldn't be converting to dictionary
        x = self._config.dataset.gcp.pixels.dict()
        df_from = list(map(list, zip(*[(v) for _, v in x.items()])))
        if reduce is not None:
            for i, _ in enumerate(df_from):
                df_from[i][0] = df_from[i][0]-reduce[0]
                df_from[i][1] = df_from[i][1]-reduce[1]

        # FIX: This is a temporary fix. I shouldn't be converting to dictionary
        x = self._config.dataset.gcp.meters.dict()
        df_to = list(map(list, zip(*[(v) for _, v in x.items()])))
        if self._config.preprocessing.image_correction.apply:
            corr_img = ip.lens_corr(
                    sample_image,
                    k1=self._config.preprocessing.image_correction.k1,
                    c=self._config.preprocessing.image_correction.c,
                    f=self._config.preprocessing.image_correction.f
                    )
        else:
            corr_img = sample_image
        M, C, _ = ip.orthorect_param(corr_img,
                                      df_from,
                                      df_to,
                                      PPM=self._config.dataset.ppm,
                                      lonlat=False)
        return (M, C)

    def _get_rotation_matrix(self):
        """Rotate matrix.

        based on:
        https://stackoverflow.com/questions/43892506/opencv-python-rotate-image-without-cropping-sides
        """
        a = 1.0   # TODO: idk why is 1.0
        height, width = self._shape
        image_center = (width/2, height/2)
        # getRotationMatrix2D needs coordinates in reverse
        # order (width, height) compared to shape
        rot_mat= cv2.getRotationMatrix2D(
            image_center,
            self._rotation_angle,
            a)
        # rotation calculates the cos and sin, taking absolutes of those.
        abs_cos = abs(rot_mat[0, 0])
        abs_sin = abs(rot_mat[0, 1])
        # find the new width and height bounds
        bound_w = int(height * abs_sin + width * abs_cos)
        bound_h = int(height * abs_cos + width * abs_sin)
        # subtract old image center (bringing image back to origo) and adding
        # the new image center coordinates
        rot_mat[0, 2] += bound_w/2 - image_center[0]
        rot_mat[1, 2] += bound_h/2 - image_center[1]
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
        """Set slice to cover the entire image"""
        w_slice = slice(0, 6000)
        h_slice = slice(0, 6000)
        self._slice = (w_slice, h_slice)

    def _rotate(self, image: np.ndarray) -> np.ndarray:
        if self._rotation_angle != 0:
            # rotate image with the new bounds and translated rotation matrix
            rotated_mat = cv2.warpAffine(
                image,
                self._rotation_matrix,
                self._bound
                )
            return rotated_mat
        return image
    def _pre_crop(self, image: np.ndarray) -> np.ndarray:
        new_image =  image[self._pre_slice[0], self._pre_slice[1]]
        self._shape = (new_image.shape[0], new_image.shape[1])
        # TODO: this shouldn't be done here. Find a better way
        self._rotation_matrix = self._get_rotation_matrix()
        return new_image

    def _crop(self, image: np.ndarray) -> np.ndarray:
        new_image =  image[self._slice[0], self._slice[1]]
        self._shape = (new_image.shape[0], new_image.shape[1])
        # TODO: this shouldn't be done here. Find a better way
        self._rotation_matrix = self._get_rotation_matrix()
        return new_image

    def apply_roi_extraction(self, image: np.ndarray, gray=True) -> np.ndarray:
        """Apply image rotation, cropping and rgb2gray"""
        # it must be in this order in order to calibrate easier
        image = self._pre_crop(image)
        image = self._rotate(image)
        image = self._crop(image)
        if gray:
            image = self._gray(image)
        return image

    def apply_image_enhancement(self, image: np.ndarray) -> np.ndarray:
        """Apply contrast- and gamma correction"""
        # img_grey = ip.color_corr(
        #     img_orth,
        #     alpha=self.enhance_alpha,
        #     beta=self.enhance_beta,
        #     gamma=self.enhance_gamma)
        return image

    def _crop_using_refs(self, image: np.ndarray) -> np.ndarray:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        xs = self._config.dataset.gcp.pixels.y
        ys = self._config.dataset.gcp.pixels.x
        xslice = slice(min(xs), max(xs)+1)
        yslice = slice(min(ys), max(ys)+1)
        image = image[xslice, yslice]
        self._shape = (image.shape[0], image.shape[1])
        self._or_params = self._get_orthorectification_params(image, reduce=(min(ys), min(xs)))
        self._rotation_matrix = self._get_rotation_matrix()
        return image

    def apply_distortion_correction(self, image: np.ndarray) ->np.ndarray:
        """Given GCP, undistort image."""
        if not self._config.dataset.gcp.apply:
            return image
        if self._or_params is None:
            LOG.error('No orthorectification parameters found')
            return image

        image = self._crop_using_refs(image)
        # apply lens distortion correction
        if self._config.preprocessing.image_correction.apply:
            image = ip.lens_corr(
                    image,
                    k1=self._config.preprocessing.image_correction.k1,
                    c =self._config.preprocessing.image_correction.c,
                    f =self._config.preprocessing.image_correction.f,
                    )

        # apply orthorectification
        image = ip.orthorect_trans(
            image,
           self._or_params[0],
           self._or_params[1]
        )
        self._shape = (image.shape[0], image.shape[1])
        # update rotation matrix such as the shape of the image changed
        self._rotation_matrix = self._get_rotation_matrix()
        return image


def main(config_path: str, video_identifier: str, save_image: bool):
    """Demonstrate basic example of video correction."""
    config: Config = Config.from_json(config_path, video_identifier)
    t0 = time.process_time()
    loader = get_loader(config_path, video_identifier)
    t1 = time.process_time()
    formatter = Formatter(config)
    t2 = time.process_time()
    image = loader.read()
    t3 = time.process_time()
    image = formatter.apply_distortion_correction(image)
    t4 = time.process_time()
    image = formatter.apply_roi_extraction(image)
    t5 = time.process_time()
    loader.end()
    t6 = time.process_time()
    print('- get_loader:', t1 - t0)
    print('- Formatter:', t2 - t1)
    print('- loader.read:', t3 - t2)
    print('- formatter.apply_distortion_correction:', t4 - t3)
    print('- formatter.apply_roi_extraction:', t5 - t4)
    print('- loader.end:', t6 - t5)

    if save_image:
        cv2.imwrite('tmp.jpg', image)
    else:
        cv2.imshow('image', cv2.resize(image, (1000, 1000)))
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
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
