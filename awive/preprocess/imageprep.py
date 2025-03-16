"""Image preparation."""

import itertools

import cv2
import numpy as np
from numpy.typing import NDArray
from awive.tools import imshow


def crop_to_gcp_area(
    img: np.ndarray, pixels_coordinates: list[list[int]]
) -> tuple[np.ndarray, list[list[int]]]:
    """Crops the image to the bounding box of the given pixel coordinates.

    Args:
        img: Original image.
        pixels_coordinates: List of (x, y) pixel coordinates of GCPs.

    Returns:
        Cropped image and updated pixel coordinates relative to the cropped area.
    """
    x_min, y_min = np.min(pixels_coordinates, axis=0)
    x_max, y_max = np.max(pixels_coordinates, axis=0)
    cropped_img = img[y_min:y_max, x_min:x_max]
    updated_coords = [[x - x_min, y - y_min] for x, y in pixels_coordinates]

    return cropped_img, updated_coords


def apply_lens_correction(
    img: np.ndarray,
    k1: float = -10.0e-6,
    c: float = 2,
    f: float = 8.0,
) -> np.ndarray:
    """Lens distortion correction based on lens characteristics.

    Args:
        img: Original image.
        k1: Barrel lens distortion parameter.
        c: Optical center.
        f: Focal length.

    Returns:
        Image corrected for lens distortion.
    """
    # define distortion coefficient vector
    dist = np.zeros((4, 1), np.float64)
    dist[0, 0] = k1

    # define camera matrix
    mtx = np.eye(3, dtype=np.float32)

    mtx[0, 2] = img.shape[0] / c  # define center x
    mtx[1, 2] = img.shape[1] / c  # define center y
    mtx[0, 0] = f  # define focal length x
    mtx[1, 1] = f  # define focal length y

    # correct image for lens distortion
    corr_img = cv2.undistort(img, mtx, dist)

    return corr_img


# def xy_coord(df: list[list[int]]) -> int:
#     """Turn longitudes and latitudes into XY coordinates using an Equirectangular
#     projection. Only applicable on a small scale.

#     Args:
#         df (pd.DataFrame): DataFrame containing columns with longitudes and latitudes.

#     Returns:
#         int: Placeholder return value.
#     """
#     # set base parameters
#     # r = 6378137  # meters according to WGS84
#     # phi_0 = df.latitude[0]
#     # cos_phi_0 = math.cos(math.radians(phi_0))

#     # # create new DataFrame containing original coordinates in metres
#     # df_new = pd.DataFrame()
#     # df_new["x"] = [r * math.radians(lon) * cos_phi_0 for lon in df.lon.values]
#     # df_new["y"] = [r * math.radians(lat) for lat in df.lat.values]

#     # return df_new
#     return 12


def build_orthorect_params(
    img: np.ndarray,
    pixels_coordinates: NDArray,
    meters_coordinates: NDArray,
    ppm: float = 100.0,
    lonlat: bool = False,
) -> tuple[NDArray, NDArray]:
    """Image orthorectification parameters based on 4 GCPs.
    GCPs need to be at water level.

    Args:
        img: Original image.
        pixels_coordinates: DataFrame containing the xy-coordinates of the GCPs in the imagery in pixels.
        meters_coordinates: DataFrame containing the real xy-coordinates of the GCPs in metres.
        required_ppm: Pixels per meter in the corrected imagery. This will be used to scale the coordinates.
        lonlat: Convert longitudes/latitudes to meters.

    Returns:
        Transformation matrix, coordinates of image corners
    """
    # img, pixels_coordinates = crop_to_gcp_area(img, pixels_coordinates)
    # if lonlat:
    #     meters_coordinates = xy_coord(meters_coordinates)

    # set points to float32
    pts1 = np.array(pixels_coordinates, dtype=np.float32)
    # # Multiple elements inside df_to by PPM
    pts2 = np.array(meters_coordinates, dtype=np.float32) * ppm

    # define transformation matrix based on GCPs
    m = cv2.getPerspectiveTransform(pts1, pts2)  # type: ignore

    # find locations of transformed image corners
    # height, width, __ = img.shape
    height = img.shape[0]
    width = img.shape[1]
    c = np.array(
        [[0, 0, 1], [width, 0, 1], [0, height, 1], [width, height, 1]],
        dtype=np.float32,
    )
    c_new = np.array(
        [(np.dot(i, m.T) / np.dot(i, m.T)[2])[:2] for i in c],
        dtype=np.float32,
    )

    c_new[:, 0] -= min(c_new[:, 0])
    c_new[:, 1] -= min(c_new[:, 1])

    # define new transformation matrix based on image corners
    # otherwise, part of the imagery will not be saved
    c_old = c[:, :2].astype(
        np.float32
    )  # it is required becuase indexing change dtypes
    m_new = cv2.getPerspectiveTransform(c_old, c_new)

    # return m_new, c_new, img
    return m_new, c_new


def apply_orthorec(
    img: np.ndarray, m: np.ndarray, c: np.ndarray
) -> np.ndarray:
    """Image orthorectification based on parameters found with orthorect_param().

    Args:
        img: Original image.
        m: Transformation matrix based on image corners.
        c: Coordinates of image corners in the orthorectified imagery.

    Returns:
        Orthorectified image.
    """

    # define corrected image dimensions based on C
    cols = int(np.ceil(max(c[:, 0])))
    rows = int(np.ceil(max(c[:, 1])))

    # orthorectify image
    corr_img = cv2.warpPerspective(img, m, (cols, rows))

    return corr_img


def apply_color_correction(
    img: np.ndarray,
    alpha: float | None = None,
    beta: float | None = None,
    gamma: float = 0.5,
) -> np.ndarray:
    """Grey scaling, contrast- and gamma correction. Both alpha and beta need to
    be defined in order to apply contrast correction.

    Args:
        img: Original image.
        alpha: Gain parameter for contrast correction.
        beta: Bias parameter for contrast correction.
        gamma: Brightness parameter for gamma correction.

    Returns:
        Gray scaled, contrast- and gamma corrected image.
    """
    # turn image into grey scale
    corr_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    if alpha and beta:
        # apply contrast correction
        corr_img = cv2.convertScaleAbs(corr_img, alpha=alpha, beta=beta)

    # apply gamma correction
    invGamma = 1.0 / gamma
    table = np.array(
        [((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]
    ).astype(np.uint8)

    corr_img = cv2.LUT(corr_img, table)

    return corr_img


if __name__ == "__main__":
    fp = "data/video.mp4"
    pixels_coordinates = [
        [596, 422],
        [916, 234],
        [3380, 1160],
        [2657, 2077],
    ]
    meters_coordinates = [
        [0.0, 2.66],
        [0.61, 6.39],
        [21.17, 8.02],
        [18.9, 0.0],
    ]
    resolution = np.mean(
        [
            np.linalg.norm(
                np.array(pixels_coordinates[i])
                - np.array(pixels_coordinates[j])
            )
            / np.linalg.norm(
                np.array(meters_coordinates[i])
                - np.array(meters_coordinates[j])
            )
            for i, j in itertools.combinations(
                range(len(pixels_coordinates)), 2
            )
        ]
    )
    print(f"Computed resolution: {resolution}")

    # read first frame from image
    cap = cv2.VideoCapture(fp)
    ret, img = cap.read()
    cap.release()
    print(f"{img.shape=}")

    # pass to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    m, c = build_orthorect_params(
        img, np.array(pixels_coordinates), np.array(meters_coordinates), ppm=50
    )
    ortho_img = apply_orthorec(img, m, c)
    print(f"{ortho_img.shape=}")
    imshow(ortho_img, "Orthorectified image")
