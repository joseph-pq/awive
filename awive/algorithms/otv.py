"""Optical Tracking Image Velocimetry."""

import argparse
import math
import itertools
import random
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from awive.config import Config
from awive.loader import Loader, make_loader
from awive.preprocess.correct_image import Formatter
from awive.tools import imshow
import logging

LOG = logging.getLogger(__name__)


def get_magnitude(kp1: cv2.KeyPoint, kp2: cv2.KeyPoint) -> float:
    """Get the distance between two keypoints."""
    return math.dist(kp1.pt, kp2.pt)  # type: ignore[attr-defined]
    # return abs(kp2.pt[0] - kp1.pt[0])


def get_angle(kp1, kp2):
    """Get angle between two key points."""
    return (
        math.atan2(kp2.pt[1] - kp1.pt[1], kp2.pt[0] - kp1.pt[0])
        * 180
        / math.pi
    )


def compute_velocity(
    kp1: cv2.KeyPoint,
    kp2: cv2.KeyPoint,
    pixels_to_meters: float,
    frames,
    fps: float,
):
    """Compute velocity in m/s

    Args:
        kp1: Begin keypoint
        kp2: End keypoint
        pixels_to_meters: Conversion factor from pixels to meters (meters / pixels)
        frames: Number of frames that the keypoint has been tracked
        fps: Frames per second
    """
    if frames == 0:
        return 0
    # pixels * (meters / pixels) * (frames / seconds) / frames
    return get_magnitude(kp1, kp2) * pixels_to_meters * fps / frames


def reject_outliers(data: NDArray[np.float32], m=2.0) -> NDArray[np.float32]:
    """Reject outliers from a dataset based on the median absolute deviation.

    Args:
        data: A numpy array of float32 values.
        m: The threshold multiplier for determining outliers.

    Returns:
        A numpy array with outliers removed.
    """
    deviation = np.abs(data - np.median(data))
    median_deviation = np.median(deviation)
    if median_deviation != 0:
        return data[(deviation / median_deviation) < m]
    else:
        return data


def compute_stats(velocity: list[list[float]], hist=False):
    """Compute statistics of the velocity
    Args:
        velocity: List of velocities
        hist: If True, show histogram of velocities

    Returns:
        Tuple of mean, max, min, std deviation, and count of velocities
    """
    v = np.array(list(itertools.chain(*velocity)))
    if v.size == 0:
        return 0, 0, 0, 0, 0
    v = reject_outliers(v)
    if v.size == 0:
        return 0, 0, 0, 0, 0
    if hist:
        pass
        # import matplotlib.pyplot as plt
        # plt.hist(v.astype(int))
        # plt.ylabel('Probability')
        # plt.xlabel('Data');
        # plt.show()

    return v.mean(), v.max(), v.min(), np.std(v), len(v)


class OTV:
    """Optical Tracking Image Velocimetry."""

    def __init__(
        self,
        config_: Config,
        prev_gray: NDArray,
        formatter: Formatter,
        lines: list[int],
    ) -> None:
        root_config = config_
        config = config_.otv
        self.formatter = formatter
        self.config = config
        self._partial_max_angle = config.partial_max_angle
        self._partial_min_angle = config.partial_min_angle
        self._final_max_angle = config.final_max_angle
        self._final_min_angle = config.final_min_angle
        self._final_min_distance = config.final_min_distance
        self._max_features = config.max_features
        self._max_level = config.lk.max_level
        self._step = config.region_step
        self._resolution = formatter.resolution
        self._pixel_to_real = 1 / root_config.preprocessing.ppm
        self.max_distance = (
            self._max_level * (2 * config.lk.radius + 1) / self._resolution
        )

        self._width = (
            root_config.preprocessing.roi[1][1]
            - root_config.preprocessing.roi[0][1]
        )
        self._height = (
            root_config.preprocessing.roi[1][0]
            - root_config.preprocessing.roi[1][1]
        )
        self.regions_heights = [
            int(line * formatter.resolution) for line in lines
        ]
        self.region_range = config_.otv.lines_width
        if config.mask_path is not None:
            self._mask: NDArray[np.uint8] | None = (
                cv2.imread(str(config.mask_path), 0) > 1
            ).astype(np.uint8)
            self._mask = cv2.resize(
                self._mask,
                (self._height, self._width),
                cv2.INTER_NEAREST,  # type: ignore[arg-type]
            )
            if self._resolution < 1:
                self._mask = cv2.resize(
                    self._mask,
                    (0, 0),
                    fx=self._resolution,
                    fy=self._resolution,
                )
        else:
            self._mask = None

        self.winsize = config.lk.winsize
        self.prev_gray = prev_gray

    def valid_displacement(self, kp1, kp2):
        magnitude = get_magnitude(
            kp1, kp2
        )  # only to limit the research window
        if magnitude > self.max_distance:
            return False
        angle = get_angle(kp1, kp2)
        if angle < 0:
            angle = angle + 360
        if angle == 0 or angle == 360:
            return True
        if self._partial_min_angle <= angle <= self._partial_max_angle:
            return True
        return False

    def valid_trajectory(self, kp1: cv2.KeyPoint, kp2: cv2.KeyPoint):
        """Final filter of keypoints"""
        magnitude = get_magnitude(kp1, kp2)
        if magnitude < self._final_min_distance:
            return False
        angle = get_angle(kp1, kp2)
        if angle < 0:
            angle = angle + 360
        if angle == 0 or angle == 360:
            return True
        if self._final_min_angle <= angle <= self._final_max_angle:
            return True
        return False

    def _apply_mask(self, image) -> NDArray:
        if self._mask is not None:
            image = image * self._mask
        return image

    def _init_subregion_list(self, dimension, width):
        ret = []
        n_regions = math.ceil(width / self._step)
        for _ in range(n_regions):
            # TODO: This is so inneficient
            if dimension == 1:
                ret.append(0)
            elif dimension == 2:
                ret.append([])
        return ret

    def predict_kps(
        self, prev_frame: NDArray, curr_frame: NDArray, kps: list[cv2.KeyPoint]
    ) -> tuple[list[cv2.KeyPoint], list[bool]]:
        """Predict keypoints using Lucas-Kanade"""
        pts2, status, _ = cv2.calcOpticalFlowPyrLK(
            prev_frame,
            curr_frame,
            cv2.KeyPoint_convert(kps),
            None,
            winSize=(self.winsize, self.winsize),
            maxLevel=self._max_level,
            criteria=(
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                self.config.lk.max_count,
                self.config.lk.epsilon,
            ),
            flags=cv2.OPTFLOW_LK_GET_MIN_EIGENVALS,
            minEigThreshold=self.config.lk.min_eigen_threshold,
        )
        kps = [
            cv2.KeyPoint(pt[0], pt[1], 1.0)  # type: ignore[call-arg]
            for pt in pts2
        ]
        return kps, status

    def run(
        self, loader: Loader, show_video=False
    ) -> dict[str, dict[str, float]]:
        """Execute OTV and get velocimetry"""
        # initialze parametrers
        detector = cv2.FastFeatureDetector_create()
        prev_frame = None
        prev_kps: list[cv2.KeyPoint] = []  # keypoints at the current frame
        traj_start_kps: list[
            cv2.KeyPoint
        ] = []  # keypoints at the start of the trajectory
        traj_start_idx: list[
            int
        ] = []  # Frame index of the start of the trajectory
        curr_kps: list[cv2.KeyPoint] = []  # keypoints predicted by LK
        masks: list[NDArray] = []

        # valid: list[list[bool]] = [
        #     []
        # ] * loader.total_frames
        # velocity_mem: list[list[float]] = [[] for _ in range(loader.total_frames)]
        velocities: list[list[float]] = [
            [] for _ in range(loader.total_frames)
        ]
        # angles: list[list[float]] = [[] for _ in range(loader.total_frames)]
        # distance: list[list[float]] = [[] for _ in range(loader.total_frames)]
        path: list[list[int]] = [[] for _ in range(loader.total_frames)]
        # keypoints_mem_current: list[list[cv2.KeyPoint]] = []
        # keypoints_mem_predicted: list[list[cv2.KeyPoint]] = []
        regions: list[list[float]] = [
            [] for _ in range(len(self.regions_heights))
        ]

        # traj_map must have the size of the image after all preprocessing
        traj_map = np.zeros_like(self.prev_gray)

        # update width and height if needed
        # TODO: Why is this needed?
        self._width = min(loader.width, self._width)
        self._height = min(loader.height, self._height)

        # subregion_velocity = self._init_subregion_list(2, self._width)
        # subregion_trajectories = self._init_subregion_list(1, self._width)

        while loader.has_images():
            # get current frame
            curr_frame = loader.read()
            if curr_frame is None:
                # TODO: This is not the best way to handle this
                break
            curr_frame = self.formatter.apply_distortion_correction(curr_frame)
            curr_frame = self.formatter.apply_roi_extraction(curr_frame)
            curr_frame = self.formatter.apply_resolution(curr_frame)
            curr_frame = self._apply_mask(curr_frame)

            # get features as a list of KeyPoints
            keypoints: list[cv2.KeyPoint] = list(
                detector.detect(curr_frame, None)
            )
            random.shuffle(keypoints)

            # Add keypoints in lists
            keypoints_to_add = min(
                self._max_features - len(prev_kps), len(keypoints)
            )
            if keypoints_to_add != 0:
                traj_start_idx.extend([loader.index] * keypoints_to_add)
                # valid[loader.index].extend([False] * keypoints_to_add)
                # velocity_mem[loader.index].extend([0] * keypoints_to_add)
                if prev_frame is None:
                    prev_kps.extend(keypoints[:keypoints_to_add])
                    traj_start_kps.extend(keypoints[:keypoints_to_add])
                    path[loader.index].extend(range(keypoints_to_add))
                else:
                    prev_kps.extend(keypoints[-keypoints_to_add:])
                    traj_start_kps.extend(keypoints[-keypoints_to_add:])

            LOG.debug(f"Analyzing frame: {loader.index}")
            if prev_frame is not None:
                curr_kps, kps_status = self.predict_kps(
                    prev_frame, curr_frame, prev_kps
                )

                k = 0  # valid keypoints counter

                for i in range(len(prev_kps)):
                    # Check if vector is valid
                    if kps_status[i] and self.valid_displacement(
                        prev_kps[i], curr_kps[i]
                    ):
                        # Trajectory didn't finished. Thus, keep its data
                        prev_kps[k] = prev_kps[i]
                        curr_kps[k] = curr_kps[i]
                        traj_start_kps[k] = traj_start_kps[i]
                        traj_start_idx[k] = traj_start_idx[i]
                        k += 1
                        # path[loader.index].append(i)
                        # velocity_mem[loader.index].append(0)
                        # valid[loader.index].append(False)
                        continue

                    # If vector is invalid, finish trajectory and process it.
                    # Check valid trajectory
                    if not self.valid_trajectory(
                        traj_start_kps[i], curr_kps[i]
                    ):
                        continue
                    # Compute velocity in valid trajectory
                    velocity = compute_velocity(
                        traj_start_kps[i],
                        curr_kps[i],
                        self._pixel_to_real / self._resolution,
                        loader.index - traj_start_idx[i],
                        loader.fps,
                    )
                    # angle = get_angle(traj_start_kps[i], curr_kps[i])

                    # sub-region computation
                    # module_start = int(keypoints_start[i].pt[1] /
                    #         self._step)
                    # module_current = int(keypoints_current[i].pt[1] /
                    #         self._step)
                    # if module_start == module_current:
                    # subregion_velocity[module_start].append(velocity_i)
                    # subregion_trajectories[module_start] += 1

                    # Add velocity to the trajectory map and regions
                    start_x = int(traj_start_kps[i].pt[1])  # type: ignore[attr-defined]
                    start_y = int(traj_start_kps[i].pt[0])  # type: ignore[attr-defined]
                    traj_map[start_x][start_y] += 100
                    for r_idx, region_x in enumerate(self.regions_heights):
                        if abs(start_x - region_x) < self.region_range:
                            regions[r_idx].append(velocity)

                    # update storage
                    # pos = i
                    # j = loader.index - 1
                    # while j >= traj_start_idx[i]:
                    #     valid[j][pos] = True
                    #     velocity_mem[j][pos] = velocity
                    #     pos = path[j][pos]
                    #     j -= 1

                    velocities[loader.index].append(velocity)
                    # angles[loader.index].append(angle)
                    # distance[loader.index].append(
                    #     velocity
                    #     / (loader.index - traj_start_idx[i])
                    #     / loader.fps
                    # )

                # Only keep until the kth keypoint in order to filter invalid
                # vectors
                prev_kps = prev_kps[:k]
                curr_kps = curr_kps[:k]
                traj_start_kps = traj_start_kps[:k]
                traj_start_idx = traj_start_idx[:k]

                LOG.debug(f"number of trajectories: {k}")

                if show_video:
                    color_frame = cv2.cvtColor(curr_frame, cv2.COLOR_GRAY2RGB)
                    output: NDArray[np.uint8] = draw_vectors(
                        color_frame,
                        prev_kps,
                        curr_kps,
                        masks,
                    )
                    imshow(output, "sparse optical flow", handle_destroy=False)
                    if cv2.waitKey(10) & 0xFF == ord("q"):
                        LOG.debug("Breaking")
                        break

            prev_frame = curr_frame.copy()
            # keypoints_mem_current.append(prev_kps)
            # keypoints_mem_predicted.append(curr_kps)

            # TODO: I guess the swap is not needed such as in the next iteration
            # the keypoints_predicted will be cleaned
            if len(curr_kps) != 0:
                prev_kps, curr_kps = curr_kps, prev_kps
        np.save("traj.npy", traj_map)

        loader.end()
        if show_video:
            cv2.destroyAllWindows()

        LOG.debug("Computing stats")
        avg, max_, min_, std_dev, count = compute_stats(velocities, show_video)
        LOG.debug(f"avg: {round(avg, 4)}")
        LOG.debug(f"max: {round(max_, 4)}")
        LOG.debug(f"min: {round(min_, 4)}")
        LOG.debug(f"std_dev: {round(std_dev, 2)}")
        LOG.debug(f"count: {count}")

        LOG.debug("Computing stats by region")
        out_json: dict[str, dict[str, float]] = {}
        for i, (sv, position) in enumerate(zip(regions, self.regions_heights)):
            out_json[str(i)] = {}
            t = np.array(sv)
            t = t[t != 0]
            if len(t) != 0:
                t = reject_outliers(t)
                m = t.mean()
            else:
                m = 0
            out_json[str(i)]["velocity"] = m
            out_json[str(i)]["count"] = len(t)
            out_json[str(i)]["position"] = position
        LOG.debug("Finished")
        return out_json


def draw_vectors(
    image: NDArray[np.uint8],
    new_list: list[cv2.KeyPoint],
    old_list: list[cv2.KeyPoint],
    masks: list[NDArray[np.uint8]],
) -> NDArray[np.uint8]:
    """Draw vectors of velocity and return the output and update mask"""
    thick = 1
    # if len(image.shape) == 3:
    #     color: tuple | int = (0, 255, 0)
    #     thick = 1
    # else:
    #     color = 255
    #     thick = 1

    # create new mask
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    for new, old in zip(new_list, old_list):
        new_pt = (int(new.pt[0]), int(new.pt[1]))  # type: ignore
        old_pt = (int(old.pt[0]), int(old.pt[1]))  # type: ignore
        distance = math.dist(new_pt, old_pt)
        intensity = min(int(255 * (distance / 30)), 255)  # TODO: change 50
        mask = cv2.line(mask, new_pt, old_pt, intensity, thick)

    # update masks list
    masks.append(mask)
    if len(masks) < 3:
        return np.zeros(image.shape, dtype=np.uint8)
    if len(masks) > 3:
        masks.pop(0)

    # generate image with mask
    total_mask = np.zeros(mask.shape, dtype=np.uint8)
    for mask_ in masks:
        total_mask = cv2.add(total_mask, mask_)
    # mask when values of total_mask are 0
    temp_mask = np.zeros_like(total_mask)
    temp_mask[total_mask == 0] = 255
    total_mask = cv2.applyColorMap(total_mask, cv2.COLORMAP_JET)
    total_mask[temp_mask == 255] = 0
    output = cv2.add(image, total_mask)
    return output


def run_otv(
    config_path: Path,
    show_video=False,
    debug=0,
) -> tuple[dict[str, dict[str, float]], np.ndarray | None]:
    """Basic example of OTV

    Processing for each frame
        1. Crop image using gcp.pixels parameter
        2. If enabled, lens correction using preprocessing.image_correction
        3. Orthorectification using relation gcp.pixels and gcp.real
        4. Pre crop
        5. Rotation
        6. Crop
        7. Convert to gray scale
    """
    config = Config.from_fp(config_path)
    loader: Loader = make_loader(config.dataset)
    formatter = Formatter(config)
    loader.has_images()
    image = loader.read()
    if image is None:
        raise ValueError("No image found")
    prev_gray = formatter.apply_distortion_correction(image)
    prev_gray = formatter.apply_roi_extraction(prev_gray)
    prev_gray = formatter.apply_resolution(prev_gray)
    otv = OTV(
        config_=config,
        prev_gray=prev_gray,
        formatter=formatter,
        lines=config.lines,
    )
    return otv.run(loader, show_video), prev_gray


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config",
        help="Config path to the config folder",
        type=Path,
    )
    parser.add_argument(
        "video_identifier",
        help="Index of the video of the json config file",
        type=str,
    )
    parser.add_argument(
        "-d", "--debug", help="Activate debug mode", type=int, default=0
    )
    parser.add_argument(
        "-v",
        "--video",
        action="store_true",
        help="Play video while processing",
    )
    parser.add_argument(
        "-s",
        "--save_image",
        action="store_true",
        help="Save image instead of showing",
    )
    args = parser.parse_args()
    velocities, image = run_otv(
        config_path=args.config,
        show_video=args.video,
        debug=args.debug,
    )
    if args.save_image and image is not None:
        print("Saving image")
        cv2.imwrite("tmp.jpg", image)
    print(f"{velocities=}")
