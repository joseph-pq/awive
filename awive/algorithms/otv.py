"""Optical Tracking Image Velocimetry."""

import argparse
import math
import random
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from awive.config import Config
from awive.loader import Loader, make_loader
from awive.preprocess.correct_image import Formatter
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


def _get_velocity(
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


def reject_outliers(data, m=2.0):
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d / mdev if mdev else 0.0
    return data[s < m]


def compute_stats(velocity, hist=False):
    v = np.array(sum(velocity, []))
    if len(v) == 0:
        return 0, 0, 0, 0, 0
    v = reject_outliers(v)
    count = len(v)
    if count == 0:
        return 0, 0, 0, 0, 0
    avg = v.mean()
    max_ = v.max()
    min_ = v.min()
    std_dev = np.std(v)

    if hist:
        pass
        # import matplotlib.pyplot as plt
        # plt.hist(v.astype(int))
        # plt.ylabel('Probability')
        # plt.xlabel('Data');
        # plt.show()

    return avg, max_, min_, std_dev, count


class OTV:
    """Optical Tracking Image Velocimetry."""

    def __init__(
        self,
        config_: Config,
        prev_gray: NDArray,
    ) -> None:
        root_config = config_
        config = config_.otv
        self._partial_max_angle = config.partial_max_angle
        self._partial_min_angle = config.partial_min_angle
        self._final_max_angle = config.final_max_angle
        self._final_min_angle = config.final_min_angle
        self._final_min_distance = config.final_min_distance
        self._max_features = config.max_features
        self._max_level = config.lk.max_level
        self._step = config.region_step
        self._resolution = config.resolution
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
        self._regions = config.lines
        self.lines_width = config_.otv.lines_width
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

        winsize = config.lk.winsize

        self.lk_params = {
            "winSize": (winsize, winsize),
            "maxLevel": self._max_level,
            "criteria": (
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                config.lk.max_count,
                config.lk.epsilon,
            ),
            "flags": cv2.OPTFLOW_LK_GET_MIN_EIGENVALS,
            "minEigThreshold": config.lk.min_eigen_threshold,
        }
        self.prev_gray = prev_gray

    def _partial_filtering(self, kp1, kp2):
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

    def _final_filtering(self, kp1: cv2.KeyPoint, kp2: cv2.KeyPoint):
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

    def _apply_mask(self, image):
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

    def run(
        self, loader: Loader, formatter: Formatter, show_video=False
    ) -> dict[str, dict[str, float]]:
        """Execute OTV and get velocimetry"""
        # initialze parametrers
        detector = cv2.FastFeatureDetector_create()
        previous_frame = None
        keypoints_current: list[
            cv2.KeyPoint
        ] = []  # keypoints at the current frame
        keypoints_start = []  # keypoints at the start of the trajectory
        time = []  # Frame index of the start of the trajectory
        keypoints_predicted: list[
            cv2.KeyPoint
        ] = []  # keypoints predicted by LK
        masks: list[NDArray] = []

        valid: list[list[bool]] = [
            []
        ] * loader.total_frames  # valid trajectory
        velocity_mem: list[list[int]] = [[]] * loader.total_frames
        velocity: list[list[float]] = [[]] * loader.total_frames
        angle: list[list[float]] = [[]] * loader.total_frames
        distance: list[list[float]] = [[]] * loader.total_frames
        path: list[list[int]] = [[]] * loader.total_frames
        keypoints_mem_current: list[list[cv2.KeyPoint]] = []
        keypoints_mem_predicted: list[list[cv2.KeyPoint]] = []
        regions: list[list[float]] = [[]] * len(self._regions)

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
            current_frame = loader.read()
            if current_frame is None:
                # TODO: This is not the best way to handle this
                break
            current_frame = formatter.apply_distortion_correction(
                current_frame
            )
            current_frame = formatter.apply_roi_extraction(current_frame)
            current_frame = cv2.resize(
                current_frame,
                (0, 0),
                fx=self._resolution,
                fy=self._resolution,
            )
            current_frame = self._apply_mask(current_frame)

            # get features as a list of KeyPoints
            keypoints: list[cv2.KeyPoint] = list(
                detector.detect(current_frame, None)
            )
            random.shuffle(keypoints)

            # Add keypoints in lists
            kepoints_to_add = min(
                self._max_features - len(keypoints_current), len(keypoints)
            )
            time.extend([loader.index] * kepoints_to_add)
            valid[loader.index].extend([False] * kepoints_to_add)
            velocity_mem[loader.index].extend([0] * kepoints_to_add)
            if previous_frame is None:
                keypoints_current.extend(keypoints[:kepoints_to_add])
                keypoints_start.extend(keypoints[:kepoints_to_add])
                path[loader.index].extend(range(kepoints_to_add))
            else:
                keypoints_current.extend(keypoints[-kepoints_to_add:])
                keypoints_start.extend(keypoints[-kepoints_to_add:])

            LOG.debug("Analyzing frame:", loader.index)
            if previous_frame is not None:
                pts1 = cv2.KeyPoint_convert(keypoints_current)
                pts2, st, _ = cv2.calcOpticalFlowPyrLK(
                    previous_frame, current_frame, pts1, None, **self.lk_params
                )

                # add predicted by Lucas-Kanade new keypoints
                keypoints_predicted = [
                    cv2.KeyPoint(pt2[0], pt2[1], 1.0) # type: ignore[arg-type]
                    for pt2 in pts2
                ]

                k = 0

                for i, keypoint in enumerate(keypoints_current):
                    valid_displacement_vector = self._partial_filtering(
                        keypoint, keypoints_predicted[i]
                    )
                    # Filter vectors of previous to current frame
                    if not (st[i] and valid_displacement_vector):
                        valid_displacement_trajectory = self._final_filtering(
                            keypoints_start[i], keypoints_current[i]
                        )
                        # Filter trajectory vectors
                        if valid_displacement_trajectory:
                            velocity_i = _get_velocity(
                                keypoints_start[i],
                                keypoints_current[i],
                                self._pixel_to_real / self._resolution,
                                loader.index - time[i],
                                loader.fps,
                            )
                            angle_i = get_angle(
                                keypoints_start[i], keypoints_current[i]
                            )

                            xx0 = int(keypoints_start[i].pt[1])  # type: ignore[attr-defined]
                            yy0 = int(keypoints_start[i].pt[0])  # type: ignore[attr-defined]
                            traj_map[xx0][yy0] += 100
                            # sub-region computation
                            # module_start = int(keypoints_start[i].pt[1] /
                            #         self._step)
                            # module_current = int(keypoints_current[i].pt[1] /
                            #         self._step)
                            # if module_start == module_current:
                            # subregion_velocity[module_start].append(velocity_i)
                            # subregion_trajectories[module_start] += 1

                            for r_idx, region in enumerate(self._regions):
                                if abs(xx0 - region) < self.lines_width:
                                    regions[r_idx].append(velocity_i)

                            # update storage
                            pos = i
                            j = loader.index - 1
                            while j >= time[i]:
                                valid[j][pos] = True
                                velocity_mem[j][pos] = velocity_i
                                pos = path[j][pos]
                                j -= 1

                            velocity[loader.index].append(velocity_i)
                            angle[loader.index].append(angle_i)
                            distance[loader.index].append(
                                velocity_i
                                * (loader.index - time[i])
                                / loader.fps
                            )

                        continue

                    # Add new displacement vector
                    keypoints_current[k] = keypoints_current[i]
                    keypoints_start[k] = keypoints_start[i]
                    keypoints_predicted[k] = keypoints_predicted[i]
                    path[loader.index].append(i)
                    velocity_mem[loader.index].append(0)
                    valid[loader.index].append(False)
                    time[k] = time[i]
                    k += 1

                # Only keep until the kth keypoint in order to filter invalid
                # vectors
                keypoints_current = keypoints_current[:k]
                keypoints_start = keypoints_start[:k]
                keypoints_predicted = keypoints_predicted[:k]
                time = time[:k]

                LOG.debug("number of trajectories:", len(keypoints_current))

                if show_video:
                    color_frame = cv2.cvtColor(
                        current_frame, cv2.COLOR_GRAY2RGB
                    )
                    output: NDArray[np.float32] = draw_vectors(
                        color_frame,
                        keypoints_predicted,
                        keypoints_current,
                        masks,
                    ).astype(np.float32)
                    # Scale to 512p as width
                    initial_shape = output.shape
                    height = int(512 * initial_shape[0] / initial_shape[1])
                    output = cv2.resize(output, (512, height))
                    cv2.imshow("sparse optical flow", output)
                    if cv2.waitKey(10) & 0xFF == ord("q"):
                        break

            previous_frame = current_frame.copy()
            keypoints_mem_current.append(keypoints_current)
            keypoints_mem_predicted.append(keypoints_predicted)

            # TODO: I guess the swap is not needed such as in the next iteration
            # the keypoints_predicted will be cleaned
            if len(keypoints_predicted) != 0:
                keypoints_predicted, keypoints_current = (
                    keypoints_current,
                    keypoints_predicted,
                )
        np.save("traj.npy", traj_map)

        loader.end()
        if show_video:
            cv2.destroyAllWindows()
        avg, max_, min_, std_dev, count = compute_stats(velocity, show_video)

        LOG.debug("avg:", round(avg, 4))
        LOG.debug("max:", round(max_, 4))
        LOG.debug("min:", round(min_, 4))
        LOG.debug("std_dev:", round(std_dev, 2))
        LOG.debug("count:", count)

        out_json: dict[str, dict[str, float]] = {}
        for i, (sv, position) in enumerate(zip(regions, self._regions)):
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
        return out_json


def draw_vectors(image, new_list, old_list, masks):
    """Draw vectors of velocity and return the output and update mask"""
    if len(image.shape) == 3:
        color = (0, 255, 0)
        thick = 1
    else:
        color = 255
        thick = 1

    # create new mask
    mask = np.zeros(image.shape, dtype=np.uint8)
    for new, old in zip(new_list, old_list):
        new_pt = (int(new.pt[0]), int(new.pt[1]))
        old_pt = (int(old.pt[0]), int(old.pt[1]))
        mask = cv2.line(mask, new_pt, old_pt, color, thick)

    # update masks list
    masks.append(mask)
    if len(masks) < 3:
        return np.zeros(image.shape)
    if len(masks) > 3:
        masks.pop(0)

    # generate image with mask
    total_mask = np.zeros(mask.shape, dtype=np.uint8)
    for mask_ in masks:
        total_mask = cv2.add(total_mask, mask_)
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
    if config.otv.resolution < 1:
        prev_gray = cv2.resize(
            prev_gray,
            (0, 0),
            fx=config.otv.resolution,
            fy=config.otv.resolution,
        )
    otv = OTV(config, prev_gray, debug)
    return otv.run(loader, formatter, show_video), prev_gray


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
