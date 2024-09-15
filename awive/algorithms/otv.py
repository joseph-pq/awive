"""Optical Tracking Image Velocimetry."""

import argparse
import math
import random
from typing import NamedTuple, Optional

import cv2
import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel

from awive.algorithms.image_velocimetry import ImageVelocimetry
from awive.config import Config, ConfigOtv, ConfigRoi
from awive.correct_image import Formatter
from awive.loader import Loader, get_loader


FOLDER_PATH = "examples/datasets"


class Stats(NamedTuple):
    """Stats of the algorithm."""

    avg: float = 0
    max_: float = 0
    min_: float = 0
    std_dev: float = 0
    count: int = 0


def get_magnitude(kp1, kp2):
    """Get the distance between two keypoints."""
    # return math.sqrt((kp2.pt[1] - kp1.pt[1])**2 + \
    # (kp2.pt[0] - kp1.pt[0]) ** 2 )
    return abs(kp2.pt[0] - kp1.pt[0])


def get_angle(kp1, kp2) -> float:
    """Get angle between two key points."""
    return math.atan2(
        kp2.pt[1] - kp1.pt[1],
        kp2.pt[0] - kp1.pt[0]
    ) * 180 / math.pi


def _get_velocity(kp1, kp2, real_distance_pixel, time, fps):
    if time == 0:
        return 0
    return get_magnitude(kp1, kp2) * real_distance_pixel * fps / time


def reject_outliers(data, m=2.):
    """Reject outliers of data."""
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d / mdev if mdev else 0.
    return data[s < m]


def compute_stats(velocity, hist=False) -> Stats:
    """Compute stats of list of velicities."""
    v = np.array(sum(velocity, []))
    v = reject_outliers(v)
    count = len(v)
    if count == 0:
        return Stats()
    avg: float = v.mean()
    max_: float = v.max()
    min_: float = v.min()
    std_dev: float = np.std(v)  # type: ignore

    if hist:
        # import matplotlib.pyplot as plt
        # plt.hist(v.astype(int))
        # plt.ylabel('Probability')
        # plt.xlabel('Data')
        # plt.show()

    return Stats(avg, max_, min_, std_dev, count)


class LKParams(BaseModel):
    """Lucas-Kanade parameters."""

    winSize: tuple[int, int]
    maxLevel: int
    criteria: tuple[int, int, float]
    flags: int
    minEigThreshold: float


class Otv(ImageVelocimetry):
    """Optical Tracking Image Velocimetry."""

    def __init__(self, config: Config, debug: bool) -> None:
        """Initialize OTV."""
        self.conf: ConfigOtv = config.otv
        self._debug: bool = debug
        self._partial_max_angle: float = self.conf.partial_max_angle
        self._partial_min_angle: float = self.conf.partial_min_angle
        self._final_max_angle: float = self.conf.final_max_angle
        self._final_min_angle: float = self.conf.final_min_angle
        self._final_min_distance: float = self.conf.final_min_distance
        self._max_features: int = self.conf.max_features
        self._radius: int = self.conf.lk.radius
        self._max_level: int = self.conf.lk.max_level
        self._step: int = self.conf.region_step
        self._resolution: int = self.conf.resolution
        self._pixel_to_real: float = self.conf.pixel_to_real

        roi: ConfigRoi = config.preprocessing.roi
        self._width: int = roi.w2 - roi.w1
        self._height: int = roi.h2 - roi.h1
        self._regions: list[int] = self.conf.lines

        # Load mask if exists
        self._mask: Optional[NDArray] = None
        if self.conf.mask_path:
            self._mask = cv2.imread(self.conf.mask_path, 0) > 1  # type: ignore
            self._mask = cv2.resize(
                self._mask.astype(np.uint8),
                (self._height, self._width),
                cv2.INTER_NEAREST  # type: ignore
            )

        # Load Lucas Kanade parameters
        self.lk_params: LKParams = LKParams(
            winSize=(self._radius, self._radius),
            maxLevel=self._max_level,
            criteria=(
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                self.conf.lk.max_count,
                self.conf.lk.epsilon
            ),
            flags=cv2.OPTFLOW_LK_GET_MIN_EIGENVALS,
            minEigThreshold=self.conf.lk.min_eigen_threshold
        )

        # Load firt images
        # self.prev_gray = prev_gray

    def _partial_filtering(self, kp1, kp2, max_distance):
        # only to limit the research window
        magnitude = get_magnitude(kp1, kp2)
        if magnitude > max_distance:
            return False
        angle = get_angle(kp1, kp2)
        if angle < 0:
            angle = angle + 360
        if angle == 0 or angle == 360:
            return True
        if self._partial_min_angle <= angle <= self._partial_max_angle:
            return True
        return False

    def _final_filtering(self, kp1, kp2):
        """Final filter of keypoints."""
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

    def run(self, loader: Loader, formatter: Formatter, show_video=False):
        """Execute OTV and get velocimetry."""
        # initialze parametrers
        detector = cv2.FastFeatureDetector_create()
        previous_frame = None
        keypts_curr: list[cv2.KeyPoint] = []  # current keypoints
        keypoints_start: list[cv2.KeyPoint] = []
        time: list[int] = []  # list of indices of Keypoints
        keypts_pred: list[cv2.KeyPoint] = []  # predicted keypoints
        masks: list[NDArray[np.uint8]] = []

        valid: list[list[bool]] = []  # Indices of valid Keypoints
        velocity_mem: list[list[float]] = []
        keypoints_mem_current: list[cv2.KeyPoint] = []
        keypoints_mem_predicted: list[cv2.KeyPoint] = []
        velocity: list[list[float]] = []
        angle: list[list[float]] = []
        distance: list[list[float]] = []
        path: list[list[int]] = []
        traj_map = np.zeros((1200, 900))

        # update width and height if needed
        if loader.image_shape[0] < self._width:
            self._width = loader.image_shape[0]
        if loader.image_shape[1] < self._height:
            self._height = loader.image_shape[1]

        # This are not used. but idk why :c
        # subregion_velocity = self._init_subregion_list(2, self._width)
        # subregion_trajectories = self._init_subregion_list(1, self._width)

        regions: list[list[float]] = [[] for _ in range(len(self._regions))]

        # Initialization
        for i in range(loader.total_frames):
            valid.append([])
            velocity_mem.append([])
            velocity.append([])
            angle.append([])
            distance.append([])
            path.append([])

        while loader.has_images():
            # get current frame
            current_frame = loader.read()
            current_frame = formatter.apply_distortion_correction(
                current_frame)
            current_frame = formatter.apply_roi_extraction(current_frame)
            # current_frame = self._apply_mask(current_frame)

            # get features as a list of KeyPoints
            keypoints: list[cv2.KeyPoint] = detector.detect(
                current_frame,
                None
            )
            random.shuffle(keypoints)

            # update lot of lists
            if previous_frame is None:
                for i, keypoint in enumerate(keypoints):
                    if len(keypts_curr) < self._max_features:
                        keypts_curr.append(keypoint)
                        keypoints_start.append(keypoint)
                        time.append(loader.index)
                        valid[loader.index].append(False)
                        velocity_mem[loader.index].append(0)
                        path[loader.index].append(i)
            else:
                for i, keypoint in enumerate(reversed(keypoints)):
                    if len(keypts_curr) < self._max_features:
                        keypts_curr.append(keypoint)
                        keypoints_start.append(keypoint)
                        time.append(loader.index)
                        valid[loader.index].append(False)
                        velocity_mem[loader.index].append(0)

            if self._debug >= 1:
                print('Analyzing frame:', loader.index)
            if previous_frame is not None:
                pts1 = cv2.KeyPoint_convert(keypts_curr)
                pts2, st, _ = cv2.calcOpticalFlowPyrLK(
                    previous_frame,
                    current_frame,
                    pts1,
                    None,
                    **self.lk_params.dict()
                )

                # add predicted by Lucas-Kanade new keypoints
                keypts_pred.clear()
                for pt2 in pts2:
                    keypts_pred.append(cv2.KeyPoint(
                        pt2[0],
                        pt2[1],
                        1.0
                    ))

                max_distance = self._max_level * (2 * self._radius + 1)
                max_distance /= self._resolution

                k = 0

                for i, keypoint in enumerate(keypts_curr):
                    partial_filter = self._partial_filtering(
                        keypoint,
                        keypts_pred[i],
                        max_distance
                    )
                    # check if the trajectory finished or the vector is invalid
                    if not (st[i] and partial_filter):
                        final_filter = self._final_filtering(
                            keypoints_start[i],
                            keypts_curr[i]
                        )
                        # check if it is a valid trajectory
                        if final_filter:
                            velocity_i = _get_velocity(
                                keypoints_start[i],
                                keypts_curr[i],
                                self._pixel_to_real / self._resolution,
                                loader.index - time[i],
                                loader.fps
                            )
                            angle_i: float = get_angle(
                                keypoints_start[i],
                                keypts_curr[i]
                            )

                            xx0 = int(keypoints_start[i].pt[1])
                            yy0 = int(keypoints_start[i].pt[0])
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
                                if abs(xx0 - region) < 15:
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
                                velocity_i *
                                (loader.index - time[i]) /
                                loader.fps
                            )

                        continue

                    # Add new displacement vector
                    keypts_curr[k] = keypts_curr[i]
                    keypoints_start[k] = keypoints_start[i]
                    keypts_pred[k] = keypts_pred[i]
                    path[loader.index].append(i)
                    velocity_mem[loader.index].append(0)
                    valid[loader.index].append(False)
                    time[k] = time[i]
                    k += 1

                # Only keep until the kth keypoint in order to filter invalid
                # vectors
                keypts_curr = keypts_curr[:k]
                keypoints_start = keypoints_start[:k]
                keypts_pred = keypts_pred[:k]
                time = time[:k]

            if self._debug >= 1:
                print('number of trajectories:', len(keypts_curr))

            if show_video:
                if previous_frame is not None:
                    color_frame = cv2.cvtColor(
                        current_frame, cv2.COLOR_GRAY2RGB)
                    output = draw_vectors(
                        color_frame,
                        keypts_pred,
                        keypts_curr,
                        masks
                    )
                    cv2.imshow("sparse optical flow", output)
                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        break
            previous_frame = current_frame.copy()
            keypoints_mem_current.append(keypts_curr)
            keypoints_mem_predicted.append(keypts_pred)

            # TODO: I guess the swap is not needed such as in the next
            # iteration the keypoints_predicted will be cleaned
            if len(keypts_pred) != 0:
                keypts_pred, keypts_curr = keypts_curr, keypts_pred
        np.save('traj.npy', traj_map)

        loader.end()
        cv2.destroyAllWindows()
        avg, max_, min_, std_dev, count = compute_stats(velocity, show_video)

        if self._debug >= 1:
            print('avg:', round(avg, 4))
            print('max:', round(max_, 4))
            print('min:', round(min_, 4))
            print('std_dev:', round(std_dev, 2))
            print('count:', count)

        out_json: dict = {}
        for i, sv in enumerate(regions):
            out_json[str(i)] = {}
            t = np.array(sv)
            t = t[t != 0]
            if len(t) != 0:
                t = reject_outliers(t)
                m = t.mean()
            else:
                m = 0
            out_json[str(i)]['velocity'] = m
            out_json[str(i)]['count'] = len(t)
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
        # TODO: Perhaps there is a more efficient way to transform this to a
        # numpy array
        new_pt = np.array([int(new.pt[0]), int(new.pt[1])])
        old_pt = np.array([int(old.pt[0]), int(old.pt[1])])
        mask = cv2.line(mask, new_pt.ravel(), old_pt.ravel(), color, thick)

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


def main(
    config_path: str,
    video_identifier: str,
    show_video: bool = False,
    debug: bool = False
):
    """Execute a basic example of OTV."""
    loader: Loader = get_loader(config_path, video_identifier)
    formatter: Formatter = Formatter(config_path, video_identifier)

    # loader.has_images()
    # image = loader.read()
    # prev_gray = formatter.apply_distortion_correction(image)
    # prev_gray = formatter.apply_roi_extraction(prev_gray)

    config = Config.from_json(config_path, video_identifier)
    otv = Otv(config, debug)
    return otv.run(loader, formatter, show_video)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "statio_name",
        help="Name of the station to be analyzed"
    )
    parser.add_argument(
        "video_identifier",
        help="Index of the video of the json config file"
    )
    parser.add_argument(
        '-p',
        '--path',
        help='Path to the config folder',
        type=str,
        default=FOLDER_PATH
    )
    parser.add_argument(
        '-d',
        '--debug',
        help='Activate debug mode',
        type=int,
        default=0
    )
    parser.add_argument(
        '-v',
        '--video',
        action='store_true',
        help='Play video while processing'
    )
    args = parser.parse_args()
    print(
        main(
            config_path=f'{args.path}/{args.statio_name}/config.json',
            video_identifier=args.video_identifier,
            show_video=args.video,
            debug=args.debug
        )
    )
