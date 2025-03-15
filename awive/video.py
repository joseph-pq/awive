"""Play  a video."""

import argparse
from pathlib import Path
import json

import cv2
import numpy as np

from awive.config import Config
from awive.loader import Loader, make_loader
from awive.preprocess.correct_image import Formatter

FOLDER_PATH = "/home/joseph/Documents/Thesis/Dataset/config"
RESIZE_RATIO = 5


def play(
    loader: Loader,
    formatter: Formatter,
    undistort=True,
    roi=True,
    time_delay=1,
    resize=False,
    wlcrop: tuple[tuple[int, int], tuple[int, int]] | None = None,
    blur=True,
    resize_factor: float | None = None,
) -> None:
    """Plays a video."""
    i: int = 0

    while loader.has_images():
        image = loader.read()
        if image is None:
            continue
        if undistort:
            image = formatter.apply_distortion_correction(image)
        if roi:
            image = formatter.apply_roi_extraction(
                image, resize_factor=resize_factor
            )
        elif wlcrop is not None:
            image = image[
                wlcrop[0][0] : wlcrop[0][1], wlcrop[1][0] : wlcrop[1][1]
            ]
        if blur:
            image = cv2.medianBlur(image, 5)
        if resize:
            lil_im = cv2.resize(image, (1000, 1000))
        else:
            lil_im = image
        cv2.imshow("Video", lil_im)
        np.save(f"images/im_{i:04}.npy", lil_im)
        if cv2.waitKey(time_delay) & 0xFF == ord("q"):
            print("Finished by key 'q'")
            break
        i += 1
    cv2.destroyAllWindows()


def main(
    config_fp: Path,
    video_identifier: str,
    undistort=True,
    roi=True,
    time_delay=1,
    resize=True,
    wlcrop=True,
    blur=True,
) -> None:
    """Read configurations and play video."""
    config = Config.from_fp(config_fp)
    loader = make_loader(config.dataset)
    formatter = Formatter(config)
    if wlcrop:
        with open(config_fp) as json_file:
            config = json.load(json_file)[video_identifier]["water_level"]
        roi2 = config["roi"]
        wr0 = (roi2[0][0], roi2[1][0])
        wr1 = (roi2[0][1], roi2[1][1])
        crop = (wr0, wr1)
    else:
        crop = None
    play(loader, formatter, undistort, roi, time_delay, resize, crop, blur)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "statio_name", help="Name of the station to be analyzed"
    )
    parser.add_argument(
        "video_identifier", help="Index of the video of the json config file"
    )
    parser.add_argument(
        "-u",
        "--undistort",
        action="store_true",
        help="Format image using distortion correction",
    )
    parser.add_argument(
        "-r",
        "--roi",
        action="store_true",
        help="Format image using selecting only roi area",
    )
    parser.add_argument(
        "-c", "--wlcrop", action="store_true", help="Water level crop"
    )
    parser.add_argument("-b", "--blur", action="store_true", help="Blur image")
    parser.add_argument(
        "-z",
        "--resize",
        action="store_true",
        help="Resizer image to 1000x1000",
    )
    parser.add_argument(
        "-t",
        "--time",
        default=1,
        type=int,
        help="Time delay between each frame (ms)",
    )
    args = parser.parse_args()
    main(
        config_fp=Path(f"{FOLDER_PATH}/{args.statio_name}.json"),
        video_identifier=args.video_identifier,
        undistort=args.undistort,
        roi=args.roi,
        time_delay=args.time,
        resize=args.resize,
        wlcrop=args.wlcrop,
        blur=args.blur,
    )
