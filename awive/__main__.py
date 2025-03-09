"""Execute awive methods."""

import time
import argparse

from awive.algorithms import (
    otv,
    sti,
    water_level_hist,
)


FOLDER_PATH = "/home/joseph/Documents/Thesis/Dataset/config"


def execute_method(method_name, station_name, video_identifier, config_path):
    """Execute a method."""
    print(method_name)
    t = time.process_time()
    if method_name == "sti":
        ret = sti.main(config_path, video_identifier)
    elif method_name == "otv":
        ret, image = otv.run_otv(config_path, video_identifier)
    else:
        ret = None
    elapsed_time = time.process_time() - t
    for i in range(len(ret)):
        print(ret[str(i)]["velocity"])
    print(f"{elapsed_time=}")


def main(station_name, video_identifier, config_path, th):
    """Execute one method methods."""
    idpp = water_level_hist.main(config_path, video_identifier)
    if idpp > th:
        execute_method("sti", station_name, video_identifier, config_path)
    else:
        execute_method("otv", station_name, video_identifier, config_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "station_name", help="Name of the station to be analyzed"
    )
    parser.add_argument(
        "video_identifier", help="Index of the video of the json config file"
    )
    parser.add_argument("th", help="threshold", type=float)
    parser.add_argument(
        "-p",
        "--path",
        help="Path to the config folder",
        type=str,
        default=FOLDER_PATH,
    )
    args = parser.parse_args()
    CONFIG_PATH = f"{args.path}/{args.station_name}.json"
    main(
        station_name=args.station_name,
        video_identifier=args.video_identifier,
        config_path=CONFIG_PATH,
        th=args.th,
    )
