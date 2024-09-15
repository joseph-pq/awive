"""Execute both methods."""
import argparse
import time

import otv
import sti


FOLDER_PATH = '/home/joseph/Documents/Thesis/Dataset/config'


def execute_method(method_name, station_name, video_identifier, config_path):
    """Execute a method."""
    t = time.process_time()
    if method_name == 'sti':
        ret = sti.main(config_path, video_identifier)
    elif method_name == 'otv':
        ret = otv.run_otv(config_path, video_identifier)
    else:
        ret = None
    elapsed_time = (time.process_time() - t)
    print(method_name)
    for i in range(len(ret)):
        print(ret[str(i)]['velocity'])
    print(elapsed_time)


def main(station_name, video_identifier, config_path):
    """Execute both methods."""
    execute_method('sti', station_name, video_identifier, config_path)
    execute_method('otv', station_name, video_identifier, config_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "station_name",
        help="Name of the station to be analyzed")
    parser.add_argument(
        "video_identifier",
        help="Index of the video of the json config file")
    parser.add_argument(
        '-p',
        '--path',
        help='Path to the config folder',
        type=str,
        default=FOLDER_PATH)
    args = parser.parse_args()
    main(
        station_name=args.station_name,
        video_identifier=args.video_identifier,
        config_path = f'{args.path}/{args.station_name}.json',
    )
