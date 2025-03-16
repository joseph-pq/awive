"""Basic usage of Loader: Load an image and plot."""

import time
from pathlib import Path

import cv2
import numpy as np
import requests
from numpy.typing import NDArray

from awive.loader import Loader, get_loader

CONFIG_PATH = "examples/basic/config.json"
VIDEO_PATH = "examples/basic/AlpineStabilised.avi"
FILE_ID = "1JreGYQEYUB4DkIk-MkE4n_-2RzSb0W27"


def download_file(file_id: str, local_filename: str) -> str:
    """Download a file from Google Drive."""
    url: str = (
        "https://drive.google.com/uc?"
        f"export=download&confirm=9iBg&id={file_id}"
    )
    CHUNK_SIZE: int = 4194304  # 4MB
    total_bytes: float = 0
    print(f"Downloading file from {url}")
    start: float = time.time()
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, "wb") as f:
            for chunk in r.iter_content(chunk_size=CHUNK_SIZE):
                total_bytes += len(chunk)
                print(
                    f"\rDownloaded: {(total_bytes / 1024 / 1024):0.2f} MB",
                    end="",
                )
                f.write(chunk)
    print("")
    print(f"Downloaded {local_filename} in {time.time() - start} seconds")
    return local_filename


def basic_plot_image(config_path: Path) -> None:
    """Use basic loader functions to read an image."""
    loader: Loader = get_loader(config_path)
    if not loader.has_images():
        raise ValueError("The video does not have images")
    image: NDArray[np.uint8] | None = loader.read()
    if image is None:
        raise ValueError("The image is None")
    cv2.imshow("image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    loader.end()


if __name__ == "__main__":
    if not Path(VIDEO_PATH).exists():
        download_file(FILE_ID, VIDEO_PATH)
    basic_plot_image(Path(CONFIG_PATH))
