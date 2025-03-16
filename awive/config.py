"""Configuration."""

from pydantic import BaseModel as RawBaseModel, Field
from numpy.typing import NDArray
from typing import Any
import numpy as np
import functools
import json
from pathlib import Path
import yaml


class BaseModel(RawBaseModel):
    @staticmethod
    def from_fp(fp: Path):
        """Load config from json."""
        if fp.suffix == ".json":
            return Config(**json.load(fp.open()))
        elif fp.suffix == ".yaml":
            return Config(**yaml.safe_load(fp.open()))
        else:
            raise ValueError(f"File extension not supported: {fp.suffix}")


class GroundTruth(BaseModel):
    """Ground truth data."""

    position: list[int]
    velocity: float


class ConfigGcp(BaseModel):
    """Configurations GCP."""

    apply: bool
    pixels: list[tuple[int, int]] = Field(
        ...,
        description="at least four pixels coordinates: [[x1,y2], ..., [x4,y4]]",
    )
    meters: list[tuple[float, float]] = Field(
        default_factory=lambda: [],
        description="at least four meters coordinates: [[x1,y2], ..., [x4,y4]]",
    )
    distances: dict[tuple[int, int], float] | None = Field(
        None, description="distances in meters between the GCPs"
    )
    ground_truth: list[GroundTruth] | None = Field(None)

    @functools.cached_property
    def pixels_coordinates(self) -> NDArray:
        """Return pixel coordinates."""
        return np.array(self.pixels)

    @functools.cached_property
    def meters_coordinates(self) -> NDArray:
        """Return meters coordinates."""
        return np.array(self.meters)

    def calculate_meters(
        self, distances: dict[tuple[int, int], float]
    ) -> list[tuple[float, float]]:
        def di(i: int, j: int):
            return distances.get((i, j)) or distances.get((j, i))

        d = np.array(
            [
                [0, di(0, 1), di(0, 2), di(0, 3)],
                [di(1, 0), 0, di(1, 2), di(1, 3)],
                [di(2, 0), di(2, 1), 0, di(2, 3)],
                [di(3, 0), di(3, 1), di(3, 2), 0],
            ]
        )
        # check if nans are present
        if np.isnan(d).any():
            raise ValueError("Not all distances between GCPs are available")
        dim = 2

        # D is the distance matrix (n x n)
        n = d.shape[0]
        # Create centering matrix
        h = np.eye(n) - np.ones((n, n)) / n
        # Square the distances
        d_squared = d**2
        # Apply double centering
        b = -0.5 * h @ d_squared @ h
        # Eigen decomposition: using numpy's eig function
        eigvals, eigvecs = np.linalg.eig(b)
        # Sort eigenvalues and eigenvectors in descending order
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx][:dim]
        eigvecs = eigvecs[:, idx][:, :dim]
        # Compute coordinates using the positive eigenvalues
        l = np.diag(np.sqrt(eigvals))  # noqa: E741
        x = eigvecs @ l
        x[:, 0] *= -1
        return x.tolist()

    def model_post_init(self, __context: Any):
        if len(self.pixels) < 4:
            raise ValueError(
                f"at least four coordinates are required: {len(self.pixels)}"
            )
        if len(self.meters) == 0 and self.distances is None:
            raise ValueError("meters or distances must be provided")
        if len(self.meters) == 0 and self.distances is not None:
            if len(self.distances) != (
                len(self.pixels) * (len(self.pixels) - 1) / 2
            ):
                self.meters = self.calculate_meters(self.distances)
            else:
                raise ValueError(
                    "distances must have the correct number of elements"
                )

        if len(self.pixels) != len(self.meters):
            raise ValueError("pixels and meters must have the same length")


class ImageCorrection(BaseModel):
    """Configuration Image Correction."""

    apply: bool
    k1: float
    c: int
    f: float


class PreProcessing(BaseModel):
    """Configurations pre-processing."""

    rotate_image: int = Field(0, description="degrees")
    pre_roi: tuple[tuple[int, int], tuple[int, int]] = Field(
        ..., description="((x1,y1), (x2,y2))"
    )
    roi: tuple[tuple[int, int], tuple[int, int]] = Field(
        ..., description="((x1,y1), (x2,y2))"
    )
    image_correction: ImageCorrection
    ppm: int = Field(
        100,
        description=(
            "Resolution in pixels per meter. This is not the video "
            "resolution, but the resolution that will be forced."
        ),
    )
    resolution: float = Field(
        1,
        description=(
            "Resolution to process the video. Use this feature when the image"
            " resolution is too big"
        ),
    )


class Dataset(BaseModel):
    """Configuration dataset."""

    image_dataset_dp: Path | None = None
    image_number_offset: int = Field(
        0, description="Offset for the image number"
    )
    image_path_prefix: str = Field("", description="Prefix for the image path")
    image_path_digits: int = Field(
        4, description="Number of digits for the image path"
    )
    video_fp: Path | None = None
    gcp: ConfigGcp

    def model_post_init(self, __context: Any):
        if self.image_dataset_dp is None and self.video_fp is None:
            raise ValueError("Image dataset path not provided")


class OtvFeatures(BaseModel):
    """Config for OTV Features."""

    maxcorner: int = Field(
        default=300,
        description="Maximum number of corners to use in feature detection",
    )
    qualitylevel: float = Field(
        default=0.2, description="Quality level for feature detection"
    )
    mindistance: int = Field(default=2, description="Minimum distance")
    blocksize: int = Field(
        default=2, description="Block size for feature detection"
    )


class OtvLucasKanade(BaseModel):
    """Config for OTV Lucas Kanade."""

    winsize: int = Field(default=15, description="Window size")
    max_level: int = Field(default=4, description="Maximum level")
    max_count: int = Field(default=20, description="Maximum count")
    epsilon: float = Field(default=0.03, description="Epsilon")
    flags: int = Field(default=0, description="Flags")
    radius: int = Field(default=7, description="Radius")
    min_eigen_threshold: float = Field(
        default=0.001, description="Minimum eigen threshold"
    )


class Otv(BaseModel):
    """Configuration OTV."""

    mask_path: Path | None = Field(None, description="Path to the mask")
    partial_min_angle: float = Field(135, description="degrees")
    partial_max_angle: float = Field(225, description="degrees")
    final_min_angle: float = Field(160, description="degrees")
    final_max_angle: float = Field(200, description="degrees")
    final_min_distance: int = Field(
        8,
        description=(
            "Minimum trajectory distances (pixeles) of tracked features "
            "from beginning of the trajectory to the end"
        ),
    )
    max_features: int = Field(
        7000, description="Maximum number of features to track between frames"
    )
    region_step: int = Field(
        240, description="Step for the region. This feature is not used."
    )
    features: OtvFeatures = Field(
        default_factory=lambda: OtvFeatures(),
        description="Features configuration",
    )
    lk: OtvLucasKanade = Field(
        default_factory=lambda: OtvLucasKanade(),
        description="Lucas Kanade configuration",
    )
    lines_width: int = Field(
        ...,
        description="Width of the lines to extract the velocity vector",
    )


class Stiv(BaseModel):
    """Configuration STIV."""

    window_shape: tuple[int, int] = Field(
        default=(51, 51), description="Window shape. only using in GMT"
    )
    filter_window: int
    overlap: int = Field(default=31, description="Overlap. only used in GMT")
    ksize: int = Field(default=7, description="Kernel size. only used in GMT")
    polar_filter_width: int
    lines_range: list[tuple[int, int]]


class WaterLevel(BaseModel):
    """Configuration Water Level."""

    buffer_length: int
    roi: tuple[tuple[int, int], tuple[int, int]]
    roi2: tuple[tuple[int, int], tuple[int, int]]
    kernel_size: int


class Config(BaseModel):
    """Config class for awive."""

    dataset: Dataset
    otv: Otv
    stiv: Stiv | None = None
    preprocessing: PreProcessing
    water_level: WaterLevel | None = None
    lines: list[int] = Field(
        ...,
        description="Height of the lines to extract the velocity vector",
    )
