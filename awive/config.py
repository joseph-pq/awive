"""Configuration."""

from pydantic import BaseModel, Field
from numpy.typing import NDArray
import numpy as np
import functools
import json
from pathlib import Path


class GroundTruth(BaseModel):
    """Ground truth data."""

    position: list[int]
    velocity: float


class ConfigGcp(BaseModel):
    """Configurations GCP."""

    apply: bool
    pixels: list[list[int]] = Field(
        ..., alias="at least four coordinates: [[x1,y2], ..., [x4,y4]]"
    )
    meters: list[list[float]] = Field(
        ..., alias="at least four coordinates: [[x1,y2], ..., [x4,y4]]"
    )
    ground_truth: list[GroundTruth]

    @functools.cached_property
    def pixels_coordinates(self) -> NDArray:
        """Return pixel coordinates."""
        return np.array(self.pixels)

    @functools.cached_property
    def meters_coordinates(self) -> NDArray:
        """Return meters coordinates."""
        return np.array(self.meters)


class ConfigRoi(BaseModel):
    """Configurations ROI."""

    h1: int
    h2: int
    w1: int
    w2: int


class ConfigImageCorrection(BaseModel):
    """Configuration Image Correction."""

    apply: bool
    k1: float
    c: int
    f: float


class PreProcessing(BaseModel):
    """Configurations pre-processing."""

    rotate_image: int
    pre_roi: ConfigRoi
    roi: ConfigRoi
    image_correction: ConfigImageCorrection


class Dataset(BaseModel):
    """Configuration dataset."""

    image_dataset: str
    image_number_offset: int
    image_path_prefix: str
    image_path_digits: int
    video_path: str
    width: int
    height: int
    ppm: int
    gcp: ConfigGcp


class ConfigOtvFeatures(BaseModel):
    """Config for OTV Features."""

    maxcorner: int
    qualitylevel: float
    mindistance: int
    blocksize: int


class ConfigOtvLucasKanade(BaseModel):
    """Config for OTV Lucas Kanade."""

    winsize: int
    max_level: int
    max_count: int
    epsilon: float
    flags: int
    radius: int
    min_eigen_threshold: float


class Otv(BaseModel):
    """Configuration OTV."""

    mask_path: str
    pixel_to_real: float
    partial_min_angle: float
    partial_max_angle: float
    final_min_angle: float
    final_max_angle: float
    final_min_distance: int
    max_features: int
    region_step: int
    resolution: int
    features: ConfigOtvFeatures
    lk: ConfigOtvLucasKanade
    lines: list[int]
    lines_width: int
    resize_factor: float | None = None


class ConfigStivLine(BaseModel):
    """Config for STIV line."""

    start: list[int]
    end: list[int]


class Stiv(BaseModel):
    """Configuration STIV."""

    window_shape: list[int]
    filter_window: int
    overlap: int
    ksize: int
    polar_filter_width: int
    lines: list[ConfigStivLine]
    resize_factor: float | None = None


class Config(BaseModel):
    """Config class for awive."""

    dataset: Dataset
    otv: Otv
    stiv: Stiv
    preprocessing: PreProcessing

    @staticmethod
    def from_json(file_path: str, video_id: str | None = None):
        """Load config from json."""
        if video_id is None:
            return Config(**json.load(Path(file_path).open()))
        return Config(**json.load(Path(file_path).open())[video_id])
