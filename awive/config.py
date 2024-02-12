"""Configuration."""

from pydantic import BaseModel
import json
from pathlib import Path


class Position(BaseModel):
    """Pixels or meters."""

    x: list[int | float]
    y: list[int | float]


class GroundTruth(BaseModel):
    """Ground truth data."""

    position: list[int]
    velocity: float


class ConfigGcp(BaseModel):
    """Configurations GCP."""

    apply: bool
    pixels: Position
    meters: Position
    ground_truth: list[GroundTruth]


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


class ConfigPreProcessing(BaseModel):
    """Configurations pre-processing."""

    rotate_image: int
    pre_roi: ConfigRoi
    roi: ConfigRoi
    image_correction: ConfigImageCorrection


class ConfigDataset(BaseModel):
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


class ConfigOtv(BaseModel):
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


class ConfigStivLine(BaseModel):
    """Config for STIV line."""

    start: list[int]
    end: list[int]


class ConfigStiv(BaseModel):
    """Configuration STIV."""

    window_shape: list[int]
    filter_window: int
    overlap: int
    ksize: int
    polar_filter_width: int
    lines: list[ConfigStivLine]


class Config(BaseModel):
    """Config class for awive."""

    dataset: ConfigDataset
    otv: ConfigOtv
    stiv: ConfigStiv
    preprocessing: ConfigPreProcessing

    @staticmethod
    def from_json(file_path: str, video_id: str | None = None):
        """Load config from json."""
        if video_id is None:
            return Config(**json.load(Path(file_path).open()))
        return Config(**json.load(Path(file_path).open())[video_id])
