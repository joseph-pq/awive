import datetime as dt
from pathlib import Path
from unittest.mock import MagicMock, Mock, mock_open, patch

import numpy as np
import pytest
import yaml

from awive.algorithms.velocimetry import process_video, velocimetry


@pytest.fixture
def mock_awive_fp(tmp_path: Path) -> Path:
    """Create a temporary awive config file path."""
    return tmp_path / "awive.yaml"


@pytest.fixture
def mock_config() -> Mock:
    """Create a mock Config object."""
    config = Mock()
    config.water_flow.profile.depths_meters.return_value = np.array(
        [
            [0.0, 0.0],
            [0.5, 5.0],
            [1.0, 10.0],
        ]
    )
    config.water_flow.profile.height = 1.0
    config.water_flow.roughness = 0.03
    config.preprocessing.ppm = 100
    config.preprocessing.resolution = 1.0
    return config


@pytest.fixture
def mock_otv_output() -> tuple[dict, None]:
    """Create mock OTV output."""
    return {
        "0": {"velocity": 1.5},
        "1": {"velocity": 2.0},
        "2": {"velocity": 2.5},
    }, None


@pytest.fixture
def mock_video_fp(tmp_path: Path) -> Path:
    """Create a temporary video file path."""
    return tmp_path / "video.mp4"


@pytest.fixture
def mock_config_for_velocimetry() -> Mock:
    """Create a mock Config object for velocimetry tests."""
    config = Mock()
    config.dataset.video_fp = Path("/old/video.mp4")
    config.preprocessing.resolution = 1.0
    config.water_flow.area = 10.0
    config.model_dump.return_value = {
        "dataset": {"video_fp": "/old/video.mp4"},
        "preprocessing": {"resolution": 1.0},
        "water_flow": {"area": 10.0},
    }
    return config


# Tests for process_video


@patch("awive.algorithms.velocimetry.run_otv")
@patch("awive.algorithms.velocimetry.get_simplest_water_flow")
@patch("builtins.open", new_callable=mock_open)
@patch("yaml.dump")
def test_process_video_without_wlevel(
    mock_yaml_dump: Mock,
    mock_file: Mock,
    mock_get_simplest: Mock,
    mock_run_otv: Mock,
    mock_awive_fp: Path,
    mock_otv_output: tuple[dict, None],
) -> None:
    """Test process_video without water level uses simplest water flow."""
    mock_run_otv.return_value = mock_otv_output
    mock_get_simplest.return_value = 15.0
    area = 10.0
    ts = dt.datetime(2026, 1, 15, 12, 0, 0, tzinfo=dt.UTC)

    process_video(awive_fp=mock_awive_fp, area=area, ts=ts, wlevel=None)

    # Verify run_otv was called
    mock_run_otv.assert_called_once_with(mock_awive_fp)

    # Verify get_simplest_water_flow was called
    mock_get_simplest.assert_called_once_with(
        area=area, velocities=mock_otv_output[0]
    )

    # Verify data was saved
    mock_file.assert_called_once_with("/root/awive/data.yaml", "w")
    mock_yaml_dump.assert_called_once()

    # Check the data structure
    saved_data = mock_yaml_dump.call_args[0][0]
    assert saved_data["timestamp"] == ts
    assert saved_data["velocimetry"] == [1.5, 2.0, 2.5]
    assert saved_data["water_flow"] == 15.0


@patch("awive.algorithms.velocimetry.run_otv")
@patch("awive.algorithms.velocimetry.get_water_flow")
@patch("awive.algorithms.velocimetry.awive.config.Config.from_fp")
@patch("builtins.open", new_callable=mock_open)
@patch("yaml.dump")
def test_process_video_with_wlevel(
    mock_yaml_dump: Mock,
    mock_file: Mock,
    mock_from_fp: Mock,
    mock_get_water_flow: Mock,
    mock_run_otv: Mock,
    mock_awive_fp: Path,
    mock_config: Mock,
    mock_otv_output: tuple[dict, None],
) -> None:
    """Test process_video with water level uses get_water_flow."""
    mock_run_otv.return_value = mock_otv_output
    mock_from_fp.return_value = mock_config
    mock_get_water_flow.return_value = 20.5
    area = 10.0
    wlevel = 0.8
    ts = dt.datetime(2026, 1, 15, 12, 0, 0, tzinfo=dt.UTC)

    process_video(awive_fp=mock_awive_fp, area=area, ts=ts, wlevel=wlevel)

    # Verify run_otv was called
    mock_run_otv.assert_called_once_with(mock_awive_fp)

    # Verify Config.from_fp was called
    mock_from_fp.assert_called_once_with(mock_awive_fp)

    # Verify get_water_flow was called
    mock_get_water_flow.assert_called_once()
    call_args = mock_get_water_flow.call_args
    assert np.array_equal(call_args[1]["vels"], np.array([1.5, 2.0, 2.5]))
    assert call_args[1]["old_depth"] == 1.0
    assert call_args[1]["roughness"] == 0.03
    assert call_args[1]["current_depth"] == wlevel

    # Verify data was saved
    mock_file.assert_called_once_with("/root/awive/data.yaml", "w")
    mock_yaml_dump.assert_called_once()

    # Check the data structure
    saved_data = mock_yaml_dump.call_args[0][0]
    assert saved_data["timestamp"] == ts
    assert saved_data["velocimetry"] == [1.5, 2.0, 2.5]
    assert saved_data["water_flow"] == 20.5


@patch("awive.algorithms.velocimetry.run_otv")
@patch("awive.algorithms.velocimetry.get_simplest_water_flow")
@patch("awive.algorithms.velocimetry.dt.datetime")
@patch("builtins.open", new_callable=mock_open)
@patch("yaml.dump")
def test_process_video_default_timestamp(
    mock_yaml_dump: Mock,
    mock_file: Mock,
    mock_datetime: Mock,
    mock_get_simplest: Mock,
    mock_run_otv: Mock,
    mock_awive_fp: Path,
    mock_otv_output: tuple[dict, None],
) -> None:
    """Test process_video uses current time when timestamp is None."""
    mock_now = dt.datetime(2026, 1, 15, 14, 30, 0, tzinfo=dt.UTC)
    mock_datetime.now.return_value = mock_now
    mock_datetime.UTC = dt.UTC
    mock_run_otv.return_value = mock_otv_output
    mock_get_simplest.return_value = 15.0

    process_video(awive_fp=mock_awive_fp, area=10.0, ts=None, wlevel=None)

    # Verify datetime.now was called
    mock_datetime.now.assert_called_once_with(dt.UTC)

    # Verify timestamp in saved data
    saved_data = mock_yaml_dump.call_args[0][0]
    assert saved_data["timestamp"] == mock_now


@patch("awive.algorithms.velocimetry.run_otv")
@patch("awive.algorithms.velocimetry.get_simplest_water_flow")
@patch("builtins.open", new_callable=mock_open)
@patch("yaml.dump")
def test_process_video_with_non_dict_values(
    mock_yaml_dump: Mock,
    mock_file: Mock,
    mock_get_simplest: Mock,
    mock_run_otv: Mock,
    mock_awive_fp: Path,
) -> None:
    """Test process_video filters non-dict values correctly."""
    # OTV output with mixed types
    otv_output = {
        "0": {"velocity": 1.5},
        "1": "not_a_dict",  # This should be filtered out
        "2": {"velocity": 2.5},
        "3": None,  # This should be filtered out
        "4": {"velocity": 3.0},
    }
    mock_run_otv.return_value = (otv_output, None)
    mock_get_simplest.return_value = 15.0

    process_video(
        awive_fp=mock_awive_fp,
        area=10.0,
        ts=dt.datetime.now(dt.UTC),
        wlevel=None,
    )

    # Verify only valid velocities are saved
    saved_data = mock_yaml_dump.call_args[0][0]
    assert saved_data["velocimetry"] == [1.5, 2.5, 3.0]


# Tests for velocimetry


@patch("awive.algorithms.velocimetry.awive.config.Config.from_fp")
@patch("awive.algorithms.velocimetry.process_video")
@patch("yaml.dump")
def test_velocimetry_updates_config(
    mock_yaml_dump: Mock,
    mock_process_video: Mock,
    mock_from_fp: Mock,
    tmp_path: Path,
) -> None:
    """Test velocimetry with new video path and resolution."""
    mock_awive_fp = tmp_path / "awive.yaml"
    mock_video_fp = tmp_path / "video.mp4"

    mock_config = Mock()
    mock_config.dataset.video_fp = Path("/old/video.mp4")
    mock_config.preprocessing.resolution = 1.0
    mock_config.water_flow.area = 10.0
    mock_config.model_dump.return_value = {
        "dataset": {"video_fp": "/old/video.mp4"},
        "preprocessing": {"resolution": 1.0},
        "water_flow": {"area": 10.0},
    }
    mock_from_fp.return_value = mock_config
    mock_yaml_dump.return_value = "mocked_yaml_string"

    velocimetry(
        awive_fp=mock_awive_fp,
        video_fp=mock_video_fp,
        ts=None,
        wlevel=None,
        resolution=0.5,
    )

    # Verify Config.from_fp was called
    mock_from_fp.assert_called_once_with(mock_awive_fp)

    # Verify video_fp and resolution were updated
    assert mock_config.dataset.video_fp == mock_video_fp
    assert mock_config.preprocessing.resolution == 0.5

    # Verify config was written
    mock_yaml_dump.assert_called_once()

    # Verify process_video was called
    mock_process_video.assert_called_once_with(
        awive_fp=mock_awive_fp,
        area=10.0,
        ts=None,
        wlevel=None,
    )


@patch("awive.algorithms.velocimetry.awive.config.Config.from_fp")
@patch("awive.algorithms.velocimetry.process_video")
@patch("yaml.dump")
def test_velocimetry_with_timestamp_and_wlevel(
    mock_yaml_dump: Mock,
    mock_process_video: Mock,
    mock_from_fp: Mock,
    tmp_path: Path,
) -> None:
    """Test velocimetry passes timestamp and water level to process_video."""
    mock_awive_fp = tmp_path / "awive.yaml"
    mock_video_fp = tmp_path / "video.mp4"

    mock_config = Mock()
    mock_config.dataset.video_fp = Path("/old/video.mp4")
    mock_config.preprocessing.resolution = 1.0
    mock_config.water_flow.area = 10.0
    mock_config.model_dump.return_value = {
        "dataset": {"video_fp": "/old/video.mp4"},
        "preprocessing": {"resolution": 1.0},
        "water_flow": {"area": 10.0},
    }
    mock_from_fp.return_value = mock_config
    mock_yaml_dump.return_value = "mocked_yaml_string"

    ts = dt.datetime(2026, 1, 15, 12, 0, 0, tzinfo=dt.UTC)
    wlevel = 0.8

    velocimetry(
        awive_fp=mock_awive_fp,
        video_fp=mock_video_fp,
        ts=ts,
        wlevel=wlevel,
        resolution=1.0,
    )

    # Verify process_video was called with correct parameters
    mock_process_video.assert_called_once_with(
        awive_fp=mock_awive_fp,
        area=10.0,
        ts=ts,
        wlevel=wlevel,
    )


@patch("awive.algorithms.velocimetry.awive.config.Config.from_fp")
@patch("awive.algorithms.velocimetry.process_video")
@patch("yaml.dump")
def test_velocimetry_default_resolution(
    mock_yaml_dump: Mock,
    mock_process_video: Mock,
    mock_from_fp: Mock,
    tmp_path: Path,
) -> None:
    """Test velocimetry uses default resolution of 1.0."""
    mock_awive_fp = tmp_path / "awive.yaml"
    mock_video_fp = tmp_path / "video.mp4"

    mock_config = Mock()
    mock_config.dataset.video_fp = Path("/old/video.mp4")
    mock_config.preprocessing.resolution = 1.0
    mock_config.water_flow.area = 10.0
    mock_config.model_dump.return_value = {
        "dataset": {"video_fp": "/old/video.mp4"},
        "preprocessing": {"resolution": 1.0},
        "water_flow": {"area": 10.0},
    }
    mock_from_fp.return_value = mock_config
    mock_yaml_dump.return_value = "mocked_yaml_string"

    velocimetry(
        awive_fp=mock_awive_fp,
        video_fp=mock_video_fp,
    )

    # Verify resolution was set to 1.0 (default)
    assert mock_config.preprocessing.resolution == 1.0
