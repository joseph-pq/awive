import pytest

from awive.config import ConfigGcp


@pytest.fixture
def config_gcp() -> ConfigGcp:
    """Fixture for creating a ConfigGcp object with default values."""
    return ConfigGcp(
        apply=True,
        pixels=[(0, 0)] * 4,
        meters=[(0.0, 0.0)] * 4,
        distances=None,
        ground_truth=None,
    )


def test_convert_str_keys_valid(config_gcp: ConfigGcp) -> None:
    """Test conversion of valid string keys to tuples."""
    input_dict = {"(0,1)": 1.0, "(2,3)": 2.0}
    expected = {(0, 1): 1.0, (2, 3): 2.0}
    converted = config_gcp.convert_str_keys_to_tuples(input_dict)
    assert converted == expected


def test_convert_str_keys_invalid(config_gcp: ConfigGcp) -> None:
    """Test error when converting invalid string keys."""
    input_dict = {"invalid": 1.0}
    with pytest.raises(ValueError) as exc:  # noqa: PT011
        config_gcp.convert_str_keys_to_tuples(input_dict)
    assert "Key 'invalid' is not a tuple" in str(exc.value)
