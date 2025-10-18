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


def test_parse_tuple_keys_valid(config_gcp: ConfigGcp) -> None:
    """Test parsing of valid string keys to tuples."""
    input_dict = {
        "(0,1)": 1.0,
        "(2,3)": 2.0,
        "4,5": 3.0,
        "  (  6 , 7 )  ": 4.0,
    }
    expected = {(0, 1): 1.0, (2, 3): 2.0, (4, 5): 3.0, (6, 7): 4.0}
    parsed = config_gcp.parse_tuple_keys(input_dict)
    assert parsed == expected


def test_parse_tuple_keys_invalid(config_gcp: ConfigGcp) -> None:
    """Test error when parsing invalid string keys."""
    input_dict = {"invalid": 1.0, "(1,2,3)": 2.0, "(a,b)": 3.0}
    with pytest.raises(ValueError) as exc:  # noqa: PT011
        config_gcp.parse_tuple_keys(input_dict)
    assert (
        "Key 'invalid' is not a valid tuple" in str(exc.value)
        or ("Key '(1,2,3)' is not a valid tuple" in str(exc.value))
        or ("Key '(a,b)' is not a valid tuple" in str(exc.value))
    )
