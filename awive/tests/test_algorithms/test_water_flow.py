import numpy as np
import pytest

from awive.algorithms.water_flow import (
    Velocity,
    get_simplest_water_flow,
    get_water_flow,
)


class TestGetSimplestWaterFlow:
    """Test the get_simplest_water_flow function."""

    def test_empty_velocities(self) -> None:
        """Test that an empty list of velocities raises a ValueError."""
        with pytest.raises(
            ValueError,
            match="Velocities list cannot be empty.",
        ):
            get_simplest_water_flow(area=10.0, velocities=[])

    @pytest.mark.parametrize(
        ("area", "velocities", "expected_flow"),
        [
            (10.0, [2.0, 3.0, 4.0], 30.0),
            (5.0, [1.0, 2.0, 3.0, 4.0], 12.5),
            (
                2.0,
                [5.0, 5.0, 5.0],
                10.0,
            ),  # Test with same velocities
            (1.0, [10.0], 10.0),  # Test with single velocity
        ],
    )
    def test_list(
        self, area: float, velocities: list[float], expected_flow: float
    ) -> None:
        """Test with a list of velocities."""
        assert (
            get_simplest_water_flow(area=area, velocities=velocities)
            == expected_flow
        )

    @pytest.mark.parametrize(
        ("area", "velocities", "expected_flow"),
        [
            (
                10.0,
                {
                    "0": {"velocity": 2.0, "count": 1, "position": 1},
                    "1": {"velocity": 4.0, "count": 1, "position": 2},
                },
                30.0,
            ),
            (
                5.0,
                {
                    "0": {"velocity": 1.0, "count": 1, "position": 1},
                    "1": {"velocity": 3.0, "count": 1, "position": 2},
                },
                10.0,
            ),
        ],
    )
    def test_get_simplest_water_flow_dict(
        self,
        area: float,
        velocities: dict[str, Velocity],
        expected_flow: float,
    ) -> None:
        """Test with a dictionary of velocities."""
        assert (
            get_simplest_water_flow(area=area, velocities=velocities)
            == expected_flow
        )


def test_velocity_type() -> None:
    """Test the Velocity TypedDict."""
    velocity: Velocity = {"velocity": 1.0, "count": 1, "position": 1}
    assert velocity["velocity"] == 1.0
    assert velocity["count"] == 1
    assert velocity["position"] == 1


def test_water_flow_w_profile() -> None:
    """Test the get_water_flow function."""
    depths = np.array(  # m
        [
            [0.28, 0.0],
            [0.48, 0.5],
            [0.58, 1.0],
            [0.68, 1.5],
            [0.88, 2.0],
            [1.18, 2.5],
            [1.48, 3.0],
            [1.28, 3.5],
            [1.18, 4.0],
            [1.18, 4.5],
            [1.18, 5.0],
            [1.18, 5.5],
            [1.08, 6.0],
            [0.88, 6.5],
            [0.78, 7.0],
            [0.78, 7.5],
            [0.58, 8.0],
        ]
    )
    vels = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 8, 7, 6, 5, 4, 3, 2, 1])  # m/s
    roughness = 8

    wf = get_water_flow(
        depths, vels, old_depth=2.0, roughness=roughness, current_depth=2.0
    )
    wf2 = get_water_flow(
        depths, vels, old_depth=3.0, roughness=roughness, current_depth=3.0
    )

    assert wf == wf2
