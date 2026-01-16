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
            match=r"Velocities list cannot be empty.",
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
            (0.0, [1.0, 2.0], 0.0),  # Test with zero area
            (10.0, [0.0, 0.0], 0.0),  # Test with zero velocities
            (7.5, [1.5, 2.5, 3.5, 4.5], 22.5),  # Test with decimal values
        ],
    )
    def test_list(
        self, area: float, velocities: list[float], expected_flow: float
    ) -> None:
        """Test with a list of velocities."""
        result = get_simplest_water_flow(area=area, velocities=velocities)
        assert np.isclose(result, expected_flow), (
            f"Expected {expected_flow}, got {result}"
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
            (
                2.0,
                {
                    "0": {"velocity": 5.0, "count": 1, "position": 1},
                },
                10.0,
            ),  # Single velocity in dict
        ],
    )
    def test_get_simplest_water_flow_dict(
        self,
        area: float,
        velocities: dict[str, Velocity],
        expected_flow: float,
    ) -> None:
        """Test with a dictionary of velocities."""
        result = get_simplest_water_flow(area=area, velocities=velocities)
        assert np.isclose(result, expected_flow), (
            f"Expected {expected_flow}, got {result}"
        )


def test_velocity_type() -> None:
    """Test the Velocity TypedDict."""
    velocity: Velocity = {"velocity": 1.0, "count": 1, "position": 1}
    assert velocity["velocity"] == 1.0
    assert velocity["count"] == 1
    assert velocity["position"] == 1


def test_water_flow_w_profile() -> None:
    """Test the get_water_flow function with depth profile."""
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

    # When depth difference is same, flow should be equal
    assert np.isclose(wf, wf2), f"Expected equal flows, got {wf} and {wf2}"
    # Flow should be positive
    assert wf > 0, f"Flow should be positive, got {wf}"


def test_water_flow_mismatched_arrays() -> None:
    """Test that mismatched depths and velocities raise an error."""
    depths = np.array([[0.5, 0.0], [1.0, 1.0]])
    vels = np.array([1.0])  # Only 1 velocity for 2 depths

    with pytest.raises(AssertionError, match="must have the same length"):
        get_water_flow(
            depths,
            vels,
            old_depth=1.0,
            roughness=8,
            current_depth=1.0,
        )


def test_water_flow_negative_depths() -> None:
    """Test handling of negative depths (sets them to zero)."""
    depths = np.array([[0.5, 0.0], [1.0, 1.0], [0.8, 2.0]])
    vels = np.array([1.0, 2.0, 1.5])

    # old_depth > max depth causes negative adjusted depths
    # Function should handle this by setting negative depths to 0
    wf = get_water_flow(
        depths,
        vels,
        old_depth=2.0,  # Higher than max depth
        roughness=8,
        current_depth=0.5,
    )
    # Should not raise error and should return a value
    assert isinstance(wf, (float, np.floating))
    assert wf >= 0, f"Flow should be non-negative, got {wf}"


def test_water_flow_custom_coefficients() -> None:
    """Test linear correction with custom a_coeff and b_coeff."""
    depths = np.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]])
    vels = np.array([1.0, 1.0, 1.0])

    # Test with coefficient 1.0 and offset 0 (no correction)
    wf1 = get_water_flow(
        depths,
        vels,
        old_depth=1.0,
        roughness=8,
        current_depth=1.0,
        a_coeff=1.0,
        b_coeff=0.0,
    )

    # Test with coefficient 0.5 and offset 10
    wf2 = get_water_flow(
        depths,
        vels,
        old_depth=1.0,
        roughness=8,
        current_depth=1.0,
        a_coeff=0.5,
        b_coeff=10.0,
    )

    # Verify linear relationship: wf2 = 0.5 * wf1 + 10
    expected = 0.5 * wf1 + 10.0
    assert np.isclose(wf2, expected), f"Expected {expected}, got {wf2}"


def test_water_flow_depth_change_increases_flow() -> None:
    """Test that increasing depth increases flow."""
    depths = np.array([[1.0, 0.0], [1.5, 1.0], [1.0, 2.0]])
    vels = np.array([2.0, 3.0, 2.0])
    roughness = 8

    # Lower depth
    wf_low = get_water_flow(
        depths,
        vels,
        old_depth=1.0,
        roughness=roughness,
        current_depth=1.0,
    )

    # Higher depth
    wf_high = get_water_flow(
        depths,
        vels,
        old_depth=1.0,
        roughness=roughness,
        current_depth=2.0,
    )

    # Higher depth should result in higher flow
    assert wf_high > wf_low, (
        f"Higher depth should increase flow: {wf_low} vs {wf_high}"
    )


def test_water_flow_single_measurement() -> None:
    """Test with minimal single measurement point."""
    depths = np.array([[1.0, 0.0]])
    vels = np.array([2.0])

    wf = get_water_flow(
        depths,
        vels,
        old_depth=1.0,
        roughness=8,
        current_depth=1.0,
    )

    # Should handle single point without error
    assert isinstance(wf, (float, np.floating))
    assert wf > 0


def test_water_flow_wide_profile() -> None:
    """Test with wide river profile (many measurement points)."""
    num_points = 50
    x_positions = np.linspace(0, 10, num_points)
    # Parabolic depth profile (deeper in middle)
    depths_values = 2.0 - 0.5 * (x_positions - 5) ** 2 / 25
    depths = np.column_stack([depths_values, x_positions])
    # Velocity profile (faster in middle)
    vels = 3.0 - 0.3 * (x_positions - 5) ** 2 / 25

    wf = get_water_flow(
        depths,
        vels,
        old_depth=1.5,
        roughness=8,
        current_depth=1.5,
    )

    # Should handle many points and produce reasonable result
    assert isinstance(wf, (float, np.floating))
    assert wf > 0, f"Flow should be positive, got {wf}"
