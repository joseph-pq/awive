from typing import TypedDict

import numpy as np
from numpy.typing import NDArray


class Velocity(TypedDict):
    """TypedDict for velocity data.

    Attributes:
        velocity: A float representing the velocity.
        unit: A string representing the unit of measurement.
    """

    velocity: float
    count: int
    position: int


def get_simplest_water_flow(
    area: float,
    velocities: list[float] | dict[str, Velocity],
) -> float:
    """Compute the simplest water flow based on area and velocities.

    Just multiple area by mean velocities.

    Args:
        area: Area of the flow.
        velocities: List of velocities or a dictionary with velocity data.

    Returns:
        float: Simplest water flow.
    """
    if not velocities:
        raise ValueError("Velocities list cannot be empty.")

    if isinstance(velocities, dict):
        velocities = [v["velocity"] for v in velocities.values()]

    mean_velocity = sum(velocities) / len(velocities)
    return area * mean_velocity


def integrate_vels_over_depth(
    depths: NDArray, vels: NDArray, width: float, roughness: float
) -> float:
    """Integrate velocities over depth to compute water flow."""
    assert len(depths) == len(vels), (
        "Depth and velocities must have the same length."
    )
    water_flow = 0
    for d, v in zip(depths, vels):
        x = np.linspace(0, d, 100000)
        if d == 0:
            segment_area = 0.0
        else:
            y = v * ((x / d) ** (1 / roughness))
            segment_area = np.trapz(y, x)
        water_flow += segment_area * width
    return water_flow


def get_water_flow(
    depths: NDArray,
    vels: NDArray,
    old_depth: float,
    roughness: float,
    current_depth: float,
) -> float:
    """Compute the water flow based on profile and velocities.

    Args:
        depths: Array of depths (N,2) (meters, meters). First column is depth,
            second is distance between depths.
        vels: Array of velocities (N,) (m/s).
        old_depth: Depth when depths were measured (meters).
        roughness: Roughness coefficient (Manning's n).
        current_depth: Current depth (meters).

    Returns:
        float: Water flow (m^3/s).
    """
    assert depths.shape[0] == vels.shape[0], (
        "Depth and velocities must have the same length."
    )
    # Calculate width as mean distance between depth points
    width = np.abs(float(np.mean(np.diff(depths[:, 1]))))  # m
    # Update depths based on current and old depth
    new_depths = depths[:, 0] + (current_depth - old_depth)
    new_depths = np.where(new_depths < 0, 0, new_depths)
    return integrate_vels_over_depth(new_depths, vels, width, roughness)
