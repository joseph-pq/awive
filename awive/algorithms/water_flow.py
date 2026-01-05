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


def get_water_flow(
    depths: NDArray,
    vels: NDArray,
    old_depth: float,
    roughness: float,
    current_depth: float,
    a_coeff: float = 0.646465,  # Parámetro de escala optimizado
    b_coeff: float = 9.95,  # Parámetro de offset optimizado
) -> float:
    """Compute water flow using Manning's equation with depth profiles.

    This function calculates the water flow rate by integrating
    velocity profiles over the cross-sectional area of a river. It
    uses Manning's equation to model the vertical velocity distribution
    and applies a linear correction based on optimized coefficients.

    The vertical velocity profile follows: v(z) = v * (z/d)^(1/n)
    where z is the depth, d is the total depth, and n is the
    roughness coefficient.

    Args:
        depths: 2D array of shape (N, 2) where each row contains
            [depth, x_position]. First column represents the riverbed
            depth at each position along the width. Second column
            represents the x-coordinate along the river width.
        vels: 1D array of surface velocities corresponding to each
            depth measurement. Must have the same length as depths.
        old_depth: Reference water depth used during velocity
            measurements.
        roughness: Manning's roughness coefficient (n). Controls the
            shape of the vertical velocity profile.
        current_depth: Current water depth for which to compute the
            flow rate.
        a_coeff: Linear correction scaling coefficient. Default is
            0.646465 (optimized).
        b_coeff: Linear correction offset coefficient. Default is 9.95
            (optimized).

    Returns:
        float: Computed water flow rate in cubic meters per second
            (m³/s), after applying the linear correction:
            a_coeff * flow + b_coeff.

    Raises:
        AssertionError: If depths and vels arrays have different
            lengths.

    Note:
        The function adjusts depths based on the difference between
        current_depth and old_depth, and sets negative depths to zero.
        The final flow rate is computed by integrating the velocity
        profile over the entire cross-section.
    """
    assert depths.shape[0] == vels.shape[0], (
        "Depth and velocities must have the same length."
    )
    # Coordinates along the river width
    x = depths[:, 1]
    # Update depths based on current and old depth
    new_depths = depths[:, 0] + (current_depth - old_depth)
    new_depths = np.where(new_depths < 0, 0, new_depths)

    # Create fine x grid
    x_fine = np.linspace(x.min(), x.max(), 10000)

    # Interpolate riverbed profile
    new_depths_fine = np.interp(x_fine, x, new_depths)  # depths along width

    # Interpolate velocities
    vels_fine = np.interp(x_fine, x, vels)  # velocity along width

    # Compute water flow per width unit (using Manning's equation)
    # v(z) = v * (z/d)^(1/n)
    # q = integrate v(z) dz from 0 to d

    q_fine = vels_fine * new_depths_fine * (roughness) / (roughness + 1)

    # Compute water flow by integrating q over the width
    water_flow = np.trapz(q_fine, x_fine)

    # Apply linear correction
    return a_coeff * water_flow + b_coeff
