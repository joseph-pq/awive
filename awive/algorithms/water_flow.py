from typing import TypedDict

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import CubicSpline


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


def get_water_flow_experimental(
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
    # Coordinates along the river width
    x = depths[:, 1]
    # Update depths based on current and old depth
    new_depths = depths[:, 0] + (current_depth - old_depth)
    new_depths = np.where(new_depths < 0, 0, new_depths)

    # Interpolate riverbed profile
    riverbed_spline = CubicSpline(x, new_depths, bc_type="natural")

    # Create fine x grid
    x_fine = np.linspace(x.min(), x.max(), 10000)
    new_depths_fine = riverbed_spline(x_fine)  # depth along width

    # Interpolate velocities
    velocity_spline = CubicSpline(x, vels, bc_type="natural")
    vels_fine = velocity_spline(x_fine)  # velocity along width

    # Compute water flow per width unit (using Manning's equation)
    # v(z) = v * (z/d)^(1/n)
    # q = integrate v(z) dz from 0 to d

    q_fine = vels_fine * new_depths_fine * (roughness) / (roughness + 1)

    # Compute water flow by integrating q over the width
    water_flow = np.trapz(q_fine, x_fine)

    return water_flow


def get_water_flow_trapz(
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
    # Update depths based on current and old depth
    new_depths = depths[:, 0] + (current_depth - old_depth)
    new_depths = np.where(new_depths < 0, 0, new_depths)

    # river width
    river_x = depths[:, 1]

    # Water flow per width unit
    q = np.zeros(new_depths.shape[0])
    i = 0

    for d, v in zip(new_depths, vels):
        x = np.linspace(0, d, 100000)
        if d == 0:
            q[i] = 0.0
        else:
            y = v * ((x / d) ** (1 / roughness))
            q[i] = np.trapz(y, x)
        i = i + 1

    water_flow = np.trapz(q, river_x)
    return water_flow


def get_water_flow_experimental_v2(
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

    return water_flow


def get_water_flow_linear_corrected(
    depths: NDArray,
    vels: NDArray,
    old_depth: float,
    roughness: float,
    current_depth: float,
    a_coeff: float = 0.646465,  # Parámetro de escala optimizado
    b_coeff: float = 9.95,  # Parámetro de offset optimizado
) -> float:
    """Compute water flow with linear correction optimized for SENAMHI data.

    Uses the linear interpolation method with post-processing correction:
    corrected_flow = a * original_flow + b

    Default parameters were fitted to minimize error against SENAMHI data.

    Args:
        depths: Array of depths (N,2) (meters, meters). First column is depth,
            second is distance between depths.
        vels: Array of velocities (N,) (m/s).
        old_depth: Depth when depths were measured (meters).
        roughness: Roughness coefficient (Manning's n).
        current_depth: Current depth (meters).
        correction_a: Linear correction scale factor.
        correction_b: Linear correction offset.

    Returns:
        float: Corrected water flow (m^3/s).
    """
    # Get base calculation using the best performing method
    base_flow = get_water_flow_experimental_v2(
        depths, vels, old_depth, roughness, current_depth
    )

    # Apply linear correction
    corrected_flow = a_coeff * base_flow + b_coeff

    # Ensure non-negative flow
    return max(0.0, corrected_flow)


def get_water_flow_depth_corrected(
    depths: NDArray,
    vels: NDArray,
    old_depth: float,
    roughness: float,
    current_depth: float,
    depth_coeff_a: float = 0.6788,  # Coeficiente base optimizado
    depth_coeff_b: float = 0.1341,  # Coeficiente dependiente de profundidad
    offset: float = 4.9530,  # Offset constante optimizado
) -> float:
    """Compute water flow with depth-dependent correction.

    Applies correction: corrected_flow = (a + b * depth) * original_flow + offset
    This accounts for the correlation between error and water depth.

    Args:
        depths: Array of depths (N,2) (meters, meters). First column is depth,
            second is distance between depths.
        vels: Array of velocities (N,) (m/s).
        old_depth: Depth when depths were measured (meters).
        roughness: Roughness coefficient (Manning's n).
        current_depth: Current depth (meters).
        depth_coeff_a: Base correction coefficient.
        depth_coeff_b: Depth-dependent correction coefficient.
        offset: Constant offset.

    Returns:
        float: Corrected water flow (m^3/s).
    """
    # Get base calculation
    base_flow = get_water_flow_experimental_v2(
        depths, vels, old_depth, roughness, current_depth
    )

    # Apply depth-dependent correction
    correction_factor = depth_coeff_a + depth_coeff_b * current_depth
    corrected_flow = correction_factor * base_flow + offset

    return max(0.0, corrected_flow)


def get_water_flow_exponential_corrected(
    depths: NDArray,
    vels: NDArray,
    old_depth: float,
    roughness: float,
    current_depth: float,
    exp_coeff: float = 40.45064695055738,  # Coeficiente exponencial optimizado
    linear_coeff: float = 0.04104328781700687,  # Coeficiente lineal optimizado
    offset: float = -23.168956556695633,  # Offset optimizado
) -> float:
    """Compute water flow with exponential correction.

    Applies correction: corrected = exp_coeff * exp(depth) + linear_coeff * flow + offset
    This non-linear approach can capture complex depth-flow relationships.

    Args:
        depths: Array of depths (N,2) (meters, meters).
        vels: Array of velocities (N,) (m/s).
        old_depth: Depth when depths were measured (meters).
        roughness: Roughness coefficient (Manning's n).
        current_depth: Current depth (meters).
        exp_coeff: Exponential correction coefficient.
        linear_coeff: Linear correction coefficient.
        offset: Constant offset.

    Returns:
        float: Corrected water flow (m^3/s).
    """
    # Get base calculation
    base_flow = get_water_flow_experimental_v2(
        depths, vels, old_depth, roughness, current_depth
    )

    # Apply exponential correction
    corrected_flow = (
        exp_coeff * np.exp(current_depth) + linear_coeff * base_flow + offset
    )

    return max(0.0, corrected_flow)


def get_water_flow_polynomial_corrected(
    depths: NDArray,
    vels: NDArray,
    old_depth: float,
    roughness: float,
    current_depth: float,
    poly_coeffs: tuple = (
        0.8125,
        0.0438,
        -0.0289,
    ),  # (a, b, c) para ax² + bx + c
    flow_coeff: float = 0.9875,  # Coeficiente multiplicador del flujo base
) -> float:
    """Compute water flow with polynomial depth correction.

    Applies correction: corrected = flow_coeff * base_flow *
                       (poly_coeffs[0] + poly_coeffs[1]*depth + poly_coeffs[2]*depth²)

    Args:
        depths: Array of depths (N,2) (meters, meters).
        vels: Array of velocities (N,) (m/s).
        old_depth: Depth when depths were measured (meters).
        roughness: Roughness coefficient (Manning's n).
        current_depth: Current depth (meters).
        poly_coeffs: Polynomial coefficients (a, b, c) for depth correction.
        flow_coeff: Base flow multiplier.

    Returns:
        float: Corrected water flow (m^3/s).
    """
    # Get base calculation
    base_flow = get_water_flow_experimental_v2(
        depths, vels, old_depth, roughness, current_depth
    )

    # Apply polynomial correction
    depth_factor = (
        poly_coeffs[0]
        + poly_coeffs[1] * current_depth
        + poly_coeffs[2] * current_depth**2
    )

    corrected_flow = flow_coeff * base_flow * depth_factor

    return max(0.0, corrected_flow)


def get_water_flow_advanced_params(
    depths: NDArray,
    vels: NDArray,
    old_depth: float,
    roughness: float,
    current_depth: float,
    velocity_exp: float = -2.533090413373657,  # Exponente para perfil de velocidad
    depth_scaling: float = 0.985343542633097,  # Factor de escalado de profundidad
    width_correction: float = 0.5320227238237,  # Factor de corrección de ancho
    roughness_modifier: float = 1.0831297709762364,  # Modificador de rugosidad
) -> float:
    """Compute water flow with advanced parameter optimization.

    This function includes additional parameters that can be tuned:
    - velocity_exp: Controls velocity profile shape (default 1/roughness)
    - depth_scaling: Scales effective depth calculation
    - width_correction: Adjusts effective channel width
    - roughness_modifier: Fine-tunes roughness coefficient

    Args:
        depths: Array of depths (N,2) (meters, meters).
        vels: Array of velocities (N,) (m/s).
        old_depth: Depth when depths were measured (meters).
        roughness: Base roughness coefficient (Manning's n).
        current_depth: Current depth (meters).
        velocity_exp: Velocity profile exponent.
        depth_scaling: Depth scaling factor.
        width_correction: Width correction factor.
        roughness_modifier: Roughness modification factor.

    Returns:
        float: Water flow (m^3/s).
    """
    assert depths.shape[0] == vels.shape[0], (
        "Depth and velocities must have the same length."
    )

    # Coordinates along the river width
    x = depths[:, 1] * width_correction

    # Update depths with scaling
    new_depths = depths[:, 0] + (current_depth - old_depth) * depth_scaling
    new_depths = np.where(new_depths < 0, 0, new_depths)

    # Create fine x grid
    x_fine = np.linspace(x.min(), x.max(), 10000)

    # Interpolate riverbed profile and velocities
    new_depths_fine = np.interp(x_fine, x, new_depths)
    vels_fine = np.interp(x_fine, x, vels)

    # Modified roughness coefficient
    effective_roughness = roughness * roughness_modifier

    # Compute water flow per width unit with custom velocity exponent
    q_fine = (
        vels_fine
        * new_depths_fine
        * effective_roughness
        / (effective_roughness + velocity_exp)
    )

    # Integrate over width
    water_flow = np.trapz(q_fine, x_fine)

    return max(0.0, water_flow)
