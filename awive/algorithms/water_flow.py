from typing import TypedDict


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
