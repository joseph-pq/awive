import cv2
import numpy as np
import numpy.typing as npt


def draw_velocities(
    velocities: dict[str, dict[str, float]],
    image: npt.NDArray[np.uint8],
    max_velocity: float = 5,
) -> npt.NDArray[np.uint8]:
    """Draw velocities arrows in an image.

    The color of those arrows depends on the velocity value. max is red and
    min is blue.

    Args:
        velocities: dictionary of velocities. keys are the velocity and values
            are dicts with three keys: "velocity", "position", and "count".
            velocity: velocity value in p/s.
            position: position of the velocity in pixels in y column
            count: number of occurrences of the velocity.
        image: image to draw the velocities on.
        max_velocity: maximum velocity value. if a velocity is higher than this
            value, it will be considered as this value.

    Returns:
        an image with the velocities arrows.
    """
    # Extract all velocity values
    velocity_values = [v["velocity"] for v in velocities.values()]

    # Determine min and max velocities
    min_velocity = min(min(velocity_values), 0)
    max_velocity = max(max(velocity_values), max_velocity)
    max_length = image.shape[1] // 4
    thickness = image.shape[0] // 100

    # Iterate over the velocities and draw arrows
    for velocity_info in velocities.values():
        velocity = velocity_info["velocity"]
        position = (2 * image.shape[0] // 3, velocity_info["position"])

        # Normalize velocity between 0 and 1
        norm_velocity = (velocity - min_velocity) / (
            max_velocity - min_velocity
        )
        # Calculate the end position of the arrow based on the velocity
        end_position = (
            position[0] - int(norm_velocity * max_length),
            position[1],
        )

        # Map normalized velocity to color
        color = (
            cv2.applyColorMap(
                np.array([[int(255 - norm_velocity * 255)]], dtype=np.uint8),
                cv2.COLORMAP_JET,
            )
            .flatten()
            .tolist()
        )

        # Draw the arrow
        cv2.arrowedLine(  # pyright: ignore[reportCallIssue]
            image,  # type: ignore[arg-type]
            position,  # pyright: ignore[reportArgumentType]
            end_position,  # pyright: ignore[reportArgumentType]
            color,
            thickness=thickness,
            tipLength=0.15,
        )

    return image


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    velocities = {
        "0": {"velocity": 1.0, "position": 100, "count": 10},
        "1": {"velocity": 3.0, "position": 200, "count": 20},
        "2": {"velocity": 4.0, "position": 300, "count": 30},
        "3": {"velocity": 2.0, "position": 400, "count": 30},
    }
    image = np.zeros((500, 500, 3), dtype=np.uint8)
    image = draw_velocities(velocities, image)
    plt.imshow(image)
    plt.show()
