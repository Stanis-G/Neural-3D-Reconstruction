
import argparse
import json
import math
from pathlib import Path

from utils import plot_camera_poses


Vector3 = tuple[float, float, float]


def normalize(v: Vector3) -> Vector3:
    length = math.sqrt(sum(x * x for x in v))
    if length < 1e-10:
        raise ValueError("Cannot normalize a zero-length vector.")
    return tuple(x / length for x in v)


def dot(a: Vector3, b: Vector3) -> float:
    return sum(x * y for x, y in zip(a, b))


def cross(a: Vector3, b: Vector3) -> Vector3:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def subtract(a: Vector3, b: Vector3) -> Vector3:
    return tuple(x - y for x, y in zip(a, b))


def camera_transform(
    position: Vector3,
    target: Vector3 = (0.0, 0.0, 0.0),
) -> list[list[float]]:
    """Create a Blender/NeRF camera-to-world matrix

    The camera's local -Z axis points toward the target
    The camera's local +Y axis is oriented as close to world-up as possible
    """
    forward = normalize(subtract(target, position))
    world_up = (0.0, 0.0, 1.0)

    # Avoid a degenerate orientation when looking vertically
    if abs(dot(forward, world_up)) > 0.999:
        world_up = (0.0, 1.0, 0.0)

    right = normalize(cross(forward, world_up))
    up = normalize(cross(right, forward))
    backward = tuple(-x for x in forward)

    # Rotation basis vectors are matrix columns
    return [
        [right[0], up[0], backward[0], position[0]],
        [right[1], up[1], backward[1], position[1]],
        [right[2], up[2], backward[2], position[2]],
        [0.0, 0.0, 0.0, 1.0],
    ]


def generate_validation_ring(
    output_dir: str | Path,
    axis: str = "z",
    axis_offset: float = 0.0,
    radius: float = 8.0,
    num_cameras: int = 36,
    width: int = 1920,
    height: int = 1080,
    fl_x: float = 2666.6664748650437,
    fl_y: float = 2743.0587138734313,
    image_prefix: str = "validation",
    start_angle_deg: float = 0.0,
) -> dict:
    """Generate a circular camera ring and save transforms.json

    Args:
        output_dir: Directory for generated transforms.json
        axis: Ring axis ('x', 'y', or 'z'). The ring lies in the plane perpendicular to this axis
        axis_offset: Offset along the selected axis relative to the origin
        radius: Distance from the ring center to each camera
        num_cameras: Number of evenly spaced cameras around 360 degrees
        width, height: Image dimensions in pixels
        fl_x, fl_y: Horizontal and vertical focal lengths in pixels
        image_prefix: Prefix for image paths stored in the JSON
        start_angle_deg: Angular offset of the first camera

    Returns:
        The generated transforms dictionary
    """
    axis = axis.lower()
    if axis not in {"x", "y", "z"}:
        raise ValueError("axis must be 'x', 'y', or 'z'")
    if radius <= 0:
        raise ValueError("radius must be positive")
    if num_cameras < 1:
        raise ValueError("num_cameras must be at least 1")
    if width <= 0 or height <= 0 or fl_x <= 0 or fl_y <= 0:
        raise ValueError("Image dimensions and focal lengths must be positive")

    origin = (0, 0, 0)

    # Coordinates that vary along the ring
    plane_axes = {
        "x": (1, 2),  # YZ plane
        "y": (0, 2),  # XZ plane
        "z": (0, 1),  # XY plane
    }
    u_axis, v_axis = plane_axes[axis]

    frames = []

    for i in range(num_cameras):
        angle = math.radians(start_angle_deg + 360.0 * i / num_cameras)

        # Calculate camera position
        position = list(origin)
        position[u_axis] += radius * math.cos(angle)
        position[v_axis] += radius * math.sin(angle)
        position[{"x": 0, "y": 1, "z": 2}[axis]] += axis_offset
        position = tuple(position)

        frames.append({
            "file_path": f"{image_prefix}_{i:03d}.png",
            "transform_matrix": camera_transform(position, (0.0, 0.0, 0.0)),
            "camera_angle_x": 2.0 * math.atan(width / (2.0 * fl_x)),
            "camera_angle_y": 2.0 * math.atan(height / (2.0 * fl_y)),
            "fl_x": fl_x,
            "fl_y": fl_y,
            "w": width,
            "h": height,
        })

    transforms = {"frames": frames}

    output_path = Path(output_dir) / "transforms.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(transforms, indent=4),
        encoding="utf-8",
    )

    return transforms


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a NeRF validation camera ring"
    )
    parser.add_argument("--output-dir", required=True, help="Output directory for transforms.json")
    parser.add_argument("--axis", choices=["x", "y", "z"], default="z")
    parser.add_argument("--axis-offset", type=float, default=0.0)
    parser.add_argument("--radius", type=float, default=8.0)
    parser.add_argument("--num-cameras", type=int, default=20)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--fl-x", type=float, default=2666.6664748650437)
    parser.add_argument("--fl-y", type=float, default=2743.0587138734313)
    parser.add_argument("--image-prefix", default="camera")
    parser.add_argument("--start-angle-deg", type=float, default=0.0)

    args = parser.parse_args()

    transforms = generate_validation_ring(
        output_dir=args.output_dir,
        axis=args.axis,
        axis_offset=args.axis_offset,
        radius=args.radius,
        num_cameras=args.num_cameras,
        width=args.width,
        height=args.height,
        fl_x=args.fl_x,
        fl_y=args.fl_y,
        image_prefix=args.image_prefix,
        start_angle_deg=args.start_angle_deg,
    )
    print(f"Saved {args.num_cameras} cameras to {args.output_dir}")

    plot_camera_poses(meta_train=[], meta_valid=transforms['frames'], experiment_dir=args.output_dir)
