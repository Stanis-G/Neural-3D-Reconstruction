import os
from PIL import Image

import torch
from torchvision.transforms.functional import pil_to_tensor
from torch.utils.data import Dataset


def pixel_to_camera_rays(pixel_coords, camera_params):
    """
    Convert pixel coordinates into camera-space ray directions.

    Args:
        pixel_coords: Pixel coordinates, shape (..., 2), where
            the last dimension contains (x, y).
        camera_params: Camera parameters.

    Returns:
        camera_directions: Camera-space ray directions, shape (..., 3).
    """
    fx = camera_params["fl_x"]
    fy = camera_params["fl_y"]
    w = camera_params["w"]
    h = camera_params["h"]

    x = pixel_coords[..., 0]
    y = pixel_coords[..., 1]

    # Principal point
    cx = w / 2
    cy = h / 2

    # Pixel -> camera space
    camera_directions = torch.stack(
        [
            (x - cx) / fx,
            -(y - cy) / fy,
            -torch.ones_like(x),
        ],
        dim=-1,
    )

    return camera_directions


def camera_to_world_rays(camera_directions, camera_params):
    """
    Convert camera-space ray directions into world-space rays.

    Args:
        camera_directions: Camera-space ray directions, shape (..., 3).
        camera_params: Camera parameters.

    Returns:
        ray_origins: Ray origins, shape (..., 3).
        ray_directions: Normalized world-space ray directions, shape (..., 3).
    """
    transform = torch.tensor(
        camera_params["transform_matrix"],
        dtype=torch.float32,
    )

    R = transform[:3, :3]
    origin = transform[:3, 3]

    # Camera space -> world space
    ray_directions = camera_directions @ R.T

    # Normalize directions
    ray_directions = ray_directions / torch.linalg.norm(
        ray_directions,
        dim=-1,
        keepdim=True,
    )

    # Same origin for every ray
    ray_origins = origin.expand_as(ray_directions)

    return ray_origins, ray_directions


def generate_rays(camera_params):
    """
    Generate rays for every pixel in an image

    Args:
        camera_params: Camera parameters

    Returns:
        ray_origins: Shape (H * W, 3).
        ray_directions: Shape (H * W, 3).
    """
    w = camera_params["w"]
    h = camera_params["h"]

    # Generate pixel coordinates
    y, x = torch.meshgrid(
        torch.arange(h, dtype=torch.float32),
        torch.arange(w, dtype=torch.float32),
        indexing="ij",
    )

    pixel_coords = torch.stack(
        [x, y],
        dim=-1,
    )  # (H, W, 2)

    # Pixel -> camera space
    camera_directions = pixel_to_camera_rays(
        pixel_coords,
        camera_params,
    )

    # Camera space -> world space
    ray_origins, ray_directions = camera_to_world_rays(
        camera_directions,
        camera_params,
    )

    # Flatten image dimensions
    ray_origins = ray_origins.reshape(-1, 3)
    ray_directions = ray_directions.reshape(-1, 3)

    return ray_origins, ray_directions


class NeRFDataset(Dataset):

    def __init__(
        self,
        data_dir,
        meta,
        num_rays_per_image,
        img_downsample,
        seed,
    ):
        """
        Build NeRF dataset and precompute camera ray directions.

        One dataset item is one ray

        Args:
            data_dir: folder containing `transforms.json` and frame images
            meta: list of dicts with paths to images and camera params
            num_rays_per_image: number of rays to sample from each image
            img_downsample: integer image downsampling factor. Affects camera intrinsics too
            seed: random seed for reproducible ray sampling
        """
        self.data_dir = data_dir
        self.meta = meta
        self.num_rays_per_image = num_rays_per_image
        self.img_downsample = img_downsample
        self.seed = seed


    def _sample_pixels(self, seed):
        """Randomly sample pixel coordinates and colors from each image"""

        generator = torch.Generator()
        generator.manual_seed(seed)

        ray_pixels = []
        pixel_colors = []
        all_camera_params = []

        for frame in self.meta:

            # Read image
            image_path = os.path.join(self.data_dir, frame["file_path"])
            image = Image.open(image_path)

            width, height = image.size

            # Apply downsampling
            width //= self.img_downsample
            height //= self.img_downsample

            image = image.resize((width, height))

            # Convert to tensor: [H, W, C], values in [0, 1]
            image_tensor = pil_to_tensor(image).float() / 255

            num_pixels = width * height

            if self.num_rays_per_image > num_pixels:
                raise ValueError(
                    f"num_rays_per_image ({self.num_rays_per_image})"
                    f"is greater than the number of pixels ({num_pixels})"
                    f"in image {image_path}"
                )

            # Sample unique pixel indices
            indices = torch.randperm(
                num_pixels,
                generator=generator,
            )[:self.num_rays_per_image]

            # Convert flattened indices to (x, y)
            ys = indices // width
            xs = indices % width
            pixels = torch.stack((xs, ys), dim=1)

            # Extract RGB colors for pixels of current image [C, num_rays_per_image]
            colors = image_tensor[:, ys, xs]

            ray_pixels.append(pixels)
            pixel_colors.append(colors)

            # Extract and modify camera instrinsics for sampled pixels (since downsampling is applied to image)
            camera_params = {
                'transform_matrix': frame['transform_matrix'],
                'camera_angle_x': frame['camera_angle_x'],
                'camera_angle_y': frame['camera_angle_y'],
                'fl_x': frame['fl_x'] / self.img_downsample,
                'fl_y': frame['fl_y'] / self.img_downsample,
                'w': width,
                'h': height,
            }

            all_camera_params.append(camera_params)

        # Return tensors of sampled pixels and modified camera params
        return torch.stack(ray_pixels), torch.stack(pixel_colors).transpose(1, 2), all_camera_params


    def resample(self, epoch):
        """Resample pixels for all images for a new training epoch"""
        seed = self.seed + epoch

        # Generate sampled pixel coordinates of shape (n_images, num_rays_per_image, 2)
        # 2 means x and y coordinates of a pixel
        # Generate sampled pixel colors of shape (n_images, num_rays_per_image, 3)
        # Get modified camera params
        self.ray_pixels, self.pixel_colors, self.all_camera_params = self._sample_pixels(seed)


    def __getitem__(self, index):
        """Sampling ray origin, direction and color"""

        # Sample pixel from tensor of shape (n_images, num_rays_per_images, coords) using single value index
        image_idx = index // self.num_rays_per_image
        ray_idx = index % self.num_rays_per_image

        # Get pixel coords in an image and color
        pixel_coords = self.ray_pixels[image_idx, ray_idx]
        pixel_color = self.pixel_colors[image_idx, ray_idx]

        # Extract camera params
        camera_params = self.all_camera_params[image_idx]

        camera_direction = pixel_to_camera_rays(
            pixel_coords,
            camera_params,
        )

        ray_origin, ray_direction = camera_to_world_rays(
            camera_direction,
            camera_params,
        )

        return ray_origin, ray_direction, pixel_color


    def __len__(self):
        """Get num of rays"""
        # Num of images * num rays per image
        return len(self.meta) * self.num_rays_per_image


def sample_points_on_ray(
    ray_origins: torch.Tensor,
    ray_directions: torch.Tensor,
    near: float,
    far: float,
    num_samples: int,
) -> torch.Tensor:
    """
    Sample 3D points along a ray.

    Args:
        ray_origins: Ray origins, shape (batch, 3)
        ray_directions: Ray directions, shape (batch, 3)
        near: Near bound
        far: Far bound
        num_samples: Number of points to sample per ray

    Returns:
        Points along the ray, shape (batch, num_samples, 3)
    """

    # Sample distances from ray origin
    t = torch.linspace(
        near,
        far,
        num_samples,
        device=ray_origins.device,
    )

    # Convert dimensions
    t = t[None, :, None]
    ray_origins = ray_origins[:, None, :]
    ray_directions = ray_directions[:, None, :]

    # Retrieve coords of points at sampled distances along a ray
    points = ray_origins + t * ray_directions
    return points


def sample_train_valid(items, valid_frac):
    """Randomly sample train and validation items from a list"""
    num_valid_samples = int(len(items) * valid_frac)
    indices = torch.randperm(len(items))
    train_indices = indices[num_valid_samples:]
    valid_indices = indices[:num_valid_samples]

    items_train = [items[i] for i in train_indices]
    items_valid = [items[i] for i in valid_indices]
    return items_train, items_valid
