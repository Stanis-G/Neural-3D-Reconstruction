import os
from tqdm import tqdm

import torch
from torchvision.io import write_png

from dataset import generate_rays


@torch.no_grad()
def render_image(
    model,
    camera_params,
    batch_size,
):
    """Render an image from a single camera"""
    # Generate rays for every pixel
    ray_origins, ray_directions = generate_rays(camera_params)

    # Move rays to inference device
    device = next(model.parameters()).device
    ray_origins = ray_origins.to(device)
    ray_directions = ray_directions.to(device)

    rendered_pixels = []

    # Process rays in batches
    for start in tqdm(
        range(0, len(ray_origins), batch_size),
        total=(len(ray_origins) + batch_size - 1) // batch_size,
        desc='Rendering',
        leave=False,
    ):
        origins = ray_origins[start:start + batch_size]
        directions = ray_directions[start:start + batch_size]

        # Run model
        pixel_colors = model(origins, directions)

        rendered_pixels.append(pixel_colors.cpu())

    # Combine batches to shape (H * W, 3)
    rendered_pixels = torch.cat(rendered_pixels, dim=0)

    # Convert (H * W, 3) -> (H, W, 3)
    image = rendered_pixels.reshape(
        camera_params["h"],
        camera_params["w"],
        3,
    )

    return image.clamp(0, 1)


def render_frames(
    model,
    meta,
    batch_size,
    img_downsample,
    render_dir,
    width=None,
    height=None,
):
    for frame_idx, frame in tqdm(enumerate(meta), total=len(meta), desc='Frames', leave=False):
        frame_width = width or frame['w']
        frame_height = height or frame['h']
        camera_params = {
            'transform_matrix': frame['transform_matrix'],
            'camera_angle_x': frame['camera_angle_x'],
            'camera_angle_y': frame['camera_angle_y'],
            'fl_x': frame['fl_x'] / img_downsample,
            'fl_y': frame['fl_y'] / img_downsample,
            "w": frame_width // img_downsample,
            "h": frame_height // img_downsample,
        }
        image = render_image(
            model=model,
            camera_params=camera_params,
            batch_size=batch_size,
        )

        # Convert [0, 1] -> [0, 255]
        write_png(
            (image * 255).byte().permute(2, 0, 1),
            os.path.join(render_dir, f"render_{frame_idx:04d}.png"),
        )
