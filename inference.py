import argparse
import os
import json
import yaml
from types import SimpleNamespace
from tqdm import tqdm

import torch
from torchvision.io import write_png

from model import NeRF
from dataset import generate_rays
from utils import create_result_dir


def load_model(model_path, model_params, device):
    model = NeRF(**model_params)

    state_dict = torch.load(
        model_path,
        map_location=device,
        weights_only=True,
    )

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model


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

    # Convert [0, 1] -> [0, 255]
    image = (image.clamp(0, 1) * 255).byte()

    return image


if __name__ == "__main__":

    # Read path to config
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config YAML file",
    )
    args = parser.parse_args()

    # Upload config
    with open(args.config) as f:
        config = SimpleNamespace(**yaml.safe_load(f))

    # Read file with image metadata
    with open(os.path.join(config.data_dir, 'transforms.json'), "r", encoding="utf-8") as f:
        meta = json.load(f)["frames"]

    model = load_model(config.model_path, config.nn_params, config.device)

    create_result_dir(config.render_dir)

    for frame_idx, frame in tqdm(enumerate(meta), total=len(meta), desc='Frames'):
        camera_params = {
            'transform_matrix': frame['transform_matrix'],
            'camera_angle_x': frame['camera_angle_x'],
            'camera_angle_y': frame['camera_angle_y'],
            'fl_x': frame['fl_x'] / config.img_downsample,
            'fl_y': frame['fl_y'] / config.img_downsample,
            "w": config.width // config.img_downsample,
            "h": config.height // config.img_downsample,
        }
        image = render_image(
            model=model,
            camera_params=camera_params,
            batch_size=config.batch_size,
        )

        write_png(
            image.permute(2, 0, 1),
            os.path.join(config.render_dir, f"render_{frame_idx:04d}.png"),
        )
