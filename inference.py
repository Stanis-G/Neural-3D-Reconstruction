import argparse
import os
import json
import yaml
from types import SimpleNamespace

import torch

from model import NeRF
from render import render_frames
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

    render_dir = create_result_dir(config.render_dir)

    render_frames(
        model=model,
        meta=meta,
        batch_size=config.batch_size,
        img_downsample=config.img_downsample,
        render_dir = render_dir,
        width=config.width,
        height=config.height,
    )
