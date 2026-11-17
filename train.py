import argparse
import os
import yaml
import json
from types import SimpleNamespace
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset import NeRFDataset, load_frame_image, sample_train_valid
from model import NeRF
from render import render_frames
from utils import create_result_dir, save_history, plot_camera_poses


def train(
    model,
    dataloader,
    optimizer,
    criterion,
    device='cpu',
    step_scheduler=None,
):
    model = model.to(device)

    model.train()
    train_loss = 0
    for ray_origin, ray_direction, pixel_color in tqdm(
        dataloader,
        desc='Training',
        leave=False,
    ):
        ray_origin = ray_origin.to(device)
        ray_direction = ray_direction.to(device)
        pixel_color = pixel_color.to(device)

        optimizer.zero_grad()

        color_pred, coarse_pred = model(ray_origin, ray_direction, return_coarse=True)
        loss = criterion(color_pred, pixel_color)
        if coarse_pred is not None:
            loss = loss + criterion(coarse_pred, pixel_color)

        loss.backward()
        optimizer.step()
        if step_scheduler is not None:
            step_scheduler.step()

        train_loss += loss.item() * pixel_color.shape[0]

    train_loss /= len(dataloader.dataset)

    return model, train_loss


def validate(
    model,
    dataloader,
    criterion,
    device='cpu',
):
    model = model.to(device)
    
    model.eval()
    valid_loss = 0
    with torch.no_grad():
        for ray_origin, ray_direction, pixel_color in tqdm(
            dataloader,
            desc='Validation',
            leave=False,
        ):
            ray_origin = ray_origin.to(device)
            ray_direction = ray_direction.to(device)
            pixel_color = pixel_color.to(device)

            color_pred = model(ray_origin, ray_direction)
            valid_loss += criterion(color_pred, pixel_color).item() * pixel_color.shape[0]

    valid_loss /= len(dataloader.dataset)

    return valid_loss


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

    torch.manual_seed(config.seed)
    model = NeRF(**config.nn_params).to(config.device)

    # Create dir with experiment results
    experiment_dir = create_result_dir(config.experiment_dir)

    # Save config
    with open(os.path.join(experiment_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(vars(config), f, sort_keys=False)

    # Read file with image metadata
    with open(os.path.join(config.data_dir, 'transforms.json'), "r", encoding="utf-8") as f:
        meta = json.load(f)["frames"]

    # Create train and validation datasets
    meta_train, meta_valid = sample_train_valid(
        meta, valid_frac=config.valid_frac, generator=torch.Generator().manual_seed(config.seed)
    )
    plot_camera_poses(
        meta_train,
        meta_valid,
        experiment_dir=experiment_dir,
    )

    train_dataset = NeRFDataset(
        data_dir=config.data_dir,
        meta=meta_train,
        num_rays_per_image=config.num_rays_per_image,
        img_downsample=config.img_downsample,
        seed=config.seed,
        white_bkgd=model.white_bkgd,
    )
    valid_dataset = NeRFDataset(
        data_dir=config.data_dir,
        meta=meta_valid,
        num_rays_per_image=getattr(config, "valid_rays_per_image", config.num_rays_per_image),
        img_downsample=config.img_downsample,
        seed=config.seed + 1,
        white_bkgd=model.white_bkgd,
    )

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=config.batch_size, shuffle=True)
    valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=config.batch_size, shuffle=False)

    valid_dataset.resample(0, "random", {})

    # Set training params
    optimizer = torch.optim.Adam(
        model.parameters(),
        **config.optimizer_params,
    )

    decay_steps = getattr(config, "lr_decay_steps", None)
    if decay_steps is not None:
        if decay_steps <= 0:
            raise ValueError("lr_decay_steps must be positive.")
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda step: 0.1 ** (step / decay_steps)
        )
    else:
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, **config.scheduler_params)

    criterion = nn.MSELoss()

    # Save original training images for visual comparison with renders
    train_original_dir = os.path.join(experiment_dir, 'renders_train', 'original')
    os.makedirs(train_original_dir, exist_ok=True)
    for i, frame in enumerate(meta_train[:config.render_num_images]):
        image, _, _ = load_frame_image(
            os.path.join(config.data_dir, frame["file_path"]), config.img_downsample, model.white_bkgd
        )
        image.save(os.path.join(train_original_dir, f"original_{i:04d}.png"))

    # Save original validation images for visual comparison with renders
    valid_original_dir = os.path.join(experiment_dir, 'renders_valid', 'original')
    os.makedirs(valid_original_dir, exist_ok=True)
    for i, frame in enumerate(meta_valid[:config.render_num_images]):
        image, _, _ = load_frame_image(
            os.path.join(config.data_dir, frame["file_path"]), config.img_downsample, model.white_bkgd
        )
        image.save(os.path.join(valid_original_dir, f"original_{i:04d}.png"))

    # Start training
    history = {
        "train_loss": [],
        "valid_loss": [],
    }
    for epoch in tqdm(range(1, config.num_epochs + 1), desc='Epochs'):
        # Sample new rays/pixels from the images
        train_dataset.resample(epoch, config.sample_strategy, config.sampler_params)
        model, train_loss = train(
            model=model,
            dataloader=train_dataloader,
            optimizer=optimizer,
            criterion=criterion,
            device=config.device,
            step_scheduler=scheduler if decay_steps is not None else None,
        )
        valid_loss = validate(
            model=model,
            dataloader=valid_dataloader,
            criterion=criterion,
            device=config.device,            
        )
        if decay_steps is None:
            scheduler.step()
        epoch_results = {
            "train_loss": train_loss,
            "valid_loss": valid_loss,
        }

        # Update history
        for k, v in epoch_results.items():
            history[k].append(v)
        save_history(history=history, experiment_dir=experiment_dir)

        # Render and save validation images
        if epoch >= config.render_from and epoch % config.render_every == 0:
            train_render_dir = os.path.join(experiment_dir, 'renders_train', f'epoch_{epoch}')
            valid_render_dir = os.path.join(experiment_dir, 'renders_valid', f'epoch_{epoch}')
            os.makedirs(train_render_dir, exist_ok=True)
            os.makedirs(valid_render_dir, exist_ok=True)
            render_frames(
                model=model,
                meta=meta_train[:config.render_num_images],
                batch_size=config.render_batch_size,
                img_downsample=config.img_downsample,
                render_dir=train_render_dir,
            )
            render_frames(
                model=model,
                meta=meta_valid[:config.render_num_images],
                batch_size=config.render_batch_size,
                img_downsample=config.img_downsample,
                render_dir=valid_render_dir,
            )

    # Save model
    torch.save(model.state_dict(), os.path.join(experiment_dir, "model.pth"))
