# Neural 3D Reconstruction

Neural rendering and 3D reconstruction experiments built around multi-view images and known camera poses. The repository is aimed at **novel view synthesis** (rendering scenes from new viewpoints) and **neural 3D reconstruction** from captured or synthetic views

Today the codebase implements a **NeRF** (Neural Radiance Fields) trainer and renderer. Planned extensions include **3D Gaussian Splatting** and utility scripts to extract triangle meshes from trained representations—those parts are not in the repo yet, but the layout is intended to grow into a small toolkit rather than a single NeRF script

## What you can do now


| Task                                                            | Entry point          |
| --------------------------------------------------------------- | -------------------- |
| Train a NeRF on your multi-view dataset                         | `train.py`           |
| Render images from a trained checkpoint                         | `inference.py`       |
| Build synthetic camera rings for evaluation / fly-through poses | `gen_valid_rings.py` |


Training optimizes a network to predict RGB along rays; rendering walks the same model over full images for each camera in `transforms.json`.

## Requirements

- Python 3.10+ (tested with PyTorch CUDA builds)
- NVIDIA GPU recommended for training (see note below)
- Dependencies: `requirements.txt` (PyTorch with CUDA 12.2 wheel index, `matplotlib`, `tqdm`)

```bash
python -m venv venv
source venv/Scripts/activate   # Windows Git Bash
pip install -r requirements.txt
```

**Training device:** `train.py` moves the model to CUDA during the training loop. Set `device: cuda` in your training config and use a GPU environment. Inference can run on CPU if you set `device: cpu` in the inference config.

## Dataset layout

Data follows the common NeRF / Blender JSON format:

```
your_dataset/
├── transforms.json
├── frame_000.png
├── frame_001.png
└── ...
```

`transforms.json` must contain a `"frames"` list. Each frame entry needs at least:

- `file_path` — image path relative to `data_dir`
- `transform_matrix` — 4×4 camera-to-world matrix
- `fl_x`, `fl_y` — focal lengths in pixels
- `w`, `h` — image size
- `camera_angle_x`, `camera_angle_y` — field-of-view angles (stored for compatibility)

Images may include an alpha channel; samplers can use it when sampling pixels (see `foreground` strategy)

## Configuration

Two templates ship with the repo:


| File                        | Purpose                                                                   |
| --------------------------- | ------------------------------------------------------------------------- |
| `default_train_config.yaml` | Hyperparameters, paths, training and periodic render settings             |
| `default_inf_config.yaml`   | Checkpoint path, architecture params (must match training), render output |


Copy them to local files (for example `config_train.yaml` and `config_inf.yaml`), edit paths and hyperparameters, and pass the paths to the scripts. Local config copies are typically gitignored so experiments stay on your machine.

Important training fields:

- `data_dir`, `experiment_dir` — input dataset and output run folder
- `nn_params` — network depth, ray sampling (`near` / `far` / `num_samples`), positional encoding sizes
- `num_rays_per_image`, `sample_strategy`, `sampler_params` — how many rays per image per epoch and how pixels are chosen
- `valid_frac` — random hold-out fraction of frames for validation loss
- `render_from`, `render_every`, `render_num_images` — when to save train/valid renders during training

Inference must reuse the same `nn_params` as training and set `model_path` to a saved `model.pth`

## Usage
### Train

```bash
python train.py --config config_train.yaml
```

Each run creates a new directory under `experiment_dir` (if the path already exists, a suffix `_1`, `_2`, … is added). The run folder contains:

- `config.yaml` — frozen copy of the training config
- `history.json`, `history.png` — train/validation MSE per epoch
- `camera_poses.png` — train vs validation camera layout
- `renders_train/`, `renders_valid/` — originals and epoch snapshots
- `model.pth` — final weights

### Inference / novel views

```bash
python inference.py --config config_inf.yaml
```

Loads `model_path`, reads all frames from `data_dir/transforms.json`, and writes PNGs to a new subdirectory under `render_dir`. Optional `width` and `height` override per-frame resolution (useful when rendering at training resolution with downsampling).

### Synthetic validation rings

For viewpoints without real photos (only poses), generate a circular camera path:

```bash
python gen_valid_rings.py --output-dir path/to/ring --axis z --radius 8.0 --num-cameras 36
```

This writes `transforms.json` and `camera_poses.png`. Image paths in the JSON are placeholders (`camera_000.png`, …); inference still renders from the trained model even if those files are missing, as long as you only need synthetic views. For training, real images must exist at the listed paths.

Supported ring axes: `x`, `y`, `z` (ring lies in the plane perpendicular to the chosen axis).

## Pixel sampling strategies

During training, rays are resampled every epoch. Set `sample_strategy` in the training config:


| Strategy     | Behavior                                                                                          |
| ------------ | ------------------------------------------------------------------------------------------------- |
| `random`     | Uniform random pixels                                                                             |
| `gaussian`   | More samples near the image center                                                                |
| `foreground` | Mix of transparent (alpha = 0) and opaque pixels; tune `transparent_fraction` in `sampler_params` |


Implementations live in `samplers.py`.

## Repository layout

| File                        | Role                                                                                                                              |
| --------------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| `train.py`                  | CLI for training: load config, split frames, build dataloaders, optimization loop, periodic renders, save checkpoint and metrics |
| `inference.py`              | CLI for loading `model.pth` and rendering all poses in `transforms.json`                                                         |
| `model.py`                  | `NeRF` module: positional encoding, MLPs for density and color, volume rendering along rays                                      |
| `dataset.py`                | `NeRFDataset` (ray batches from multi-view images), ray geometry helpers, train/valid split, point sampling along rays           |
| `render.py`                 | Full-image rendering: generate all pixel rays, batched forward passes, save PNGs                                                 |
| `samplers.py`               | Pixel index samplers used when choosing training rays each epoch                                                                 |
| `utils.py`                  | Experiment directory creation (auto-increment if exists), loss history JSON/plots, camera pose visualization                     |
| `gen_valid_rings.py`        | Standalone tool to emit circular camera paths as `transforms.json` for evaluation trajectories                                   |
| `default_train_config.yaml` | Documented template for training                                                                                                 |
| `default_inf_config.yaml`   | Documented template for inference                                                                                                |
| `requirements.txt`          | Python dependencies (PyTorch CUDA wheels + plotting/progress)                                                                    |

## Roadmap

- **NeRF** — current focus: train, validate, render
- **3D Gaussian Splatting** — planned as an additional representation and training path for faster real-time-style rendering
- **Mesh extraction** — planned scripts to recover explicit surface meshes from trained models

Contributions and experiments should stay config-driven (YAML) and follow the same pattern: dataset → train → artifact directory → inference / export.

## Tips

- Match `near` / `far` to your scene scale (distance from cameras to content along rays)
- Increase `num_rays_per_image` and `num_epochs` for quality; use `img_downsample` for faster iteration
- Keep `nn_params` identical between training and inference configs
- Compare `renders_valid/original/` with `renders_valid/epoch_*` during training to spot overfitting or bad bounds

