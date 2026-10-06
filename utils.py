import os
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def create_result_dir(path: str | Path) -> Path:
    path = Path(path)

    if not path.exists():
        path.mkdir(parents=True)
        return path

    i = 1
    while True:
        new_path = path.parent / f"{path.name}_{i}"

        if not new_path.exists():
            new_path.mkdir(parents=True)
            return new_path

        i += 1


def plot_history(history: dict[str, list], save_fig: str | None = None):
    x = range(len(next(iter(history.values()))))

    fig, ax = plt.subplots()

    for key, values in history.items():
        ax.plot(x, values, label=key)

    ax.set_xlabel("Epoch")
    ax.legend()
    ax.grid()
    fig.tight_layout()

    if save_fig is not None:
        fig.savefig(save_fig)

    return fig


def save_history(history, experiment_dir):
    history_path = os.path.join(experiment_dir, "history.json")

    with open(history_path, "w") as f:
        json.dump(history, f, indent=4)

    plot_history(
        history=history,
        save_fig=os.path.join(experiment_dir, "history.png"),
    )


def plot_camera_poses(meta_train, meta_valid, experiment_dir):
    """
    Plot camera positions for train and validation datasets

    Args:
        meta_train: List of dictionaries containing camera metadata.
        meta_valid: List of dictionaries containing camera metadata.
        save_path: Directory where camera_poses.png will be saved.
    """

    train_positions = np.array([
        np.asarray(item["transform_matrix"])[:3, 3]
        for item in meta_train
    ])

    valid_positions = np.array([
        np.asarray(item["transform_matrix"])[:3, 3]
        for item in meta_valid
    ])

    fig = plt.figure(figsize=(14, 6))

    # 3D view
    ax1 = fig.add_subplot(121, projection="3d")

    ax1.scatter(
        train_positions[:, 0],
        train_positions[:, 1],
        train_positions[:, 2],
        label="Train",
        s=20,
    )

    ax1.scatter(
        valid_positions[:, 0],
        valid_positions[:, 1],
        valid_positions[:, 2],
        label="Valid",
        s=20,
    )

    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")
    ax1.set_title("Camera positions — 3D")
    ax1.legend()

    # Top-down view (X-Y)
    ax2 = fig.add_subplot(122)

    ax2.scatter(
        train_positions[:, 0],
        train_positions[:, 1],
        label="Train",
        s=20,
    )

    ax2.scatter(
        valid_positions[:, 0],
        valid_positions[:, 1],
        label="Valid",
        s=20,
    )

    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_title("Camera positions — top view")
    ax2.axis("equal")
    ax2.legend()

    plt.tight_layout()

    output_path = os.path.join(experiment_dir, "camera_poses.png")

    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
