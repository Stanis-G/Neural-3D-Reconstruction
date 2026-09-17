from pathlib import Path
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
