from pathlib import Path

import matplotlib.pyplot as plt
import torch
import yaml
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import classification_report, confusion_matrix

from dataset import build_dataloaders, find_data_root
from model import build_model
from train import get_device

# Sequential single-hue blue ramp (light -> dark), from the project's validated
# data-viz palette: near-zero recedes toward the surface, high values read dark.
SEQUENTIAL_BLUE = LinearSegmentedColormap.from_list(
    "sequential_blue",
    [
        "#cde2fb",
        "#9ec5f4",
        "#6da7ec",
        "#3987e5",
        "#256abf",
        "#184f95",
        "#0d366b",
    ],
)
PRIMARY_INK = "#0b0b0b"
MUTED_INK = "#898781"


def main():
    root_dir = Path(__file__).resolve().parent.parent
    config = yaml.safe_load((root_dir / "config.yaml").read_text())
    checkpoint_path = root_dir / config["paths"]["checkpoint_dir"] / "best_model.pt"

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    classes = checkpoint["classes"]

    data_root = find_data_root()
    _, _, test_loader, _ = build_dataloaders(
        data_root,
        img_size=config["data"]["img_size"],
        batch_size=config["data"]["batch_size"],
        val_split=config["data"]["val_split"],
        test_split=config["data"]["test_split"],
        num_workers=config["data"]["num_workers"],
        seed=config["data"]["seed"],
    )

    device = get_device()
    model = build_model(
        len(classes), backbone=config["model"]["backbone"], pretrained=False
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    report = classification_report(all_labels, all_preds, target_names=classes, digits=3)
    print(report)

    output_dir = root_dir / config["paths"]["output_dir"]
    output_dir.mkdir(exist_ok=True)
    (output_dir / "eval_report.txt").write_text(report)

    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(
        figsize=(max(8, len(classes) * 0.4), max(8, len(classes) * 0.4)),
        facecolor="#fcfcfb",
    )
    im = ax.imshow(cm, cmap=SEQUENTIAL_BLUE)
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=90, fontsize=6, color=MUTED_INK)
    ax.set_yticklabels(classes, fontsize=6, color=MUTED_INK)
    ax.set_xlabel("Predicted", color=PRIMARY_INK)
    ax.set_ylabel("True", color=PRIMARY_INK)
    ax.set_title("Confusion matrix", color=PRIMARY_INK)
    cbar = fig.colorbar(im)
    cbar.ax.yaxis.set_tick_params(color=MUTED_INK, labelcolor=MUTED_INK)
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=150, facecolor="#fcfcfb")
    print(f"Saved report and confusion matrix to {output_dir}")


if __name__ == "__main__":
    main()
