import time
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from tqdm import tqdm

from dataset import build_dataloaders, find_data_root
from model import build_model


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def run_epoch(model, loader, criterion, optimizer, device, train):
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for images, labels in tqdm(loader, leave=False):
            images, labels = images.to(device), labels.to(device)
            if train:
                optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            if train:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += images.size(0)
    return total_loss / total, correct / total


def main():
    root_dir = Path(__file__).resolve().parent.parent
    config = yaml.safe_load((root_dir / "config.yaml").read_text())

    data_root = find_data_root()
    print(f"Using data root: {data_root}")

    train_loader, val_loader, _, classes = build_dataloaders(
        data_root,
        img_size=config["data"]["img_size"],
        batch_size=config["data"]["batch_size"],
        val_split=config["data"]["val_split"],
        test_split=config["data"]["test_split"],
        num_workers=config["data"]["num_workers"],
        seed=config["data"]["seed"],
    )
    print(f"Classes ({len(classes)}): {classes}")

    device = get_device()
    print(f"Using device: {device}")

    model = build_model(
        num_classes=len(classes),
        backbone=config["model"]["backbone"],
        pretrained=config["model"]["pretrained"],
    ).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=config["train"]["label_smoothing"])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["train"]["lr"],
        weight_decay=config["train"]["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["train"]["epochs"]
    )

    checkpoint_dir = root_dir / config["paths"]["checkpoint_dir"]
    checkpoint_dir.mkdir(exist_ok=True)

    best_val_acc = 0.0
    history = []

    for epoch in range(1, config["train"]["epochs"] + 1):
        start = time.time()
        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, optimizer, device, train=True
        )
        val_loss, val_acc = run_epoch(
            model, val_loader, criterion, optimizer, device, train=False
        )
        scheduler.step()
        elapsed = time.time() - start

        print(
            f"Epoch {epoch}/{config['train']['epochs']} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} ({elapsed:.1f}s)"
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {"model_state": model.state_dict(), "classes": classes, "config": config},
                checkpoint_dir / "best_model.pt",
            )
            print(f"  Saved new best model (val_acc={val_acc:.4f})")

    output_dir = root_dir / config["paths"]["output_dir"]
    output_dir.mkdir(exist_ok=True)
    with open(output_dir / "history.yaml", "w") as f:
        yaml.safe_dump(history, f)


if __name__ == "__main__":
    main()
