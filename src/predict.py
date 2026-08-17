import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from PIL import Image

from dataset import build_transforms
from model import build_model
from train import get_device


def predict(image_path, top_k=3):
    root_dir = Path(__file__).resolve().parent.parent
    config = yaml.safe_load((root_dir / "config.yaml").read_text())
    checkpoint_path = root_dir / config["paths"]["checkpoint_dir"] / "best_model.pt"

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    classes = checkpoint["classes"]

    device = get_device()
    model = build_model(
        len(classes), backbone=config["model"]["backbone"], pretrained=False
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    _, eval_tf = build_transforms(config["data"]["img_size"])
    image = Image.open(image_path).convert("RGB")
    tensor = eval_tf(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1).squeeze(0)

    top_probs, top_idxs = probs.topk(min(top_k, len(classes)))
    return [(classes[idx], prob.item()) for prob, idx in zip(top_probs, top_idxs)]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict crop disease from an image")
    parser.add_argument("image_path", type=str, help="Path to a leaf image")
    parser.add_argument("--top-k", type=int, default=3, help="Number of top predictions to show")
    args = parser.parse_args()

    results = predict(args.image_path, top_k=args.top_k)
    print(f"\nPredictions for {args.image_path}:")
    for label, prob in results:
        print(f"  {label}: {prob * 100:.2f}%")
