import csv
import glob
import json
import os
import shutil
from pathlib import Path

from class_mapping import (
    BANANA_FOLDER_TO_CLASS,
    CASSAVA_LABEL_TO_CLASS,
    PLANTDOC_TO_PLANTVILLAGE,
    RICE_FOLDER_TO_CLASS,
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def latest_version_dir(cache_glob):
    versions = sorted(glob.glob(os.path.expanduser(cache_glob)))
    if not versions:
        raise FileNotFoundError(f"Nothing found under {cache_glob}")
    return Path(versions[-1])


def link_image(src_path, dest_dir, prefix):
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / f"{prefix}_{src_path.name}"
    if not dest_path.exists():
        os.symlink(src_path.resolve(), dest_path)


def link_folder_images(src_dir, dest_dir, prefix):
    count = 0
    for f in Path(src_dir).iterdir():
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS:
            link_image(f, dest_dir, prefix)
            count += 1
    return count


def add_plantvillage(combined_root):
    root = latest_version_dir(
        "~/.cache/kagglehub/datasets/abdallahalidev/plantvillage-dataset/versions/*"
    )
    color_dirs = [p for p in root.rglob("color") if p.is_dir()]
    color_root = color_dirs[0]

    total = 0
    for class_dir in color_root.iterdir():
        if class_dir.is_dir():
            total += link_folder_images(
                class_dir, combined_root / class_dir.name, "plantvillage"
            )
    print(f"PlantVillage: linked {total} images across {len(list(color_root.iterdir()))} classes")


def add_plantdoc(combined_root):
    root = latest_version_dir(
        "~/.cache/kagglehub/datasets/nirmalsankalana/plantdoc-dataset/versions/*"
    )
    total = 0
    unmapped = set()
    for split in ("train", "test"):
        split_dir = root / split
        if not split_dir.is_dir():
            continue
        for class_dir in split_dir.iterdir():
            if not class_dir.is_dir():
                continue
            mapped = PLANTDOC_TO_PLANTVILLAGE.get(class_dir.name)
            if mapped is None:
                unmapped.add(class_dir.name)
                continue
            total += link_folder_images(class_dir, combined_root / mapped, "plantdoc")
    if unmapped:
        print(f"PlantDoc: WARNING skipped unmapped classes: {sorted(unmapped)}")
    print(f"PlantDoc: linked {total} images (merged into existing PlantVillage classes)")


def add_rice(combined_root):
    root = latest_version_dir(
        "~/.cache/kagglehub/datasets/nirmalsankalana/rice-leaf-disease-image/versions/*"
    )
    total = 0
    for folder_name, class_name in RICE_FOLDER_TO_CLASS.items():
        src_dir = root / folder_name
        if src_dir.is_dir():
            total += link_folder_images(src_dir, combined_root / class_name, "rice")
    print(f"Rice: linked {total} images across {len(RICE_FOLDER_TO_CLASS)} classes")


def add_banana(combined_root):
    root = latest_version_dir(
        "~/.cache/kagglehub/datasets/sujaykapadnis/banana-disease-recognition-dataset/versions/*"
    )
    # Only "Original Images" -- "Augmented images" are near-duplicates of the same source
    # photos, which would leak across train/val/test splits if included.
    originals_root = root / "Banana Disease Recognition Dataset" / "Original Images" / "Original Images"
    total = 0
    for folder_name, class_name in BANANA_FOLDER_TO_CLASS.items():
        src_dir = originals_root / folder_name
        if src_dir.is_dir():
            total += link_folder_images(src_dir, combined_root / class_name, "banana")
    print(f"Banana: linked {total} images across {len(BANANA_FOLDER_TO_CLASS)} classes (originals only)")


def add_cassava(combined_root):
    root = Path(
        os.path.expanduser(
            "~/.cache/kagglehub/competitions/cassava-leaf-disease-classification"
        )
    )
    train_images_dir = root / "train_images"
    train_csv = root / "train.csv"

    total = 0
    with open(train_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            class_name = CASSAVA_LABEL_TO_CLASS[int(row["label"])]
            src_path = train_images_dir / row["image_id"]
            if src_path.exists():
                link_image(src_path, combined_root / class_name, "cassava")
                total += 1
    print(f"Cassava: linked {total} images across {len(CASSAVA_LABEL_TO_CLASS)} classes")


def main():
    root_dir = Path(__file__).resolve().parent.parent
    combined_root = root_dir / "data" / "combined"

    if combined_root.exists():
        shutil.rmtree(combined_root)
    combined_root.mkdir(parents=True)

    add_plantvillage(combined_root)
    add_plantdoc(combined_root)
    add_rice(combined_root)
    add_banana(combined_root)
    add_cassava(combined_root)

    class_dirs = sorted(p for p in combined_root.iterdir() if p.is_dir())
    total_images = sum(1 for _ in combined_root.rglob("*") if _.is_file() or _.is_symlink())
    print(f"\nCombined dataset: {len(class_dirs)} classes, {total_images} images")
    print(f"Root: {combined_root}")


if __name__ == "__main__":
    main()
