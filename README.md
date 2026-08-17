# Crop Disease Classifier

An image classifier that identifies crop diseases from leaf photos, built with
PyTorch (transfer learning on EfficientNet-B0). Trained on a merged dataset
spanning 54 classes across 18 crop species, combining lab-condition images
(PlantVillage) with field-condition images (PlantDoc, Cassava, Rice, Banana)
to reduce the lab-to-field generalization gap.

## What this covers

**Species:** Apple, Blueberry, Cherry, Corn (maize), Grape, Orange, Peach,
Pepper (bell), Potato, Raspberry, Soybean, Squash, Strawberry, Tomato, Cassava,
Rice, Banana.

**54 classes total** — most are `<crop>___<disease>` (e.g.
`Tomato___Early_blight`) or `<crop>___healthy`.

**Not covered:** Yam and Cocoa were investigated but excluded — no usable
public leaf-disease dataset exists for yam, and the only available cocoa
dataset is pod images (not leaves) under a non-commercial license. General
pest/insect damage (as opposed to fungal/bacterial/viral disease) is also out
of scope — see [Known limitations](#known-limitations).

## Data sources

| Source | Images | Classes | Kaggle handle |
|---|---|---|---|
| PlantVillage | 54,305 | 38 | `abdallahalidev/plantvillage-dataset` |
| PlantDoc | 2,922 | (merged into PlantVillage classes) | `nirmalsankalana/plantdoc-dataset` |
| Rice | 5,932 | 4 | `nirmalsankalana/rice-leaf-disease-image` |
| Banana | 408 | 7 | `sujaykapadnis/banana-disease-recognition-dataset` |
| Cassava | 21,397 | 5 | Kaggle competition `cassava-leaf-disease-classification` |

**Combined: 84,964 images across 54 classes.**

PlantDoc is field-condition photos (phones, natural backgrounds) labeled with
its own taxonomy (e.g. `Tomato_Early_blight_leaf`, and generic `<species>_leaf`
for healthy). These are mapped onto the matching PlantVillage class (see
`src/class_mapping.py`) so each class contains both lab and field examples of
the same disease, rather than becoming a separate lookalike class. Banana's
"Augmented images" folder is intentionally excluded — those are near-duplicate
transforms of the "Original Images" and would leak across train/val/test
splits if included.

## Project structure

```
crop-disease-classifier/
├── config.yaml           hyperparameters
├── requirements.txt
├── src/
│   ├── class_mapping.py  PlantDoc/Cassava/Rice/Banana -> unified class-name maps
│   ├── prepare_data.py   merges all downloaded sources into data/combined/ via symlinks
│   ├── dataset.py        ImageFolder + stratified split + augmentation + weighted sampler
│   ├── model.py          EfficientNet-B0 transfer-learning backbone
│   ├── train.py          training loop, checkpointing, MPS/CUDA/CPU device selection
│   ├── evaluate.py       test-set classification report + confusion matrix
│   └── predict.py        single-image inference
├── checkpoints/          best_model.pt                                 (gitignored -- regenerate by training)
├── outputs/              eval_report.txt, confusion_matrix.png, history.yaml (gitignored)
└── data/                 combined/ dataset built by prepare_data.py    (gitignored)
```

## Setup

Requires Python 3.14+ and a Kaggle account (free) for dataset downloads.

```bash
cd crop-disease-classifier
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Kaggle authentication

1. Go to kaggle.com → Settings → API → "Create New Token" — downloads `kaggle.json`.
2. Place it at `~/.kaggle/kaggle.json` and lock down permissions:
   ```bash
   mkdir -p ~/.kaggle
   mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
   chmod 600 ~/.kaggle/kaggle.json
   ```

The Cassava dataset is a Kaggle **competition**, which requires accepting its
rules once before the API will allow downloading it:
visit https://kaggle.com/competitions/cassava-leaf-disease-classification/rules
and accept, using the same Kaggle account tied to your API token.

## Downloading the data

```bash
python3 -c "import kagglehub; print(kagglehub.dataset_download('abdallahalidev/plantvillage-dataset'))"
python3 -c "import kagglehub; print(kagglehub.dataset_download('nirmalsankalana/plantdoc-dataset'))"
python3 -c "import kagglehub; print(kagglehub.dataset_download('nirmalsankalana/rice-leaf-disease-image'))"
python3 -c "import kagglehub; print(kagglehub.dataset_download('sujaykapadnis/banana-disease-recognition-dataset'))"
python3 -c "import kagglehub; print(kagglehub.competition_download('cassava-leaf-disease-classification'))"
```

These are cached under `~/.cache/kagglehub/` and only need to be downloaded once.

## Preparing the combined dataset

Merges all five sources into `data/combined/`, one folder per class, using
symlinks (no data duplication) and the class-name mapping in `class_mapping.py`:

```bash
python3 src/prepare_data.py
```

Re-run this any time a new source is added or the mapping changes — it wipes
and rebuilds `data/combined/` from scratch each time, so it's always safe to
re-run.

## Training

```bash
caffeinate -i python3 src/train.py
```

`caffeinate -i` is macOS-only and prevents the system from sleeping mid-run,
which otherwise can stall training for hours if the lid closes or the display
sleeps. On Windows, use `powercfg /change standby-timeout-ac 0` (or disable
sleep in Settings) before training; on Linux, `systemd-inhibit --what=sleep
python3 src/train.py` is the equivalent.

- Backbone: EfficientNet-B0, ImageNet-pretrained, fine-tuned end-to-end.
- Data split: 70% train / 15% val / 15% test, stratified by class.
- Training uses a `WeightedRandomSampler` so classes are sampled ~evenly
  per batch regardless of raw size — Cassava alone is ~4,000 images/class
  versus ~100-300/class for most other classes, so plain shuffling would
  have let it dominate gradient updates.
- Best checkpoint (by validation accuracy) is saved to `checkpoints/best_model.pt`.
- All hyperparameters live in `config.yaml`.

On an M4 MacBook Pro (MPS backend), expect roughly 20-25 minutes/epoch on the
full 54-class combined dataset (~85k images), so ~5-6 hours for the default
15 epochs.

## Evaluation

```bash
caffeinate -i python3 src/evaluate.py
```

Runs the best checkpoint against the held-out test set and writes:
- `outputs/eval_report.txt` — precision/recall/F1 per class
- `outputs/confusion_matrix.png` — full 54x54 confusion matrix heatmap

## Predicting on a single image

```bash
python3 src/predict.py path/to/leaf_image.jpg --top-k 3
```

Prints the top-k predicted classes with confidence scores. The model always
returns its best guess among the 54 known classes — it has no "unknown /
out of distribution" detection, so a confident-looking prediction on an
image of something it wasn't trained on (a different crop, a non-leaf
photo, pest damage) should not be trusted at face value.

## Results

**Combined 54-class model:** 94.9% test accuracy (macro F1 0.952).

Most classes score 0.93-1.0 F1. The clear weak spot is **Cassava**, where
4 of its 5 classes score 0.53-0.75 F1 (only the majority class,
`Cassava___Mosaic_Disease`, scores well at 0.941). This tracks with the
broader Kaggle competition history for this exact dataset — the diseases
visually overlap, the labels are known to have some noise from crowdsourced
collection, and competition-winning solutions needed larger backbones,
ensembling, and test-time augmentation to exceed ~90%. A single EfficientNet-B0
run landing in this range on Cassava specifically is an expected baseline,
not a bug.

Rice and most PlantVillage/PlantDoc-merged classes perform strongly; Banana's
per-class test counts are small (3-13 images), so its reported metrics carry
more sampling noise than the other classes.

## Known limitations

- **Domain shift on real-world photos**: adding PlantDoc measurably helped
  (a manually tested real-world tomato photo went from a low-confidence,
  scattered prediction to a confident, correct-crop prediction), but did not
  fully close the gap — another real photo remained confidently
  *mis*-classified as the wrong crop entirely. Expect the model to be
  noticeably more reliable on PlantVillage-style single-leaf, plain-background
  photos than on cluttered field/garden photos.
- **No pest/insect-damage class**: a real test photo showing tomato leaves
  with insect feeding damage (ragged chewed holes) was misclassified, because
  no class for that damage type exists in the tomato taxonomy — the model
  had no correct label available to give, regardless of image quality.
  Banana has an `Insect_Pest` class; tomato and other crops do not.
  This classifier targets disease identification, not general plant-damage
  triage.
- **Cassava accuracy ceiling**: see [Results](#results) above.
- **Coverage**: 18 species total. Common Nigerian staples Yam and Cocoa are
  not covered (see [What this covers](#what-this-covers)).
