"""
Experiment 05 — build low-label-budget YOLO datasets

For each of the 4 saved publication datasets, randomly selects tiles from the
original tiled_dataset train and val splits (fixed seed, reused identically
across both variants and repeats) until the cumulative number of annotated
objects reaches TARGET_OBJECTS +/- OBJECT_TOLERANCE (default 20 +/- 5,
override via EXP05_TARGET_OBJECTS / EXP05_OBJECT_TOLERANCE env vars) — train
and val are budgeted independently. Materializes two dataset folders ready
for YOLO training:

  A_limited_no_enrichment/limited_dataset/
      images/train, labels/train   <- object-count-budgeted subset of tiled_dataset train
      images/val,   labels/val     <- object-count-budgeted subset of tiled_dataset val
      data.yaml (test points at the full, original tiled_dataset/images/test)

  B_limited_pseudo_enrichment/limited_enriched_dataset/
      images/train, labels/train   <- SAME labeled subset + the FULL
                                       pseudo_labels / tiled_unlabeled pool
      images/val,   labels/val     <- SAME subset as variant A (untouched)
      data.yaml (test points at the full, original tiled_dataset/images/test)

The test split is never copied — both variants' data.yaml reference the
original tiled_dataset/images/test directly (absolute path), so evaluation
stays on the full, unmodified held-out set and is directly comparable to
exp04's test metrics.

A per-dataset sample_manifest.json records exactly which tile stems were
selected/added (and the achieved object counts), so the experiment is
reproducible and auditable.

Usage (from repo root, run once before run.py):
    python experiments/exp05_low_label_enricher_eval/build_datasets.py
    EXP05_TARGET_OBJECTS=20 EXP05_OBJECT_TOLERANCE=5 python experiments/exp05_low_label_enricher_eval/build_datasets.py
"""

import json
import random
import shutil
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.exp05_low_label_enricher_eval.config import (
    DATASETS, TARGET_OBJECTS, OBJECT_TOLERANCE, N_SELECTION_TRIALS, DATA_SELECTION_SEED,
    source_dataset_root, variant_dataset_dir, dataset_out_dir,
)

IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".tif", ".tiff"]


def _find_image(img_dir: Path, stem: str) -> Path:
    for ext in IMAGE_EXTS:
        candidate = img_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No image found for stem '{stem}' in {img_dir}")


def _stems(img_dir: Path) -> list:
    return sorted(p.stem for p in img_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def _count_objects(lbl_path: Path) -> int:
    if not lbl_path.exists():
        return 0
    return len([line for line in lbl_path.read_text().splitlines() if line.strip()])


def _copy_pair(img_dir: Path, lbl_dir: Path, stem: str, out_img_dir: Path, out_lbl_dir: Path):
    img_path = _find_image(img_dir, stem)
    lbl_path = lbl_dir / f"{stem}.txt"

    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(img_path, out_img_dir / img_path.name)
    if lbl_path.exists():
        shutil.copy2(lbl_path, out_lbl_dir / lbl_path.name)
    else:
        (out_lbl_dir / f"{stem}.txt").write_text("")


def select_by_object_budget(img_dir: Path, lbl_dir: Path, target: int, tolerance: int,
                             rng: random.Random, n_trials: int = N_SELECTION_TRIALS) -> tuple:
    """
    Randomly shuffle tiles and take a prefix (stopping as soon as the
    cumulative object count reaches `target`); repeat for `n_trials` shuffles
    and keep the prefix whose total object count is closest to `target`.
    Returns (selected_stems, total_objects).
    """
    stems = _stems(img_dir)
    counts = {s: _count_objects(lbl_dir / f"{s}.txt") for s in stems}

    best_selection, best_total, best_diff = [], 0, float("inf")

    for _ in range(n_trials):
        order = stems[:]
        rng.shuffle(order)

        selection, total = [], 0
        for stem in order:
            if total >= target:
                break
            selection.append(stem)
            total += counts[stem]

        diff = abs(total - target)
        if diff < best_diff:
            best_selection, best_total, best_diff = selection, total, diff
            if diff <= tolerance:
                break  # good enough, stop searching

    return sorted(best_selection), best_total


def write_yaml(out_dir: Path, test_dir: Path) -> Path:
    data = {
        "path": str(out_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": str(test_dir.resolve()),
        "names": ["object"],
    }
    yaml_path = out_dir / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(data, f)
    return yaml_path


def build_for_dataset(dataset: dict) -> dict:
    name = dataset["name"]
    src_root = source_dataset_root(dataset) / "tiled_dataset"
    train_img_dir = src_root / "images" / "train"
    train_lbl_dir = src_root / "labels" / "train"
    val_img_dir = src_root / "images" / "val"
    val_lbl_dir = src_root / "labels" / "val"
    test_img_dir = src_root / "images" / "test"

    pool_root = source_dataset_root(dataset)
    pseudo_lbl_dir = pool_root / "pseudo_labels"
    pseudo_img_dir = pool_root / "tiled_unlabeled" / "images"

    rng = random.Random(DATA_SELECTION_SEED)
    train_stems, train_objects = select_by_object_budget(train_img_dir, train_lbl_dir, TARGET_OBJECTS, OBJECT_TOLERANCE, rng)
    val_stems, val_objects = select_by_object_budget(val_img_dir, val_lbl_dir, TARGET_OBJECTS, OBJECT_TOLERANCE, rng)
    pseudo_stems = _stems(pseudo_img_dir)

    # --- variant A: limited labeled subset only ---
    a_dir = variant_dataset_dir(name, "A_limited_no_enrichment")
    if a_dir.exists():
        shutil.rmtree(a_dir)
    for stem in train_stems:
        _copy_pair(train_img_dir, train_lbl_dir, stem, a_dir / "images" / "train", a_dir / "labels" / "train")
    for stem in val_stems:
        _copy_pair(val_img_dir, val_lbl_dir, stem, a_dir / "images" / "val", a_dir / "labels" / "val")
    write_yaml(a_dir, test_img_dir)

    # --- variant B: same limited labeled subset + full pseudo pool in train ---
    b_dir = variant_dataset_dir(name, "B_limited_pseudo_enrichment")
    if b_dir.exists():
        shutil.rmtree(b_dir)
    for stem in train_stems:
        _copy_pair(train_img_dir, train_lbl_dir, stem, b_dir / "images" / "train", b_dir / "labels" / "train")
    for stem in pseudo_stems:
        _copy_pair(pseudo_img_dir, pseudo_lbl_dir, stem, b_dir / "images" / "train", b_dir / "labels" / "train")
    for stem in val_stems:
        _copy_pair(val_img_dir, val_lbl_dir, stem, b_dir / "images" / "val", b_dir / "labels" / "val")
    write_yaml(b_dir, test_img_dir)

    manifest = {
        "dataset": name,
        "target_objects": TARGET_OBJECTS,
        "object_tolerance": OBJECT_TOLERANCE,
        "data_selection_seed": DATA_SELECTION_SEED,
        "n_original_train": len(_stems(train_img_dir)),
        "n_selected_train": len(train_stems),
        "train_objects_achieved": train_objects,
        "selected_train_stems": train_stems,
        "n_original_val": len(_stems(val_img_dir)),
        "n_selected_val": len(val_stems),
        "val_objects_achieved": val_objects,
        "selected_val_stems": val_stems,
        "n_pseudo_pool": len(pseudo_stems),
        "pseudo_pool_stems": pseudo_stems,
        "variant_A_dir": str(a_dir),
        "variant_A_train_size": len(train_stems),
        "variant_B_dir": str(b_dir),
        "variant_B_train_size": len(train_stems) + len(pseudo_stems),
        "test_dir_shared": str(test_img_dir),
    }

    manifest_path = dataset_out_dir(name) / "sample_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    return manifest


if __name__ == "__main__":
    print(f"Target objects: {TARGET_OBJECTS} +/- {OBJECT_TOLERANCE}  (train and val budgeted independently)")

    for dataset in DATASETS:
        print(f"\n{'=' * 60}\nBuilding limited-label datasets: {dataset['name']}\n{'=' * 60}")
        manifest = build_for_dataset(dataset)

        train_diff = manifest["train_objects_achieved"] - TARGET_OBJECTS
        val_diff = manifest["val_objects_achieved"] - TARGET_OBJECTS
        train_flag = "" if abs(train_diff) <= OBJECT_TOLERANCE else "  [OUT OF TOLERANCE]"
        val_flag = "" if abs(val_diff) <= OBJECT_TOLERANCE else "  [OUT OF TOLERANCE]"

        print(f"  labeled train: {manifest['n_selected_train']} tiles / {manifest['n_original_train']} available "
              f"-> {manifest['train_objects_achieved']} objects{train_flag}")
        print(f"  labeled val:   {manifest['n_selected_val']} tiles / {manifest['n_original_val']} available "
              f"-> {manifest['val_objects_achieved']} objects{val_flag}")
        print(f"  pseudo pool:   {manifest['n_pseudo_pool']} tiles (added to variant B train only)")
        print(f"  A train size:  {manifest['variant_A_train_size']} tiles  ->  {manifest['variant_A_dir']}")
        print(f"  B train size:  {manifest['variant_B_train_size']} tiles  ->  {manifest['variant_B_dir']}")

    print("\nDone. Next: python experiments/exp05_low_label_enricher_eval/run.py")
