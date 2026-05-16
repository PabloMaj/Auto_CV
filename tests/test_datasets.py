from pathlib import Path
import pytest 

ROOT_DIR = Path(r"C:\projects\agent_cv\data\data_structured\crop_line_uav")


# =========================
# DATASET DISCOVERY
# =========================
def get_datasets():
    return [
        p for p in ROOT_DIR.iterdir()
        if p.is_dir() and (p / "images").exists() and (p / "labels").exists()
    ]


@pytest.fixture(params=get_datasets(), ids=lambda p: p.name)
def dataset(request):
    return request.param


# =========================
# 1. IMAGE ↔ LABEL CONSISTENCY
# =========================
def test_image_label_consistency(dataset):
    for split in ["train", "val", "test"]:
        img_dir = dataset / "images" / split
        label_dir = dataset / "labels" / split

        assert img_dir.exists(), f"Missing directory: {img_dir}"
        assert label_dir.exists(), f"Missing directory: {label_dir}"

        images = {p.stem for p in img_dir.glob("*")}
        labels = {p.stem for p in label_dir.glob("*.txt")}

        missing_labels = images - labels
        missing_images = labels - images

        assert not missing_labels, (
            f"{dataset.name}/{split}: missing labels for {missing_labels}"
        )
        assert not missing_images, (
            f"{dataset.name}/{split}: missing images for {missing_images}"
        )


# =========================
# 2. YOLO LABEL FORMAT VALIDITY
# =========================
def test_yolo_label_format(dataset):
    for split in ["train", "val", "test"]:
        label_dir = dataset / "labels" / split
        assert label_dir.exists(), f"Missing label dir: {label_dir}"

        for label_file in label_dir.glob("*.txt"):
            lines = label_file.read_text().strip().split("\n")

            for line in lines:
                if not line.strip():
                    continue

                parts = line.split()

                assert len(parts) == 5, (
                    f"{label_file}: invalid YOLO format -> {line}"
                )

                _, x, y, w, h = parts
                x, y, w, h = map(float, [x, y, w, h])

                assert 0.0 <= x <= 1.0, f"{label_file}: x out of range"
                assert 0.0 <= y <= 1.0, f"{label_file}: y out of range"
                assert 0.0 <= w <= 1.0, f"{label_file}: w out of range"
                assert 0.0 <= h <= 1.0, f"{label_file}: h out of range"


# =========================
# 3. DATA LEAKAGE CHECK
# =========================
def test_no_data_leakage(dataset):
    splits = ["train", "val", "test", "unlabelled"]

    split_sets = {}

    for split in splits:
        img_dir = dataset / "images" / split
        if not img_dir.exists():
            continue

        split_sets[split] = {p.name for p in img_dir.glob("*")}

    keys = list(split_sets.keys())

    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = keys[i], keys[j]

            intersection = split_sets[a] & split_sets[b]

            assert not intersection, (
                f"{dataset.name}: data leakage between {a} and {b}: {intersection}"
            )


# =========================
# 4. DUPLICATE IMAGE CHECK
# =========================
def test_no_duplicate_images(dataset):
    seen = set()

    for split in ["train", "val", "test", "unlabelled"]:
        img_dir = dataset / "images" / split
        if not img_dir.exists():
            continue

        for img in img_dir.glob("*"):
            assert img.name not in seen, (
                f"{dataset.name}: duplicate image found: {img.name}"
            )
            seen.add(img.name)
