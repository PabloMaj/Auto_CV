import json
import random
import shutil
from pathlib import Path

import cv2
from PIL import Image


class DataProcessorDataset:
    def __init__(self, input_root, output_root, seed=42):
        self.input_root = Path(input_root)
        self.output_root = Path(output_root)
        self.seed = seed
        random.seed(seed)

    # =========================
    # MAIN ENTRY
    # =========================
    def process_all(self):
        datasets = self._find_datasets()

        for json_path, images_dir, dataset_name in datasets:
            print(f"[INFO] Processing: {dataset_name}")

            data = self._load_json(json_path)

            if not self._is_valid_dataset(data, images_dir):
                print(f"[SKIP] Not enough RGB images: {dataset_name}")
                continue

            self._process_dataset(data, images_dir, dataset_name)

    # =========================
    # FIND DATASETS
    # =========================
    def _find_datasets(self):
        datasets = []

        for domain in ["maize", "sugarbeet", "sunflower"]:
            domain_path = self.input_root / domain
            if not domain_path.exists():
                continue

            for item in domain_path.iterdir():
                if item.is_dir():
                    json_path = item.with_suffix(".json")
                    if json_path.exists():
                        dataset_name = f"{domain}_{item.name}"
                        datasets.append((json_path, item, dataset_name))

        return datasets

    # =========================
    # LOAD JSON
    # =========================
    def _load_json(self, path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    # =========================
    # VALIDATION
    # =========================
    def _is_valid_dataset(self, data, images_dir):
        count = 0

        for img in data.get("images", []):
            img_path = self._resolve_image_path(img, images_dir)

            if img_path is None:
                continue

            if not img_path.exists():
                continue

            try:
                with Image.open(img_path) as im:
                    if im.mode == "RGB":
                        count += 1
            except Exception as e:
                print(f"[ERROR] {img_path} -> {e}")

        return count >= 8

    # =========================
    # PROCESS DATASET
    # =========================
    def _process_dataset(self, data, images_dir, dataset_name):
        out_dir = self.output_root / dataset_name

        splits = {
            "train": [],
            "val": [],
            "test": [],
            "unlabelled": []
        }

        images = data["images"]
        random.shuffle(images)

        splits["train"] = images[:2]
        splits["val"] = images[2:4]
        splits["test"] = images[4:6]
        splits["unlabelled"] = images[6:]

        # folders
        for split in splits:
            (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
            if split != "unlabelled":
                (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

        # vis folder
        (out_dir / "images" / "vis_temp").mkdir(parents=True, exist_ok=True)

        # category mapping
        categories = data.get("categories", [])
        cat_map = {c["id"]: i for i, c in enumerate(categories)}

        # annotations index
        ann_map = {}
        for ann in data.get("annotations", []):
            ann_map.setdefault(ann["image_id"], []).append(ann)

        # process
        for split, imgs in splits.items():
            for img in imgs:
                self._write_sample(
                    img,
                    ann_map.get(img["id"], []),
                    cat_map,
                    images_dir,
                    out_dir,
                    split
                )

        self._write_yaml(out_dir, categories)

    # =========================
    # WRITE SAMPLE
    # =========================
    def _write_sample(self, img, annotations, cat_map, images_dir, out_dir, split):
        src_img = self._resolve_image_path(img, images_dir)

        if src_img is None or not src_img.exists():
            return

        dst_img = out_dir / "images" / split / src_img.name
        shutil.copy2(src_img, dst_img)

        # VISUALIZATION
        if split in ["train", "val", "test"]:
            vis_path = out_dir / "images" / "vis_temp" / src_img.name
            self._draw_vis(src_img, annotations, vis_path)

        # unlabelled = no labels
        if split == "unlabelled":
            return

        label_path = out_dir / "labels" / split / (src_img.stem + ".txt")

        w, h = img["width"], img["height"]

        lines = []
        for ann in annotations:
            if ann.get("iscrowd"):
                continue

            cls = cat_map[ann["category_id"]]
            x, y, bw, bh = ann["bbox"]

            x_center = (x + bw / 2) / w
            y_center = (y + bh / 2) / h
            bw /= w
            bh /= h

            lines.append(f"{cls} {x_center} {y_center} {bw} {bh}")

        label_path.write_text("\n".join(lines), encoding="utf-8")

    # =========================
    # VISUALIZATION
    # =========================
    def _draw_vis(self, img_path, annotations, out_path):
        image = cv2.imread(str(img_path))
        if image is None:
            return

        for ann in annotations:
            x, y, w, h = ann["bbox"]

            x1, y1 = int(x), int(y)
            x2, y2 = int(x + w), int(y + h)

            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

        cv2.imwrite(str(out_path), image)

    # =========================
    # PATH RESOLVE
    # =========================
    def _resolve_image_path(self, img, images_dir):
        """
        FIX: ignore broken JSON 'path' field.
        Match ONLY by file_name in dataset folder.
        """

        file_name = img.get("file_name")
        if not file_name:
            return None

        # recursive search (bezpieczne dla Twojej struktury)
        for p in images_dir.rglob(file_name):
            return p

        return None

    # =========================
    # YAML
    # =========================
    def _write_yaml(self, out_dir, categories):
        names = [c["name"] for c in categories]

        yaml_content = f"""
path: {out_dir}
train: images/train
val: images/val
test: images/test
nc: {len(names)}
names: {names}
"""

        (out_dir / "data.yaml").write_text(yaml_content, encoding="utf-8")


if __name__ == "__main__":

    processor = DataProcessorDataset(
        input_root=r"C:\projects\agent_cv\data\data_raw\crop_line_uav",
        output_root=r"C:\projects\agent_cv\data\data_structured\crop_line_uav",
        seed=42
    )
    processor.process_all()
