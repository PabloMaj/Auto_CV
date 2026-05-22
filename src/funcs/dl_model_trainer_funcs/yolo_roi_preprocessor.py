import yaml
from pathlib import Path
from typing import List
import cv2


class YOLOROIPreprocessor:
    """
    Split-safe YOLO ROI tiling (train/val/test preserved).
    """

    def __init__(
        self,
        input_root: Path,
        output_root: Path,
        tile_size: int = 640,
        overlap: float = 0.5
    ):
        self.input_root = input_root
        self.output_root = output_root

        self.tile_size = tile_size
        self.stride = int(tile_size * (1 - overlap))

        self.splits = ["train", "val", "test"]

    # -----------------------------
    def load_labels(self, path: Path):
        if not path.exists():
            return []

        with open(path, "r") as f:
            lines = f.read().strip().splitlines()

        out = []
        for line in lines:
            c, x, y, w, h = line.split()
            out.append((int(c), float(x), float(y), float(w), float(h)))
        return out

    def save_labels(self, path: Path, labels):
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            for c, x, y, w, h in labels:
                f.write(f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

    # -----------------------------
    def yolo_to_xyxy(self, box, w, h):
        c, x, y, bw, bh = box
        x1 = (x - bw / 2) * w
        y1 = (y - bh / 2) * h
        x2 = (x + bw / 2) * w
        y2 = (y + bh / 2) * h
        return c, x1, y1, x2, y2

    def xyxy_to_yolo(self, c, x1, y1, x2, y2, size):
        x1 = max(0, min(size, x1))
        y1 = max(0, min(size, y1))
        x2 = max(0, min(size, x2))
        y2 = max(0, min(size, y2))

        if x2 <= x1 or y2 <= y1:
            return None

        cx = (x1 + x2) / 2 / size
        cy = (y1 + y2) / 2 / size
        bw = (x2 - x1) / size
        bh = (y2 - y1) / size

        return (c, cx, cy, bw, bh)

    # -----------------------------
    def process_split(self, split: str):

        img_dir = self.input_root / "images" / split
        lbl_dir = self.input_root / "labels" / split

        out_img_dir = self.output_root / "images" / split
        out_lbl_dir = self.output_root / "labels" / split

        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_lbl_dir.mkdir(parents=True, exist_ok=True)

        images = list(img_dir.rglob("*.jpg")) + list(img_dir.rglob("*.png"))

        for img_path in images:

            label_path = lbl_dir / f"{img_path.stem}.txt"

            img = cv2.imread(str(img_path))
            if img is None:
                continue

            h, w = img.shape[:2]
            labels = self.load_labels(label_path)

            tile_id = 0

            for y in range(0, h, self.stride):
                for x in range(0, w, self.stride):

                    x2 = min(x + self.tile_size, w)
                    y2 = min(y + self.tile_size, h)
                    x1 = x2 - self.tile_size
                    y1 = y2 - self.tile_size

                    if x1 < 0 or y1 < 0:
                        continue

                    tile = img[y1:y2, x1:x2]

                    if tile.shape[:2] != (self.tile_size, self.tile_size):
                        continue

                    new_labels = []

                    for box in labels:
                        c, bx1, by1, bx2, by2 = self.yolo_to_xyxy(box, w, h)

                        ix1 = max(bx1, x1)
                        iy1 = max(by1, y1)
                        ix2 = min(bx2, x2)
                        iy2 = min(by2, y2)

                        if ix2 <= ix1 or iy2 <= iy1:
                            continue

                        rel = self.xyxy_to_yolo(
                            c,
                            ix1 - x1,
                            iy1 - y1,
                            ix2 - x1,
                            iy2 - y1,
                            self.tile_size
                        )

                        if rel:
                            new_labels.append(rel)

                    out_name = f"{img_path.stem}_{tile_id}.jpg"

                    cv2.imwrite(str(out_img_dir / out_name), tile)
                    self.save_labels(out_lbl_dir / f"{img_path.stem}_{tile_id}.txt", new_labels)

                    tile_id += 1

    # -----------------------------
    def generate(self):
        for split in self.splits:
            self.process_split(split)

    # -----------------------------
    def create_yaml(self, class_names: List[str]):

        data = {
            "path": str(self.output_root.resolve()),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "names": class_names
        }

        yaml_path = self.output_root / "data.yaml"

        with open(yaml_path, "w") as f:
            yaml.dump(data, f)

        return yaml_path
