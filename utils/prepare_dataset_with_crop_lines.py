import os
import json
import cv2
from pathlib import Path


class LabelmeLineToYoloLines:
    def __init__(self, dataset_root, output_dir, class_name="crop_line"):
        self.dataset_root = Path(dataset_root)
        self.output_dir = Path(output_dir)

        self.class_map = {class_name: 0}

        self.splits = ["train", "val", "test", "unlabelled"]

        for split in self.splits:
            (self.output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
            (self.output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    def convert(self):
        for split in self.splits:
            img_dir = self.dataset_root / "images" / split

            if not img_dir.exists():
                continue

            json_files = list(img_dir.glob("*.json"))

            for json_path in json_files:
                self._process_file(json_path, split)

        self._create_yaml()

    def _process_file(self, json_path: Path, split: str):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        image_name = data.get("imagePath")
        image_path = json_path.parent / image_name

        if not image_path.exists():
            print(f"[WARN] Missing image: {image_path}")
            return

        img = cv2.imread(str(image_path))
        if img is None:
            print(f"[WARN] Cannot read image: {image_path}")
            return

        h, w = img.shape[:2]

        label_lines = []

        # ❗ IMPORTANT: ignore annotations for unlabelled split
        if split != "unlabelled":
            for shape in data.get("shapes", []):
                if shape["shape_type"] != "line":
                    continue

                label = shape["label"]
                if label not in self.class_map:
                    self.class_map[label] = len(self.class_map)

                class_id = self.class_map[label]

                (x1, y1), (x2, y2) = shape["points"]

                x1, y1 = x1 / w, y1 / h
                x2, y2 = x2 / w, y2 / h

                label_lines.append(
                    f"{class_id} {x1:.6f} {y1:.6f} {x2:.6f} {y2:.6f}"
                )

        # save labels only for labeled splits
        if split != "unlabelled":
            out_label = self.output_dir / "labels" / split / f"{image_path.stem}.txt"
            with open(out_label, "w", encoding="utf-8") as f:
                f.write("\n".join(label_lines))

        # copy image
        out_img = self.output_dir / "images" / split / image_path.name
        if not out_img.exists():
            out_img.write_bytes(image_path.read_bytes())

    def _create_yaml(self):
        yaml_path = self.output_dir / "data.yaml"

        names = {v: k for k, v in self.class_map.items()}

        content = [
            f"path: {self.output_dir.resolve()}",
            "train: images/train",
            "val: images/val",
            "test: images/test",
            "names:",
        ]

        for i in sorted(names.keys()):
            content.append(f"  {i}: {names[i]}")

        with open(yaml_path, "w", encoding="utf-8") as f:
            f.write("\n".join(content))


if __name__ == "__main__":

    ROOT_PATH = "C:/projects/agent_cv/"
    path_to_images_with_annotations = ROOT_PATH + "data/data_line_annotation/"

    for dataset_name in os.listdir(path_to_images_with_annotations):

        converter = LabelmeLineToYoloLines(
            dataset_root=path_to_images_with_annotations + dataset_name,
            output_dir=ROOT_PATH + f"data/data_structured/crop_line_uav/{dataset_name}"
        )

        converter.convert()