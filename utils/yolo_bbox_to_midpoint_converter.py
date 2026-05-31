from pathlib import Path
import shutil
import yaml
import os


class YoloToMidpointConverter:

    def __init__(self, dataset_path: str, output_path: str):
        self.dataset_path = Path(dataset_path)
        self.output_path = Path(output_path)

    def convert(self):
        self.output_path.mkdir(parents=True, exist_ok=True)

        self._copy_images()
        self._convert_labels()
        self._copy_unlabelled()
        self._copy_yaml()

        print(f"Dataset was saved to: {self.output_path}")

    def _copy_images(self):
        src_images = self.dataset_path / "images"

        if not src_images.exists():
            return

        dst_images = self.output_path / "images"

        for split_dir in src_images.iterdir():
            if not split_dir.is_dir():
                continue
            if split_dir.name.lower() in {"vis_temp"}:
                continue
            shutil.copytree(split_dir, dst_images / split_dir.name, dirs_exist_ok=True)

    def _convert_labels(self):
        src_labels = self.dataset_path / "labels"

        if not src_labels.exists():
            return

        dst_labels = self.output_path / "labels"

        for split_dir in src_labels.iterdir():

            if not split_dir.is_dir():
                continue

            out_split = dst_labels / split_dir.name
            out_split.mkdir(parents=True, exist_ok=True)

            for txt_file in split_dir.glob("*.txt"):

                out_file = out_split / txt_file.name

                midpoint_lines = []
                with open(txt_file, "r") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) < 5:
                            continue

                        cls_id = parts[0]
                        xc = parts[1]
                        yc = parts[2]

                        midpoint_lines.append(f"{cls_id} {xc} {yc}")

                with open(out_file, "w") as f:
                    f.write("\n".join(midpoint_lines))

    def _copy_unlabelled(self):
        src_unlabelled = self.dataset_path / "images" / "unlabelled"

        if src_unlabelled.exists():
            shutil.copytree(src_unlabelled, self.output_path / "images" / "unlabelled", dirs_exist_ok=True)

    def _copy_yaml(self):
        src_yaml = self.dataset_path / "data.yaml"

        if not src_yaml.exists():
            return

        with open(src_yaml, "r") as f:
            data = yaml.safe_load(f)

        data["task"] = "midpoint"

        with open(self.output_path / "data.yaml", "w") as f:
            yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


if __name__ == "__main__":

    ROOT_PATH = 'C:/projects/agent_cv/data/data_structured/crop_line_uav/'

    for dataset_name in os.listdir(ROOT_PATH):

        if "_midpoints" not in dataset_name:
            print(dataset_name)
            converter = YoloToMidpointConverter(
                dataset_path=ROOT_PATH + dataset_name,
                output_path=ROOT_PATH + dataset_name + "_midpoints"
            )
            converter.convert()
