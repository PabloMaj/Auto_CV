from pathlib import Path
import shutil
from src.funcs.dl_model_trainer_funcs.yolo_roi_preprocessor import YOLOROIPreprocessor
from src.funcs.dl_model_trainer_funcs.yolo_trainer import YOLOTrainer


class YOLOPipeline:
    """
    Split-safe ROI pipeline (fixed version).
    """

    def __init__(
        self,
        raw_dataset: Path,
        processed_dataset: Path,
        yolo_train_artifacts_save_path: Path,
        class_names: list,
        pseudo_images_dir: Path = None,
        pseudo_labels_dir: Path = None,
    ):
        self.raw_dataset = raw_dataset
        self.processed_dataset = processed_dataset
        self.yolo_train_artifacts_save_path = yolo_train_artifacts_save_path
        self.class_names = class_names
        self.pseudo_images_dir = pseudo_images_dir
        self.pseudo_labels_dir = pseudo_labels_dir

        # print(os.listdir(str(self.pseudo_images_dir)))
        # print(os.listdir(str(self.pseudo_labels_dir)))

    #  -----------------------------
    def copy_pseudo_labeled_data(self):
        """
        Copies tiled pseudo-labeled samples
        directly into TRAIN split.
        """

        if self.pseudo_images_dir is None or self.pseudo_labels_dir is None:
            print("No pseudo-labeled dataset provided.")
            return

        target_images_dir = self.processed_dataset / "images" / "train"
        target_labels_dir = self.processed_dataset / "labels" / "train"

        target_images_dir.mkdir(parents=True, exist_ok=True)
        target_labels_dir.mkdir(parents=True, exist_ok=True)

        image_extensions = [".jpg", ".jpeg", ".png", ".tif", ".tiff"]

        copied = 0

        print("LABEL DIR:", self.pseudo_labels_dir)
        print("FILES FOUND:", list(self.pseudo_labels_dir.glob("*.txt")))

        for label_file in self.pseudo_labels_dir.glob("*.txt"):

            stem = label_file.stem

            image_file = None

            for ext in image_extensions:

                candidate = self.pseudo_images_dir / f"{stem}{ext}"

                if candidate.exists():
                    image_file = candidate
                    break

            if image_file is None:
                print(f"[WARNING] Missing image for: {label_file.name}")
                continue

            shutil.copy2(image_file, target_images_dir / image_file.name)
            shutil.copy2(label_file, target_labels_dir / label_file.name)

            copied += 1

        print(f"Copied {copied} pseudo-labeled train samples.")

    # -----------------------------
    def run_preprocessing(self):
        pre = YOLOROIPreprocessor(
            input_root=self.raw_dataset,
            output_root=self.processed_dataset,
            tile_size=640,
            overlap=0.5
        )

        # split-safe generation
        pre.generate()

        # add pseudo labeled samples ONLY to train split
        self.copy_pseudo_labeled_data()

        yaml_path = pre.create_yaml(self.class_names)

        return yaml_path

    # -----------------------------
    def run_training(self, yaml_path: Path, epochs=50, batch=16):

        trainer = YOLOTrainer(
            model_weights="yolov8m.pt",
            data_yaml=str(yaml_path),
            output_dir=Path(self.yolo_train_artifacts_save_path)
        )

        results = trainer.train(
            epochs=epochs,
            imgsz=640,
            batch=batch,
            save=True
        )
        print(f"Training results: {results}")

        self.model_save_path = self.yolo_train_artifacts_save_path / "weights" / "best.pt"

        return self.model_save_path

    # -----------------------------
    def run(self):
        yaml_path = self.run_preprocessing()
        best_model = self.run_training(yaml_path)

        return {
            "yaml": yaml_path,
            "best_model": best_model
        }


def main():

    pipeline = YOLOPipeline(
        raw_dataset=Path(r"C:\projects\agent_cv\data\data_structured\crop_line_uav\maize_3_nerac_2016_1"),
        processed_dataset=Path(r"C:\projects\agent_cv\data\data_structured\crop_line_uav\maize_3_nerac_2016_1_roi640"),
        yolo_train_artifacts_save_path=Path(r"C:\projects\agent_cv\yolo_train_artifacts\crop_line_uav\maize_3_nerac_2016_1_roi640"),
        class_names=["crop"]
    )

    result = pipeline.run()

    print("\n===== PIPELINE DONE =====")
    print(f"YOLO YAML: {result['yaml']}")
    print(f"BEST MODEL: {result['best_model']}")


if __name__ == "__main__":
    main()
