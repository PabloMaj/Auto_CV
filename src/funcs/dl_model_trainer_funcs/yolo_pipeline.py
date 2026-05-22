from pathlib import Path

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
        class_names: list
    ):
        self.raw_dataset = raw_dataset
        self.processed_dataset = processed_dataset
        self.yolo_train_artifacts_save_path = yolo_train_artifacts_save_path
        self.class_names = class_names

    # -----------------------------
    def run_preprocessing(self):
        pre = YOLOROIPreprocessor(
            input_root=self.raw_dataset,
            output_root=self.processed_dataset,
            tile_size=640,
            overlap=0.5
        )

        # 🔥 split-safe generation
        pre.generate()

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
