import json
from pathlib import Path
from ultralytics import YOLO


class YOLOTrainer:
    def __init__(self, model_weights: str, data_yaml: str, output_dir: Path):
        self.model_weights = model_weights
        self.data_yaml = data_yaml
        self.output_dir = output_dir

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.model = YOLO(f"resources/{model_weights}")

    def train(self, **train_kwargs):
        results = self.model.train(
            data=self.data_yaml, project=self.output_dir.parent,
            name=self.output_dir.name, exist_ok=True, **train_kwargs)
        history_path = self.output_dir / "train_history.json"
        best_val_path = self.output_dir / "best_val_metrics.json"

        history = getattr(results, "results_dict", {}) or {}
        best_metrics = getattr(results, "best", {}) or {}

        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)

        with open(best_val_path, "w") as f:
            json.dump(best_metrics, f, indent=2)

        return results
