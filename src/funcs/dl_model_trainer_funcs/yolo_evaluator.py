import json
from pathlib import Path
from ultralytics import YOLO


class YOLOEvaluator:
    def __init__(self, weights_path: str, data_yaml: str, output_dir: Path):
        self.weights_path = weights_path
        self.data_yaml = data_yaml
        self.output_dir = output_dir

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.model = YOLO(weights_path)

    def test(self, **test_kwargs):
        results = self.model.val(
            data=self.data_yaml,
            split="test",
            **test_kwargs
        )

        metrics_path = self.output_dir / "test_metrics.json"

        metrics = getattr(results, "results_dict", {}) or {}

        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        return results
