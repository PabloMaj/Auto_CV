
from pathlib import Path
from src.logger import log

def dataset_agent(state):
    path = Path(state["dataset_path"])

    images = []

    if path.exists():
        images = list(path.glob("*"))

    state["dataset_report"] = {
        "num_images": len(images),
        "annotation_type": "bbox",
        "image_type": "rgb"
    }

    log(state, f"Dataset analyzed: {len(images)} images")

    return state
