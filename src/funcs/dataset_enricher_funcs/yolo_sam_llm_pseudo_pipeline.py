import random
import shutil
from pathlib import Path
from typing import List

import cv2
import yaml

from src.funcs.dl_model_trainer_funcs.yolo_roi_preprocessor import YOLOROIPreprocessor
from src.funcs.dataset_enricher_funcs.sam_model import SamSingleton
from src.funcs.dataset_enricher_funcs.sam_auto_labeler import SAM3AutoLabeler
from src.funcs.dataset_enricher_funcs.prompt_optimizer import SAM3PromptOptimizer
from src.funcs.dataset_enricher_funcs.loaders import load_images_and_labels
from src.utils.logger import get_logger
# from src.utils.cuda import cuda_cleanup


logger = get_logger(__name__)


class YOLOSAMLLMPseudoPipeline:

    def __init__(self, dataset_root: Path, unlabeled_root: Path, output_root: Path, class_names: List[str],
                 task: str = "detect", tile_size: int = 640, overlap: float = 0.5, sam_model_path: str = "resources/sam3.pt",
                 llm_model: str = "gemma3:latest", random_seed: int = 42):

        self.dataset_root = Path(dataset_root)
        self.unlabeled_root = Path(unlabeled_root)
        self.output_root = Path(output_root)

        self.class_names = class_names
        self.task = task

        self.tile_size = tile_size
        self.overlap = overlap

        self.sam_model_path = sam_model_path

        self.random_seed = random_seed
        random.seed(random_seed)

        self.prompt_optimizer = SAM3PromptOptimizer(sam_model_path=self.sam_model_path, llm_model=llm_model,
                                                    task=task, max_iters=5, max_desc_words=10)

    def preprocess_labeled_dataset(self):

        logger.info("ROI preprocessing labeled dataset")

        out_root = self.output_root / "tiled_dataset"

        preprocessor = YOLOROIPreprocessor(input_root=self.dataset_root, output_root=out_root,
                                           tile_size=self.tile_size, overlap=self.overlap)
        preprocessor.generate()
        preprocessor.create_yaml(self.class_names)

        return out_root

    def tile_unlabeled(self):

        logger.info("Tiling unlabeled dataset")

        out_dir = self.output_root / "tiled_unlabeled/images"
        out_dir.mkdir(parents=True, exist_ok=True)

        images = (list(self.unlabeled_root.glob("*.jpg")) + list(self.unlabeled_root.glob("*.png")))

        stride = int(self.tile_size * (1 - self.overlap))

        for img_path in images:

            img = cv2.imread(str(img_path))
            if img is None:
                continue

            h, w = img.shape[:2]
            tile_id = 0

            for y in range(0, h, stride):
                for x in range(0, w, stride):

                    x2, y2 = min(x + self.tile_size, w), min(y + self.tile_size, h)
                    x1, y1 = x2 - self.tile_size, y2 - self.tile_size

                    if x1 < 0 or y1 < 0:
                        continue

                    tile = img[y1:y2, x1:x2]

                    if tile.shape[:2] != (self.tile_size, self.tile_size):
                        continue

                    cv2.imwrite(str(out_dir / f"{img_path.stem}_tile_{tile_id}.jpg"), tile)

                    tile_id += 1

        return out_dir

    def optimize_prompt(self, tiled_root: Path, workdir: Path,):

        logger.info("Optimizing SAM prompt (AP50 loop)")

        train_images, train_labels = load_images_and_labels(tiled_root / "images/train", tiled_root / "labels/train")
        val_images, val_labels = load_images_and_labels(tiled_root / "images/val", tiled_root / "labels/val")

        best = self.prompt_optimizer.optimize(train_images=train_images, train_yolo_labels=train_labels,
                                              val_images=val_images, val_yolo_labels=val_labels, workdir=workdir)

        return best["desc"]

    def sam_pseudo_label(self, tiled_unlabeled_dir: Path, prompt: str):

        logger.info("Running SAM pseudo labeling")

        out_dir = self.output_root / "pseudo_labels"
        out_dir.mkdir(parents=True, exist_ok=True)

        sam = SamSingleton(model_path=self.sam_model_path)
        labeler = SAM3AutoLabeler(sam=sam, task=self.task, class_id=0)

        labeler.label_folder(image_dir=tiled_unlabeled_dir, out_label_dir=out_dir,
                             prompt_filter=prompt, visualize=False)

        del labeler
        del sam
        SamSingleton._instance = None

        # cuda_cleanup()

        return out_dir

    def build_dataset(self, tiled_root: Path, tiled_unlabeled_dir: Path, pseudo_dir: Path):

        logger.info("Building final YOLO dataset")

        final_root = self.output_root / "final_dataset"

        for split in ["train", "val", "test"]:

            src_img = tiled_root / f"images/{split}"
            src_lbl = tiled_root / f"labels/{split}"

            dst_img = final_root / f"images/{split}"
            dst_lbl = final_root / f"labels/{split}"

            dst_img.mkdir(parents=True, exist_ok=True)
            dst_lbl.mkdir(parents=True, exist_ok=True)

            for p in src_img.glob("*.*"):
                shutil.copy(p, dst_img / p.name)

            for p in src_lbl.glob("*.txt"):
                shutil.copy(p, dst_lbl / p.name)

        train_img = final_root / "images/train"
        train_lbl = final_root / "labels/train"

        for img_path in tiled_unlabeled_dir.glob("*.*"):

            lbl_path = pseudo_dir / f"{img_path.stem}.txt"

            if not lbl_path.exists():
                continue

            shutil.copy(img_path, train_img / img_path.name)
            shutil.copy(lbl_path, train_lbl / lbl_path.name)

        with open(final_root / "data.yaml", "w") as f:
            yaml.dump(
                {
                    "path": str(final_root.resolve()),
                    "train": "images/train",
                    "val": "images/val",
                    "test": "images/test",
                    "names": self.class_names,
                },
                f,
            )

        return final_root

    def run(self):

        logger.info("START PIPELINE")

        tiled_root = self.preprocess_labeled_dataset()
        tiled_unlabeled = self.tile_unlabeled()
        prompt = self.optimize_prompt(tiled_root=tiled_root, workdir=self.output_root / "prompt_opt")
        pseudo_dir = self.sam_pseudo_label(tiled_unlabeled_dir=tiled_unlabeled, prompt=prompt)
        final_dataset = self.build_dataset(tiled_root=tiled_root, tiled_unlabeled_dir=tiled_unlabeled, pseudo_dir=pseudo_dir)

        logger.info(f"DONE: {final_dataset}")

        return final_dataset


if __name__ == "__main__":

    pipeline = YOLOSAMLLMPseudoPipeline(
        dataset_root=Path("C:/projects/agent_cv/data/data_structured/crop_line_uav/sugarbeet_3_charmont_2017_1"),
        unlabeled_root=Path("C:/projects/agent_cv/data/data_structured/crop_line_uav/sugarbeet_3_charmont_2017_1/images/unlabelled"),
        output_root=Path("C:/projects/agent_cv/outputs/maize_pipeline"),
        class_names=["maize"],
        llm_model="gemma3:latest",
        task="detect",
        tile_size=640,
        overlap=0.5
    )
    final = pipeline.run()

    print("FINAL DATASET:", final)
