from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


class YOLOPredictor:
    """
    Sliding-window inference for large UAV images + merging duplicates.
    """

    def __init__(
        self,
        model_path: str,
        tile_size: int = 640,
        overlap: float = 0.5,
        conf: float = 0.25
    ):
        self.model = YOLO(model_path)
        self.tile_size = tile_size
        self.overlap = overlap
        self.stride = int(tile_size * (1 - overlap))
        self.conf = conf

    # -----------------------------
    # utils
    # -----------------------------
    def xyxy_shift(self, boxes, dx, dy):
        shifted = []
        for b in boxes:
            x1, y1, x2, y2, conf, cls = b
            shifted.append([x1 + dx, y1 + dy, x2 + dx, y2 + dy, conf, cls])
        return shifted

    # -----------------------------
    # IoU
    # -----------------------------
    def iou(self, a, b):
        x1 = max(a[0], b[0])
        y1 = max(a[1], b[1])
        x2 = min(a[2], b[2])
        y2 = min(a[3], b[3])

        inter = max(0, x2 - x1) * max(0, y2 - y1)
        if inter == 0:
            return 0

        area_a = (a[2] - a[0]) * (a[3] - a[1])
        area_b = (b[2] - b[0]) * (b[3] - b[1])

        return inter / (area_a + area_b - inter)

    # -----------------------------
    # NMS (global merge)
    # -----------------------------
    def nms(self, boxes, iou_thr=0.5):
        boxes = sorted(boxes, key=lambda x: x[4], reverse=True)

        keep = []
        while boxes:
            best = boxes.pop(0)
            keep.append(best)

            boxes = [
                b for b in boxes
                if not (self.iou(best, b) > iou_thr and best[5] == b[5])
            ]

        return keep

    # -----------------------------
    # ownership zone filter
    # -----------------------------
    def _ownership_zone(self, tile_x1, tile_y1, tile_x2, tile_y2, img_w, img_h):
        """
        Each tile owns predictions whose midpoint falls in its central strip.
        Edge tiles extend their zone to the image boundary so the full image
        is covered without gaps.
        """
        margin = self.overlap / 2 * self.tile_size
        zx1 = 0 if tile_x1 == 0 else tile_x1 + margin
        zx2 = img_w if tile_x2 >= img_w else tile_x1 + self.tile_size - margin
        zy1 = 0 if tile_y1 == 0 else tile_y1 + margin
        zy2 = img_h if tile_y2 >= img_h else tile_y1 + self.tile_size - margin
        return zx1, zy1, zx2, zy2

    # -----------------------------
    # sliding window inference
    # -----------------------------
    def predict(self, image: np.ndarray):
        h, w = image.shape[:2]

        all_boxes = []

        for y in range(0, h, self.stride):
            for x in range(0, w, self.stride):

                x2 = min(x + self.tile_size, w)
                y2 = min(y + self.tile_size, h)
                x1 = x2 - self.tile_size
                y1 = y2 - self.tile_size

                tile = image[y1:y2, x1:x2]

                results = self.model.predict(tile, conf=self.conf, verbose=False)

                if len(results) == 0:
                    continue

                r = results[0]

                if r.boxes is None:
                    continue

                boxes = r.boxes.xyxy.cpu().numpy()
                confs = r.boxes.conf.cpu().numpy()
                clss = r.boxes.cls.cpu().numpy()

                zx1, zy1, zx2, zy2 = self._ownership_zone(x1, y1, x2, y2, w, h)

                for i in range(len(boxes)):
                    bx1, by1, bx2, by2 = boxes[i]
                    mx = (bx1 + bx2) / 2 + x1
                    my = (by1 + by2) / 2 + y1
                    if zx1 <= mx < zx2 and zy1 <= my < zy2:
                        all_boxes.append([
                            bx1 + x1, by1 + y1, bx2 + x1, by2 + y1,
                            float(confs[i]),
                            int(clss[i])
                        ])

        final_boxes = self.nms(all_boxes, iou_thr=0.75)

        output = []
        for bx1, by1, bx2, by2, conf, cls in final_boxes:
            output.append({
                "bbox": [bx1, by1, bx2, by2],
                "label": self.model.names[int(cls)],
                "confidence": conf
            })

        return output

    # -----------------------------
    # visualize
    # -----------------------------
    def draw(self, image, boxes):
        out = image.copy()

        for b in boxes:
            x1, y1, x2, y2 = b["bbox"]
            conf = b["confidence"]
            label = b["label"]

            cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(out, f"{label}:{conf:.2f}", (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 1)

        return out


if __name__ == "__main__":

    # ścieżki
    ROOT_PATH = r"C:\projects\agent_cv"
    MODEL_PATH = Path(ROOT_PATH) / "yolo_train_artifacts" / "crop_line_uav" / "maize_3_nerac_2016_1_roi640" / "weights" / "best.pt"
    IMAGE_PATH = Path(ROOT_PATH) / "data" / "data_structured" / "crop_line_uav" / "maize_3_nerac_2016_1" / "images" / "test" / "20.png"
    OUTPUT_PATH = Path(ROOT_PATH) / "result.jpg"

    image = cv2.imread(str(IMAGE_PATH))
    if image is None:
        raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

    predictor = YOLOPredictor(model_path=MODEL_PATH, tile_size=640, overlap=0.5, conf=0.25)
    boxes = predictor.predict(image)

    print(boxes)

    print(f"Wykryto obiektów: {len(boxes)}")

    # wizualizacja
    result = predictor.draw(image, boxes)

    # zapis
    cv2.imwrite(OUTPUT_PATH, result)

    # podgląd
    cv2.imshow("Result", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
