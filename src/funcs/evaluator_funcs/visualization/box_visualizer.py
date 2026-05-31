from pathlib import Path
import cv2


class BoxVisualizer:

    @staticmethod
    def visualize(image_path, tp_all, fp_all, fn_all):

        img = cv2.imread(str(image_path))
        if img is None:
            return None

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image_name = Path(image_path).name

        def draw(box, color, label):

            x1, y1, x2, y2 = map(int, box)

            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        for item in tp_all:
            if item["image"] == image_name:
                draw(item["bbox"], (0, 255, 0), "TP")

        for item in fp_all:
            if item["image"] == image_name:
                draw(item["bbox"], (255, 0, 0), "FP")

        for item in fn_all:
            if item["image"] == image_name:
                draw(item["bbox"], (255, 255, 0), "FN")

        return img
