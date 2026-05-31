from pathlib import Path
import cv2


class PointVisualizer:

    @staticmethod
    def visualize(image_path, tp_all, fp_all, fn_all):

        img = cv2.imread(str(image_path))
        if img is None:
            return None

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        name = Path(image_path).name

        def draw(pt, color, label):
            x, y = map(int, pt)
            cv2.circle(img, (x, y), 4, color, -1)
            cv2.putText(img, label, (x + 3, y - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        for t in tp_all:
            if t["image"] == name:
                draw(t["point"], (0, 255, 0), "TP")

        for f in fp_all:
            if f["image"] == name:
                draw(f["point"], (255, 0, 0), "FP")

        for f in fn_all:
            if f["image"] == name:
                draw(f["point"], (255, 255, 0), "FN")

        return img
