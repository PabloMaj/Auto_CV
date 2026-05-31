from pathlib import Path
import cv2


class LineVisualizer:

    @staticmethod
    def visualize(image_path, tp_all, fp_all, fn_all):

        img = cv2.imread(str(image_path))
        if img is None:
            return None

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        name = Path(image_path).name

        def draw(line, color, label):

            x1, y1, x2, y2 = map(int, line)

            cv2.line(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, label, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        for t in tp_all:
            if t["image"] == name:
                draw(t["line"], (0, 255, 0), "TP")

        for f in fp_all:
            if f["image"] == name:
                draw(f["line"], (255, 0, 0), "FP")

        for f in fn_all:
            if f["image"] == name:
                draw(f["line"], (255, 255, 0), "FN")

        return img
