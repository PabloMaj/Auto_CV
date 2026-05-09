
from src.logger import log

BASELINE_CODE = '''
import cv2
import numpy as np

class Predictor:

    def predict(self, image):

        exg = 2 * image[:,:,1] - image[:,:,0] - image[:,:,2]

        exg = np.clip(exg, 0, 255).astype(np.uint8)

        _, thresh = cv2.threshold(
            exg,
            40,
            255,
            cv2.THRESH_BINARY
        )

        kernel = np.ones((3,3), np.uint8)

        thresh = cv2.morphologyEx(
            thresh,
            cv2.MORPH_OPEN,
            kernel
        )

        num_labels, _, _, _ = cv2.connectedComponentsWithStats(
            thresh
        )

        return max(0, num_labels - 1)


if __name__ == "__main__":
    print("Predictor class initialized successfully")
    print("Ready for inference on images")
'''

def programmer_agent(state):

    if state["iteration"] == 0:
        state["predictor_code"] = BASELINE_CODE
        log(state, "Baseline predictor generated")

    else:
        state["predictor_code"] += "\n# improvement iteration"
        log(state, "Predictor improved")

    return state
