BASE_PROGRAMMER_PROMPT = """
USER TASK:
{user_prompt}

DESIRED OUTPUT SPECIFICATION:
{desired_output_specification}

GENERAL REQUIREMENTS:
- Return ONLY Python source code.
- Wrap the source code between:
  {code_start_token}
  and
  {code_end_token}

IMPLEMENTATION REQUIREMENTS:
- Main implementation must be inside class Predictor.
- desired form of the prediction:
    predictor = Predictor()
    results = predictor.predict(img_path), where img_path is the path to a single image file.
- default parameters for predict method should be set to allow direct usage without additional tuning.
- Additional helper classes and functions are allowed.
- Saveable as generated_solution.py.
- Must contain:

if __name__ == "__main__":

- Code must be executable.
- No pseudocode.
- No TODO placeholders.
- Include imports.
- Prefer robust and readable implementations.
- Handle edge cases.
- Avoid unnecessary complexity.
{dl_model_section}"""

DL_MODEL_SECTION = """
DEEP LEARNING MODEL AVAILABILITY:
- A trained deep learning object detection model may be available and can be used if it is beneficial for solving the task.
- If the deep learning model is used, it can be loaded from the following path:
    MODEL_PATH = {model_path}
- Example usage:

    from src.funcs.dl_model_trainer_funcs.yolo_predictor import YOLOPredictor

    image = cv2.imread(str(IMAGE_PATH))
    predictor = YOLOPredictor(model_path=MODEL_PATH, tile_size=640, overlap=0.5, conf=0.25)
    boxes = predictor.predict(image)

- When using the deep learning model through the example predictor shown above, the returned `boxes` variable already
 follows the desired output format and can be used directly in the solution.
- Usage of the deep learning model is OPTIONAL.
- The solution may use either:
    - the provided deep learning model,
    - classical computer vision techniques (OpenCV),
    - or a combination of both.
- Choose the approach that is most appropriate, robust, and efficient for the given task.
- Do not force the use of the deep learning model if a classical computer vision solution is more suitable.
"""
