from ultralytics.models.sam import SAM3SemanticPredictor


class SamSingleton:
    _instance = None
    _model = None

    def __new__(cls, model_path: str = None, conf_thresh: float = 0.25, save_flag: bool = False):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            if model_path is None:
                raise ValueError("Model path must be provided on first initialization.")
            overrides = dict(conf=conf_thresh, task="segment", mode="predict",
                             model=model_path, half=True, save=save_flag)
            cls._model = SAM3SemanticPredictor(overrides=overrides)
        return cls._instance

    def predict(self, path_to_image: str = None, prompt_filter: str = None):
        self._model.set_image(path_to_image)
        return self._model(text=[prompt_filter])
