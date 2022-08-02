from mlbenchmarks import vision
from typing import TypedDict


class VisionModelsDict(TypedDict):
    classifier: vision.Classifier


class MLModelsDict(TypedDict):
    vision: VisionModelsDict


class MLModelsProvider:
    def __init__(self):
        self.models: MLModelsDict = {
            "vision": {
                "classifier": vision.classification.Classifier(),
            }
        }

    def __call__(self) -> MLModelsDict:
        return self.models

ml_models_provider = MLModelsProvider()