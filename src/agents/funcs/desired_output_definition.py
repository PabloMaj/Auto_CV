# src/agents/funcs/desired_output_definition.py

from dataclasses import dataclass
from typing import Dict


@dataclass
class DesiredOutputDefinition:
    name: str
    description: str
    structure: str
    example: str


class DesiredOutputRegistry:

    DEFINITIONS: Dict[str, DesiredOutputDefinition] = {
        "scalar": DesiredOutputDefinition(
            name="scalar",
            description="Single scalar numeric prediction.",
            structure="float | int",
            example="0.873"
        ),

        "object_detection": DesiredOutputDefinition(
            name="object_detection",
            description="List of bounding boxes with class labels and confidence.",
            structure=(
                "List[Dict] where each Dict contains: "
                "bbox=[x1, y1, x2, y2], label=str, confidence=float"
            ),
            example='[{"bbox": [10, 20, 100, 120], "label": "cat", "confidence": 0.98}]'
        ),

        "segmentation_masks": DesiredOutputDefinition(
            name="segmentation_masks",
            description="Binary or multiclass segmentation masks.",
            structure="np.ndarray with shape [H, W] or [C, H, W]",
            example="mask.shape == (512, 512)"
        ),
    }

    @classmethod
    def get(cls, output_name: str) -> DesiredOutputDefinition:
        if output_name not in cls.DEFINITIONS:
            raise ValueError(f"Unknown desired output: {output_name}")

        return cls.DEFINITIONS[output_name]


if __name__ == "__main__":
    for name, definition in DesiredOutputRegistry.DEFINITIONS.items():
        print(f"Name: {definition.name}")
        print(f"Description: {definition.description}")
        print(f"Structure: {definition.structure}")
        print(f"Example: {definition.example}")
        print("-" * 40)
