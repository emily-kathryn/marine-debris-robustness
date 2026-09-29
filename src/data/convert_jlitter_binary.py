
import json
from pathlib import Path


def convert_to_binary_coco(
    input_json: Path,
    output_json: Path,
) -> None:

    with open(input_json) as f:
        data = json.load(f)

    id_to_name = {
        category["id"]: category["name"]
        for category in data["categories"]
    }

    new_annotations = []

    for annotation in data["annotations"]:

        new_annotation = annotation.copy()

        original_id = annotation["category_id"]
        original_name = id_to_name[original_id]

        # Preserve the original J-Litter category
        # for later error analysis.
        new_annotation["original_category_id"] = original_id
        new_annotation["original_category_name"] = original_name

        # Canonical project class
        new_annotation["category_id"] = 1

        new_annotations.append(new_annotation)

    converted = {
        **data,

        "categories": [
            {
                "id": 1,
                "name": "debris",
                "supercategory": "marine_debris",
            }
        ],

        "annotations": new_annotations,
    }

    output_json.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(output_json, "w") as f:
        json.dump(converted, f)

    print(f"Saved: {output_json}")
    print(f"Images retained: {len(converted['images'])}")
    print(
        f"Debris annotations: "
        f"{len(converted['annotations'])}"
    )


if __name__ == "__main__":

    jlitter_root = Path(
        "/content/jlitter/J-Litter"
    )

    output_dir = Path(
        "/content/drive/MyDrive/"
        "MarineDebrisProject/data/jlitter/processed"
    )

    convert_to_binary_coco(
        jlitter_root / "coco.json",
        output_dir / "jlitter_binary.json",
    )
