
import json
from pathlib import Path

TRASH_CATEGORIES = {
    "trash_etc",
    "trash_fabric",
    "trash_fishing_gear",
    "trash_metal",
    "trash_paper",
    "trash_plastic",
    "trash_rubber",
    "trash_wood",
}


def convert_to_binary_coco(input_json: Path, output_json: Path) -> None:
    with open(input_json) as f:
        data = json.load(f)

    id_to_name = {
        category["id"]: category["name"]
        for category in data["categories"]
    }

    new_annotations = []

    for annotation in data["annotations"]:
        original_name = id_to_name[annotation["category_id"]]

        if original_name in TRASH_CATEGORIES:
            new_annotation = annotation.copy()

            new_annotation["original_category_id"] = annotation["category_id"]
            new_annotation["original_category_name"] = original_name
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

    output_json.parent.mkdir(parents=True, exist_ok=True)

    with open(output_json, "w") as f:
        json.dump(converted, f)

    print(f"Saved: {output_json}")
    print(f"Images retained: {len(converted['images'])}")
    print(f"Debris annotations: {len(converted['annotations'])}")


if __name__ == "__main__":
    material_root = Path("/content/trashcan/dataset/material_version")

    output_dir = Path(
        "/content/drive/MyDrive/MarineDebrisProject/data/trashcan/processed"
    )

    convert_to_binary_coco(
        material_root / "instances_train_trashcan.json",
        output_dir / "trashcan_train_binary.json",
    )

    convert_to_binary_coco(
        material_root / "instances_val_trashcan.json",
        output_dir / "trashcan_val_binary.json",
    )
