
import json
import re
from pathlib import Path

from sklearn.model_selection import GroupShuffleSplit


def source_group(filename: str) -> str:
    """
    Derive a source-sequence group from a J-Litter filename.

    Examples:
        HPD2246C1H_121530_background.jpg -> HPD2246C1H
        6K1744C1H-070_10400.jpg         -> 6K1744C1H-070
    """
    stem = Path(filename).stem

    # Remove explicit background suffix.
    stem = re.sub(r"_background$", "", stem)

    # Remove final frame/time index.
    stem = re.sub(r"_\d+$", "", stem)

    return stem


def build_records(coco_data: dict) -> list[dict]:
    annotated_ids = {
        annotation["image_id"]
        for annotation in coco_data["annotations"]
    }

    records = []

    for image in coco_data["images"]:
        records.append(
            {
                "image_id": image["id"],
                "file_name": image["file_name"],
                "group": source_group(image["file_name"]),
                "has_debris": image["id"] in annotated_ids,
            }
        )

    return records


def create_grouped_split(
    records: list[dict],
    random_state: int = 42,
) -> dict[str, list[dict]]:

    indices = list(range(len(records)))
    groups = [record["group"] for record in records]

    # First split:
    # ~80% train, ~20% temporary.
    split1 = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=random_state,
    )

    train_idx, temp_idx = next(
        split1.split(
            indices,
            groups=groups,
        )
    )

    temp_groups = [
        records[i]["group"]
        for i in temp_idx
    ]

    # Second split:
    # divide temporary set evenly into validation/test.
    split2 = GroupShuffleSplit(
        n_splits=1,
        test_size=0.50,
        random_state=random_state,
    )

    val_relative, test_relative = next(
        split2.split(
            temp_idx,
            groups=temp_groups,
        )
    )

    val_idx = [
        temp_idx[i]
        for i in val_relative
    ]

    test_idx = [
        temp_idx[i]
        for i in test_relative
    ]

    return {
        "train": [
            records[i]
            for i in train_idx
        ],
        "val": [
            records[i]
            for i in val_idx
        ],
        "test": [
            records[i]
            for i in test_idx
        ],
    }


def summarize_split(
    name: str,
    items: list[dict],
) -> None:

    total = len(items)

    debris = sum(
        item["has_debris"]
        for item in items
    )

    background = total - debris

    unique_groups = len(
        {
            item["group"]
            for item in items
        }
    )

    debris_percent = (
        debris / total * 100
        if total
        else 0
    )

    print(name.upper())
    print(f"Images: {total}")
    print(f"Debris images: {debris}")
    print(f"Background images: {background}")
    print(f"Debris %: {debris_percent:.2f}")
    print(f"Groups: {unique_groups}")
    print()


def verify_no_group_leakage(
    splits: dict[str, list[dict]],
) -> None:

    train_groups = {
        item["group"]
        for item in splits["train"]
    }

    val_groups = {
        item["group"]
        for item in splits["val"]
    }

    test_groups = {
        item["group"]
        for item in splits["test"]
    }

    train_val_overlap = len(
        train_groups & val_groups
    )

    train_test_overlap = len(
        train_groups & test_groups
    )

    val_test_overlap = len(
        val_groups & test_groups
    )

    print(
        "Train/Val overlap:",
        train_val_overlap,
    )

    print(
        "Train/Test overlap:",
        train_test_overlap,
    )

    print(
        "Val/Test overlap:",
        val_test_overlap,
    )

    if any(
        [
            train_val_overlap,
            train_test_overlap,
            val_test_overlap,
        ]
    ):
        raise RuntimeError(
            "Source-group leakage detected."
        )


def save_manifest(
    splits: dict[str, list[dict]],
    output_path: Path,
) -> None:

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(output_path, "w") as f:
        json.dump(
            splits,
            f,
            indent=2,
        )

    print(f"Saved: {output_path}")


if __name__ == "__main__":

    jlitter_root = Path(
        "/content/jlitter/J-Litter"
    )

    coco_path = jlitter_root / "coco.json"

    output_path = Path(
        "/content/drive/MyDrive/"
        "MarineDebrisProject/data/jlitter/splits/"
        "jlitter_grouped_split.json"
    )

    with open(coco_path) as f:
        jlitter = json.load(f)

    records = build_records(jlitter)

    print("Images:", len(records))
    print(
        "Unique groups:",
        len(
            {
                record["group"]
                for record in records
            }
        ),
    )
    print()

    splits = create_grouped_split(
        records,
        random_state=42,
    )

    for split_name, items in splits.items():
        summarize_split(
            split_name,
            items,
        )

    verify_no_group_leakage(splits)

    save_manifest(
        splits,
        output_path,
    )
