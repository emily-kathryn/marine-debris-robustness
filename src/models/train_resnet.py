"""
ResNet-18 baseline for binary marine debris classification.

Dataset: J-Litter
Classes:
    0 = Background
    1 = Marine debris

Uses grouped training and validation splits.
The test split is intentionally excluded from training.
"""

import argparse
import json
import random
import shutil
import zipfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


SEED = 42
IMAGE_SIZE = 224


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class JlitterDataset(Dataset):
    def __init__(self, records, image_dir, training=False):
        self.records = records
        self.image_dir = Path(image_dir)

        transform_list = [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        ]

        if training:
            transform_list.append(
                transforms.RandomHorizontalFlip(p=0.5)
            )

        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        self.transform = transforms.Compose(transform_list)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]

        image_path = self.image_dir / record["file_name"]

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image = self.transform(image)

        label = float(record["has_debris"])

        return image, torch.tensor(label, dtype=torch.float32)


def prepare_images(zip_path, cache_dir):
    """
    Copy the dataset ZIP from Drive to local Colab storage
    and extract it for faster training.
    """

    cache_dir = Path(cache_dir)

    local_zip = Path("/content/J-Litter.zip")
    image_dir = cache_dir / "J-Litter" / "images"

    if not image_dir.exists():
        print("Preparing images...")

        if (
            not local_zip.exists()
            or local_zip.stat().st_size != zip_path.stat().st_size
        ):
            print("Copying ZIP to local storage...")
            shutil.copy2(zip_path, local_zip)

        print("Extracting images...")

        cache_dir.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(local_zip) as archive:
            archive.extractall(cache_dir)

    if not image_dir.is_dir():
        raise FileNotFoundError(
            f"Image directory not found: {image_dir}"
        )

    return image_dir


def validate_splits(splits, image_dir):
    """
    Check that images exist and that groups do not
    overlap between train, validation, and test.
    """

    group_sets = {}
    image_id_sets = {}

    for split_name in ["train", "val", "test"]:
        records = splits[split_name]

        group_sets[split_name] = {
            record["group"] for record in records
        }

        image_id_sets[split_name] = {
            record["image_id"] for record in records
        }

        missing = [
            record["file_name"]
            for record in records
            if not (image_dir / record["file_name"]).is_file()
        ]

        if missing:
            raise FileNotFoundError(
                f"{split_name}: {len(missing)} missing images. "
                f"Example: {missing[:3]}"
            )

        if len(image_id_sets[split_name]) != len(records):
            raise ValueError(
                f"Duplicate image IDs in {split_name}"
            )

    split_names = ["train", "val", "test"]

    for i in range(len(split_names)):
        for j in range(i + 1, len(split_names)):
            first = split_names[i]
            second = split_names[j]

            if group_sets[first] & group_sets[second]:
                raise ValueError(
                    f"Group leakage between {first} and {second}"
                )

            if image_id_sets[first] & image_id_sets[second]:
                raise ValueError(
                    f"Image leakage between {first} and {second}"
                )

    print("Dataset validation passed.")
    print("No overlapping groups or image IDs.")

    for split_name in split_names:
        records = splits[split_name]

        debris_count = sum(
            bool(record["has_debris"])
            for record in records
        )

        print(
            f"{split_name}: {len(records)} images, "
            f"{debris_count} debris"
        )


def create_model(device):
    weights = models.ResNet18_Weights.DEFAULT

    model = models.resnet18(weights=weights)

    model.fc = nn.Linear(
        model.fc.in_features,
        1,
    )

    return model.to(device)


def calculate_metrics(labels, predictions):
    return {
        "accuracy": float(
            accuracy_score(labels, predictions)
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(labels, predictions)
        ),
        "precision": float(
            precision_score(
                labels, predictions, zero_division=0
            )
        ),
        "recall": float(
            recall_score(
                labels, predictions, zero_division=0
            )
        ),
        "f1": float(
            f1_score(
                labels, predictions, zero_division=0
            )
        ),
        "confusion_matrix": confusion_matrix(
            labels, predictions, labels=[0, 1]
        ).tolist(),
    }


def run_epoch(
    model,
    loader,
    criterion,
    device,
    optimizer=None,
):
    training = optimizer is not None

    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    all_labels = []
    all_predictions = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        if training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(training):
            logits = model(images).squeeze(1)

            loss = criterion(logits, labels)

            if training:
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * images.size(0)

        probabilities = torch.sigmoid(logits)

        predictions = (
            probabilities >= 0.5
        ).int()

        all_labels.extend(
            labels.cpu().numpy().astype(int).tolist()
        )

        all_predictions.extend(
            predictions.cpu().numpy().tolist()
        )

    metrics = calculate_metrics(
        all_labels,
        all_predictions,
    )

    metrics["loss"] = total_loss / len(loader.dataset)

    return metrics


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--zip",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--splits",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.0001,
    )

    args = parser.parse_args()

    set_seed(SEED)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "GPU unavailable. Enable a T4 GPU in Colab."
        )

    device = torch.device("cuda")

    print("Device:", torch.cuda.get_device_name(0))

    args.output.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(args.splits) as file:
        splits = json.load(file)

    image_dir = prepare_images(
        args.zip,
        Path("/content/jlitter_cache"),
    )

    validate_splits(splits, image_dir)

    train_dataset = JlitterDataset(
        splits["train"],
        image_dir,
        training=True,
    )

    val_dataset = JlitterDataset(
        splits["val"],
        image_dir,
        training=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    model = create_model(device)

    positive_count = sum(
        record["has_debris"]
        for record in splits["train"]
    )

    negative_count = (
        len(splits["train"]) - positive_count
    )

    positive_weight = torch.tensor(
        [negative_count / positive_count],
        dtype=torch.float32,
        device=device,
    )

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=positive_weight
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=0.0001,
    )

    history = []
    best_f1 = -1.0

    print("\nStarting ResNet-18 training...")

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer,
        )

        val_metrics = run_epoch(
            model,
            val_loader,
            criterion,
            device,
        )

        history.append({
            "epoch": epoch,
            "train": train_metrics,
            "validation": val_metrics,
        })

        print(f"\nEpoch {epoch}/{args.epochs}")

        print(
            f"Training loss: "
            f"{train_metrics['loss']:.4f}"
        )

        print(
            f"Validation loss: "
            f"{val_metrics['loss']:.4f}"
        )

        print(
            f"Validation accuracy: "
            f"{val_metrics['accuracy']:.4f}"
        )

        print(
            f"Validation F1: "
            f"{val_metrics['f1']:.4f}"
        )

        with open(
            args.output / "history.json", "w"
        ) as file:
            json.dump(history, file, indent=2)

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "validation_metrics": val_metrics,
                    "seed": SEED,
                    "model": "resnet18",
                    "pretrained": True,
                    "class_names": [
                        "background",
                        "debris",
                    ],
                },
                args.output / "best_model.pth",
            )

            with open(
                args.output / "best_metrics.json", "w"
            ) as file:
                json.dump(
                    val_metrics,
                    file,
                    indent=2,
                )

            print("Saved new best model.")

    print("\nTraining complete.")
    print(f"Best validation F1: {best_f1:.4f}")
    print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()
