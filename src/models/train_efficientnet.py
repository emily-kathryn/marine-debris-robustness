"""
EfficientNet-B0 Baseline Experiment

Project: Marine Debris Detection and Robustness
Dataset: J-Litter
Task: Binary image classification

Classes:
    0 = Background
    1 = Marine debris

Purpose:
    Compare EfficientNet-B0 against our original
    ResNet-18 baseline.

The test dataset is not used during training.
"""

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models

from train_resnet import (
    JlitterDataset,
    prepare_images,
    run_epoch,
    set_seed,
    validate_splits,
)


SEED = 42


def create_efficientnet(device):
    """Create pretrained EfficientNet-B0."""

    weights = models.EfficientNet_B0_Weights.DEFAULT

    model = models.efficientnet_b0(
        weights=weights
    )

    # Replace ImageNet classifier with binary classifier
    input_features = model.classifier[-1].in_features

    model.classifier[-1] = nn.Linear(
        input_features,
        1
    )

    return model.to(device)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--zip",
        type=Path,
        required=True
    )

    parser.add_argument(
        "--splits",
        type=Path,
        required=True
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=5
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.0001
    )

    args = parser.parse_args()

    if args.epochs < 1:
        raise ValueError("Epochs must be at least 1.")

    set_seed(SEED)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "GPU unavailable. Enable T4 GPU in Colab."
        )

    device = torch.device("cuda")

    print("GPU:", torch.cuda.get_device_name(0))

    args.output.mkdir(
        parents=True,
        exist_ok=True
    )

    # Load saved dataset splits
    with open(args.splits) as file:
        splits = json.load(file)

    # Load original J-Litter images
    image_dir = prepare_images(
        args.zip,
        Path("/content/jlitter_cache")
    )

    # Verify dataset integrity
    validate_splits(
        splits,
        image_dir
    )

    # Same training transformations as original baseline
    train_dataset = JlitterDataset(
        splits["train"],
        image_dir,
        training=True
    )

    # No augmentation during validation
    val_dataset = JlitterDataset(
        splits["val"],
        image_dir,
        training=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    # Initialize EfficientNet
    model = create_efficientnet(device)

    # Account for class imbalance
    positive_count = sum(
        int(record["has_debris"])
        for record in splits["train"]
    )

    negative_count = (
        len(splits["train"]) - positive_count
    )

    positive_weight = torch.tensor(
        [negative_count / positive_count],
        dtype=torch.float32,
        device=device
    )

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=positive_weight
    )

    # Same optimizer as original ResNet baseline
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=0.0001
    )

    configuration = {
        "experiment": "efficientnet_b0_baseline",
        "model": "efficientnet_b0",
        "pretrained": True,
        "seed": SEED,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": 0.0001,
        "image_size": 224,
        "selection_metric": "validation_f1",
        "classification_threshold": 0.5,
        "dataset": "J-Litter",
        "training_augmentation": "horizontal_flip",
        "test_set_used": False
    }

    with open(
        args.output / "config.json", "w"
    ) as file:
        json.dump(
            configuration,
            file,
            indent=2
        )

    history = []

    best_f1 = -1.0
    best_epoch = None

    print("\n=== EFFICIENTNET-B0 TRAINING ===")
    print("Dataset: J-Litter")
    print("Training images:", len(train_dataset))
    print("Validation images:", len(val_dataset))
    print("Epochs:", args.epochs)

    for epoch in range(1, args.epochs + 1):

        # Training
        train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer
        )

        # Validation
        val_metrics = run_epoch(
            model,
            val_loader,
            criterion,
            device
        )

        history.append({
            "epoch": epoch,
            "train": train_metrics,
            "validation": val_metrics
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
            f"Validation precision: "
            f"{val_metrics['precision']:.4f}"
        )

        print(
            f"Validation recall: "
            f"{val_metrics['recall']:.4f}"
        )

        print(
            f"Validation F1: "
            f"{val_metrics['f1']:.4f}"
        )

        # Save training history
        with open(
            args.output / "history.json", "w"
        ) as file:
            json.dump(
                history,
                file,
                indent=2
            )

        # Save best-performing checkpoint
        if val_metrics["f1"] > best_f1:

            best_f1 = val_metrics["f1"]
            best_epoch = epoch

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "validation_metrics": val_metrics,
                    "seed": SEED,
                    "model": "efficientnet_b0",
                    "pretrained": True,
                    "class_names": [
                        "background",
                        "debris"
                    ]
                },
                args.output / "best_model.pth"
            )

            with open(
                args.output / "best_metrics.json", "w"
            ) as file:
                json.dump(
                    val_metrics,
                    file,
                    indent=2
                )

            print("Saved new best model.")

    print("\n=== TRAINING COMPLETE ===")

    print("Best epoch:", best_epoch)
    print(f"Best validation F1: {best_f1:.4f}")
    print("Results saved to:", args.output)


if __name__ == "__main__":
    main()
