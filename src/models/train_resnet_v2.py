"""
ResNet-18 V2: Color Augmentation Experiment

Dataset: J-Litter
Task: Binary marine debris classification

Changes from baseline:
    - Mild color augmentation during training

Unchanged:
    - ResNet-18 architecture
    - Pretrained ImageNet weights
    - Data splits
    - Learning rate
    - Batch size
    - Loss function
    - Optimizer
    - Number of epochs

The test set is not used.
"""

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

from train_resnet import (
    IMAGE_SIZE,
    JlitterDataset,
    create_model,
    prepare_images,
    run_epoch,
    set_seed,
    validate_splits,
)


SEED = 42


class AugmentedJlitterDataset(JlitterDataset):
    """
    J-Litter dataset with additional color augmentation.

    Augmentation is applied only to training images.
    """

    def __init__(self, records, image_dir, training=False):
        super().__init__(
            records,
            image_dir,
            training=training,
        )

        if training:
            self.transform = transforms.Compose([
                transforms.Resize(
                    (IMAGE_SIZE, IMAGE_SIZE)
                ),

                transforms.RandomHorizontalFlip(
                    p=0.5
                ),

                transforms.ColorJitter(
                    brightness=0.15,
                    contrast=0.15,
                    saturation=0.10,
                    hue=0.02,
                ),

                transforms.ToTensor(),

                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])


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

    if args.epochs < 1:
        raise ValueError("Epochs must be at least 1.")

    set_seed(SEED)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "GPU unavailable. Enable a T4 GPU in Colab."
        )

    device = torch.device("cuda")

    print("GPU:", torch.cuda.get_device_name(0))

    args.output.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Load dataset splits
    with open(args.splits) as file:
        splits = json.load(file)

    # Prepare images
    image_dir = prepare_images(
        args.zip,
        Path("/content/jlitter_cache"),
    )

    # Check data integrity
    validate_splits(
        splits,
        image_dir,
    )

    # Training dataset with color augmentation
    train_dataset = AugmentedJlitterDataset(
        splits["train"],
        image_dir,
        training=True,
    )

    # Validation dataset without augmentation
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

    # Same ResNet-18 as original baseline
    model = create_model(device)

    # Same class weighting as original baseline
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
        device=device,
    )

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=positive_weight
    )

    # Same optimizer and regularization
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=0.0001,
    )

    # Save experiment settings
    configuration = {
        "experiment": "resnet18_color_augmentation",
        "seed": SEED,
        "model": "resnet18",
        "pretrained": True,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": 0.0001,
        "selection_metric": "validation_f1",
        "classification_threshold": 0.5,
        "augmentation": {
            "horizontal_flip": 0.5,
            "brightness": 0.15,
            "contrast": 0.15,
            "saturation": 0.10,
            "hue": 0.02,
        },
    }

    with open(
        args.output / "config.json", "w"
    ) as file:
        json.dump(
            configuration,
            file,
            indent=2,
        )

    history = []
    best_f1 = -1.0
    best_epoch = None

    print("\n=== RESNET-18 V2 TRAINING ===")
    print("Experiment: Color Augmentation")
    print("Epochs:", args.epochs)

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
            f"Validation recall: "
            f"{val_metrics['recall']:.4f}"
        )

        print(
            f"Validation F1: "
            f"{val_metrics['f1']:.4f}"
        )

        # Save complete training history
        with open(
            args.output / "history.json", "w"
        ) as file:
            json.dump(
                history,
                file,
                indent=2,
            )

        # Save best checkpoint based on validation F1
        if val_metrics["f1"] > best_f1:

            best_f1 = val_metrics["f1"]
            best_epoch = epoch

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "validation_metrics": val_metrics,
                    "seed": SEED,
                    "model": "resnet18",
                    "pretrained": True,
                    "experiment": "color_augmentation",
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

    print("\n=== TRAINING COMPLETE ===")

    print("Best epoch:", best_epoch)
    print(f"Best validation F1: {best_f1:.4f}")

    print(
        "Results saved to:",
        args.output,
    )


if __name__ == "__main__":
    main()
