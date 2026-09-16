"""
Train a ResNet18 classifier on the Food11 dataset, with MLflow tracking.

Usage:
    uv run python ./src/food11/train.py --dataset mini --epochs 5 --lr 0.001 --batch-size 32
    uv run python ./src/food11/train.py --dataset processed --epochs 10 --lr 0.0001 --batch-size 64
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "food11"

DATASET_DIRS = {
    "mini": Path("data/food11_processed_mini"),
    "processed": Path("data/food11_processed"),
}

NUM_CLASSES = 11
IMAGE_SIZE = 128


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a ResNet18 on Food11")
    parser.add_argument(
        "--dataset",
        choices=["mini", "processed"],
        default="mini",
        help="Which processed dataset to train on (default: mini)",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def build_dataloaders(dataset_root: Path, batch_size: int):
    # ImageNet normalization stats, since we're fine-tuning a pretrained resnet18.
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            normalize,
        ]
    )

    train_dir = dataset_root / "training"
    val_dir = dataset_root / "validation"
    test_dir = dataset_root / "evaluation"

    train_ds = datasets.ImageFolder(train_dir, transform=transform)
    val_ds = datasets.ImageFolder(val_dir, transform=transform)
    test_ds = datasets.ImageFolder(test_dir, transform=transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader, train_ds.classes


def build_model(num_classes: int, device: torch.device) -> nn.Module:
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model.to(device)


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    torch.set_grad_enabled(train)
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        if train:
            optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        if train:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    torch.set_grad_enabled(True)
    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def main() -> None:
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset_root = DATASET_DIRS[args.dataset]
    if not dataset_root.exists():
        raise SystemExit(
            f"Dataset folder not found at '{dataset_root}'. "
            "Run ./src/food11/data.py first, or dvc pull the data."
        )

    train_loader, val_loader, test_loader, classes = build_dataloaders(
        dataset_root, args.batch_size
    )

    model = build_model(NUM_CLASSES, device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run():
        mlflow.log_params(
            {
                "dataset": args.dataset,
                "epochs": args.epochs,
                "lr": args.lr,
                "batch_size": args.batch_size,
                "model": "resnet18",
                "num_classes": NUM_CLASSES,
                "image_size": IMAGE_SIZE,
            }
        )

        for epoch in range(args.epochs):
            train_loss, train_acc = run_epoch(
                model, train_loader, criterion, optimizer, device, train=True
            )
            val_loss, val_acc = run_epoch(
                model, val_loader, criterion, optimizer, device, train=False
            )

            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("val_accuracy", val_acc, step=epoch)

            print(
                f"Epoch {epoch + 1}/{args.epochs} "
                f"train_loss={train_loss:.4f} "
                f"val_loss={val_loss:.4f} "
                f"val_accuracy={val_acc:.4f}"
            )

        test_loss, test_acc = run_epoch(
            model, test_loader, criterion, optimizer, device, train=False
        )
        mlflow.log_metric("test_accuracy", test_acc)
        print(f"Final test_accuracy={test_acc:.4f}")

        example_input = next(iter(train_loader))[0][:1].cpu()
        mlflow.pytorch.log_model(
            model,
            "model",
            input_example=example_input,
            serialization_format = "pickle",
        )


if __name__ == "__main__":
    main()
    