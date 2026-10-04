import argparse
import os
from pathlib import Path

import torch
import torch.nn as nn
from src.data.dataset import ACNE04Dataset, get_default_transforms
from src.models.transfer_models import build_transfer_model
from src.utils.metrics import compute_classification_metrics
from torch.utils.data import DataLoader
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Transfer Learning Models on ACNE04"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mobilenet_v2",
        choices=["mobilenet_v2", "efficientnet_b0", "resnet50", "vit"],
    )
    parser.add_argument(
        "--metadata",
        type=str,
        default="data/metadata/acne04_metadata.csv",
        help="Path to metadata CSV",
    )
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_dir", type=str, default="outputs/checkpoints")
    return parser.parse_args()


def main():
    args = parse_args()
    print(f"=== Starting Training Pipeline: {args.model.upper()} on {args.device} ===")

    train_tf, eval_tf = get_default_transforms(image_size=args.img_size)

    # Initialize Datasets
    if not os.path.exists(args.metadata):
        print(f"Error: {args.metadata} not found! Please run notebooks/01_acne04_dataset_exploration.ipynb first.")
        return

    train_dataset = ACNE04Dataset(metadata_csv=args.metadata, split="train", transform=train_tf)
    val_dataset = ACNE04Dataset(metadata_csv=args.metadata, split="val", transform=eval_tf)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # Initialize Model
    model = build_transfer_model(model_name=args.model, num_classes=4, pretrained=True)
    model.to(args.device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_f1 = 0.0
    os.makedirs(args.save_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0

        for images, targets in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [Train]"):
            images, targets = images.to(args.device), targets.to(args.device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        scheduler.step()
        train_loss = running_loss / len(train_dataset)

        # Validation Step
        model.eval()
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for images, targets in tqdm(val_loader, desc=f"Epoch {epoch}/{args.epochs} [Val]"):
                images = images.to(args.device)
                outputs = model(images)
                preds = outputs.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_targets.extend(targets.numpy())

        metrics = compute_classification_metrics(all_targets, all_preds)
        print(f"Epoch {epoch} | Loss: {train_loss:.4f} | Val Acc: {metrics['accuracy']:.4f} | Val F1: {metrics['macro_f1']:.4f} | Val MAE: {metrics['mae']:.4f}")

        if metrics["macro_f1"] > best_val_f1:
            best_val_f1 = metrics["macro_f1"]
            best_path = Path(args.save_dir) / f"{args.model}_best.pth"
            torch.save(model.state_dict(), best_path)
            print(f"--> Saved new best model checkpoint to {best_path} (F1: {best_val_f1:.4f})")

    print("\nTraining completed successfully!")


if __name__ == "__main__":
    main()
