import argparse
import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import ACNE04Dataset
from src.data.preprocessing import (
    STAGE_DESCRIPTIONS,
    build_pipeline_by_stage,
    compute_class_weights,
    get_weighted_sampler,
)
from src.models.transfer_models import build_transfer_model
from src.utils.metrics import compute_classification_metrics


def parse_args():
    parser = argparse.ArgumentParser(
        description="Ablation and Transfer Learning Training on ACNE04"
    )
    # Model Selection
    parser.add_argument(
        "--model",
        type=str,
        default="mobilenet_v2",
        choices=["mobilenet_v2", "efficientnet_b0", "resnet50", "vit"],
        help="Candidate architecture for transfer learning benchmark",
    )
    # Preprocessing Ablation Stage (1 through 7)
    parser.add_argument(
        "--stage",
        type=int,
        default=1,
        choices=[1, 2, 3, 4, 5, 6, 7],
        help="Incremental preprocessing stage (1: Base, 2: +Aug, 3: +CLAHE, 4: +WB, 5: +Face, 6: +Skin, 7: +Imbalance)",
    )
    # Preprocessing Hyperparameters
    parser.add_argument(
        "--clahe_clip",
        type=float,
        default=2.0,
        help="CLAHE contrast clip limit",
    )
    parser.add_argument(
        "--wb_method",
        type=str,
        default="gray_world",
        choices=["gray_world", "shades_of_gray"],
        help="Color constancy method",
    )
    parser.add_argument(
        "--face_margin",
        type=float,
        default=0.15,
        help="Bounding box padding margin for face crop",
    )
    parser.add_argument(
        "--skin_blend",
        type=float,
        default=0.15,
        help="Background blend factor for skin segmentation (0.0=black, 0.15=dimmed)",
    )
    # Class Imbalance Handling
    parser.add_argument(
        "--imbalance_method",
        type=str,
        default="auto",
        choices=["auto", "none", "class_weights", "ros"],
        help="Imbalance strategy (auto uses class_weights when stage >= 7)",
    )
    # Training Parameters
    parser.add_argument(
        "--metadata",
        type=str,
        default="data/metadata/acne04_metadata.csv",
        help="Path to metadata CSV",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="outputs/checkpoints",
        help="Directory to save model checkpoints",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 75)
    print(f" EXPERIMENT: {args.model.upper()} | PREPROCESSING STAGE {args.stage}/7")
    print(f" {STAGE_DESCRIPTIONS[args.stage]}")
    print(f" Device: {args.device} | Image Size: {args.img_size}x{args.img_size}")
    print("=" * 75)

    if not os.path.exists(args.metadata):
        print(f"Error: {args.metadata} not found! Please run notebooks/01_acne04_dataset_exploration.ipynb first.")
        return

    # Build Stage-specific Preprocessing Pipelines
    train_tf = build_pipeline_by_stage(
        stage=args.stage,
        image_size=args.img_size,
        is_train=True,
        clahe_clip=args.clahe_clip,
        wb_method=args.wb_method,
        face_margin=args.face_margin,
        skin_blend=args.skin_blend,
    )
    eval_tf = build_pipeline_by_stage(
        stage=args.stage,
        image_size=args.img_size,
        is_train=False,
        clahe_clip=args.clahe_clip,
        wb_method=args.wb_method,
        face_margin=args.face_margin,
        skin_blend=args.skin_blend,
    )

    # Initialize Datasets
    train_dataset = ACNE04Dataset(metadata_csv=args.metadata, split="train", transform=train_tf)
    val_dataset = ACNE04Dataset(metadata_csv=args.metadata, split="val", transform=eval_tf)

    print(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")

    # Resolve Imbalance Strategy
    imbalance_mode = args.imbalance_method
    if imbalance_mode == "auto":
        imbalance_mode = "class_weights" if args.stage >= 7 else "none"

    train_labels = [label for _, label in train_dataset]
    train_sampler = None
    loss_weights = None

    if imbalance_mode == "class_weights":
        loss_weights = compute_class_weights(train_labels, num_classes=4).to(args.device)
        print(f"[Imbalance Mode: Class Weights] Computed Loss Weights: {loss_weights.cpu().numpy().round(3)}")
    elif imbalance_mode == "ros":
        train_sampler = get_weighted_sampler(train_labels, num_classes=4)
        print("[Imbalance Mode: ROS] Active WeightedRandomSampler in DataLoader")
    else:
        print("[Imbalance Mode: None] Standard unweighted training")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=2,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
    )

    # Initialize Model
    model = build_transfer_model(model_name=args.model, num_classes=4, pretrained=True)
    model.to(args.device)

    criterion = nn.CrossEntropyLoss(weight=loss_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_f1 = 0.0
    os.makedirs(args.save_dir, exist_ok=True)
    checkpoint_name = f"{args.model}_stage{args.stage}_best.pth"
    best_path = Path(args.save_dir) / checkpoint_name

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
        print(
            f"Epoch {epoch:02d} | Loss: {train_loss:.4f} | "
            f"Val Acc: {metrics['accuracy']:.4f} | "
            f"Val F1: {metrics['macro_f1']:.4f} | "
            f"Val MAE: {metrics['mae']:.4f}"
        )

        if metrics["macro_f1"] > best_val_f1:
            best_val_f1 = metrics["macro_f1"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_name": args.model,
                    "stage": args.stage,
                    "state_dict": model.state_dict(),
                    "macro_f1": best_val_f1,
                    "metrics": metrics,
                },
                best_path,
            )
            print(f"  --> Saved new best checkpoint: {best_path} (Macro F1: {best_val_f1:.4f})")

    print(f"\nTraining completed! Best checkpoint: {best_path} (F1: {best_val_f1:.4f})")


if __name__ == "__main__":
    main()
