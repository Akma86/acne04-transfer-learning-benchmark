import os
from pathlib import Path
from typing import Callable, Optional, Tuple

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class ACNE04Dataset(Dataset):
    """PyTorch Dataset for the ACNE04 Acne Severity Classification Benchmark.

    Args:
        metadata_csv (str or Path): Path to the generated metadata CSV file.
        split (str): One of 'train', 'val', or 'test'.
        transform (Optional[Callable]): Optional image transformation pipeline.
        root_dir (Optional[Path]): Base project directory to resolve relative paths.
    """

    def __init__(
        self,
        metadata_csv: str | Path,
        split: str = "train",
        transform: Optional[Callable] = None,
        root_dir: Optional[Path] = None,
    ):
        self.metadata_df = pd.read_csv(metadata_csv)
        if split not in ["train", "val", "test", "all"]:
            raise ValueError(
                f"Invalid split: {split}. Choose from ['train', 'val', 'test', 'all']"
            )

        if split != "all":
            self.metadata_df = self.metadata_df[
                self.metadata_df["split"] == split
            ].reset_index(drop=True)

        self.transform = transform
        self.root_dir = (
            Path(root_dir) if root_dir else Path(__file__).resolve().parents[2]
        )

    def __len__(self) -> int:
        return len(self.metadata_df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        row = self.metadata_df.iloc[idx]
        image_path = self.root_dir / row["relative_path"]

        image = Image.open(image_path).convert("RGB")
        label = int(row["label"])

        if self.transform:
            image = self.transform(image)

        return image, label


def get_default_transforms(
    image_size: int = 224,
) -> Tuple[transforms.Compose, transforms.Compose]:
    """Returns standard ImageNet normalization and augmentation pipelines."""
    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(
                brightness=0.1, contrast=0.1, saturation=0.1
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )

    eval_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )

    return train_transform, eval_transform
