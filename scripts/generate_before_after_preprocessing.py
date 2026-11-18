"""Generate a multi-stage Before-After Preprocessing and Data Augmentation visual for Bab III."""

from __future__ import annotations

import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image, ImageOps

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocessing import (
    CLAHETransform,
    ColorConstancyTransform,
    FaceCropTransform,
    SafeRandomRotation,
    SkinMaskTransform,
)

RAW_DIR = PROJECT_ROOT / "data" / "raw" / "acne04"
OUTPUT_FIG_DIR = PROJECT_ROOT / "outputs" / "figures"
PAPER_FIG_DIR = PROJECT_ROOT / "docs" / "paper" / "Gambar"
OUTPUT_PATH_REPO = OUTPUT_FIG_DIR / "preprocessing_augmentation_before_after.png"
OUTPUT_PATH_PAPER = PAPER_FIG_DIR / "preprocessing_augmentation_before_after.png"

SAMPLES = [
    ("level_1_moderate/levle1_117.jpg", "Subjek 1: Level 1 (Moderate)\n6–20 Lesi"),
    ("level_3_very_severe/levle3_106.jpg", "Subjek 2: Level 3 (Very Severe)\n>50 Lesi"),
]


def pad_to_square(im: Image.Image, target_size: int = 224, fill: tuple[int, int, int] = (241, 245, 249)) -> Image.Image:
    """Letterbox raw image cleanly so aspect ratio and raw background clutter remain visible."""
    rgb = im.convert("RGB")
    return ImageOps.pad(rgb, (target_size, target_size), method=Image.Resampling.LANCZOS, color=fill)


def build_stage_images(img_path: Path, rot_seed: int, jitter_seed: int) -> list[tuple[Image.Image, str]]:
    """Run actual src.data.preprocessing transforms to capture Before, Preprocessing, and Augmentation outputs."""
    face_crop = FaceCropTransform(margin=0.15, min_size=(100, 100))
    skin_mask = SkinMaskTransform(blend_alpha=0.15)
    color_const = ColorConstancyTransform(method="gray_world", power_p=6)
    clahe = CLAHETransform(clip_limit=2.0, tile_grid_size=(8, 8))
    resize = T.Resize((224, 224), interpolation=T.InterpolationMode.BILINEAR)

    with Image.open(img_path) as im:
        raw = im.convert("RGB")
        orig_w, orig_h = raw.size

    raw_panel = pad_to_square(raw, target_size=224)
    cropped = face_crop(raw)
    masked = skin_mask(cropped)
    roi_panel = resize(masked)

    color_norm = color_const(masked)
    preprocessed_224 = resize(clahe(color_norm))

    # Augmentation 1: Horizontal Flip + SafeRandomRotation (reflection-padded)
    flipped = TF.hflip(preprocessed_224)
    np.random.seed(rot_seed)
    angle_deg = float(np.random.uniform(-15.0, 15.0))
    np.random.seed(rot_seed)
    rotator = SafeRandomRotation(degrees=15.0)
    aug_geom = rotator(flipped)

    # Augmentation 2: Mild ColorJitter on top of geometric augmentation
    random.seed(jitter_seed)
    torch.manual_seed(jitter_seed)
    jitter = T.ColorJitter(brightness=0.10, contrast=0.10, saturation=0.10, hue=0.03)
    aug_full = jitter(aug_geom)

    return [
        (raw_panel, f"{orig_w}×{orig_h} px"),
        (roi_panel, "ROI Wajah + Soft Mask"),
        (preprocessed_224, "224×224 px (Input Eval/Uji)"),
        (aug_geom, f"Flip + Rotasi ({angle_deg:+.1f}°)"),
        (aug_full, "+ ColorJitter (±10%)"),
    ]


def main() -> None:
    OUTPUT_FIG_DIR.mkdir(parents=True, exist_ok=True)
    PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)

    col_titles = [
        "(a) Citra Asli\n(Before)",
        "(b) Face Crop &\nSkin Masking",
        "(c) Gray World + CLAHE\n(After Preprocessing)",
        "(d) Augmentasi Geometrik\n(H-Flip + Safe Rotation)",
        "(e) Augmentasi Gabungan\n(+ Mild ColorJitter)",
    ]

    fig = plt.figure(figsize=(15.5, 6.8), dpi=300, facecolor="white")
    gs = fig.add_gridspec(
        nrows=2,
        ncols=5,
        left=0.125,
        right=0.985,
        top=0.81,
        bottom=0.08,
        wspace=0.10,
        hspace=0.22,
    )

    sample_configs = [
        (SAMPLES[0], 6, 101),
        (SAMPLES[1], 7, 202),
    ]

    axes_grid = np.empty((2, 5), dtype=object)
    for r_idx, ((rel_path, row_label), rot_seed, seed) in enumerate(sample_configs):
        stages = build_stage_images(RAW_DIR / rel_path, rot_seed=rot_seed, jitter_seed=seed)
        for c_idx, (pil_im, sub_note) in enumerate(stages):
            ax = fig.add_subplot(gs[r_idx, c_idx])
            axes_grid[r_idx, c_idx] = ax
            ax.imshow(pil_im)
            ax.set_xticks([])
            ax.set_yticks([])

            # Border tint by phase (neutral slate for Before, navy for Preprocessing, crimson-rose for Augmentation)
            edge_color = "#64748B" if c_idx == 0 else ("#143C6E" if c_idx in (1, 2) else "#B91C1C")
            lw = 1.6 if c_idx in (0, 2, 4) else 1.1
            for spine in ax.spines.values():
                spine.set_edgecolor(edge_color)
                spine.set_linewidth(lw)

            if r_idx == 0:
                ax.set_title(col_titles[c_idx], fontsize=10.5, fontweight="bold", color="#0F172A", pad=9)

            ax.set_xlabel(sub_note, fontsize=8.8, color="#334155", labelpad=4)
            if c_idx == 0:
                ax.set_ylabel(
                    row_label,
                    fontsize=10.0,
                    fontweight="bold",
                    color="#143C6E",
                    labelpad=10,
                )

    # Top phase group banners matching the 5 columns
    fig.canvas.draw()
    group_specs = [
        (0, 0, "BEFORE", "#475569", "#F1F5F9"),
        (1, 2, "AFTER PREPROCESSING (Deterministic Eval & Train Base)", "#143C6E", "#EFF6FF"),
        (3, 4, "AFTER DATA AUGMENTATION (Train Only)", "#B91C1C", "#FEF2F2"),
    ]
    for c_start, c_end, label, text_color, bg_color in group_specs:
        pos0 = axes_grid[0, c_start].get_position()
        pos1 = axes_grid[0, c_end].get_position()
        x0, x1 = pos0.x0, pos1.x1
        y_banner = 0.925
        h_banner = 0.048
        rect = patches.FancyBboxPatch(
            (x0, y_banner),
            x1 - x0,
            h_banner,
            boxstyle="round,pad=0.004,rounding_size=0.008",
            transform=fig.transFigure,
            facecolor=bg_color,
            edgecolor=text_color,
            linewidth=1.3,
        )
        fig.add_artist(rect)
        fig.text(
            0.5 * (x0 + x1),
            y_banner + 0.5 * h_banner,
            label,
            ha="center",
            va="center",
            fontsize=10.0,
            fontweight="bold",
            color=text_color,
        )

    fig.savefig(OUTPUT_PATH_REPO, dpi=300)
    fig.savefig(OUTPUT_PATH_PAPER, dpi=300)
    plt.close(fig)
    print(f"[OK] Saved {OUTPUT_PATH_REPO} and {OUTPUT_PATH_PAPER}")


if __name__ == "__main__":
    main()
