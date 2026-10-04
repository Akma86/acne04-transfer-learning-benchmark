"""
Preprocessing and Pre-flight Pipeline for ACNE04 Severity Classification.

Implements the 7-step incremental preprocessing & ablation protocol:
1. Resize 224x224 + ImageNet Normalization
2. Data Augmentation (Horizontal Flip, Light Rotation, Color Jitter)
3. CLAHE (Contrast Limited Adaptive Histogram Equalization)
4. Color Constancy / White Balance (Gray World & Shades of Gray)
5. Face Detection & Cropping (OpenCV Haar Cascade with fallback)
6. Skin Segmentation & Masking (HSV + YCbCr color spaces)
7. Class Imbalance Handling (Class-weighted Loss, WeightedRandomSampler / ROS)
"""

import os
# Prevent OpenMP runtime collision (common on Windows with PyTorch + OpenCV in Jupyter)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import cv2
# Prevent OpenCV from spawning conflicting background thread pools with PyTorch
cv2.setNumThreads(0)

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import WeightedRandomSampler
import torchvision.transforms as T


# ==============================================================================
# 1. Base Transform: Resize & ImageNet Normalization
# ==============================================================================

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_base_transforms(image_size: int = 224) -> Tuple[T.Compose, T.Compose]:
    """Stage 1: Standard baseline resize + ImageNet normalization."""
    train_transform = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    eval_transform = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return train_transform, eval_transform


# ==============================================================================
# 2. Data Augmentation (Train only)
# ==============================================================================

class SafeRandomRotation:
    """Random rotation keeping image format consistent."""
    def __init__(self, degrees: float = 15.0):
        self.degrees = degrees

    def __call__(self, img: Image.Image) -> Image.Image:
        angle = np.random.uniform(-self.degrees, self.degrees)
        return img.rotate(angle, resample=Image.BILINEAR)


def get_augmented_transforms(image_size: int = 224) -> Tuple[T.Compose, T.Compose]:
    """Stage 2: Adds random horizontal flips, mild rotations, and color jitter."""
    train_transform = T.Compose([
        T.Resize((image_size, image_size)),
        T.RandomHorizontalFlip(p=0.5),
        SafeRandomRotation(degrees=15.0),
        T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.03),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    eval_transform = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return train_transform, eval_transform


# ==============================================================================
# 3. CLAHE (Contrast Limited Adaptive Histogram Equalization)
# ==============================================================================

class CLAHETransform:
    """Applies CLAHE on the Luminance (L) channel in LAB color space.

    Preserves dermatological color fidelity while enhancing lesion boundaries.
    """
    def __init__(self, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def __call__(self, img: Image.Image) -> Image.Image:
        np_img = np.ascontiguousarray(np.array(img))
        # Convert RGB to LAB
        lab = cv2.cvtColor(np_img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)

        # Apply CLAHE to L-channel
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
        cl = clahe.apply(l)

        # Merge and convert back to RGB
        merged_lab = cv2.merge((cl, a, b))
        enhanced_rgb = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2RGB)
        return Image.fromarray(enhanced_rgb)


def apply_clahe(img: Image.Image, clip_limit: float = 2.0) -> Image.Image:
    """Functional wrapper for CLAHE transformation."""
    return CLAHETransform(clip_limit=clip_limit)(img)


# ==============================================================================
# 4. Color Constancy / White Balance (Gray World & Shades of Gray)
# ==============================================================================

class ColorConstancyTransform:
    """Applies illumination normalization via Gray World or Shades of Gray algorithm."""
    def __init__(self, method: str = "gray_world", power_p: int = 6):
        self.method = method.lower()
        self.power_p = power_p

    def __call__(self, img: Image.Image) -> Image.Image:
        np_img = np.ascontiguousarray(np.array(img, dtype=np.float32))

        if self.method == "gray_world":
            # Gray World Assumption: average scene color is neutral gray
            mean_r = np.mean(np_img[:, :, 0]) + 1e-6
            mean_g = np.mean(np_img[:, :, 1]) + 1e-6
            mean_b = np.mean(np_img[:, :, 2]) + 1e-6
            mean_gray = (mean_r + mean_g + mean_b) / 3.0

            np_img[:, :, 0] = np.clip(np_img[:, :, 0] * (mean_gray / mean_r), 0, 255)
            np_img[:, :, 1] = np.clip(np_img[:, :, 1] * (mean_gray / mean_g), 0, 255)
            np_img[:, :, 2] = np.clip(np_img[:, :, 2] * (mean_gray / mean_b), 0, 255)

        elif self.method == "shades_of_gray":
            # Shades of Gray using Minkowski p-norm
            p = self.power_p
            e_r = (np.mean(np_img[:, :, 0] ** p)) ** (1.0 / p) + 1e-6
            e_g = (np.mean(np_img[:, :, 1] ** p)) ** (1.0 / p) + 1e-6
            e_b = (np.mean(np_img[:, :, 2] ** p)) ** (1.0 / p) + 1e-6
            norm = np.sqrt(e_r**2 + e_g**2 + e_b**2) + 1e-6

            # Scaled factor relative to equal-energy illuminant (1/sqrt(3))
            target = 1.0 / np.sqrt(3.0)
            np_img[:, :, 0] = np.clip(np_img[:, :, 0] * (target / (e_r / norm)), 0, 255)
            np_img[:, :, 1] = np.clip(np_img[:, :, 1] * (target / (e_g / norm)), 0, 255)
            np_img[:, :, 2] = np.clip(np_img[:, :, 2] * (target / (e_b / norm)), 0, 255)

        return Image.fromarray(np_img.astype(np.uint8))


def apply_color_constancy(img: Image.Image, method: str = "gray_world") -> Image.Image:
    """Functional wrapper for color constancy / white balance."""
    return ColorConstancyTransform(method=method)(img)


# ==============================================================================
# 5. Face Detection & Cropping (OpenCV Haar Cascade with Fallback)
# ==============================================================================

class FaceCropTransform:
    """Detects primary face bounding box with safety padding.

    Gracefully falls back to original image if no face is detected
    (crucial for close-up dermatological macro shots).
    """
    def __init__(self, margin: float = 0.15, min_size: Tuple[int, int] = (100, 100)):
        self.margin = margin
        self.min_size = min_size
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

    def __call__(self, img: Image.Image) -> Image.Image:
        np_img = np.array(img)
        gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)

        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=self.min_size
        )

        if len(faces) == 0:
            # Fallback: image might already be a tightly framed cheek/chin lesion crop
            return img

        # Pick the largest detected face box
        x, y, w, h = max(faces, key=lambda b: b[2] * b[3])

        # Add margin padding
        pad_w = int(w * self.margin)
        pad_h = int(h * self.margin)

        img_h, img_w = np_img.shape[:2]
        x1 = max(0, x - pad_w)
        y1 = max(0, y - pad_h)
        x2 = min(img_w, x + w + pad_w)
        y2 = min(img_h, y + h + pad_h)

        cropped = np.ascontiguousarray(np_img[y1:y2, x1:x2])
        return Image.fromarray(cropped)


def crop_face(img: Image.Image, margin: float = 0.15) -> Image.Image:
    """Functional wrapper for face detection and cropping."""
    return FaceCropTransform(margin=margin)(img)


# ==============================================================================
# 6. Skin Segmentation & Masking (HSV + YCbCr)
# ==============================================================================

class SkinMaskTransform:
    """Segments skin regions using combined HSV + YCbCr color thresholds.

    Attenuates hair, background, and medical bibs to highlight epidermal acne lesions.
    """
    def __init__(self, blend_alpha: float = 0.15):
        self.blend_alpha = blend_alpha  # 0.0 = black background, 0.15 = dimmed background

    def __call__(self, img: Image.Image) -> Image.Image:
        np_img = np.ascontiguousarray(np.array(img))

        # 1. HSV Skin Thresholding
        hsv = cv2.cvtColor(np_img, cv2.COLOR_RGB2HSV)
        lower_hsv = np.array([0, 30, 60], dtype=np.uint8)
        upper_hsv = np.array([25, 255, 255], dtype=np.uint8)
        mask_hsv = cv2.inRange(hsv, lower_hsv, upper_hsv)

        # 2. YCbCr Skin Thresholding
        ycbcr = cv2.cvtColor(np_img, cv2.COLOR_RGB2YCrCb)
        lower_ycb = np.array([80, 135, 85], dtype=np.uint8)   # Y, Cr, Cb
        upper_ycb = np.array([255, 180, 135], dtype=np.uint8)
        mask_ycb = cv2.inRange(ycbcr, lower_ycb, upper_ycb)

        # 3. Fuse masks
        combined_mask = cv2.bitwise_or(mask_hsv, mask_ycb)

        # 4. Morphological clean-up (remove salt-and-pepper noise)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        cleaned_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_OPEN, kernel, iterations=1)

        # 5. Apply mask with optional soft alpha blending for context
        mask_3d = (cleaned_mask[:, :, None] / 255.0).astype(np.float32)
        background = np_img.astype(np.float32) * self.blend_alpha
        foreground = np_img.astype(np.float32) * mask_3d
        blended = np.clip(foreground + background * (1.0 - mask_3d), 0, 255).astype(np.uint8)

        return Image.fromarray(blended)


def apply_skin_mask(img: Image.Image, blend_alpha: float = 0.15) -> Image.Image:
    """Functional wrapper for skin segmentation masking."""
    return SkinMaskTransform(blend_alpha=blend_alpha)(img)


# ==============================================================================
# 6.1 Deep Background Removal & Face Isolation (rembg + Auto-Crop)
# ==============================================================================

class RemoveBackgroundTransform:
    """Removes non-facial backgrounds using deep salient segmentation (U2Net/rembg).

    Features:
    - Eliminates walls, medical bibs, backdrop clutter, and second persons in the background.
    - Robust macro fallback: if the input is an extreme close-up of a cheek (foreground ratio < 20%),
      it gracefully falls back to skin thresholding so acne lesions are never erased.
    - Optional auto-cropping to the non-zero face bounding box to maximize facial lesion focus.
    """
    def __init__(
        self,
        bg_color: Union[str, Tuple[int, int, int]] = "black",
        crop_to_face: bool = True,
        padding: float = 0.05,
        session: Any = None,
    ):
        self.crop_to_face = crop_to_face
        self.padding = padding
        self.session = session

        if isinstance(bg_color, str):
            if bg_color.lower() == "black":
                self.bg_color = (0, 0, 0)
            elif bg_color.lower() == "white":
                self.bg_color = (255, 255, 255)
            elif bg_color.lower() == "gray":
                self.bg_color = (128, 128, 128)
            else:
                self.bg_color = (0, 0, 0)
        else:
            self.bg_color = bg_color

    def __call__(self, img: Image.Image) -> Image.Image:
        try:
            import rembg
        except ImportError:
            # Fallback to skin mask if rembg is unavailable
            return SkinMaskTransform(blend_alpha=0.0)(img)

        np_img = np.ascontiguousarray(np.array(img))

        # Lazy init persistent session if not passed
        if self.session is None:
            self.session = rembg.new_session("u2net")

        # 1. Run salient foreground segmentation with cached session
        rgba = rembg.remove(img, session=self.session)
        alpha = np.array(rgba)[:, :, 3]
        fg_ratio = np.count_nonzero(alpha > 15) / alpha.size

        # 2. Safety fallback for extreme macro close-ups (e.g. tight cheek shots)
        if fg_ratio < 0.20:
            hsv = cv2.cvtColor(np_img, cv2.COLOR_RGB2HSV)
            mask_hsv = cv2.inRange(hsv, np.array([0, 30, 60]), np.array([25, 255, 255]))
            ycbcr = cv2.cvtColor(np_img, cv2.COLOR_RGB2YCrCb)
            mask_ycb = cv2.inRange(ycbcr, np.array([80, 135, 85]), np.array([255, 180, 135]))
            skin_mask = cv2.bitwise_or(mask_hsv, mask_ycb)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
            skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
            skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel, iterations=2)
            alpha = skin_mask

        # 3. Optional framing crop to face bounding box
        if self.crop_to_face:
            rows = np.any(alpha > 15, axis=1)
            cols = np.any(alpha > 15, axis=0)
            if np.any(rows) and np.any(cols):
                ymin, ymax = np.where(rows)[0][[0, -1]]
                xmin, xmax = np.where(cols)[0][[0, -1]]
                h, w = alpha.shape
                pad_h = int((ymax - ymin) * self.padding)
                pad_w = int((xmax - xmin) * self.padding)
                ymin = max(0, ymin - pad_h)
                ymax = min(h, ymax + pad_h)
                xmin = max(0, xmin - pad_w)
                xmax = min(w, xmax + pad_w)
                np_img = np_img[ymin:ymax, xmin:xmax]
                alpha = alpha[ymin:ymax, xmin:xmax]

        # 4. Composite onto background color
        mask_3d = (alpha[:, :, None] / 255.0).astype(np.float32)
        bg_arr = np.full_like(np_img, self.bg_color, dtype=np.float32)
        composite = np.clip(
            np_img.astype(np.float32) * mask_3d + bg_arr * (1.0 - mask_3d),
            0,
            255
        ).astype(np.uint8)
        return Image.fromarray(composite)


def apply_remove_bg(
    img: Image.Image,
    bg_color: str = "black",
    crop_to_face: bool = True,
    session: Any = None,
) -> Image.Image:
    """Functional wrapper for deep background removal and face isolation."""
    return RemoveBackgroundTransform(bg_color=bg_color, crop_to_face=crop_to_face, session=session)(img)


# ==============================================================================
# 7. Class Imbalance Handling (Loss Weights & Sampler)
# ==============================================================================

def compute_class_weights(
    labels: Union[List[int], np.ndarray, pd.Series],
    num_classes: int = 4,
    method: str = "balanced"
) -> torch.Tensor:
    """Computes cost-sensitive class weights for CrossEntropyLoss.

    Args:
        labels: Array-like of integer target labels (0 to num_classes - 1).
        num_classes: Total number of severity categories (4 for ACNE04).
        method: 'balanced' (standard inverse frequency) or 'effective_num' (Cui et al. CVPR 2019).
    """
    labels_arr = np.array(labels)
    class_counts = np.bincount(labels_arr, minlength=num_classes)
    total_samples = len(labels_arr)

    if method == "balanced":
        # w_c = N / (K * N_c)
        weights = total_samples / (num_classes * np.maximum(class_counts, 1).astype(np.float32))
    elif method == "effective_num":
        # Effective Number of Samples (beta = 0.999)
        beta = 0.999
        effective_num = 1.0 - np.power(beta, np.maximum(class_counts, 1))
        weights = (1.0 - beta) / np.array(effective_num, dtype=np.float32)
        weights = weights / np.sum(weights) * num_classes
    else:
        weights = np.ones(num_classes, dtype=np.float32)

    return torch.tensor(weights, dtype=torch.float32)


def get_weighted_sampler(
    labels: Union[List[int], np.ndarray, pd.Series],
    num_classes: int = 4
) -> WeightedRandomSampler:
    """Constructs a WeightedRandomSampler (Random Oversampling / ROS) for DataLoader.

    Ensures minority classes (e.g. Level 3 severe) are drawn uniformly per mini-batch.
    """
    labels_arr = np.array(labels)
    class_counts = np.bincount(labels_arr, minlength=num_classes)
    class_weights = 1.0 / np.maximum(class_counts, 1).astype(np.float32)

    sample_weights = np.array([class_weights[y] for y in labels_arr])
    sampler = WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True
    )
    return sampler


# ==============================================================================
# Master Preprocessing Pipeline Builder (Ablation Stages 1 to 7)
# ==============================================================================

STAGE_DESCRIPTIONS = {
    1: "Stage 1 (Raw Baseline)     : Resize 224x224 + ImageNet Normalization",
    2: "Stage 2 (+ Augmentation)   : Stage 1 + Random Horizontal Flip, Rotation (15 deg), Color Jitter",
    3: "Stage 3 (+ CLAHE)          : Stage 2 + Contrast Limited Adaptive Histogram Equalization",
    4: "Stage 4 (+ Color Constancy): Stage 3 + Gray World White Balance Normalization",
    5: "Stage 5 (+ Face Crop)      : Stage 4 + Face Detection & Cropping (OpenCV Haar Cascade)",
    6: "Stage 6 (+ Skin Masking)   : Stage 5 + Skin Segmentation (HSV & YCbCr Color Thresholds)",
    7: "Stage 7 (+ Imbalance Sol)  : Stage 6 + Cost-Sensitive Class Weights / Weighted Sampler (ROS)",
}


def build_pipeline_by_stage(
    stage: int = 1,
    image_size: int = 224,
    is_train: bool = True,
    clahe_clip: float = 2.0,
    wb_method: str = "gray_world",
    face_margin: float = 0.15,
    skin_blend: float = 0.15,
    remove_bg: bool = False,
    nobg_color: str = "black",
    nobg_crop_to_face: bool = True,
) -> T.Compose:
    """Builds the cumulative transformation pipeline up to the specified stage (1 through 7).

    Args:
        stage: Integer 1 to 7 corresponding to the incremental ablation study.
        image_size: Target square image dimension (e.g., 224).
        is_train: Whether this pipeline is for training (True) or validation/testing (False).
        clahe_clip: Clip limit for CLAHE.
        wb_method: 'gray_world' or 'shades_of_gray'.
        face_margin: Bounding box padding for Haar face crop.
        skin_blend: Background dimming blend factor for skin mask.
        remove_bg: If True, applies deep salient background removal & face focus.
        nobg_color: Fill color for removed background ('black', 'white', or 'gray').
        nobg_crop_to_face: If True, crops to the foreground face bounding box.
    """
    if stage < 1 or stage > 7:
        raise ValueError(f"Stage must be an integer between 1 and 7 (received {stage}).")

    pre_resize_transforms: List[Callable] = []

    # Optional deep background removal & face isolation
    if remove_bg:
        pre_resize_transforms.append(
            RemoveBackgroundTransform(bg_color=nobg_color, crop_to_face=nobg_crop_to_face)
        )

    # Stages 5 & 6 happen on spatial resolution before final model resize
    if stage >= 5 and not (remove_bg and nobg_crop_to_face):
        pre_resize_transforms.append(FaceCropTransform(margin=face_margin))

    if stage >= 6:
        pre_resize_transforms.append(SkinMaskTransform(blend_alpha=skin_blend))

    # Stage 4: Color Constancy
    if stage >= 4:
        pre_resize_transforms.append(ColorConstancyTransform(method=wb_method))

    # Stage 3: CLAHE
    if stage >= 3:
        pre_resize_transforms.append(CLAHETransform(clip_limit=clahe_clip))

    # Core spatial resizing
    spatial_transforms = [T.Resize((image_size, image_size))]

    # Stage 2: Data Augmentation (Train split only)
    if stage >= 2 and is_train:
        spatial_transforms.extend([
            T.RandomHorizontalFlip(p=0.5),
            SafeRandomRotation(degrees=15.0),
            T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.03),
        ])

    # Final normalization
    final_transforms = [
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]

    all_transforms = pre_resize_transforms + spatial_transforms + final_transforms
    return T.Compose(all_transforms)
