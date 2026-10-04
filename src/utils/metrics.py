from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    precision_score,
    recall_score,
)


def compute_classification_metrics(
    y_true: List[int] | np.ndarray,
    y_pred: List[int] | np.ndarray,
) -> Dict[str, float]:
    """Computes comprehensive evaluation metrics for multi-class acne severity grading.

    Includes:
    - Overall Top-1 Accuracy
    - Macro-averaged Precision, Recall, and F1-Score (essential for imbalanced classes)
    - Mean Absolute Error (MAE): reflects ordinal distance error between grades (0, 1, 2, 3)
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "macro_precision": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }
    return metrics


def plot_confusion_matrix(
    y_true: List[int] | np.ndarray,
    y_pred: List[int] | np.ndarray,
    class_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
):
    """Plots and optionally saves a normalized confusion matrix heatmap."""
    if class_names is None:
        class_names = [
            "Level 0 (Normal)",
            "Level 1 (Mild)",
            "Level 2 (Moderate)",
            "Level 3 (Severe)",
        ]

    cm = confusion_matrix(y_true, y_pred, normalize="true")

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=False,
    )
    plt.title(
        "Normalized Confusion Matrix (Acne Severity)",
        fontweight="bold",
        pad=12,
    )
    plt.xlabel("Predicted Class")
    plt.ylabel("True Ground Truth")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
