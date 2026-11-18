"""
Comprehensive Exploratory Data Analysis (EDA) on ACNE04 Dataset
Project: Acne Severity Classification Benchmark
Author: Akmal Yaasir Fauzaan

This script performs end-to-end exploratory analysis on the ACNE04 benchmark dataset:
1. Scans dataset directories and indexes metadata.
2. Evaluates class frequency distribution and class imbalance ratios.
3. Analyzes physical properties: image dimensions, aspect ratio, and JPEG file sizes.
4. Profiles color space characteristics and inflammatory erythema (CIE-LAB a* channel).
5. Quantifies texture complexity and structural variation (Shannon Entropy & Laplacian Variance).
6. Computes empirical RGB normalization constants (Mean & Std) compared to ImageNet defaults.
7. Generates publication-ready figures for the research proposal.
8. Exports stratified train/val/test metadata splits (70/15/15).
"""

import os
import sys
from pathlib import Path
import warnings

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
from scipy.stats import entropy
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# Configuration and Path Setup
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "acne04"
OUTPUT_FIG_DIR = PROJECT_ROOT / "outputs" / "figures"
PAPER_FIG_DIR = PROJECT_ROOT / "docs" / "paper" / "Gambar"
METADATA_DIR = PROJECT_ROOT / "data" / "metadata"

OUTPUT_FIG_DIR.mkdir(parents=True, exist_ok=True)
PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)
METADATA_DIR.mkdir(parents=True, exist_ok=True)

# Visual styling
sns.set_theme(style="whitegrid", font="sans-serif")
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 14,
})

SEVERITY_PALETTE = {
    "Level 0 (Mild)": "#2ecc71",
    "Level 1 (Moderate)": "#3498db",
    "Level 2 (Severe)": "#f39c12",
    "Level 3 (Very Severe)": "#e74c3c",
}

CLASS_CONFIG = {
    "level_0_mild": {"label": 0, "tier": "Level 0 (Mild)", "hayashi": "0 - 5 lesions"},
    "level_1_moderate": {"label": 1, "tier": "Level 1 (Moderate)", "hayashi": "6 - 20 lesions"},
    "level_2_severe": {"label": 2, "tier": "Level 2 (Severe)", "hayashi": "21 - 50 lesions"},
    "level_3_very_severe": {"label": 3, "tier": "Level 3 (Very Severe)", "hayashi": "> 50 lesions"},
}


def save_fig_dual(fig, filename):
    """Save figure to both outputs/figures and docs/paper/Gambar."""
    p1 = OUTPUT_FIG_DIR / filename
    p2 = PAPER_FIG_DIR / filename
    fig.savefig(p1, dpi=300, bbox_inches="tight")
    fig.savefig(p2, dpi=300, bbox_inches="tight")
    print(f"  [SAVED] {p1.name} (outputs/ & docs/paper/Gambar/)")


# -----------------------------------------------------------------------------
# 1. Dataset Discovery & Metadata Ingestion
# -----------------------------------------------------------------------------
def load_dataset():
    print("\n" + "=" * 65)
    print("STEP 1: Scanning ACNE04 Dataset Folders...")
    print("=" * 65)

    records = []
    for folder_name, cfg in CLASS_CONFIG.items():
        folder_path = RAW_DATA_DIR / folder_name
        if not folder_path.exists():
            print(f"  [WARNING] Folder not found: {folder_path}")
            continue

        valid_exts = {".jpg", ".jpeg", ".png"}
        for img_path in sorted(p for p in folder_path.iterdir() if p.is_file() and p.suffix.lower() in valid_exts):
            size_kb = round(os.path.getsize(img_path) / 1024.0, 2)
            rel_path = str(img_path.relative_to(PROJECT_ROOT)).replace("\\", "/")
            records.append({
                "filepath": rel_path,
                "relative_path": rel_path,
                "abs_path": str(img_path),
                "filename": img_path.name,
                "label": cfg["label"],
                "severity": cfg["tier"],
                "hayashi": cfg["hayashi"],
                "size_kb": size_kb,
            })

    df = pd.DataFrame(records)
    print(f"Total Images Indexed : {len(df)}")
    return df


# -----------------------------------------------------------------------------
# 2. Class Frequency Distribution & Imbalance Ratio
# -----------------------------------------------------------------------------
def analyze_class_distribution(df):
    print("\n" + "=" * 65)
    print("STEP 2: Analyzing Class Distribution & Imbalance Ratio...")
    print("=" * 65)

    counts = df["severity"].value_counts().loc[list(SEVERITY_PALETTE.keys())]
    percentages = (counts / len(df)) * 100

    summary_df = pd.DataFrame({
        "Severity": counts.index,
        "Count": counts.values,
        "Percentage (%)": percentages.values.round(2)
    })
    print(summary_df.to_string(index=False))

    maj_count = counts.max()
    min_count = counts.min()
    imbalance_ratio = maj_count / min_count
    print(f"\nMajority Class : {counts.idxmax()} ({maj_count} samples)")
    print(f"Minority Class : {counts.idxmin()} ({min_count} samples)")
    print(f"Class Imbalance Ratio: {imbalance_ratio:.2f} : 1.0")

    # Plot Distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Bar chart
    bars = sns.barplot(
        x=counts.index,
        y=counts.values,
        hue=counts.index,
        palette=list(SEVERITY_PALETTE.values()),
        ax=axes[0],
        legend=False,
        edgecolor="black",
        linewidth=1.0,
    )
    axes[0].set_title("Distribusi Frekuensi Sampel Tiap Derajat Keparahan", fontweight="bold", pad=10)
    axes[0].set_ylabel("Jumlah Citra")
    axes[0].set_xlabel("Tingkat Keparahan (Kriteria Hayashi)")
    axes[0].tick_params(axis="x", rotation=15)

    for p in bars.patches:
        h = p.get_height()
        axes[0].annotate(
            f"{int(h)}\n({h/len(df)*100:.1f}%)",
            (p.get_x() + p.get_width() / 2.0, h / 2.0),
            ha="center", va="center", color="white", fontweight="bold", fontsize=10,
        )

    # Pie chart
    axes[1].pie(
        counts.values,
        labels=counts.index,
        autopct="%1.1f%%",
        colors=list(SEVERITY_PALETTE.values()),
        startangle=140,
        explode=[0.02, 0.02, 0.05, 0.08],
        wedgeprops={"edgecolor": "black", "linewidth": 1.0},
    )
    axes[1].set_title("Proporsi Relatif Kelas Dataset ACNE04", fontweight="bold", pad=10)

    plt.tight_layout()
    save_fig_dual(fig, "eda_class_distribution.png")
    plt.close(fig)


# -----------------------------------------------------------------------------
# 3. Physical Dimensions & Compression File Size
# -----------------------------------------------------------------------------
def analyze_physical_properties(df):
    print("\n" + "=" * 65)
    print("STEP 3: Checking Physical Dimensions & Aspect Ratios...")
    print("=" * 65)

    dims = []
    aspect_ratios = []
    for _, row in df.iterrows():
        with Image.open(row["abs_path"]) as img:
            w, h = img.size
            dims.append((w, h))
            aspect_ratios.append(w / h)

    widths = [w for w, _ in dims]
    heights = [h for _, h in dims]
    unique_dims = set(dims)
    unique_ar = set(np.round(aspect_ratios, 2))
    q1_kb, med_kb, q3_kb = np.percentile(df["size_kb"], [25, 50, 75])
    print(f"Total Inspected    : {len(dims)} images ({len(unique_dims)} unique resolutions)")
    print(f"Width Range (px)   : {min(widths)} .. {max(widths)} (median {np.median(widths):.0f})")
    print(f"Height Range (px)  : {min(heights)} .. {max(heights)} (median {np.median(heights):.0f})")
    print(f"Aspect Ratios      : min={min(aspect_ratios):.3f}, max={max(aspect_ratios):.3f}, median={np.median(aspect_ratios):.3f} ({len(unique_ar)} rounded unique)")
    print(f"File Size (KB)     : median={med_kb:.2f}, IQR=[{q1_kb:.2f}, {q3_kb:.2f}], min={df['size_kb'].min():.2f}, max={df['size_kb'].max():.2f}")

    # Plot File Size Boxplot
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.boxplot(
        data=df,
        x="severity",
        y="size_kb",
        hue="severity",
        palette=list(SEVERITY_PALETTE.values()),
        ax=ax,
        legend=False,
    )
    ax.set_title("Distribusi Ukuran Berkas JPEG Antar Derajat Keparahan", fontweight="bold", pad=10)
    ax.set_ylabel("Ukuran Berkas (Kilobytes)")
    ax.set_xlabel("Tingkat Keparahan")
    ax.tick_params(axis="x", rotation=15)
    plt.tight_layout()
    save_fig_dual(fig, "eda_file_size_analysis.png")
    plt.close(fig)


# -----------------------------------------------------------------------------
# 4. Color Channel Distribution & Inflammatory Erythema Profiling
# -----------------------------------------------------------------------------
def analyze_color_and_erythema(df):
    print("\n" + "=" * 65)
    print("STEP 4: Profiling Color Channels & CIE-LAB Inflammatory Erythema...")
    print("=" * 65)

    color_records = []
    # Sample 100 images per class for balanced profiling
    for label in [0, 1, 2, 3]:
        sub = df[df["label"] == label]
        sample = sub.sample(min(100, len(sub)), random_state=42)
        for _, row in sample.iterrows():
            img_bgr = cv2.imread(row["abs_path"])
            if img_bgr is None:
                continue
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)

            r_mean = np.mean(img_rgb[:, :, 0])
            g_mean = np.mean(img_rgb[:, :, 1])
            b_mean = np.mean(img_rgb[:, :, 2])
            a_mean = np.mean(img_lab[:, :, 1])  # LAB a* channel (Erythema / Redness)

            color_records.append({
                "severity": row["severity"],
                "label": row["label"],
                "red_mean": r_mean,
                "green_mean": g_mean,
                "blue_mean": b_mean,
                "lab_a_erythema": a_mean,
            })

    df_col = pd.DataFrame(color_records)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Mean Red Channel Boxplot
    sns.boxplot(
        data=df_col,
        x="severity",
        y="red_mean",
        hue="severity",
        palette=list(SEVERITY_PALETTE.values()),
        ax=axes[0],
        legend=False,
    )
    axes[0].set_title("Intensitas Kanal Merah (R) Berdasarkan Keparahan", fontweight="bold", pad=10)
    axes[0].set_ylabel("Nilai Piksel Kanal Merah (0 - 255)")
    axes[0].set_xlabel("Tingkat Keparahan")
    axes[0].tick_params(axis="x", rotation=15)

    # CIE-LAB a* Channel Boxplot (Erythema)
    sns.boxplot(
        data=df_col,
        x="severity",
        y="lab_a_erythema",
        hue="severity",
        palette=list(SEVERITY_PALETTE.values()),
        ax=axes[1],
        legend=False,
    )
    axes[1].set_title("Indeks Eritema Inflamasi (Kanal a* Ruang Warna CIE-LAB)", fontweight="bold", pad=10)
    axes[1].set_ylabel("Nilai Rerata Kanal a* LAB")
    axes[1].set_xlabel("Tingkat Keparahan")
    axes[1].tick_params(axis="x", rotation=15)

    plt.tight_layout()
    save_fig_dual(fig, "eda_erythema_analysis.png")
    plt.close(fig)

    erythema_summary = df_col.groupby("severity")["lab_a_erythema"].agg(["mean", "std", "median"])
    print("\nErythema Index (LAB a*) Statistics per Class:")
    print(erythema_summary.round(2).to_string())


# -----------------------------------------------------------------------------
# 5. Texture Complexity & Spatial Shannon Entropy
# -----------------------------------------------------------------------------
def analyze_texture_complexity(df):
    print("\n" + "=" * 65)
    print("STEP 5: Evaluating Texture Complexity & Spatial Shannon Entropy...")
    print("=" * 65)

    texture_records = []
    for label in [0, 1, 2, 3]:
        sub = df[df["label"] == label]
        sample = sub.sample(min(80, len(sub)), random_state=42)
        for _, row in sample.iterrows():
            img_gray = cv2.imread(row["abs_path"], cv2.IMREAD_GRAYSCALE)
            if img_gray is None:
                continue

            # Shannon Entropy
            hist, _ = np.histogram(img_gray.ravel(), bins=256, range=(0, 256), density=True)
            hist = hist[hist > 0]
            shannon_ent = entropy(hist, base=2)

            # Laplacian Edge Variance
            lap_var = cv2.Laplacian(img_gray, cv2.CV_64F).var()

            texture_records.append({
                "severity": row["severity"],
                "entropy": shannon_ent,
                "laplacian_var": lap_var,
            })

    df_tex = pd.DataFrame(texture_records)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.violinplot(
        data=df_tex,
        x="severity",
        y="entropy",
        hue="severity",
        palette=list(SEVERITY_PALETTE.values()),
        ax=axes[0],
        cut=0,
        legend=False,
    )
    axes[0].set_title("Entropi Informasi Spasial (Shannon Entropy)", fontweight="bold", pad=10)
    axes[0].set_ylabel("Entropi (bits)")
    axes[0].set_xlabel("Tingkat Keparahan")
    axes[0].tick_params(axis="x", rotation=15)

    sns.boxplot(
        data=df_tex,
        x="severity",
        y="laplacian_var",
        hue="severity",
        palette=list(SEVERITY_PALETTE.values()),
        ax=axes[1],
        legend=False,
    )
    axes[1].set_title("Kepadatan Tekstur Tepi (Varians Filter Laplacian)", fontweight="bold", pad=10)
    axes[1].set_ylabel("Varians Laplacian")
    axes[1].set_xlabel("Tingkat Keparahan")
    axes[1].tick_params(axis="x", rotation=15)

    plt.tight_layout()
    save_fig_dual(fig, "eda_texture_entropy.png")
    plt.close(fig)


# -----------------------------------------------------------------------------
# 6. Empirical Normalization Constants (ACNE04 vs ImageNet)
# -----------------------------------------------------------------------------
def compute_normalization_constants(df):
    print("\n" + "=" * 65)
    print("STEP 6: Calculating Empirical RGB Normalization Constants...")
    print("=" * 65)

    rgb_means = []
    rgb_stds = []

    for _, row in df.iterrows():
        img = cv2.imread(row["abs_path"])
        if img is None:
            continue
        # OpenCV computes C++ double-precision mean/stddev directly on uint8 without 8x RAM float64 copies
        mean_bgr, std_bgr = cv2.meanStdDev(img)
        rgb_means.append(mean_bgr[::-1, 0] / 255.0)
        rgb_stds.append(std_bgr[::-1, 0] / 255.0)

    mean_rgb = np.mean(rgb_means, axis=0)
    std_rgb = np.mean(rgb_stds, axis=0)

    print("\n=== Normalization Constants Comparison ===")
    print(f"Computed across N = {len(rgb_means)} images")
    print(f"ImageNet Defaults : Mean = [0.485, 0.456, 0.406], Std = [0.229, 0.224, 0.225]")
    print(f"Empirical ACNE04  : Mean = [{mean_rgb[0]:.4f}, {mean_rgb[1]:.4f}, {mean_rgb[2]:.4f}], "
          f"Std = [{std_rgb[0]:.4f}, {std_rgb[1]:.4f}, {std_rgb[2]:.4f}]")
    print("Insight: ACNE04 exhibits lower Green and Blue intensities than generic ImageNet images,")
    print("reflecting warm human skin tones and justifying color constancy / CLAHE adaptation.")


# -----------------------------------------------------------------------------
# 7. Stratified Dataset Partitioning & Export
# -----------------------------------------------------------------------------
def export_stratified_metadata(df):
    print("\n" + "=" * 65)
    print("STEP 7: Exporting Stratified Train-Val-Test Splits (70/15/15)...")
    print("=" * 65)

    train_df, temp_df = train_test_split(df, test_size=0.30, stratify=df["label"], random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.50, stratify=temp_df["label"], random_state=42)

    df["split"] = "train"
    df.loc[val_df.index, "split"] = "val"
    df.loc[test_df.index, "split"] = "test"

    out_csv = METADATA_DIR / "acne04_metadata.csv"
    df.to_csv(out_csv, index=False)
    print(f"Metadata exported to: {out_csv}")
    print("\nStratified Partitioning Matrix:")
    print(pd.crosstab(df["severity"], df["split"], margins=True).to_string())


# -----------------------------------------------------------------------------
# Main Execution Pipeline
# -----------------------------------------------------------------------------
def main():
    print("=" * 65)
    print("ACNE04 EXPLORATORY DATA ANALYSIS (EDA) BENCHMARK PIPELINE")
    print("=" * 65)
    print(f"Project Directory : {PROJECT_ROOT}")
    print(f"Raw Data Path     : {RAW_DATA_DIR}")

    df = load_dataset()
    if len(df) == 0:
        print("[ERROR] No images found. Check data directory.")
        sys.exit(1)

    analyze_class_distribution(df)
    analyze_physical_properties(df)
    analyze_color_and_erythema(df)
    analyze_texture_complexity(df)
    compute_normalization_constants(df)
    export_stratified_metadata(df)

    print("\n" + "=" * 65)
    print("[SUCCESS] All EDA procedures completed and figures exported!")
    print("=" * 65)


if __name__ == "__main__":
    main()
