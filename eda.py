# %% [markdown]
# # Exploratory Data Analysis (EDA) on ACNE04 Dataset
# **Project:** Acne Severity Classification Benchmark  
# **Author:** Akmal Yaasir Fauzaan  
# 
# *Tip:* Click **"Run Cell"** or press **Shift + Enter** on any cell below to execute directly inside the IDE Interactive Window!

# %%
import os
import re
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image

# IDE plotting configurations
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (12, 5)

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "acne_1024"
OUTPUT_FIG_DIR = PROJECT_ROOT / "outputs" / "figures"
METADATA_DIR = PROJECT_ROOT / "data" / "metadata"

OUTPUT_FIG_DIR.mkdir(parents=True, exist_ok=True)
METADATA_DIR.mkdir(parents=True, exist_ok=True)

print(f"[OK] Root Directory     : {PROJECT_ROOT}")
print(f"[OK] Raw Data Directory : {RAW_DATA_DIR}")
print(f"[OK] Data Exists        : {RAW_DATA_DIR.exists()}")

# %% [markdown]
# ## 1. Scan Dataset Folders & Extract Image Metadata

# %%
CLASS_CONFIG = {
    "acne0_1024": {"label": 0, "tier": "Level 0 (Normal / Clear)", "hayashi": "0-5 Lesions"},
    "acne1_1024": {"label": 1, "tier": "Level 1 (Mild)", "hayashi": "6-20 Lesions"},
    "acne2_1024": {"label": 2, "tier": "Level 2 (Moderate)", "hayashi": "21-50 Lesions"},
    "acne3_1024": {"label": 3, "tier": "Level 3 (Severe)", "hayashi": ">50 Lesions"}
}

records = []
print("Scanning ACNE04 dataset images...")

for folder_name, cfg in CLASS_CONFIG.items():
    folder_path = RAW_DATA_DIR / folder_name
    if not folder_path.exists():
        continue

    for img_path in folder_path.glob("*.jpg"):
        size_kb = round(os.path.getsize(img_path) / 1024, 2)
        records.append({
            "filepath": str(img_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "filename": img_path.name,
            "label": cfg["label"],
            "severity": cfg["tier"],
            "hayashi": cfg["hayashi"],
            "size_kb": size_kb
        })

df = pd.DataFrame(records)
print(f"Total Images Indexed: {len(df)}")
df.head(10)

# %% [markdown]
# ## 2. Class Distribution & Imbalance Ratio

# %%
class_counts = df["severity"].value_counts().loc[[
    "Level 0 (Normal / Clear)",
    "Level 1 (Mild)",
    "Level 2 (Moderate)",
    "Level 3 (Severe)"
]]

colors = ["#2ecc71", "#3498db", "#f39c12", "#e74c3c"]

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Bar Chart
sns.barplot(x=class_counts.index, y=class_counts.values, ax=axes[0], palette=colors)
axes[0].set_title("ACNE04 Class Frequency Distribution", fontweight="bold", pad=10)
axes[0].set_ylabel("Number of Samples")
axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=15)

for p in axes[0].patches:
    axes[0].annotate(
        f"{int(p.get_height())} ({p.get_height()/len(df)*100:.1f}%)",
        (p.get_x() + p.get_width() / 2., p.get_height() + 10),
        ha='center', va='bottom', fontweight='bold', fontsize=10
    )

# Pie Chart
axes[1].pie(
    class_counts.values,
    labels=class_counts.index,
    autopct='%1.1f%%',
    colors=colors,
    startangle=140,
    explode=[0.02, 0.02, 0.05, 0.08]
)
axes[1].set_title("Proportional Class Breakdown", fontweight="bold", pad=10)

plt.tight_layout()
fig_path = OUTPUT_FIG_DIR / "class_distribution.png"
plt.savefig(fig_path, dpi=300)
plt.show()

imbalance_ratio = class_counts.max() / class_counts.min()
print(f"--> Saved chart to: {fig_path}")
print(f"--> Class Imbalance Ratio (Majority / Minority): {imbalance_ratio:.2f} : 1.0")

# %% [markdown]
# ## 3. File Size (Compression Complexity) vs Severity

# %%
plt.figure(figsize=(10, 5))
sns.boxplot(data=df, x="severity", y="size_kb", palette=colors)
plt.title("JPEG File Size Distribution Across Acne Severity Tiers", fontweight="bold")
plt.ylabel("File Size (Kilobytes)")
plt.xlabel("Severity Tier")
plt.xticks(rotation=15)
plt.tight_layout()

file_size_fig = OUTPUT_FIG_DIR / "file_size_analysis.png"
plt.savefig(file_size_fig, dpi=300)
plt.show()

# %% [markdown]
# ## 4. Qualitative Image Gallery (Visual Comparison)

# %%
samples_per_class = 3
fig, axes = plt.subplots(4, samples_per_class, figsize=(12, 14))

for row_idx, label in enumerate([0, 1, 2, 3]):
    subset = df[df["label"] == label]
    sample_df = subset.sample(n=min(samples_per_class, len(subset)), random_state=42)

    for col_idx, (_, row) in enumerate(sample_df.iterrows()):
        img = Image.open(PROJECT_ROOT / row["filepath"])
        axes[row_idx, col_idx].imshow(img)
        axes[row_idx, col_idx].set_title(f"{row['severity']}\n{row['filename']}", fontsize=9)
        axes[row_idx, col_idx].axis("off")

plt.suptitle("ACNE04 Qualitative Morphology Across Severity Levels", fontsize=14, fontweight="bold", y=0.99)
plt.tight_layout()
gallery_fig = OUTPUT_FIG_DIR / "sample_gallery.png"
plt.savefig(gallery_fig, dpi=300)
plt.show()

print(f"--> Qualitative visual gallery saved to: {gallery_fig}")

# %% [markdown]
# ## 5. Export Structured Metadata with Train-Val-Test Splits

# %%
from sklearn.model_selection import train_test_split

train_df, temp_df = train_test_split(df, test_size=0.30, stratify=df["label"], random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.50, stratify=temp_df["label"], random_state=42)

df["split"] = "train"
df.loc[val_df.index, "split"] = "val"
df.loc[test_df.index, "split"] = "test"

output_csv = METADATA_DIR / "acne04_metadata.csv"
df.to_csv(output_csv, index=False)
print(f"--> Generated metadata with stratified splits: {output_csv}")
display(pd.crosstab(df["severity"], df["split"], margins=True))

print("\n=== EDA Pipeline Successfully Completed! ===")
