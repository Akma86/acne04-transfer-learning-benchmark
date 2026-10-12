import os
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import shutil

PROJECT_ROOT = Path("d:/MAIN DATA/Documents/Semester 7/Propo")
OUTPUT_FIG_DIR = PROJECT_ROOT / "outputs" / "figures"
OUTPUT_FIG_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACT_DIR = Path("C:/Users/booma/.gemini/antigravity-ide/brain/72b5f847-e128-4552-b87c-30635b164978")

# Highly consistent, high-quality representative samples:
# (a) Normal/Clear: levle0_1.jpg
# (b) Mild: levle1_101.jpg
# (c) Moderate: levle2_100.jpg
# (d) Severe: levle3_101.jpg
selected_files = [
    ("acne0_1024", "levle0_1.jpg"),
    ("acne1_1024", "levle1_101.jpg"),
    ("acne2_1024", "levle2_100.jpg"),
    ("acne3_1024", "levle3_101.jpg"),
]

def create_collage(titles, output_filename, show_titles=True):
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.4), dpi=300)
    
    for idx, (ax, (folder, fname)) in enumerate(zip(axes, selected_files)):
        img_path = PROJECT_ROOT / "data" / "raw" / "acne_1024" / folder / fname
        img = Image.open(img_path)
        ax.imshow(img)
        ax.axis("off")
        
        if show_titles and titles:
            ax.set_title(titles[idx], fontsize=12, fontweight="bold", pad=8)
            
    plt.tight_layout()
    out_path = OUTPUT_FIG_DIR / output_filename
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved {out_path.name}")
    
    if ARTIFACT_DIR.exists():
        shutil.copy(out_path, ARTIFACT_DIR / output_filename)
        print(f"[OK] Copied to artifact: {output_filename}")

# 1. Versi Bahasa Indonesia lengkap dengan kriteria Hayashi
titles_id = [
    "(a) Level 0: Normal / Clear\n(0–5 Lesi)",
    "(b) Level 1: Ringan / Mild\n(6–20 Lesi)",
    "(c) Level 2: Sedang / Moderate\n(21–50 Lesi)",
    "(d) Level 3: Berat / Severe\n(>50 Lesi)",
]
create_collage(titles_id, "acne04_collage_1x4_id.png")

# 2. Versi Bahasa Inggris (Standar Paper / Benchmark)
titles_en = [
    "(a) Level 0: Normal / Clear\n(0–5 Lesions)",
    "(b) Level 1: Mild\n(6–20 Lesions)",
    "(c) Level 2: Moderate\n(21–50 Lesions)",
    "(d) Level 3: Severe\n(>50 Lesions)",
]
create_collage(titles_en, "acne04_collage_1x4_en.png")

# 3. Versi Simpel / Minimalis (Cocok untuk caption formal di naskah proposal Word/LaTeX)
titles_simple = [
    "(a) Level 0",
    "(b) Level 1",
    "(c) Level 2",
    "(d) Level 3",
]
create_collage(titles_simple, "acne04_collage_1x4_simple.png")

# 4. Versi Polos tanpa text di atas gambar (seamless stitch)
create_collage(None, "acne04_collage_1x4_plain.png", show_titles=False)

print("\nSemua variasi kolase berhasil digenerate di outputs/figures/!")
