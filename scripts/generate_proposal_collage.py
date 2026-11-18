"""Generate standardized 1x4 ACNE04 severity collage figures from Layered-Labs/ACNE04."""

from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image, ImageOps

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "acne04"
OUTPUT_FIG_DIR = PROJECT_ROOT / "outputs" / "figures"
PAPER_FIG_DIR = PROJECT_ROOT / "docs" / "paper" / "Gambar"

OUTPUT_FIG_DIR.mkdir(parents=True, exist_ok=True)
PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)

SELECTED_FILES = [
    ("level_0_mild", "levle0_1.jpg"),
    ("level_1_moderate", "levle1_101.jpg"),
    ("level_2_severe", "levle2_100.jpg"),
    ("level_3_very_severe", "levle3_101.jpg"),
]


def load_square_panel(path: Path, size: int = 512) -> Image.Image:
    with Image.open(path) as im:
        rgb = im.convert("RGB")
        return ImageOps.fit(rgb, (size, size), method=Image.Resampling.LANCZOS)


def create_collage(titles: list[str] | None, output_filename: str, copy_to_paper: bool = False) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.4), dpi=300)

    for idx, (ax, (folder, fname)) in enumerate(zip(axes, SELECTED_FILES)):
        img = load_square_panel(RAW_DIR / folder / fname)
        ax.imshow(img)
        ax.axis("off")
        if titles:
            ax.set_title(titles[idx], fontsize=12, fontweight="bold", pad=8)

    plt.tight_layout()
    out_path = OUTPUT_FIG_DIR / output_filename
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    if copy_to_paper:
        fig.savefig(PAPER_FIG_DIR / output_filename, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved {output_filename}")


def main() -> None:
    titles_id = [
        "(a) Level 0: Mild\n(0–5 Lesi)",
        "(b) Level 1: Moderate\n(6–20 Lesi)",
        "(c) Level 2: Severe\n(21–50 Lesi)",
        "(d) Level 3: Very Severe\n(>50 Lesi)",
    ]
    create_collage(titles_id, "acne04_collage_1x4_id.png", copy_to_paper=True)

    titles_en = [
        "(a) Level 0: Mild\n(0–5 Lesions)",
        "(b) Level 1: Moderate\n(6–20 Lesions)",
        "(c) Level 2: Severe\n(21–50 Lesions)",
        "(d) Level 3: Very Severe\n(>50 Lesions)",
    ]
    create_collage(titles_en, "acne04_collage_1x4_en.png")

    titles_simple = ["(a) Level 0", "(b) Level 1", "(c) Level 2", "(d) Level 3"]
    create_collage(titles_simple, "acne04_collage_1x4_simple.png")
    create_collage(None, "acne04_collage_1x4_plain.png")


if __name__ == "__main__":
    main()
