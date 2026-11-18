"""Download and export the canonical Layered-Labs/ACNE04 dataset from Hugging Face."""

import io
from collections import Counter
from pathlib import Path
from PIL import Image
from datasets import Image as HFImage, load_dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "data" / "raw" / "acne04"

LEVEL_FOLDERS = {
    0: "level_0_mild",
    1: "level_1_moderate",
    2: "level_2_severe",
    3: "level_3_very_severe",
}


def main() -> None:
    print("Loading Layered-Labs/ACNE04 from Hugging Face...")
    ds = load_dataset("Layered-Labs/ACNE04", split="train")
    ds_raw = ds.cast_column("image", HFImage(decode=False))

    print(f"Total records loaded: {len(ds_raw)}")
    level_counts: Counter[int] = Counter()
    severity_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    formats: Counter[str] = Counter()
    dims: list[tuple[int, int]] = []
    sizes_kb: list[float] = []
    seen_targets: set[Path] = set()

    for idx, row in enumerate(ds_raw):
        level = int(row["level"])
        severity = str(row["severity"])
        file_name = Path(str(row["file_name"])).name
        img_info = row["image"]
        raw_bytes = img_info.get("bytes")
        raw_path = img_info.get("path")

        folder = OUT_DIR / LEVEL_FOLDERS[level]
        folder.mkdir(parents=True, exist_ok=True)
        out_path = folder / file_name

        if out_path in seen_targets:
            stem = Path(file_name).stem
            suffix = Path(file_name).suffix or ".jpg"
            out_path = folder / f"{stem}_{idx}{suffix}"
        seen_targets.add(out_path)

        if raw_bytes is not None:
            out_path.write_bytes(raw_bytes)
            with Image.open(io.BytesIO(raw_bytes)) as im:
                formats[im.format or "UNKNOWN"] += 1
                dims.append(im.size)
            sizes_kb.append(len(raw_bytes) / 1024.0)
        elif raw_path:
            with Image.open(raw_path) as im:
                formats[im.format or "UNKNOWN"] += 1
                dims.append(im.size)
                im.save(out_path)
            sizes_kb.append(out_path.stat().st_size / 1024.0)

        level_counts[level] += 1
        severity_counts[severity] += 1
        split_counts[str(row["split"])] += 1

    widths = [w for w, _ in dims]
    heights = [h for _, h in dims]
    aspects = [round(w / h, 4) for w, h in dims]
    sorted_sizes = sorted(sizes_kb)
    mid = len(sorted_sizes) // 2

    print("\n--- Verification Summary ---")
    print("Levels   :", dict(sorted(level_counts.items())))
    print("Severity :", dict(severity_counts))
    print("HF Fold0 :", dict(split_counts))
    print("Formats  :", dict(formats))
    print(f"Unique resolutions: {len(set(dims))}")
    print(f"Width  range: {min(widths)} .. {max(widths)} px (median {sorted(widths)[mid]})")
    print(f"Height range: {min(heights)} .. {max(heights)} px (median {sorted(heights)[mid]})")
    print(f"Aspect range: {min(aspects)} .. {max(aspects)} (unique={len(set(aspects))})")
    print(f"Median file size: {sorted_sizes[mid]:.2f} KB (min={sorted_sizes[0]:.2f}, max={sorted_sizes[-1]:.2f})")
    print(f"Saved files to: {OUT_DIR}")


if __name__ == "__main__":
    main()

