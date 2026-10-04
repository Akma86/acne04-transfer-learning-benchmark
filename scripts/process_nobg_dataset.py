"""Offline Dataset Pre-processing: Background Removal & Facial Focus.

Batch-processes raw ACNE04 images to remove non-facial backdrops (walls, clothing,
hospital furniture, personnel) and tightly crop/frame the facial region using
salient U2Net segmentation with skin-color fallback.

Pre-caching these images offline eliminates on-the-fly U2Net inference latency
during model training (saving ~20-25 minutes per training epoch).

Usage:
    # Test on first 10 images:
    python scripts/process_nobg_dataset.py --limit 10

    # Full dataset processing (all 1,377 images):
    python scripts/process_nobg_dataset.py
"""

import argparse
import os
import sys
import time
from pathlib import Path

# OpenMP multi-threading safety guards for Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
cv2.setNumThreads(0)
import rembg
import pandas as pd
from PIL import Image
from tqdm import tqdm

from src.data.preprocessing import apply_remove_bg


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch offline pre-processing of ACNE04 for background removal and facial isolation."
    )
    parser.add_argument(
        "--metadata",
        type=str,
        default="data/metadata/acne04_metadata.csv",
        help="Source metadata CSV with raw image filepaths",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/processed/acne_nobg",
        help="Directory to save background-removed face images",
    )
    parser.add_argument(
        "--output_metadata",
        type=str,
        default="data/metadata/acne04_metadata_nobg.csv",
        help="Destination metadata CSV referencing the pre-processed images",
    )
    parser.add_argument(
        "--bg_color",
        type=str,
        default="black",
        choices=["black", "white", "gray"],
        help="Background replacement color",
    )
    parser.add_argument(
        "--no_crop",
        action="store_true",
        help="Disable auto-cropping to the facial bounding box",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of images to process (useful for rapid sanity verification)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-process images even if the output file already exists",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 75)
    print(" ACNE04 OFFLINE PREPROCESSING: DEEP BACKGROUND REMOVAL & FACE ISOLATION")
    print(f" Source Metadata   : {args.metadata}")
    print(f" Output Directory  : {args.output_dir}")
    print(f" Output Metadata   : {args.output_metadata}")
    print(f" Background Fill   : {args.bg_color}")
    print(f" Facial Auto-Crop  : {not args.no_crop}")
    print(f" Overwrite Existing: {args.overwrite}")
    if args.limit:
        print(f" Sample Limit      : {args.limit} images")
    print("=" * 75)

    meta_path = Path(args.metadata)
    if not meta_path.exists():
        print(f"[ERROR] Source metadata not found at: {meta_path}")
        print("Please ensure data/metadata/acne04_metadata.csv is generated first.")
        sys.exit(1)

    df = pd.read_csv(meta_path)
    total_raw = len(df)
    print(f"[INFO] Loaded metadata with {total_raw} records.")

    if args.limit:
        df = df.iloc[: args.limit].copy()
        print(f"[INFO] Processing a subset of {len(df)} records (--limit {args.limit}).")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[INFO] Initializing U2Net segmentation session...")
    session = rembg.new_session("u2net")
    print("[INFO] Model loaded successfully. Starting batch extraction...")

    processed_records = []
    skipped_count = 0
    newly_processed = 0

    crop_face = not args.no_crop
    start_time = time.time()

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Extracting Faces"):
        src_path = Path(row["filepath"])
        out_filename = f"{Path(row['filename']).stem}.jpg"
        out_filepath = out_dir / out_filename

        if out_filepath.exists() and not args.overwrite:
            skipped_count += 1
        else:
            try:
                img = Image.open(src_path).convert("RGB")
                processed_img = apply_remove_bg(
                    img=img,
                    bg_color=args.bg_color,
                    crop_to_face=crop_face,
                    session=session,
                )
                processed_img.save(out_filepath, "JPEG", quality=95)
                newly_processed += 1
            except Exception as e:
                print(f"\n[WARN] Failed to process {src_path.name}: {e}. Copying raw fallback.")
                if not out_filepath.exists():
                    img.save(out_filepath, "JPEG", quality=95)

        # Store metadata pointing to processed image
        new_row = row.copy()
        new_row["filepath"] = str(out_filepath).replace("\\", "/")
        new_row["is_nobg_processed"] = True
        processed_records.append(new_row)

    elapsed = time.time() - start_time

    # Save destination metadata
    out_df = pd.DataFrame(processed_records)
    out_meta_path = Path(args.output_metadata)
    out_meta_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_meta_path, index=False)

    print("\n" + "=" * 75)
    print(" PREPROCESSING SUMMARY")
    print(f" Total Processed  : {newly_processed}")
    print(f" Existing Skipped : {skipped_count}")
    print(f" Total Records    : {len(out_df)}")
    print(f" Elapsed Time     : {elapsed:.2f} seconds ({elapsed / max(1, newly_processed):.2f}s per new image)")
    print(f" Saved Metadata   : {out_meta_path}")
    print(f" Saved Directory  : {out_dir}")
    print("=" * 75)
    print("\n[NEXT STEP] To train a transfer model on these background-free face images:")
    print(f"  python src/train.py --model mobilenet_v2 --stage 3 --metadata {args.output_metadata}")


if __name__ == "__main__":
    main()
