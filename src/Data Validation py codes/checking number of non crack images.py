import pandas as pd
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

METADATA_FILE = Path(r"path/to/your/metadata.csv")

def run_dual_check():
    if not METADATA_FILE.exists():
        print(f"Error: Metadata file not found at '{METADATA_FILE}'")
        return

    print(f"Loading metadata from '{METADATA_FILE}'...")
    df = pd.read_csv(METADATA_FILE)
    non_crack_df = df[df['class'] == 'non_crack'].copy()
    total_to_check = len(non_crack_df)
    print(f"Found {total_to_check} 'non_crack' samples to check.")

    if total_to_check == 0:
        print("No 'non_crack' samples found in the metadata file.")
        return

    filename_mismatches = 0
    not_black_masks = 0
    problem_mask_files = []

    for index, row in tqdm(non_crack_df.iterrows(), total=total_to_check, desc="Checking non-crack samples"):
        image_path = Path(row['image_path'])
        mask_path = Path(row['mask_path'])

        if 'noncrack' not in image_path.name.lower():
            filename_mismatches += 1

        if not mask_path.exists():
            print(f"\nWarning: Mask file not found, skipping pixel check: {mask_path}")
            not_black_masks += 1
            problem_mask_files.append(f"{mask_path.name} (File not found)")
            continue

        try:
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if cv2.countNonZero(mask) > 0:
                not_black_masks += 1
                problem_mask_files.append(mask_path.name)
        except Exception as e:
            print(f"\nError reading or processing mask file {mask_path.name}: {e}")
            not_black_masks += 1
            problem_mask_files.append(f"{mask_path.name} (Read error)")

    print("\n--- Dual-Check Validation Report ---")
    print(f"\n[Check 1: Filename Consistency]")
    if filename_mismatches == 0:
        print(f"  \u2705 PASS: All {total_to_check} 'non_crack' labels correspond to filenames containing 'noncrack'.")
    else:
        print(
            f"  \u274c FAIL: Found {filename_mismatches} samples labeled 'non_crack' where the filename did not contain 'noncrack'.")

    print(f"\n[Check 2: Mask Pixel Content]")
    if not_black_masks == 0:
        print(f"  \u2705 PASS: All {total_to_check} masks for 'non_crack' samples are completely black.")
    else:
        print(f"  \u274c FAIL: Found {not_black_masks} masks for 'non_crack' samples that contain non-black pixels.")
        if problem_mask_files:
            print("\n  --- List of First 10 Problematic Mask Files ---")
            for i, fname in enumerate(problem_mask_files[:10]):
                print(f"  {i + 1}: {fname}")

    print("\n--- Overall Result ---")
    if filename_mismatches == 0 and not_black_masks == 0:
        print("\u2728 Your dataset's negative samples have passed the dual-check perfectly!")
    else:
        print("Issues were found. Please review the report above.")

if __name__ == "__main__":
    run_dual_check()