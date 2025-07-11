import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

DATASET_DIR = Path(r"path/to/your/dataset")
EXPECTED_SIZE = (512, 512)

def check_data_consistency():
    if not DATASET_DIR.is_dir():
        print(f"Error: Dataset directory not found at '{DATASET_DIR}'")
        return

    dimension_errors = []
    channel_errors = []
    mask_value_errors = []

    splits = ['train', 'validation', 'test']
    print("--- Starting Data Consistency Verification ---")

    for split in splits:
        print(f"\n--- Checking '{split}' set ---")
        image_dir = DATASET_DIR / split / 'images'
        mask_dir = DATASET_DIR / split / 'masks'

        for img_path in tqdm(list(image_dir.glob('*.*')), desc=f"Verifying {split} images"):
            try:
                img = cv2.imread(str(img_path))
                if img is None:
                    dimension_errors.append(f"{img_path} (could not be read)")
                    continue
                if img.shape[0] != EXPECTED_SIZE[0] or img.shape[1] != EXPECTED_SIZE[1]:
                    dimension_errors.append(f"{img_path} (is {img.shape[0]}x{img.shape[1]})")
                if len(img.shape) != 3 or img.shape[2] != 3:
                    channel_errors.append(f"{img_path} (has {len(img.shape)} dims, shape: {img.shape})")
            except Exception as e:
                print(f"Error processing image {img_path.name}: {e}")

        for mask_path in tqdm(list(mask_dir.glob('*.*')), desc=f"Verifying {split} masks "):
            try:
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    dimension_errors.append(f"{mask_path} (could not be read)")
                    continue
                if mask.shape[0] != EXPECTED_SIZE[0] or mask.shape[1] != EXPECTED_SIZE[1]:
                    dimension_errors.append(f"{mask_path} (is {mask.shape[0]}x{mask.shape[1]})")
                unique_values = np.unique(mask)
                if not np.all(np.isin(unique_values, [0, 255])):
                    mask_value_errors.append(f"{mask_path} (contains values other than 0 and 255: {unique_values})")
            except Exception as e:
                print(f"Error processing mask {mask_path.name}: {e}")

    print("\n" + "=" * 50)
    print("--- DATA CONSISTENCY REPORT ---")
    print("=" * 50)

    if not dimension_errors:
        print(
            f"\n\u2705 PASS: All images and masks have the correct dimensions ({EXPECTED_SIZE[0]}x{EXPECTED_SIZE[1]}).")
    else:
        print(f"\n\u274c FAIL: Found {len(dimension_errors)} files with incorrect dimensions.")
        for error in dimension_errors[:5]: print(f"  - {error}")

    if not channel_errors:
        print(f"\n\u2705 PASS: All images are 3-channel color images.")
    else:
        print(f"\n\u274c FAIL: Found {len(channel_errors)} images with incorrect channel counts.")
        for error in channel_errors[:5]: print(f"  - {error}")

    if not mask_value_errors:
        print(f"\n\u2705 PASS: All masks are binary (contain only 0s and 255s).")
    else:
        print(f"\n\u274c FAIL: Found {len(mask_value_errors)} masks with non-binary values.")
        for error in mask_value_errors[:5]: print(f"  - {error}")

    print("\n" + "=" * 50)
    if not dimension_errors and not channel_errors and not mask_value_errors:
        print("\n\u2728 FINAL VERDICT: Your dataset has passed all consistency checks.")
    else:
        print("\n\u274c FINAL VERDICT: Issues found. Please review the report details.")

if __name__ == "__main__":
    check_data_consistency()