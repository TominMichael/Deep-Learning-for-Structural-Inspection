from pathlib import Path
from tqdm import tqdm

DATASET_DIR = Path("./final_dataset_ready_for_training")

def verify_all_pairs():
    if not DATASET_DIR.is_dir():
        print(f"Error: Dataset directory not found at '{DATASET_DIR}'")
        print("Please make sure the path is correct and you have run the data preparation scripts.")
        return

    splits_to_check = ['train', 'validation', 'test']
    total_missing_masks = 0
    total_extra_masks = 0

    print("--- Starting Comprehensive Image-Mask Pair Verification ---")

    for split in splits_to_check:
        print(f"\n--- Checking '{split}' set ---")
        image_dir = DATASET_DIR / split / 'images'
        mask_dir = DATASET_DIR / split / 'masks'

        if not image_dir.is_dir() or not mask_dir.is_dir():
            print(f"  \u274c FAIL: '{split}' set is incomplete. Missing 'images' or 'masks' directory.")
            continue

        image_stems = {p.stem for p in image_dir.glob('*.*')}
        mask_stems = {p.stem for p in mask_dir.glob('*.*')}

        missing_masks = image_stems - mask_stems
        if not missing_masks:
            print(f"  \u2705 PASS: All {len(image_stems)} images have a corresponding mask.")
        else:
            print(f"  \u274c FAIL: Found {len(missing_masks)} images that are MISSING a mask.")
            total_missing_masks += len(missing_masks)
            for i, stem in enumerate(list(missing_masks)[:5]):
                print(f"    - Missing mask for image: {stem}")

        extra_masks = mask_stems - image_stems
        if not extra_masks:
            print(f"  \u2705 PASS: All {len(mask_stems)} masks have a corresponding image.")
        else:
            print(f"  \u274c FAIL: Found {len(extra_masks)} masks that have NO corresponding image (extra masks).")
            total_extra_masks += len(extra_masks)
            for i, stem in enumerate(list(extra_masks)[:5]):
                print(f"    - Extra mask with no image: {stem}")

    print("\n--- Overall Verification Summary ---")
    if total_missing_masks == 0 and total_extra_masks == 0:
        print("\u2728 SUCCESS! Your dataset is perfectly paired and consistent across all splits.")
        print("Data preparation is definitively complete.")
    else:
        print("Issues found in the dataset structure. Please review the errors listed above.")

if __name_ == "__main__":
    verify_all_pairs()