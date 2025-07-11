import pandas as pd
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

METADATA_FILE = Path("./metadata.csv")
OUTPUT_STATS_FILE = Path("./metadata_with_stats.csv")

def analyze_dataset_for_outliers():
    if not METADATA_FILE.exists():
        print(f"Error: Metadata file not found at '{METADATA_FILE}'")
        return

    print("Loading metadata and preparing for analysis...")
    df = pd.read_csv(METADATA_FILE)

    brightness_list = []
    contrast_list = []
    crack_pixel_percent_list = []
    is_empty_mask_list = []

    print("Analyzing all images and masks. This may take some time...")
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Analyzing files"):
        try:
            img_gray = cv2.imread(row['image_path'], cv2.IMREAD_GRAYSCALE)
            if img_gray is None:
                brightness_list.append(np.nan)
                contrast_list.append(np.nan)
            else:
                brightness_list.append(img_gray.mean())
                contrast_list.append(img_gray.std())

            mask = cv2.imread(row['mask_path'], cv2.IMREAD_GRAYSCALE)
            if mask is None:
                crack_pixel_percent_list.append(np.nan)
                is_empty_mask_list.append(True)
            else:
                crack_pixels = cv2.countNonZero(mask)
                total_pixels = mask.size
                crack_pixel_percent_list.append((crack_pixels / total_pixels) * 100)
                is_empty_mask_list.append(crack_pixels == 0)

        except Exception as e:
            print(f"\nError processing row {index}: {e}")
            brightness_list.append(np.nan)
            contrast_list.append(np.nan)
            crack_pixel_percent_list.append(np.nan)
            is_empty_mask_list.append(True)

    df['brightness'] = brightness_list
    df['contrast'] = contrast_list
    df['crack_pixel_percent'] = crack_pixel_percent_list
    df['is_empty_mask'] = is_empty_mask_list

    df.to_csv(OUTPUT_STATS_FILE, index=False)

    print("\n--- Outlier Analysis Complete ---")
    print(f"New file with statistics created at: '{OUTPUT_STATS_FILE}'")

    print("\n--- Potential Issues Found ---")

    annotation_errors = df[(df['class'] == 'crack') & (df['is_empty_mask'] == True)]
    if not annotation_errors.empty:
        print(f"\n\u274c Found {len(annotation_errors)} images labeled 'crack' but have an empty (all-black) mask:")
        print(annotation_errors[['image_path', 'class', 'is_empty_mask']].head())
    else:
        print("\n\u2705 PASS: No 'crack' images with empty masks were found.")

    print("\nFurther analysis can be done by sorting the new CSV file.")

if __name_ == "__main__":
    analyze_dataset_for_outliers()