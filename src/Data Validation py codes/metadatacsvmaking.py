import pandas as pd
from pathlib import Path
from tqdm import tqdm

DATASET_DIR = Path(r"path/to/your/dataset")
OUTPUT_CSV_PATH = DATASET_DIR / "metadata.csv"

def generate_metadata():
    if not DATASET_DIR.is_dir():
        print(f"Error: Dataset directory not found at '{DATASET_DIR}'")
        print("Please run the 'final_prepare_dataset.py' script first.")
        return

    metadata_list = []
    splits = ['train', 'validation', 'test']

    print("Scanning dataset folders to generate metadata...")

    for split in splits:
        image_dir = DATASET_DIR / split / 'images'
        mask_dir = DATASET_DIR / split / 'masks'

        if not image_dir.is_dir():
            print(f"Warning: '{split}/images' directory not found. Skipping.")
            continue

        for image_path in tqdm(list(image_dir.glob('*.*')), desc=f"Processing '{split}' set"):
            mask_path = mask_dir / image_path.name

            if mask_path.exists():
                if 'noncrack' in image_path.name:
                    image_class = 'non_crack'
                else:
                    image_class = 'crack'

                metadata_list.append({
                    'image_path': str(image_path),
                    'mask_path': str(mask_path),
                    'class': image_class,
                    'split': split
                })

    if not metadata_list:
        print("No image-mask pairs were found. Cannot create metadata file.")
        return

    df = pd.DataFrame(metadata_list)
    df.to_csv(OUTPUT_CSV_PATH, index=False)

    print(f"\nMetadata file created successfully at: '{OUTPUT_CSV_PATH}'")
    print(f"Total entries in metadata: {len(df)}")
    print("\n--- First 5 rows of metadata.csv ---")
    print(df.head())
    print("\n--- Split counts ---")
    print(df['split'].value_counts())

if __name__     == "__main__":
    generate_metadata()