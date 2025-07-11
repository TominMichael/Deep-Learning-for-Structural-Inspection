import shutil
import random
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split



BASE_DIR = Path(r"path/to/your/directory")
OUTPUT_DIR = BASE_DIR / "final_dataset_ready_for_training"


SOURCE_PAIRS = [
    (BASE_DIR / "22crackimages_512x512", BASE_DIR / "22crackmaskimages_512x512"),
    (BASE_DIR / "446crackimages_512x512", BASE_DIR / "446crackmaskimages_512x512"),
    (BASE_DIR / "3556crackimages_512x512", BASE_DIR / "3556crackmaskimages_512x512"),
    (BASE_DIR / "4697crackimages_512x512", BASE_DIR / "4697crackmaskimages_512x512"),
    (BASE_DIR / "4890crackimages_512x512", BASE_DIR / "4890crackmasks_512x512"),
    (BASE_DIR / "1411noncrackimages_512x512", BASE_DIR / "1411noncrackmaskimages_512x512"),
    (BASE_DIR / "12200noncrackimages_512x512", BASE_DIR / "12200noncrackmaskimages_512x512"),
]

TRAIN_SIZE = 0.8
VAL_SIZE = 0.1
TEST_SIZE = 0.1




def consolidate_and_rename_files():
    """
    Gathers all files, renames them to be unique, and saves them
    to a temporary consolidated directory.
    """
    temp_img_dir = BASE_DIR / "temp_consolidated_images"
    temp_mask_dir = BASE_DIR / "temp_consolidated_masks"
    temp_img_dir.mkdir(exist_ok=True)
    temp_mask_dir.mkdir(exist_ok=True)

    all_pairs = []
    print("Consolidating and renaming files...")

    for img_dir, mask_dir in tqdm(SOURCE_PAIRS, desc="Processing source folders"):
        source_prefix = img_dir.name

        for img_path in img_dir.glob('*.*'):
         
            new_img_name = f"{source_prefix}_{img_path.name}"
            new_mask_name = f"{source_prefix}_{img_path.stem}.png"

            original_mask_path = mask_dir / (img_path.stem + '.png')  
            if original_mask_path.exists():
              
                dest_img_path = temp_img_dir / new_img_name
                dest_mask_path = temp_mask_dir / new_mask_name

               
                shutil.copy(img_path, dest_img_path)
                shutil.copy(original_mask_path, dest_mask_path)

                all_pairs.append((dest_img_path, dest_mask_path))

    return all_pairs


def split_and_copy_files(all_file_pairs):
    """Splits data into train/val/test and copies to final directories."""
    print("\nShuffling and splitting data...")
    random.seed(42)
    random.shuffle(all_file_pairs)

    train_pairs, temp_pairs = train_test_split(all_file_pairs, test_size=(VAL_SIZE + TEST_SIZE), random_state=42)
    val_pairs, test_pairs = train_test_split(temp_pairs, test_size=(TEST_SIZE / (VAL_SIZE + TEST_SIZE)),
                                             random_state=42)

    print(f"Total files: {len(all_file_pairs)}")
    print(f"Train set size: {len(train_pairs)}")
    print(f"Validation set size: {len(val_pairs)}")
    print(f"Test set size: {len(test_pairs)}")


    def copy_set(pairs, set_name):
        img_dest = OUTPUT_DIR / set_name / 'images'
        mask_dest = OUTPUT_DIR / set_name / 'masks'
        img_dest.mkdir(parents=True, exist_ok=True)
        mask_dest.mkdir(parents=True, exist_ok=True)
        for img_path, mask_path in tqdm(pairs, desc=f"Copying {set_name} files"):
            shutil.copy(img_path, img_dest)
            shutil.copy(mask_path, mask_dest)

    copy_set(train_pairs, 'train')
    copy_set(val_pairs, 'validation')
    copy_set(test_pairs, 'test')


def main():
   
    if OUTPUT_DIR.exists():
        print(f"Removing old output directory: {OUTPUT_DIR}")
        shutil.rmtree(OUTPUT_DIR)

    consolidated_pairs = consolidate_and_rename_files()

    
    split_and_copy_files(consolidated_pairs)

    
    print("\nCleaning up temporary folders...")
    temp_img_dir = BASE_DIR / "temp_consolidated_images"
    temp_mask_dir = BASE_DIR / "temp_consolidated_masks"
    shutil.rmtree(temp_img_dir)
    shutil.rmtree(temp_mask_dir)

    print("\nDataset preparation is successfully complete!")
    print(f"Your final dataset is ready in: '{OUTPUT_DIR}'")


if __name__ == "__main__":
    main()