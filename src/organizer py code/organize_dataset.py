import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# --- Configuration ---

# 1. Define the base directory of your project
BASE_DIR = r"C:\Users\HP\Desktop\droneaimodel"

# 2. Define the new directories you want to create
TARGET_IMAGE_DIR = os.path.join(BASE_DIR, "data", "images")
TARGET_MASK_DIR = os.path.join(BASE_DIR, "data", "masks")
METADATA_FILE = os.path.join(BASE_DIR, "metadata.csv")

# 3. List your source folders and their corresponding class
# Format: (image_folder_name, mask_folder_name, class_label)
SOURCE_FOLDERS = [
    ("22crackimages_512x512", "22crackmaskimages_512x512", "crack"),
    ("446crackimages_512x512", "446crackmaskimages_512x512", "crack"),
    ("1411noncrackimages_512x512", "1411noncrackmaskimages_512x512", "non_crack"),
    ("3556crackimages_512x512", "3556crackmaskimages_512x512", "crack"),
    ("4697crackimages_512x512", "4697crackmaskimages_512x512", "crack"),
    ("4890crackimages_512x512", "4890crackmasks_512x512", "crack"),
    ("12200noncrackimages_512x512", "12200noncrackmaskimages_512x512", "non_crack"),
]


# --- Main Script ---

def main():
    """
    Main function to organize the dataset and create metadata.
    """
    print("Starting dataset organization...")

    # Create the target directories if they don't exist
    os.makedirs(TARGET_IMAGE_DIR, exist_ok=True)
    os.makedirs(TARGET_MASK_DIR, exist_ok=True)
    print(f"Created target directories: {TARGET_IMAGE_DIR} and {TARGET_MASK_DIR}")

    metadata_list = []

    # Use tqdm for a progress bar
    with tqdm(total=sum(len(os.listdir(os.path.join(BASE_DIR, img_f))) for img_f, _, _ in SOURCE_FOLDERS),
              desc="Processing files") as pbar:
        for img_folder_name, mask_folder_name, class_label in SOURCE_FOLDERS:
            image_folder_path = os.path.join(BASE_DIR, img_folder_name)
            mask_folder_path = os.path.join(BASE_DIR, mask_folder_name)

            if not os.path.exists(image_folder_path) or not os.path.exists(mask_folder_path):
                print(f"Warning: Skipping pair as a folder is missing: {img_folder_name}, {mask_folder_name}")
                continue

            # Iterate through each image in the source folder
            for image_filename in os.listdir(image_folder_path):
                # Assume mask has the same filename as the image.
                # This is a common convention. If your mask has a suffix like '_mask',
                # you would need to adjust the line below.
                # e.g., mask_filename = f"{os.path.splitext(image_filename)[0]}_mask.png"
                mask_filename = image_filename

                source_image_path = os.path.join(image_folder_path, image_filename)
                source_mask_path = os.path.join(mask_folder_path, mask_filename)

                # --- Validation Step: Check if the corresponding mask exists ---
                if not os.path.exists(source_mask_path):
                    print(f"Warning: Mask not found for image {source_image_path}. Skipping.")
                    pbar.update(1)
                    continue

                # Define destination paths
                target_image_path = os.path.join(TARGET_IMAGE_DIR, image_filename)
                target_mask_path = os.path.join(TARGET_MASK_DIR, mask_filename)

                # Copy files to the new unified directories
                # shutil.copy2 preserves file metadata (like creation time)
                shutil.copy2(source_image_path, target_image_path)
                shutil.copy2(source_mask_path, target_mask_path)

                # Add info to our metadata list
                metadata_list.append({
                    "image_id": image_filename,
                    "mask_id": mask_filename,
                    "class": class_label
                })
                pbar.update(1)

    print("\nFile copying complete. Creating metadata file...")

    # Convert the list of dictionaries to a pandas DataFrame
    df = pd.DataFrame(metadata_list)

    # --- Stratified Train-Validation-Test Split ---
    # We split the data while keeping the percentage of 'crack' and 'non_crack' samples
    # the same across all splits. This is crucial for imbalanced datasets.

    # Step 1: Split into training (70%) and a temporary set (30%)
    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=42,  # for reproducibility
        stratify=df['class']
    )

    # Step 2: Split the temporary set into validation (15%) and test (15%)
    # This is 0.5 (50%) of the 30% temp set.
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=42,  # for reproducibility
        stratify=temp_df['class']
    )

    # Add the 'split' column to each dataframe
    train_df['split'] = 'train'
    val_df['split'] = 'validation'
    test_df['split'] = 'test'

    # Combine them back into a single dataframe
    final_df = pd.concat([train_df, val_df, test_df])

    # Save the final dataframe to a CSV file
    final_df.to_csv(METADATA_FILE, index=False)

    print(f"\nMetadata file '{METADATA_FILE}' created successfully.")
    print("Dataset organization is complete.")
    print("\nSplit counts:")
    print(final_df['split'].value_counts())
    print("\nClass distribution in splits:")
    print(final_df.groupby('split')['class'].value_counts(normalize=True))


if __name__ == "__main__":
    main()