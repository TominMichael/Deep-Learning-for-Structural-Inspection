import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm  


INPUT_DIR = Path(
    r"")

#
OUTPUT_DIR = INPUT_DIR.parent / "Negative_Masks"




def create_black_masks():
    """
    Loops through all images in the input directory, creates a corresponding
    all-black mask for each, and saves it to the output directory.
    """
    # Check if the input directory exists
    if not INPUT_DIR.is_dir():
        print(f"Error: Input directory not found at '{INPUT_DIR}'")
        return

    # Create the output directory if it's not already there
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Masks will be saved to: {OUTPUT_DIR}")

    # Find all image files (jpg, png, jpeg) in the input folder
    image_files = list(INPUT_DIR.glob('*.jpg')) + \
                  list(INPUT_DIR.glob('*.png')) + \
                  list(INPUT_DIR.glob('*.jpeg'))

    if not image_files:
        print(f"No image files were found in '{INPUT_DIR}'")
        return

    print(f"Found {len(image_files)} images to process.")

    
    for image_path in tqdm(image_files, desc="Creating Black Masks"):
        try:
         
            img = cv2.imread(str(image_path))

           
            if img is None:
                print(f"\nWarning: Could not read image file {image_path.name}. Skipping.")
                continue

            height, width, _ = img.shape

           
            black_mask = np.zeros((height, width), dtype=np.uint8)

            
            mask_filename = image_path.stem + '.png'
            output_path = OUTPUT_DIR / mask_filename

           
            cv2.imwrite(str(output_path), black_mask)

        except Exception as e:
            print(f"\nAn error occurred while processing {image_path.name}: {e}")

    print("\nProcessing complete!")
    print(f"All {len(image_files)} black masks have been generated in '{OUTPUT_DIR}'")


if __name__ == "__main__":
    create_black_masks()