import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm


INPUT_DIR = Path(r"")


OUTPUT_DIR = INPUT_DIR.parent / (INPUT_DIR.name + "_512x512")


TARGET_SIZE = 512




def resize_and_pad(image, target_size):
    """
    Resizes an image to fit within a square canvas of target_size,
    preserving the aspect ratio and padding with black.
    """
    
    h, w = image.shape[:2]

    
    scale = target_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)

   
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC

   
    resized = cv2.resize(image, (new_w, new_h), interpolation=interp)

    
    canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)

    
    top_pad = (target_size - new_h) // 2
    left_pad = (target_size - new_w) // 2

    
    canvas[top_pad:top_pad + new_h, left_pad:left_pad + new_w] = resized

    return canvas


def process_folder():
    """Main function to find, process, and save images."""

    
    if not INPUT_DIR.is_dir():
        print(f"Error: Input directory not found at '{INPUT_DIR}'")
        return

    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output will be saved to: {OUTPUT_DIR}")

    
    image_files = list(INPUT_DIR.glob('*.jpg')) + \
                  list(INPUT_DIR.glob('*.png')) + \
                  list(INPUT_DIR.glob('*.jpeg'))

    if not image_files:
        print(f"No images found in '{INPUT_DIR}'")
        return

    
    for image_path in tqdm(image_files, desc=f"Converting images in {INPUT_DIR.name}"):
        try:
            
            img = cv2.imread(str(image_path))
            if img is None:
                print(f"Warning: Could not read {image_path.name}. Skipping.")
                continue

            
            processed_img = resize_and_pad(img, TARGET_SIZE)

            
            output_path = OUTPUT_DIR / (image_path.stem + '.png')
            cv2.imwrite(str(output_path), processed_img)

        except Exception as e:
            print(f"An error occurred with {image_path.name}: {e}")

    print("\nConversion complete!")


if __name__ == "__main__":
    process_folder()