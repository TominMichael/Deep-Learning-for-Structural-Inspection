import os


folder_path = r""


expected_total_images = 12200


file_extension = ".png"

padding_length = 5



def find_missing_images(path, total_count, ext, padding):
    """
    Finds missing sequentially numbered images in a folder.

    Args:
        path (str): The directory path to search in.
        total_count (int): The highest expected image number.
        ext (str): The file extension (e.g., ".jpg").
        padding (int): The zero-padding length of the filename number.

    Returns:
        list: A sorted list of missing filenames.
    """
    print(f"--- Checking Folder: {path} ---")

   
    if not os.path.isdir(path):
        print(f"\nError: The folder specified does not exist.")
        print("Please check the 'folder_path' variable in the script.")
        return None

    expected_files = set()
    for i in range(1, total_count + 1):
        filename = f"{i:0{padding}d}{ext}"
        expected_files.add(filename)

    print(f"Generated {len(expected_files)} expected filenames (from 1 to {total_count}).")

    try:
        actual_files = set(os.listdir(path))
        print(f"Found {len(actual_files)} files/folders in the directory.")
    except Exception as e:
        print(f"Error reading directory: {e}")
        return None

    
    missing_files = expected_files - actual_files

    
    return sorted(list(missing_files))

if __name__ == "__main__":
    missing_list = find_missing_images(
        path=folder_path,
        total_count=expected_total_images,
        ext=file_extension,
        padding=padding_length
    )

    if missing_list is not None:
        print("\n--- Results ---")
        if not missing_list:
            print(f"✅ Success! No missing images found in the sequence of {expected_total_images}.")
        else:
            print(f"🔥 Found {len(missing_list)} missing images:")
            for filename in missing_list:
                print(filename)