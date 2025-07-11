import os


folder_path = r""


start_num = 1
end_num = 2801



try:
    
    existing_files = set(os.listdir(folder_path))
except FileNotFoundError:
    print(f"Error: The folder '{folder_path}' was not found. Please provide the correct path.")
    exit()


missing_files = []

print("Searching for missing files...")


for number in range(start_num, end_num + 1):
   
    expected_file_name = f"{number:05d}.json"

    
    if expected_file_name not in existing_files:
        missing_files.append(expected_file_name)



if not missing_files:
    print("\nCongratulations! No files are missing in this sequence.")
else:
    print(f"\nFound a total of {len(missing_files)} missing files. They are:")

    for file_name in missing_files:
        print(file_name)