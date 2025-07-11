import json
import os
from PIL import Image, ImageDraw

json_directory = r""
output_directory = r""

if not os.path.exists(output_directory):
    os.makedirs(output_directory)
    print(f"Created output directory: {output_directory}")

try:
    files_in_directory = os.listdir(json_directory)
except FileNotFoundError:
    print(f"Error: The directory '{json_directory}' was not found.")
    print("Please make sure you have set the correct path in the 'json_directory' variable.")
    exit()

for filename in files_in_directory:
    if filename.endswith('.json'):
        json_path = os.path.join(json_directory, filename)
        print(f"Processing {filename}...")
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            image_height = data.get('imageHeight')
            image_width = data.get('imageWidth')
            if image_height is None or image_width is None:
                print(f"  - Skipping {filename}: 'imageHeight' or 'imageWidth' not found.")
                continue
            mask_image = Image.new('L', (image_width, image_height), 0)
            draw = ImageDraw.Draw(mask_image)
            for shape in data.get('shapes', []):
                if shape.get('shape_type') == 'polygon' and shape.get('points'):
                    polygon = [tuple(point) for point in shape['points']]
                    draw.polygon(polygon, outline=255, fill=255)
            output_filename = os.path.splitext(filename)[0] + '.png'
            output_path = os.path.join(output_directory, output_filename)
            mask_image.save(output_path)
            print(f"  - Successfully created mask: {output_filename}")
        except json.JSONDecodeError:
            print(f"  - Skipping {filename}: Invalid JSON format.")
        except Exception as e:
            print(f"  - An unexpected error occurred while processing {filename}: {e}")

