import os
from PIL import Image
import argparse

def convert_png_to_jpg(source_folder, target_folder):
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    for filename in os.listdir(source_folder):
        if filename.endswith(".png"):
            png_image_path = os.path.join(source_folder, filename)
            jpg_image_path = os.path.join(target_folder, filename.replace(".png", ".jpg"))
            with Image.open(png_image_path) as img:
                rgb_img = img.convert('RGB')
                rgb_img.save(jpg_image_path, "JPEG")
            print(f"Converted {filename} to {jpg_image_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert PNG images to JPG")
    parser.add_argument('--source_folder', type=str, help="Path to the folder containing PNG images")
    parser.add_argument('--target_folder', type=str, help="Path to the folder to save JPG images")
    args = parser.parse_args()

    convert_png_to_jpg(args.source_folder, args.target_folder)