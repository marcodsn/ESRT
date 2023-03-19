import os
import argparse
import torch
import torchvision.transforms.functional as TF
from PIL import Image

# Define command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--scale_factor", type=float, required=True, help="scale factor for downscaling")
parser.add_argument("--input_dir", type=str, required=True, help="path to input directory")
parser.add_argument("--output_dir", type=str, required=True, help="path to output directory")
parser.add_argument("--print_step", type=int, default=10, help="print progress every print_step images")
args = parser.parse_args()

# Create output directory if it doesn't exist
if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

# Loop through input directory and resize images
i = 1
for file_name in os.listdir(args.input_dir):
    if file_name.endswith(".jpg") or file_name.endswith(".jpeg") or file_name.endswith(".png"):
        # Load image
        img_path = os.path.join(args.input_dir, file_name)
        img = Image.open(img_path)

        # Define scale factor
        scale_factor = args.scale_factor

        # Calculate output size
        width, height = img.size
        new_width = int(width / scale_factor)
        new_height = int(height / scale_factor)

        # Resize image using bicubic interpolation
        resized_img = TF.resize(img, (new_height, new_width), interpolation=Image.BICUBIC)

        # Save resized image
        output_path = os.path.join(args.output_dir, file_name)
        resized_img.save(output_path)

        # Print progress
        if i % args.print_step == 0:
            print(f"Resized image {i} of {len(os.listdir(args.input_dir))}")
        i = i + 1
