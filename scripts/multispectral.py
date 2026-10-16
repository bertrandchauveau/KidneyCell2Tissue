import os
import cv2
import numpy as np
import argparse

#create a 6-channel image from H&E and whole cell masks
def main():
    # 1. Set up the Argument Parser
    parser = argparse.ArgumentParser(description="Create a 6-channel image from H&E and cell masks.")

    # Add arguments
    parser.add_argument('--rgb_dir', type=str, required=True, help='Directory containing the RGB image')
    parser.add_argument('--mask_dir', type=str, required=True, help='Directory containing the mask image')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the output TIFF')
    parser.add_argument('--rgb_name', type=str, default='RGB.jpg', help='Name of the RGB file (default: RGB.jpg)')
    parser.add_argument('--mask_name', type=str, default='mask.png', help='Name of the mask file (default: mask.png)')

    # 2. Parse the arguments from the CLI
    args = parser.parse_args()

    # Define full paths
    rgb_path = os.path.join(args.rgb_dir, args.rgb_name)
    mask_path = os.path.join(args.mask_dir, args.mask_name)
    output_filename = os.path.splitext(args.rgb_name)[0] + '.tiff'
    output_path = os.path.join(args.output_dir, output_filename)

    # 3. Check if files exist before processing
    if not os.path.exists(rgb_path):
        print(f"Error: RGB file not found at {rgb_path}")
        return
    if not os.path.exists(mask_path):
        print(f"Error: Mask file not found at {mask_path}")
        return
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print(f"Created output directory: {args.output_dir}")

    # 4. Processing Logic
    # Load RGB tile image (BGR => RGB)
    img_array = cv2.imread(rgb_path)[:, :, ::-1].copy()
    
    # Load whole cell tile mask
    molec_array = cv2.imread(mask_path)

    if img_array.shape[:2] != molec_array.shape[:2]:
        print("Error: RGB image and Mask dimensions do not match!")
        return

    # Set a whitish background that in theory would make the mask and the H&E a bit more alike in terms of color space
    molec_array = np.where(molec_array == 0, 250, molec_array)

    # Concatenate images and save
    # stack in 6 channels (H, W, 6)
    image_multi_array = np.concatenate([img_array, molec_array], axis=2)
    
    # Transpose to CHW-order (Channels, Height, Width)
    image_multi_array = np.transpose(image_multi_array, (2, 0, 1))

    # Save as tiff
    success = cv2.imwritemulti(output_path, image_multi_array)
    
    if success:
        print(f"Successfully saved: {output_path}")
    else:
        print("Error: Failed to save the image.")

if __name__ == "__main__":
    main()
