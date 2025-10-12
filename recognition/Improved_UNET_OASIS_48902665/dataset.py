import os
import cv2

"""
OASIS Dataset Loader (Step 1)
-----------------------------
Purpose: Loads a single MRI slice and its corresponding segmentation mask from the OASIS dataset.
"""

# Root path to the OASIS dataset directory
DATA_ROOT = r"D:\3710_Pattern_Recognition_Project\OASIS_dataset\OASIS"

def load_single_sample(split="train"):
    """
    Loads a single image-segmentation pair from the specified dataset split.
    """
    # Construct paths to the image and segmentation folders for the specified split
    img_folder = os.path.join(DATA_ROOT, f"keras_png_slices_{split}")
    seg_folder = os.path.join(DATA_ROOT, f"keras_png_slices_seg_{split}")
    
    # Get list of all PNG files in the image folder (filter out non-PNG files)
    img_files = [f for f in os.listdir(img_folder) if f.endswith(".png")]
    
    # Validate that there are PNG files in the image folder
    if not img_files:
        raise ValueError(f"No PNG files found in image folder: {img_folder}\nCheck if the dataset is properly downloaded.")
    
    # Select the first image file in the list to load
    first_img_filename = img_files[0]
    print(f"Found image file: {first_img_filename}")
    
    # Convert image filename to matching segmentation filename using the dataset's naming rule:
    # Replace "case_" prefix with "seg_" (e.g., "case_001_slice_0.nii.png" → "seg_001_slice_0.nii.png")
    seg_filename = first_img_filename.replace("case_", "seg_")
    
    # Create full paths to the image and segmentation files
    img_path = os.path.join(img_folder, first_img_filename)
    seg_path = os.path.join(seg_folder, seg_filename)
    
    # Verify the segmentation file exists before attempting to load
    if not os.path.exists(seg_path):
        raise FileNotFoundError(
            f"Segmentation file not found: {seg_path}\n"
            f"Expected naming pattern: 'seg_' instead of 'case_' in filename. "
            f"Check if the segmentation folder contains {seg_filename}."
        )
    
    # Load both files as grayscale images (MRI slices are single-channel)
    # cv2.IMREAD_GRAYSCALE ensures output is 2D array (H, W)
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    segmentation = cv2.imread(seg_path, cv2.IMREAD_GRAYSCALE)
    
    # Validate that images loaded successfully (cv2 returns None if loading fails)
    if image is None:
        raise IOError(f"Failed to load image file. Possible causes: corrupted file or unsupported format.\nPath: {img_path}")
    if segmentation is None:
        raise IOError(f"Failed to load segmentation file. Possible causes: corrupted file or unsupported format.\nPath: {seg_path}")
    
    # Print confirmation and details for verification
    print(f"Successfully loaded sample pair:")
    print(f"  Image: {first_img_filename} (shape: {image.shape})")
    print(f"  Segmentation: {seg_filename} (shape: {segmentation.shape})")
    
    return image, segmentation

# Execute the function when the script is run directly (for testing/verification)
if __name__ == "__main__":
    load_single_sample(split="train")
    