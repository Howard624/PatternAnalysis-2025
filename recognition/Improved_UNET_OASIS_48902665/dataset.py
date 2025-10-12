import os
import cv2
import torch
from torch.utils.data import Dataset

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

# ------------------------------
# Step 2: PyTorch Dataset Class (new)
# ------------------------------
class OASISDataset(Dataset):
    """PyTorch Dataset for OASIS MRI slices and segmentation masks"""
    
    def __init__(self, split="train"):
        # Initialize paths
        self.img_folder = os.path.join(DATA_ROOT, f"keras_png_slices_{split}")
        self.seg_folder = os.path.join(DATA_ROOT, f"keras_png_slices_seg_{split}")
        
        # Get valid PNG files (reusing Step 1 logic)
        self.img_files = [f for f in os.listdir(self.img_folder) if f.endswith(".png")]
        if not self.img_files:
            raise ValueError(f"No PNG files in image folder: {self.img_folder}")
        
        # Validate all segmentation files exist
        self._validate_segmentations()

    def _validate_segmentations(self):
        """Check all images have matching segmentation files"""
        missing = []
        for img_file in self.img_files:
            seg_file = img_file.replace("case_", "seg_")
            if not os.path.exists(os.path.join(self.seg_folder, seg_file)):
                missing.append(seg_file)
        
        if missing:
            raise FileNotFoundError(f"Missing segmentations: {missing[:5]}...")

    def __len__(self):
        """Return total number of samples"""
        return len(self.img_files)

    def __getitem__(self, idx):
        """Load and return sample at specified index"""
        # Get filenames
        img_file = self.img_files[idx]
        seg_file = img_file.replace("case_", "seg_")
        
        # Load images (using Step 1's loading method)
        img_path = os.path.join(self.img_folder, img_file)
        seg_path = os.path.join(self.seg_folder, seg_file)
        
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        segmentation = cv2.imread(seg_path, cv2.IMREAD_GRAYSCALE)
        
        # Validate loading
        if image is None or segmentation is None:
            raise IOError(f"Failed to load sample {idx}: {img_file}")
        
        # Preprocess for PyTorch
        image = image / 255.0  # Normalize to [0, 1]
        segmentation = segmentation / 255.0
        
        # Convert to tensors with channel dimension (C, H, W)
        image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
        seg_tensor = torch.tensor(segmentation, dtype=torch.float32).unsqueeze(0)
        
        return image_tensor, seg_tensor


# ------------------------------
# Test both components
# ------------------------------
if __name__ == "__main__":
    # Test Step 1: Single sample loader
    print("=== Testing Single Sample Loader ===")
    load_single_sample(split="train")
    
    # Test Step 2: PyTorch Dataset
    print("\n=== Testing PyTorch Dataset ===")
    dataset = OASISDataset(split="train")
    print(f"Total samples in dataset: {len(dataset)}")
    
    # Get first sample from dataset
    img_tensor, seg_tensor = dataset[0]
    print(f"Image tensor shape: {img_tensor.shape} (expected: [1, 256, 256])")
    print(f"Seg tensor shape: {seg_tensor.shape} (expected: [1, 256, 256])")
    print(f"Value range: {img_tensor.min():.2f} to {img_tensor.max():.2f} (expected: 0.00 to 1.00)")
    