import os
import torch
import cv2
import numpy as np
from modules import ImprovedUNET  # Import your model architecture

# Configuration - update paths as needed
CONFIG = {
    "model_path": "trained_unet.pth",  # Path to saved model
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "input_mri_path": "example_mri.png",  # Input MRI slice (grayscale)
    "image_size": (256, 256)  # Must match training input size
}

def load_trained_model(model_path, device):
    """Load the trained UNET model from checkpoint"""
    # Initialize model with correct architecture
    model = ImprovedUNET(in_channels=1, out_channels=4)
    # Load saved weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    # Move to target device and set to evaluation mode
    model.to(device)
    model.eval()
    return model

def preprocess_mri(image_path, target_size):
    """
    Preprocess a raw MRI slice for model input:
    1. Load as grayscale
    2. Resize to target size
    3. Normalize to [0, 1]
    4. Convert to PyTorch tensor with batch/channel dimensions
    """
    # Load image as grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not load MRI from {image_path}")
    
    # Resize to match model input
    img_resized = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
    
    # Normalize to [0, 1] (matches training data)
    img_normalized = img_resized / 255.0  # Assumes 8-bit input (0-255)
    
    # Convert to tensor: (H, W) → (1, 1, H, W) [batch, channel, H, W]
    img_tensor = torch.tensor(
        img_normalized, 
        dtype=torch.float32
    ).unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
    
    return img_tensor, img_resized  # Return tensor (for model) and resized image (for later visualization)

def main():
    print("=== Brain Tissue Segmentation Predictor ===")
    
    # Step 1: Load trained model
    print(f"1. Loading model from {CONFIG['model_path']}...")
    model = load_trained_model(CONFIG["model_path"], CONFIG["device"])
    print("   Model loaded successfully!")
    
    # Step 2: Preprocess input MRI
    print(f"2. Preprocessing input from {CONFIG['input_mri_path']}...")
    input_tensor, input_image = preprocess_mri(
        CONFIG["input_mri_path"], 
        CONFIG["image_size"]
    )
    input_tensor = input_tensor.to(CONFIG["device"])
    print("   Input preprocessing complete!")
    
    # Step 3: Run inference (basic prediction)
    print("3. Running segmentation prediction...")
    with torch.no_grad():  # Disable gradient calculation for speed
        output = model(input_tensor)  # Shape: (1, 4, 256, 256)
    
    # Convert model output to class indices (0-3)
    pred_indices = torch.argmax(output, dim=1).squeeze().cpu().numpy()  # Shape: (256, 256)
    print("   Prediction complete!")
    
    # Temporary: Print prediction stats
    unique_classes = np.unique(pred_indices)
    print(f"   Detected tissue classes: {unique_classes} (0=BG, 1=CSF, 2=GM, 3=WM)")

if __name__ == "__main__":
    main()
    