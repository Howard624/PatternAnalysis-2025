import os
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from modules import ImprovedUNET

# Configuration - Updated with your specific MRI path
CONFIG = {
    "model_path": "trained_unet.pth",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "input_mri_path": r"D:\3710_Pattern_Recognition_Project\OASIS_dataset\OASIS\keras_png_slices_test\case_441_slice_0.nii.png",  # Your MRI path
    "image_size": (256, 256),
    "output_plot_path": "segmentation_result.png",
    "class_colors": {
        0: [0, 0, 0],      # Background (Black)
        1: [0, 0, 255],    # CSF (Blue)
        2: [0, 255, 0],    # Gray Matter (Green)
        3: [255, 0, 0]     # White Matter (Red)
    },
    "class_names": {
        0: "Background",
        1: "CSF",
        2: "Gray Matter",
        3: "White Matter"
    }
}

def load_trained_model(model_path, device):
    """Load trained UNET model and set to evaluation mode"""
    model = ImprovedUNET(in_channels=1, out_channels=4)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"✅ Model loaded successfully on {device}")
    return model

def preprocess_mri(image_path, target_size):
    """Preprocess MRI for model input (resize + normalize)"""
    # Load as grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"❌ Could not load MRI from {image_path}. Check if the path is correct.")
    
    # Resize to model input size
    resized = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
    
    # Normalize to [0, 1] range
    normalized = resized / 255.0
    
    # Convert to tensor with batch/channel dimensions
    tensor = torch.tensor(normalized, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    print(f"✅ Preprocessed MRI: {img.shape} → {resized.shape}")
    return tensor, resized

def color_segmentation_mask(pred_indices, class_colors):
    """Convert class indices (0-3) to colored segmentation mask"""
    # Initialize RGB image
    height, width = pred_indices.shape
    colored = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Apply colors for each class
    for class_idx, color in class_colors.items():
        colored[pred_indices == class_idx] = color
    
    return colored

def visualize_results(mri_image, pred_mask, class_colors, class_names, save_path):
    """Create side-by-side plot of input MRI and segmentation result"""
    plt.figure(figsize=(12, 6))
    
    # Plot 1: Input MRI
    plt.subplot(1, 2, 1)
    plt.imshow(mri_image, cmap="gray")
    plt.title("Input MRI Slice")
    plt.axis("off")
    
    # Plot 2: Segmentation Result
    plt.subplot(1, 2, 2)
    plt.imshow(pred_mask)
    plt.title("Tissue Segmentation")
    plt.axis("off")
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=tuple(c/255 for c in color), label=name)
        for name, color in zip(class_names.values(), class_colors.values())
    ]
    plt.legend(handles=legend_elements, loc="center left", bbox_to_anchor=(1, 0.5))
    
    # Save plot
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    print(f"✅ Visualization saved to {save_path}")

def main():
    print("=== Brain Tissue Segmentation Predictor ===")
    
    # 1. Load trained model
    model = load_trained_model(CONFIG["model_path"], CONFIG["device"])
    
    # 2. Preprocess input MRI
    input_tensor, preprocessed_img = preprocess_mri(
        CONFIG["input_mri_path"],
        CONFIG["image_size"]
    )
    input_tensor = input_tensor.to(CONFIG["device"])
    
    # 3. Run inference
    print("🔍 Running segmentation prediction...")
    with torch.no_grad():
        output = model(input_tensor)  # Shape: (1, 4, 256, 256)
    
    # Convert to class indices (0-3)
    pred_indices = torch.argmax(output, dim=1).squeeze().cpu().numpy()
    
    # 4. Generate colored segmentation mask
    colored_mask = color_segmentation_mask(pred_indices, CONFIG["class_colors"])
    
    # 5. Show detected tissues
    detected_classes = np.unique(pred_indices)
    print("\nDetected Tissues:")
    for cls in detected_classes:
        print(f"  - {CONFIG['class_names'][cls]}")
    
    # 6. Save visualization
    visualize_results(
        mri_image=preprocessed_img,
        pred_mask=colored_mask,
        class_colors=CONFIG["class_colors"],
        class_names=CONFIG["class_names"],
        save_path=CONFIG["output_plot_path"]
    )

if __name__ == "__main__":
    main()
    