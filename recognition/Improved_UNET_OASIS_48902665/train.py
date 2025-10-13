import os
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

# Import project components
from dataset import create_dataloader  # Data loading pipeline
from modules import ImprovedUNET       # Segmentation model
from utils import DiceLoss, dice_score # Loss/metric functions

# Training configuration - adjust based on experiment needs
CONFIG = {
    "device": "cuda" if torch.cuda.is_available() else "cpu",  # Auto-select GPU/CPU
    "epochs": 5,                   # Total training cycles
    "batch_size": 8,                # Samples per batch (GPU-memory dependent)
    "lr": 1e-4,                     # Initial learning rate
    "weight_decay": 1e-5,           # L2 regularization to prevent overfitting
    "model_save_path": "trained_unet.pth",  # Best model checkpoint
    "plot_save_path": "training_curves.png" # Loss/metric visualization
}


    