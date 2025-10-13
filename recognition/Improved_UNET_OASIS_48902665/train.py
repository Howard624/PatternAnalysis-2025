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
    "epochs": 1,                   # Total training cycles
    "batch_size": 12,                # Samples per batch (GPU-memory dependent)
    "lr": 1.5e-4,                     # Initial learning rate
    "weight_decay": 1e-5,           # L2 regularization to prevent overfitting
    "model_save_path": "trained_unet.pth",  # Best model checkpoint
    "plot_save_path": "training_curves.png" # Loss/metric visualization
}

def verify_gpu_usage():
    """Verify GPU availability and print details"""
    if CONFIG["device"] == "cuda":
        print(f"✅ GPU detected: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA version: {torch.version.cuda}")
        print(f"   Total memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        print("⚠️ No GPU detected. Training will use CPU (this may be slow).")

def init_training():
    """Initialize model, loss, optimizer, and data loaders"""
    # Verify GPU usage first
    verify_gpu_usage()

    # Initialize model and move to target device
    model = ImprovedUNET(in_channels=1, out_channels=1).to(CONFIG["device"])
    
    # Loss function: Dice + BCE (balances overlap and pixel-wise errors)
    criterion = DiceLoss(include_bce=True, bce_weight=0.5)
    
    # Optimizer with regularization
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=CONFIG["lr"],
        weight_decay=CONFIG["weight_decay"]
    )
    
    # Learning rate scheduler (reduces LR on validation plateau)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    
    # Initialize data loaders for train/val/test splits
    train_loader = create_dataloader("train", CONFIG["batch_size"], shuffle=True)
    val_loader = create_dataloader("validate", CONFIG["batch_size"], shuffle=False)
    test_loader = create_dataloader("test", CONFIG["batch_size"], shuffle=False)
    
    return model, criterion, optimizer, scheduler, train_loader, val_loader, test_loader

def train_one_epoch(model, criterion, optimizer, dataloader, device):
    """Train model for one epoch; return avg loss and Dice score"""
    model.train()  # Enable training mode (batch norm/dropout active)
    total_loss, total_dice = 0.0, 0.0
    
    # Iterate over batches with progress tracking
    for images, masks in tqdm(dataloader, desc="Training"):
        images, masks = images.to(device), masks.to(device)
        
        # Forward pass: predict masks
        outputs = model(images)
        loss = criterion(outputs, masks)
        
        # Backward pass: update weights
        optimizer.zero_grad()  # Reset gradients
        loss.backward()        # Compute gradients
        optimizer.step()       # Update parameters
        
        # Accumulate metrics (weighted by batch size)
        total_loss += loss.item() * images.size(0)
        total_dice += dice_score(outputs, masks) * images.size(0)
    
    # Return averages over entire dataset
    return total_loss / len(dataloader.dataset), total_dice / len(dataloader.dataset)

def validate_one_epoch(model, criterion, dataloader, device):
    """Validate model for one epoch (no training); return avg loss and Dice score"""
    model.eval()  # Disable training-specific layers
    total_loss, total_dice = 0.0, 0.0
    
    # No gradient calculation for efficiency
    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc="Validation"):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            # Accumulate metrics
            total_loss += loss.item() * images.size(0)
            total_dice += dice_score(outputs, masks) * images.size(0)
    
    return total_loss / len(dataloader.dataset), total_dice / len(dataloader.dataset)

def test_model(model, criterion, dataloader, device):
    """Evaluate best model on test set; print final metrics"""
    model.eval()
    total_loss, total_dice = 0.0, 0.0
    
    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc="Testing"):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            total_loss += loss.item() * images.size(0)
            total_dice += dice_score(outputs, masks) * images.size(0)
    
    # Calculate and print test metrics
    avg_loss = total_loss / len(dataloader.dataset)
    avg_dice = total_dice / len(dataloader.dataset)
    print(f"\nTest Results - Loss: {avg_loss:.4f}, Dice: {avg_dice:.4f}")
    return avg_loss, avg_dice

def plot_curves(train_losses, val_losses, train_dices, val_dices, save_path):
    """Plot training/validation loss and Dice curves for visualization"""
    plt.figure(figsize=(12, 5))
    
    # Loss curve (lower = better)
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train")
    plt.plot(val_losses, label="Validation")
    plt.title("Loss Curves")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    
    # Dice score curve (higher = better segmentation)
    plt.subplot(1, 2, 2)
    plt.plot(train_dices, label="Train")
    plt.plot(val_dices, label="Validation")
    plt.title("Dice Score")
    plt.xlabel("Epoch")
    plt.ylabel("Score (0-1)")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Curves saved to {save_path}")

def main():
    """Main training pipeline: initialize → train → validate → test → visualize"""
    # Initialize components
    model, criterion, optimizer, scheduler, train_loader, val_loader, test_loader = init_training()
    
    # Track metrics across epochs
    train_losses, val_losses = [], []
    train_dices, val_dices = [], []
    best_val_dice = 0.0  # Track best validation performance
    
    print("Starting training...")
    for epoch in range(1, CONFIG["epochs"] + 1):
        print(f"\nEpoch {epoch}/{CONFIG['epochs']}")
        
        # Run training and validation
        train_loss, train_dice = train_one_epoch(model, criterion, optimizer, train_loader, CONFIG["device"])
        val_loss, val_dice = validate_one_epoch(model, criterion, val_loader, CONFIG["device"])
        
        # Save metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_dices.append(train_dice)
        val_dices.append(val_dice)
        
        # Adjust learning rate based on validation loss
        scheduler.step(val_loss)
        
        # Print epoch summary
        print(f"Train - Loss: {train_loss:.4f}, Dice: {train_dice:.4f}")
        print(f"Val   - Loss: {val_loss:.4f}, Dice: {val_dice:.4f}")
        
        # Save best model (highest validation Dice)
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save(model.state_dict(), CONFIG["model_save_path"])
            print(f"Saved best model (Val Dice: {best_val_dice:.4f})")
    
    # Generate and save training curves
    plot_curves(train_losses, val_losses, train_dices, val_dices, CONFIG["plot_save_path"])
    
    # Evaluate best model on test set
    best_model = ImprovedUNET().to(CONFIG["device"])
    best_model.load_state_dict(torch.load(CONFIG["model_save_path"]))
    test_model(best_model, criterion, test_loader, CONFIG["device"])

if __name__ == "__main__":
    main()
    