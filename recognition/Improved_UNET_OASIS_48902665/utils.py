import torch
import torch.nn.functional as F

class DiceLoss(torch.nn.Module):
    """
    Dice Loss for segmentation tasks (optional BCE combination).
    Dice measures overlap between predicted and target masks (1 = perfect overlap).
    Includes smoothing to avoid division by zero.
    """
    def __init__(self, include_bce=False, bce_weight=0.5, smooth=1e-6):
        super().__init__()
        self.include_bce = include_bce  # Combine with BCE loss if True
        self.bce_weight = bce_weight    # Weight for BCE component
        self.smooth = smooth            # Smoothing factor

    def forward(self, preds, targets):
        # Flatten predictions and targets to 1D tensors
        preds = preds.view(-1)
        targets = targets.view(-1)
        
        # Calculate Dice coefficient
        intersection = (preds * targets).sum()
        dice = (2.0 * intersection + self.smooth) / (preds.sum() + targets.sum() + self.smooth)
        dice_loss = 1.0 - dice  # Minimize loss → maximize overlap
        
        # Add BCE loss for better gradient flow if enabled
        if self.include_bce:
            bce_loss = F.binary_cross_entropy(preds, targets)
            return (self.bce_weight * bce_loss) + ((1 - self.bce_weight) * dice_loss)
        
        return dice_loss

def dice_score(preds, targets, threshold=0.5, smooth=1e-6):
    """
    Calculate Dice score (metric for segmentation quality).
    Args:
        preds: Model outputs (probabilities)
        targets: Ground-truth masks
        threshold: Cutoff for converting probabilities to binary (0/1) masks
    Returns:
        Score between 0 (no overlap) and 1 (perfect overlap)
    """
    # Convert probabilities to binary masks using threshold
    preds = (preds > threshold).float()
    
    # Flatten tensors for easy calculation
    preds = preds.view(-1)
    targets = targets.view(-1)
    
    # Compute Dice score
    intersection = (preds * targets).sum()
    return (2.0 * intersection + smooth) / (preds.sum() + targets.sum() + smooth)
    