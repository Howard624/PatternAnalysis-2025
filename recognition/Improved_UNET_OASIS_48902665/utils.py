import torch
import torch.nn.functional as F

# KEEP ONLY THIS: Multi-class Dice score (for 4 brain tissues)
def multi_class_dice_score(predictions, targets, num_classes=4, epsilon=1e-6):
    """
    Calculate mean Dice score for 4-class brain segmentation (Background, CSF, Gray Matter, White Matter).
    
    Args:
        predictions: Model output (4-channel probabilities) → Shape: (batch, 4, H, W)
        targets: Normalized target masks (1-channel, values 0.0/0.3333/0.6667/1.0) → Shape: (batch, 1, H, W)
        num_classes: Fixed to 4 (matches OASIS dataset)
        epsilon: Small value to avoid division by zero
    
    Returns:
        mean_dice: Average Dice score across all 4 classes (range: 0-1)
    """
    # Convert model predictions to class indices (0-3)
    pred_indices = torch.argmax(predictions, dim=1, keepdim=True)
    
    # Convert normalized targets to class indices (0-3)
    target_indices = torch.round(targets * 3).long()
    
    # One-hot encode both predictions and targets (4 channels)
    pred_one_hot = F.one_hot(pred_indices.squeeze(1), num_classes=num_classes).permute(0, 3, 1, 2).float()
    target_one_hot = F.one_hot(target_indices.squeeze(1), num_classes=num_classes).permute(0, 3, 1, 2).float()
    
    # Calculate Dice per class
    class_dice = []
    for class_idx in range(num_classes):
        pred_class = pred_one_hot[:, class_idx, :, :]
        target_class = target_one_hot[:, class_idx, :, :]
        
        intersection = torch.sum(pred_class * target_class, dim=[1, 2])
        union = torch.sum(pred_class, dim=[1, 2]) + torch.sum(target_class, dim=[1, 2])
        
        dice_per_batch = (2 * intersection + epsilon) / (union + epsilon)
        class_dice.append(torch.mean(dice_per_batch))
    
    # Average across all 4 classes
    mean_dice = torch.mean(torch.tensor(class_dice))
    return mean_dice

# Optional: Keep other utilities (e.g., visualization functions) if you use them
# def plot_segmentation(input_img, pred_mask, true_mask):
#     # ... (if you need this for debugging)