import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------
# Reusable Component: Residual Block
# ------------------------------
class ResidualBlock(nn.Module):
    """
    Residual Block with Batch Normalization (Improvement over standard conv blocks)
    
    Purpose:
        - Mitigates vanishing gradients in deep networks by adding residual connections
        - Stabilizes training with batch normalization
        - Maintains or increases feature map depth through two 3x3 convolutions
    
    Architecture:
        Input → Conv2d → BatchNorm → ReLU → Conv2d → BatchNorm → Add(Input) → ReLU
        (Shortcut connection adjusts dimensions if input/output channels/stride differ)
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        # First convolution: reduces spatial dimensions (if stride > 1) and adjusts channels
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, 
            stride=stride, padding=1, bias=False  # Bias unnecessary with BatchNorm
        )
        self.bn1 = nn.BatchNorm2d(out_channels)  # Stabilizes training distribution
        
        # Second convolution: refines features without changing spatial dimensions
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, 
            stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut connection: matches input dimensions to output dimensions if needed
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                # 1x1 convolution to adjust channels/stride efficiently
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        # Main path: two conv-batchnorm-relu blocks
        out = F.relu(self.bn1(self.conv1(x)))  # First conv + normalization + activation
        out = self.bn2(self.conv2(out))        # Second conv + normalization (no activation yet)
        
        # Residual connection: add original input (or adjusted shortcut) to output
        out += self.shortcut(x)
        out = F.relu(out)  # Final activation after residual addition
        
        return out


# ------------------------------
# Main Model: Improved UNET for MRI Segmentation
# ------------------------------
class ImprovedUNET(nn.Module):
    """
    Improved UNET Architecture for Medical Image Segmentation
    
    Key Enhancements Over Basic UNET:
        - Uses ResidualBlocks instead of standard conv blocks (deeper networks possible)
        - Integrates Batch Normalization (faster convergence, better stability)
        - Maintains critical skip connections between encoder and decoder (preserves fine details)
    
    Input: Grayscale MRI slice (shape: [batch_size, 1, 256, 256])
    Output: 4-channel segmentation mask (shape: [batch_size, 4, 256, 256])
            Each channel corresponds to a tissue class probability
    """
    def __init__(self, in_channels=1, out_channels=4):  # Updated to 4 output channels
        super(ImprovedUNET, self).__init__()
        # ------------------------------
        # Encoder (Downsampling Path)
        # Purpose: Capture context and reduce spatial dimensions
        # ------------------------------
        self.enc1 = ResidualBlock(in_channels, 64)  # Input (1→64 channels, 256x256)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # Downsample to 128x128
        
        self.enc2 = ResidualBlock(64, 128)  # 64→128 channels, 128x128
        self.pool2 = nn.MaxPool2d(2, 2)  # Downsample to 64x64
        
        self.enc3 = ResidualBlock(128, 256)  # 128→256 channels, 64x64
        self.pool3 = nn.MaxPool2d(2, 2)  # Downsample to 32x32
        
        self.enc4 = ResidualBlock(256, 512)  # 256→512 channels, 32x32
        self.pool4 = nn.MaxPool2d(2, 2)  # Downsample to 16x16
        
        # ------------------------------
        # Bottleneck (Most Compressed Features)
        # Purpose: Capture high-level context with maximum channel depth
        # ------------------------------
        self.bottleneck = ResidualBlock(512, 1024)  # 512→1024 channels, 16x16
        
        # ------------------------------
        # Decoder (Upsampling Path)
        # Purpose: Recover spatial resolution and combine with encoder details via skip connections
        # ------------------------------
        # Upsample + merge with enc4 features (512 channels)
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)  # 16x16→32x32
        self.dec4 = ResidualBlock(1024, 512)  # 512 (upconv) + 512 (enc4) → 1024→512
        
        # Upsample + merge with enc3 features (256 channels)
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)  # 32x32→64x64
        self.dec3 = ResidualBlock(512, 256)  # 256 + 256 → 512→256
        
        # Upsample + merge with enc2 features (128 channels)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)  # 64x64→128x128
        self.dec2 = ResidualBlock(256, 128)  # 128 + 128 → 256→128
        
        # Upsample + merge with enc1 features (64 channels)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)  # 128x128→256x256
        self.dec1 = ResidualBlock(128, 64)  # 64 + 64 → 128→64
        
        # ------------------------------
        # Final Output Layer
        # Purpose: Generate segmentation mask matching input dimensions
        # ------------------------------
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)  # 64→4 channels (one per class)
        # Removed sigmoid - using CrossEntropyLoss which includes softmax

    def forward(self, x):
        """Forward pass through the network: encoder → bottleneck → decoder → output"""
        # Encoder: capture features at multiple scales
        enc1 = self.enc1(x)               # (1, 256, 256) → (64, 256, 256)
        enc2 = self.enc2(self.pool1(enc1))  # (64, 256, 256) → (128, 128, 128)
        enc3 = self.enc3(self.pool2(enc2))  # (128, 128, 128) → (256, 64, 64)
        enc4 = self.enc4(self.pool3(enc3))  # (256, 64, 64) → (512, 32, 32)
        
        # Bottleneck: highest-level features
        bottleneck = self.bottleneck(self.pool4(enc4))  # (512, 32, 32) → (1024, 16, 16)
        
        # Decoder: upsample and merge with encoder features via skip connections
        dec4 = self.upconv4(bottleneck)              # (1024, 16, 16) → (512, 32, 32)
        dec4 = torch.cat([dec4, enc4], dim=1)        # Merge with enc4 → (1024, 32, 32)
        dec4 = self.dec4(dec4)                       # → (512, 32, 32)
        
        dec3 = self.upconv3(dec4)                    # (512, 32, 32) → (256, 64, 64)
        dec3 = torch.cat([dec3, enc3], dim=1)        # Merge with enc3 → (512, 64, 64)
        dec3 = self.dec3(dec3)                       # → (256, 64, 64)
        
        dec2 = self.upconv2(dec3)                    # (256, 64, 64) → (128, 128, 128)
        dec2 = torch.cat([dec2, enc2], dim=1)        # Merge with enc2 → (256, 128, 128)
        dec2 = self.dec2(dec2)                       # → (128, 128, 128)
        
        dec1 = self.upconv1(dec2)                    # (128, 128, 128) → (64, 256, 256)
        dec1 = torch.cat([dec1, enc1], dim=1)        # Merge with enc1 → (128, 256, 256)
        dec1 = self.dec1(dec1)                       # → (64, 256, 256)
        
        # Generate final segmentation mask (4 channels for 4 classes)
        out = self.final_conv(dec1)  # (64, 256, 256) → (4, 256, 256)
        
        return out


# ------------------------------
# Model Validation (Run Directly)
# ------------------------------
if __name__ == "__main__":
    # Test with sample input matching dataset output: (batch_size=2, channels=1, height=256, width=256)
    sample_input = torch.randn(2, 1, 256, 256)
    print(f"Testing ImprovedUNET with input shape: {sample_input.shape}")
    
    # Initialize model with 4 output channels
    model = ImprovedUNET(in_channels=1, out_channels=4)
    
    # Perform forward pass (verify no errors and correct output)
    with torch.no_grad():  # Disable gradient calculation for speed
        output = model(sample_input)
    
    # Validate output properties
    expected_shape = (2, 4, 256, 256)
    assert output.shape == expected_shape, \
        f"Output shape mismatch! Expected {expected_shape}, got {output.shape}"
    
    print(f"Validation passed! Output shape: {output.shape}")
    print("ImprovedUNET (4-class) is ready for training.")
    