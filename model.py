import torch
from torch import nn

class DoubleConv(nn.Module):
    """
    Double Convolution Block - A building block used in U-Net architecture
    
    This module performs two consecutive convolution operations:
    1. Conv2d -> ReLU -> Conv2d -> ReLU
    
    This pattern is commonly used in U-Net to:
    - Extract features at each resolution level
    - Maintain spatial dimensions with padding=1
    - Apply non-linearity with ReLU activation
    
    Args:
        in_c (int): Number of input channels
        out_c (int): Number of output channels
    """
    def __init__(self, in_c, out_c):
        super().__init__()
        # Sequential container that applies operations in order
        self.net = nn.Sequential(
            # First convolution: 3x3 kernel, padding=1 keeps spatial dimensions same
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1), 
            nn.ReLU(inplace=True),  # ReLU activation (inplace=True saves memory)
            
            # Second convolution: 3x3 kernel, padding=1 keeps spatial dimensions same
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1), 
            nn.ReLU(inplace=True),  # ReLU activation
        )

    def forward(self, x):
        """
        Forward pass through the double convolution block
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_c, height, width)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_c, height, width)
        """
        return self.net(x)
    


class SimpleUNet(nn.Module):
    """
    Simplified U-Net Architecture for Lane Detection
    
    U-Net is a convolutional neural network architecture originally designed for 
    biomedical image segmentation. It consists of:
    
    1. ENCODER (Contracting Path): Captures context through downsampling
       - Applies convolutions and pooling to reduce spatial dimensions
       - Increases feature channels to capture more complex patterns
    
    2. DECODER (Expanding Path): Enables precise localization through upsampling
       - Uses transposed convolutions to increase spatial dimensions
       - Combines high-level features with low-level details via skip connections
    
    This simplified version has:
    - One encoding level (enc1 -> pool -> enc2)
    - One decoding level (upsample -> concatenate -> dec1)
    - Skip connection between encoder and decoder
    
    Perfect for lane detection as it:
    - Preserves spatial information needed for pixel-level lane classification
    - Combines global context (road structure) with local details (lane markings)
    """
    def __init__(self):
        super().__init__()
        
        # ENCODER LAYERS
        # First encoding block: RGB input (3 channels) -> 32 feature maps
        self.enc1 = DoubleConv(3, 32)
        
        # Pooling layer: Reduces spatial dimensions by half (downsampling)
        # MaxPool2d(2) means 2x2 kernel with stride=2, so H,W become H/2, W/2
        self.pool = nn.MaxPool2d(2)
        
        # Second encoding block: 32 -> 64 feature maps (deeper features)
        self.enc2 = DoubleConv(32, 64)
        
        # DECODER LAYERS
        # Transposed convolution (upsampling): 64 -> 32 channels, doubles spatial dimensions
        # ConvTranspose2d is the opposite of Conv2d - increases spatial dimensions
        self.up = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        
        # Decoding block: 64 -> 32 feature maps
        # Input is 64 channels because we concatenate upsampled features (32) with skip connection (32)
        self.dec1 = DoubleConv(64, 32)
        
        # OUTPUT LAYER
        # Final 1x1 convolution: 32 -> 1 channel (binary lane mask)
        # 1x1 kernel acts as a learnable classifier for each pixel
        self.outc = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x):
        """
        Forward pass through the U-Net architecture
        
        Data Flow:
        1. Input image (H, W, 3) goes through first encoder
        2. Pooling reduces dimensions to (H/2, W/2)
        3. Second encoder extracts deeper features
        4. Upsampling restores dimensions to (H, W)
        5. Skip connection concatenates low-level and high-level features
        6. Decoder processes combined features
        7. Output layer produces binary lane mask
        
        Args:
            x (torch.Tensor): Input image tensor of shape (batch_size, 3, height, width)
                             Expected to be RGB images normalized to [0,1] or [-1,1]
            
        Returns:
            torch.Tensor: Output lane mask of shape (batch_size, 1, height, width)
                         Values are logits (before sigmoid) for binary classification
        """
        
        # ENCODER PATH (Downsampling)
        # e1: First encoding - extracts low-level features (edges, textures)
        # Shape: (batch, 3, H, W) -> (batch, 32, H, W)
        e1 = self.enc1(x)
        
        # e2: Pool + Second encoding - extracts high-level features (shapes, patterns)
        # Shape: (batch, 32, H, W) -> (batch, 32, H/2, W/2) -> (batch, 64, H/2, W/2)
        e2 = self.enc2(self.pool(e1))
        
        # DECODER PATH (Upsampling)
        # d1: Upsample to restore spatial dimensions
        # Shape: (batch, 64, H/2, W/2) -> (batch, 32, H, W)
        d1 = self.up(e2)
        
        # SKIP CONNECTION: Concatenate upsampled features with encoder features
        # This preserves fine-grained spatial information lost during downsampling
        # torch.cat concatenates along channel dimension (dim=1)
        # Shape: [(batch, 32, H, W), (batch, 32, H, W)] -> (batch, 64, H, W)
        d1 = torch.cat([d1, e1], dim=1)
        
        # FINAL PROCESSING
        # Apply final decoder and output layer
        # Shape: (batch, 64, H, W) -> (batch, 32, H, W) -> (batch, 1, H, W)
        out = self.outc(self.dec1(d1))
        
        # Return logits (apply sigmoid during training/inference for probabilities)
        return out


        
