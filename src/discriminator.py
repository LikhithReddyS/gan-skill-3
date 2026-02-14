"""
Discriminator module for DCGAN.
Implements the Discriminator network that classifies images as real or fake.
"""

import torch
import torch.nn as nn


class Discriminator(nn.Module):
    """
    DCGAN Discriminator Network.
    
    Classifies images of shape (batch_size, 3, 64, 64) as real or fake,
    outputting a probability in range [0, 1].
    
    Architecture:
        Input: x (batch_size, 3, 64, 64)
        -> Conv2d(3, 64) + LeakyReLU -> (batch_size, 64, 32, 32)
        -> Conv2d(64, 128) + BatchNorm + LeakyReLU + Dropout -> (batch_size, 128, 16, 16)
        -> Conv2d(128, 256) + BatchNorm + LeakyReLU + Dropout -> (batch_size, 256, 8, 8)
        -> Conv2d(256, 512) + BatchNorm + LeakyReLU + Dropout -> (batch_size, 512, 4, 4)
        -> Conv2d(512, 1) + Sigmoid -> (batch_size, 1, 1, 1)
    """
    
    def __init__(
        self,
        channels: int = 3,
        feature_maps: int = 64,
        dropout: float = 0.25
    ):
        """
        Initialize the Discriminator.
        
        Args:
            channels: Number of input image channels (3 for RGB).
            feature_maps: Base number of feature maps (multiplied in layers).
            dropout: Dropout probability.
        """
        super(Discriminator, self).__init__()
        
        self.channels = channels
        self.feature_maps = feature_maps
        
        self.conv_layers = nn.Sequential(
            # Input: (3, 64, 64) -> Output: (64, 32, 32)
            nn.Conv2d(channels, feature_maps, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Input: (64, 32, 32) -> Output: (128, 16, 16)
            nn.Conv2d(feature_maps, feature_maps * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(dropout),
            
            # Input: (128, 16, 16) -> Output: (256, 8, 8)
            nn.Conv2d(feature_maps * 2, feature_maps * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(dropout),
            
            # Input: (256, 8, 8) -> Output: (512, 4, 4)
            nn.Conv2d(feature_maps * 4, feature_maps * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(dropout),
            
            # Input: (512, 4, 4) -> Output: (1, 1, 1)
            nn.Conv2d(feature_maps * 8, 1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights using normal distribution."""
        if isinstance(module, nn.Conv2d):
            nn.init.normal_(module.weight, 0.0, 0.02)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.normal_(module.weight, 1.0, 0.02)
            nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Classify images as real or fake.
        
        Args:
            x: Input images of shape (batch_size, channels, 64, 64).
            
        Returns:
            Probability of being real, shape (batch_size, 1, 1, 1).
        """
        return self.conv_layers(x)
    
    def classify(self, x: torch.Tensor) -> torch.Tensor:
        """
        Classify images and return flattened probabilities.
        
        Args:
            x: Input images of shape (batch_size, channels, 64, 64).
            
        Returns:
            Probability of being real, shape (batch_size,).
        """
        return self.forward(x).view(-1)


def get_discriminator(channels: int = 3, feature_maps: int = 64, dropout: float = 0.25) -> Discriminator:
    """
    Factory function to create a Discriminator.
    
    Args:
        channels: Number of input channels.
        feature_maps: Base number of feature maps.
        dropout: Dropout probability.
        
    Returns:
        Discriminator instance.
    """
    return Discriminator(channels=channels, feature_maps=feature_maps, dropout=dropout)


if __name__ == "__main__":
    # Test the discriminator
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create discriminator
    discriminator = Discriminator().to(device)
    
    # Print model summary
    print("\nDiscriminator Architecture:")
    print(discriminator)
    
    # Count parameters
    total_params = sum(p.numel() for p in discriminator.parameters())
    trainable_params = sum(p.numel() for p in discriminator.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Test forward pass
    x = torch.randn(4, 3, 64, 64, device=device)
    output = discriminator(x)
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output values: {output.squeeze()}")
