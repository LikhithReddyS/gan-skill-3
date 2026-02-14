"""
Generator module for DCGAN.
Implements the Generator network that transforms latent vectors into images.
"""

import torch
import torch.nn as nn


class Generator(nn.Module):
    """
    DCGAN Generator Network.
    
    Transforms a latent vector z of shape (batch_size, latent_dim) into 
    an image of shape (batch_size, 3, 64, 64) in range [-1, 1].
    
    Architecture:
        Input: z (batch_size, latent_dim)
        -> Dense + Reshape to (batch_size, 512, 4, 4)
        -> ConvTranspose2d(512, 256) + BatchNorm + ReLU -> (batch_size, 256, 8, 8)
        -> ConvTranspose2d(256, 128) + BatchNorm + ReLU -> (batch_size, 128, 16, 16)
        -> ConvTranspose2d(128, 64) + BatchNorm + ReLU -> (batch_size, 64, 32, 32)
        -> ConvTranspose2d(64, 3) + Tanh -> (batch_size, 3, 64, 64)
    """
    
    def __init__(
        self,
        latent_dim: int = 100,
        feature_maps: int = 64,
        channels: int = 3     
    ):
        """
        Initialize the Generator.
        
        Args:
            latent_dim: Dimension of the latent vector z.
            feature_maps: Base number of feature maps (multiplied in layers).
            channels: Number of output image channels (3 for RGB).
        """
        super(Generator, self).__init__()
        
        self.latent_dim = latent_dim
        self.feature_maps = feature_maps
        self.channels = channels
        
        # Initial projection from latent space
        self.project = nn.Sequential(
            nn.Linear(latent_dim, 512 * 4 * 4),
            nn.BatchNorm1d(512 * 4 * 4),
            nn.ReLU(True)
        )
        
        # Convolutional layers
        self.conv_layers = nn.Sequential(
            # Input: (512, 4, 4) -> Output: (256, 8, 8)
            nn.ConvTranspose2d(512, feature_maps * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.ReLU(True),
            
            # Input: (256, 8, 8) -> Output: (128, 16, 16)
            nn.ConvTranspose2d(feature_maps * 4, feature_maps * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.ReLU(True),
            
            # Input: (128, 16, 16) -> Output: (64, 32, 32)
            nn.ConvTranspose2d(feature_maps * 2, feature_maps, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps),
            nn.ReLU(True),
            
            # Input: (64, 32, 32) -> Output: (3, 64, 64)
            nn.ConvTranspose2d(feature_maps, channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights using normal distribution."""
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(module.weight, 0.0, 0.02)
        elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
            nn.init.normal_(module.weight, 1.0, 0.02)
            nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, 0.0, 0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Generate images from latent vectors.
        
        Args:
            z: Latent vector of shape (batch_size, latent_dim).
            
        Returns:
            Generated images of shape (batch_size, channels, 64, 64).
        """
        # Project and reshape
        x = self.project(z)
        x = x.view(-1, 512, 4, 4)
        
        # Upsample through conv layers
        x = self.conv_layers(x)
        
        return x
    
    def generate(self, num_samples: int, device: torch.device = None) -> torch.Tensor:
        """
        Generate random samples.
        
        Args:
            num_samples: Number of samples to generate.
            device: Device to generate samples on.
            
        Returns:
            Generated images of shape (num_samples, channels, 64, 64).
        """
        if device is None:
            device = next(self.parameters()).device
        
        z = torch.randn(num_samples, self.latent_dim, device=device)
        return self.forward(z)


def get_generator(latent_dim: int = 100, feature_maps: int = 64, channels: int = 3) -> Generator:
    """
    Factory function to create a Generator.
    
    Args:
        latent_dim: Dimension of the latent vector.
        feature_maps: Base number of feature maps.
        channels: Number of output channels.
        
    Returns:
        Generator instance.
    """
    return Generator(latent_dim=latent_dim, feature_maps=feature_maps, channels=channels)


if __name__ == "__main__":
    # Test the generator
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create generator
    generator = Generator(latent_dim=100).to(device)
    
    # Print model summary
    print("\nGenerator Architecture:")
    print(generator)
    
    # Count parameters
    total_params = sum(p.numel() for p in generator.parameters())
    trainable_params = sum(p.numel() for p in generator.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Test forward pass
    z = torch.randn(4, 100, device=device)
    output = generator(z)
    print(f"\nInput shape: {z.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")


