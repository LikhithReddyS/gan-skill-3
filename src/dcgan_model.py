"""
DCGAN Model module.
Wraps Generator and Discriminator into a unified DCGAN class.
"""

import sys
import torch
import torch.nn as nn
from typing import Tuple, Optional
from pathlib import Path

# Add src directory to path for imports when running standalone
sys.path.insert(0, str(Path(__file__).parent))

from generator import Generator, get_generator
from discriminator import Discriminator, get_discriminator


class DCGAN(nn.Module):
    """
    Deep Convolutional Generative Adversarial Network (DCGAN).
    
    Wraps the Generator and Discriminator networks and provides
    utilities for training, saving, and loading.
    """
    
    def __init__(
        self,
        latent_dim: int = 100,
        g_feature_maps: int = 64,
        d_feature_maps: int = 64,
        channels: int = 3,
        dropout: float = 0.25
    ):
        """
        Initialize the DCGAN.
        
        Args:
            latent_dim: Dimension of the latent vector z.
            g_feature_maps: Base feature maps for Generator.
            d_feature_maps: Base feature maps for Discriminator.
            channels: Number of image channels.
            dropout: Dropout probability for Discriminator.
        """
        super(DCGAN, self).__init__()
        
        self.latent_dim = latent_dim
        self.channels = channels
        
        # Create Generator and Discriminator
        self.generator = get_generator(
            latent_dim=latent_dim,
            feature_maps=g_feature_maps,
            channels=channels
        )
        
        self.discriminator = get_discriminator(
            channels=channels,
            feature_maps=d_feature_maps,
            dropout=dropout
        )
    
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through both networks.
        
        Args:
            z: Latent vector of shape (batch_size, latent_dim).
            
        Returns:
            Tuple of (generated_images, discriminator_output).
        """
        fake_images = self.generator(z)
        d_output = self.discriminator(fake_images)
        return fake_images, d_output
    
    def generate(self, num_samples: int, device: Optional[torch.device] = None) -> torch.Tensor:
        """
        Generate images from random latent vectors.
        
        Args:
            num_samples: Number of images to generate.
            device: Device to use for generation.
            
        Returns:
            Generated images of shape (num_samples, channels, 64, 64).
        """
        if device is None:
            device = next(self.parameters()).device
        
        z = torch.randn(num_samples, self.latent_dim, device=device)
        return self.generator(z)
    
    def save(self, checkpoint_dir: str, epoch: int):
        """
        Save model checkpoints.
        
        Args:
            checkpoint_dir: Directory to save checkpoints.
            epoch: Current epoch number.
        """
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Save generator
        g_path = checkpoint_dir / f"G_epoch_{epoch:03d}.pt"
        torch.save(self.generator.state_dict(), g_path)
        
        # Save discriminator
        d_path = checkpoint_dir / f"D_epoch_{epoch:03d}.pt"
        torch.save(self.discriminator.state_dict(), d_path)
        
        print(f"Saved checkpoints at epoch {epoch}")
    
    def load(self, checkpoint_dir: str, epoch: int, device: Optional[torch.device] = None):
        """
        Load model checkpoints.
        
        Args:
            checkpoint_dir: Directory containing checkpoints.
            epoch: Epoch number to load.
            device: Device to load the model on.
        """
        checkpoint_dir = Path(checkpoint_dir)
        
        # Load generator
        g_path = checkpoint_dir / f"G_epoch_{epoch:03d}.pt"
        self.generator.load_state_dict(torch.load(g_path, map_location=device))
        
        # Load discriminator
        d_path = checkpoint_dir / f"D_epoch_{epoch:03d}.pt"
        self.discriminator.load_state_dict(torch.load(d_path, map_location=device))
        
        print(f"Loaded checkpoints from epoch {epoch}")
    
    def count_parameters(self) -> Tuple[int, int]:
        """
        Count model parameters.
        
        Returns:
            Tuple of (generator_params, discriminator_params).
        """
        g_params = sum(p.numel() for p in self.generator.parameters())
        d_params = sum(p.numel() for p in self.discriminator.parameters())
        return g_params, d_params
    
    def summary(self):
        """Print model summary."""
        g_params, d_params = self.count_parameters()
        print("=" * 60)
        print("DCGAN Model Summary")
        print("=" * 60)
        print(f"Latent dimension: {self.latent_dim}")
        print(f"Image channels: {self.channels}")
        print("-" * 60)
        print(f"Generator parameters: {g_params:,}")
        print(f"Discriminator parameters: {d_params:,}")
        print(f"Total parameters: {g_params + d_params:,}")
        print("=" * 60)


def get_dcgan(
    latent_dim: int = 100,
    g_feature_maps: int = 64,
    d_feature_maps: int = 64,
    channels: int = 3,
    dropout: float = 0.25
) -> DCGAN:
    """
    Factory function to create a DCGAN.
    
    Args:
        latent_dim: Dimension of the latent vector.
        g_feature_maps: Base feature maps for Generator.
        d_feature_maps: Base feature maps for Discriminator.
        channels: Number of image channels.
        dropout: Dropout probability for Discriminator.
        
    Returns:
        DCGAN instance.
    """
    return DCGAN(
        latent_dim=latent_dim,
        g_feature_maps=g_feature_maps,
        d_feature_maps=d_feature_maps,
        channels=channels,
        dropout=dropout
    )


if __name__ == "__main__":
    # Test the DCGAN model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create DCGAN
    dcgan = get_dcgan().to(device)
    
    # Print summary
    dcgan.summary()
    
    # Test generation
    print("\nTesting generation...")
    fake_images = dcgan.generate(4, device)
    print(f"Generated images shape: {fake_images.shape}")
    print(f"Generated images range: [{fake_images.min():.3f}, {fake_images.max():.3f}]")
    
    # Test forward pass
    print("\nTesting forward pass...")
    z = torch.randn(4, 100, device=device)
    fake_images, d_output = dcgan(z)
    print(f"Fake images shape: {fake_images.shape}")
    print(f"Discriminator output shape: {d_output.shape}")
    print(f"Discriminator output values: {d_output.squeeze()}")
