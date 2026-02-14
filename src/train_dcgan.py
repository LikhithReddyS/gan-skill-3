"""
Training script for DCGAN on cancer histopathology images.
Implements the full training pipeline with checkpointing, logging, and sample generation.
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image, make_grid
import numpy as np

# Add project root and src to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
SRC_DIR = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SRC_DIR))

from generator import Generator
from discriminator import Discriminator
from data_loader import get_dataloaders, create_sample_dataset
from utils.config import load_config
from utils.logger import Logger


class DCGANTrainer:
    """
    Trainer class for DCGAN.
    Handles the full training loop with stabilization tricks.
    """
    
    def __init__(
        self,
        data_dir: str = "data/breast_cancer",
        checkpoint_dir: str = "checkpoints",
        samples_dir: str = "samples",
        log_dir: str = "logs",
        latent_dim: int = 100,
        g_feature_maps: int = 64,
        d_feature_maps: int = 64,
        lr: float = 0.0002,
        beta1: float = 0.5,
        beta2: float = 0.999,
        batch_size: int = 64,
        image_size: int = 64,
        label_smoothing: float = 0.9,
        gradient_clip: float = 1.0,
        device: str = None
    ):
        """
        Initialize the DCGAN trainer.
        
        Args:
            data_dir: Directory containing the dataset.
            checkpoint_dir: Directory to save model checkpoints.
            samples_dir: Directory to save generated samples.
            log_dir: Directory to save training logs.
            latent_dim: Dimension of latent vector z.
            g_feature_maps: Base feature maps for Generator.
            d_feature_maps: Base feature maps for Discriminator.
            lr: Learning rate for Adam optimizer.
            beta1: Beta1 for Adam optimizer.
            beta2: Beta2 for Adam optimizer.
            batch_size: Training batch size.
            image_size: Input image size.
            label_smoothing: Real label value (< 1.0 for smoothing).
            gradient_clip: Maximum gradient norm for clipping.
            device: Device to train on ('cuda' or 'cpu').
        """
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        # Store parameters
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.image_size = image_size
        self.label_smoothing = label_smoothing
        self.gradient_clip = gradient_clip
        
        # Create directories
        self.checkpoint_dir = Path(checkpoint_dir)
        self.samples_dir = Path(samples_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.samples_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize models
        self.generator = Generator(
            latent_dim=latent_dim,
            feature_maps=g_feature_maps,
            channels=3
        ).to(self.device)
        
        self.discriminator = Discriminator(
            channels=3,
            feature_maps=d_feature_maps,
            dropout=0.25
        ).to(self.device)
        
        # Initialize optimizers
        self.g_optimizer = optim.Adam(
            self.generator.parameters(),
            lr=lr,
            betas=(beta1, beta2)
        )
        
        self.d_optimizer = optim.Adam(
            self.discriminator.parameters(),
            lr=lr,
            betas=(beta1, beta2)
        )
        
        # Loss function
        self.criterion = nn.BCELoss()
        
        # Initialize data loader
        self.data_dir = data_dir
        
        # Logger
        experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.logger = Logger(log_dir=log_dir, experiment_name=experiment_name)
        
        # Fixed noise for consistent sample generation
        self.fixed_noise = torch.randn(64, latent_dim, device=self.device)
        
        # Training history
        self.g_losses = []
        self.d_losses = []
        self.d_real_acc = []
        self.d_fake_acc = []
    
    def _get_labels(self, size: int, real: bool) -> torch.Tensor:
        """
        Create labels with optional smoothing.
        
        Args:
            size: Number of labels.
            real: Whether these are real labels.
            
        Returns:
            Label tensor.
        """
        if real:
            # Label smoothing: use 0.9 instead of 1.0
            labels = torch.full((size,), self.label_smoothing, device=self.device)
        else:
            labels = torch.zeros(size, device=self.device)
        return labels
    
    def train_discriminator(self, real_images: torch.Tensor) -> dict:
        """
        Train the discriminator for one step.
        
        Args:
            real_images: Batch of real images.
            
        Returns:
            Dictionary of discriminator metrics.
        """
        batch_size = real_images.size(0)
        
        # Zero gradients
        self.d_optimizer.zero_grad()
        
        # Train on real images
        real_labels = self._get_labels(batch_size, real=True)
        real_output = self.discriminator(real_images).view(-1)
        d_loss_real = self.criterion(real_output, real_labels)
        
        # Train on fake images
        noise = torch.randn(batch_size, self.latent_dim, device=self.device)
        fake_images = self.generator(noise)
        fake_labels = self._get_labels(batch_size, real=False)
        fake_output = self.discriminator(fake_images.detach()).view(-1)
        d_loss_fake = self.criterion(fake_output, fake_labels)
        
        # Total discriminator loss
        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        
        # Gradient clipping
        if self.gradient_clip > 0:
            nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.gradient_clip)
        
        self.d_optimizer.step()
        
        # Calculate accuracy
        real_acc = (real_output > 0.5).float().mean().item()
        fake_acc = (fake_output < 0.5).float().mean().item()
        
        return {
            "d_loss": d_loss.item(),
            "d_loss_real": d_loss_real.item(),
            "d_loss_fake": d_loss_fake.item(),
            "d_real_acc": real_acc,
            "d_fake_acc": fake_acc
        }
    
    def train_generator(self) -> dict:
        """
        Train the generator for one step.
        
        Returns:
            Dictionary of generator metrics.
        """
        # Zero gradients
        self.g_optimizer.zero_grad()
        
        # Generate fake images
        noise = torch.randn(self.batch_size, self.latent_dim, device=self.device)
        fake_images = self.generator(noise)
        
        # We want discriminator to think these are real
        real_labels = self._get_labels(self.batch_size, real=True)
        output = self.discriminator(fake_images).view(-1)
        
        # Generator loss
        g_loss = self.criterion(output, real_labels)
        g_loss.backward()
        
        # Gradient clipping
        if self.gradient_clip > 0:
            nn.utils.clip_grad_norm_(self.generator.parameters(), self.gradient_clip)
        
        self.g_optimizer.step()
        
        return {
            "g_loss": g_loss.item()
        }
    
    def save_samples(self, epoch: int, num_samples: int = 64):
        """
        Generate and save sample images.
        
        Args:
            epoch: Current epoch number.
            num_samples: Number of samples to generate.
        """
        self.generator.eval()
        with torch.no_grad():
            fake_images = self.generator(self.fixed_noise[:num_samples])
            # Denormalize from [-1, 1] to [0, 1]
            fake_images = (fake_images + 1) / 2
            
            # Save as grid
            grid = make_grid(fake_images, nrow=8, padding=2, normalize=False)
            save_path = self.samples_dir / f"samples_epoch_{epoch:03d}.png"
            save_image(grid, save_path)
            
        self.generator.train()
        print(f"Saved samples to {save_path}")
    
    def save_checkpoint(self, epoch: int):
        """
        Save model checkpoints.
        
        Args:
            epoch: Current epoch number.
        """
        # Save generator
        g_path = self.checkpoint_dir / f"G_epoch_{epoch:03d}.pt"
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.generator.state_dict(),
            'optimizer_state_dict': self.g_optimizer.state_dict(),
        }, g_path)
        
        # Save discriminator
        d_path = self.checkpoint_dir / f"D_epoch_{epoch:03d}.pt"
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.discriminator.state_dict(),
            'optimizer_state_dict': self.d_optimizer.state_dict(),
        }, d_path)
        
        print(f"Saved checkpoint at epoch {epoch}")
    
    def load_checkpoint(self, epoch: int):
        """
        Load model checkpoints.
        
        Args:
            epoch: Epoch number to load.
        """
        # Load generator
        g_path = self.checkpoint_dir / f"G_epoch_{epoch:03d}.pt"
        g_checkpoint = torch.load(g_path, map_location=self.device)
        self.generator.load_state_dict(g_checkpoint['model_state_dict'])
        self.g_optimizer.load_state_dict(g_checkpoint['optimizer_state_dict'])
        
        # Load discriminator
        d_path = self.checkpoint_dir / f"D_epoch_{epoch:03d}.pt"
        d_checkpoint = torch.load(d_path, map_location=self.device)
        self.discriminator.load_state_dict(d_checkpoint['model_state_dict'])
        self.d_optimizer.load_state_dict(d_checkpoint['optimizer_state_dict'])
        
        print(f"Loaded checkpoint from epoch {epoch}")
    
    def train(
        self,
        epochs: int = 100,
        save_interval: int = 10,
        sample_interval: int = 5,
        log_interval: int = 50
    ):
        """
        Train the DCGAN.
        
        Args:
            epochs: Number of epochs to train.
            save_interval: Save checkpoint every N epochs.
            sample_interval: Save samples every N epochs.
            log_interval: Log metrics every N batches.
        """
        print("=" * 60)
        print("Starting DCGAN Training")
        print("=" * 60)
        print(f"Epochs: {epochs}")
        print(f"Batch size: {self.batch_size}")
        print(f"Image size: {self.image_size}")
        print(f"Latent dim: {self.latent_dim}")
        print(f"Device: {self.device}")
        print("=" * 60)
        
        # Get data loaders
        train_loader, _, _ = get_dataloaders(
            data_dir=self.data_dir,
            image_size=self.image_size,
            batch_size=self.batch_size,
            num_workers=0  # Use 0 for Windows compatibility
        )
        
        print(f"Training batches per epoch: {len(train_loader)}")
        
        # Training loop
        global_step = 0
        for epoch in range(1, epochs + 1):
            epoch_d_loss = 0.0
            epoch_g_loss = 0.0
            epoch_d_real_acc = 0.0
            epoch_d_fake_acc = 0.0
            num_batches = 0
            
            for batch_idx, real_images in enumerate(train_loader):
                real_images = real_images.to(self.device)
                
                # Train discriminator
                d_metrics = self.train_discriminator(real_images)
                
                # Train generator
                g_metrics = self.train_generator()
                
                # Accumulate metrics
                epoch_d_loss += d_metrics["d_loss"]
                epoch_g_loss += g_metrics["g_loss"]
                epoch_d_real_acc += d_metrics["d_real_acc"]
                epoch_d_fake_acc += d_metrics["d_fake_acc"]
                num_batches += 1
                global_step += 1
                
                # Log every N batches
                if batch_idx % log_interval == 0:
                    print(
                        f"Epoch [{epoch}/{epochs}] "
                        f"Batch [{batch_idx}/{len(train_loader)}] "
                        f"D_loss: {d_metrics['d_loss']:.4f} "
                        f"G_loss: {g_metrics['g_loss']:.4f} "
                        f"D(x): {d_metrics['d_real_acc']:.2f} "
                        f"D(G(z)): {1 - d_metrics['d_fake_acc']:.2f}"
                    )
            
            # Average metrics for epoch
            avg_d_loss = epoch_d_loss / num_batches
            avg_g_loss = epoch_g_loss / num_batches
            avg_d_real_acc = epoch_d_real_acc / num_batches
            avg_d_fake_acc = epoch_d_fake_acc / num_batches
            
            # Store history
            self.d_losses.append(avg_d_loss)
            self.g_losses.append(avg_g_loss)
            self.d_real_acc.append(avg_d_real_acc)
            self.d_fake_acc.append(avg_d_fake_acc)
            
            # Log to file
            self.logger.log_metrics({
                "epoch": epoch,
                "d_loss": avg_d_loss,
                "g_loss": avg_g_loss,
                "d_real_acc": avg_d_real_acc,
                "d_fake_acc": avg_d_fake_acc
            }, step=epoch)
            
            print(
                f"\nEpoch [{epoch}/{epochs}] Summary: "
                f"D_loss: {avg_d_loss:.4f} "
                f"G_loss: {avg_g_loss:.4f}\n"
            )
            
            # Save samples
            if epoch % sample_interval == 0:
                self.save_samples(epoch)
            
            # Save checkpoint
            if epoch % save_interval == 0:
                self.save_checkpoint(epoch)
        
        # Save final checkpoint
        self.save_checkpoint(epochs)
        self.save_samples(epochs)
        
        print("=" * 60)
        print("Training Complete!")
        print("=" * 60)
        
        # Close logger
        self.logger.close()


def main():
    """Main function to run training."""
    parser = argparse.ArgumentParser(description="Train DCGAN on cancer images")
    
    # Data arguments
    parser.add_argument("--data_dir", type=str, default="data/breast_cancer",
                        help="Directory containing the dataset")
    parser.add_argument("--create_sample_data", action="store_true",
                        help="Create sample synthetic dataset for testing")
    
    # Model arguments
    parser.add_argument("--latent_dim", type=int, default=100,
                        help="Dimension of latent vector z")
    parser.add_argument("--g_feature_maps", type=int, default=64,
                        help="Base feature maps for Generator")
    parser.add_argument("--d_feature_maps", type=int, default=64,
                        help="Base feature maps for Discriminator")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Training batch size")
    parser.add_argument("--lr", type=float, default=0.0002,
                        help="Learning rate")
    parser.add_argument("--beta1", type=float, default=0.5,
                        help="Beta1 for Adam optimizer")
    parser.add_argument("--beta2", type=float, default=0.999,
                        help="Beta2 for Adam optimizer")
    
    # Stabilization arguments
    parser.add_argument("--label_smoothing", type=float, default=0.9,
                        help="Label smoothing value for real labels")
    parser.add_argument("--gradient_clip", type=float, default=1.0,
                        help="Gradient clipping norm")
    
    # Checkpoint arguments
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints",
                        help="Directory to save checkpoints")
    parser.add_argument("--samples_dir", type=str, default="samples",
                        help="Directory to save samples")
    parser.add_argument("--log_dir", type=str, default="logs",
                        help="Directory to save logs")
    parser.add_argument("--save_interval", type=int, default=10,
                        help="Save checkpoint every N epochs")
    parser.add_argument("--sample_interval", type=int, default=5,
                        help="Save samples every N epochs")
    parser.add_argument("--log_interval", type=int, default=50,
                        help="Log every N batches")
    
    # Device
    parser.add_argument("--device", type=str, default=None,
                        help="Device to train on (cuda or cpu)")
    
    args = parser.parse_args()
    
    # Create sample data if requested
    if args.create_sample_data:
        print("Creating sample dataset...")
        create_sample_dataset(args.data_dir, num_samples=1000)
    
    # Create trainer
    trainer = DCGANTrainer(
        data_dir=args.data_dir,
        checkpoint_dir=args.checkpoint_dir,
        samples_dir=args.samples_dir,
        log_dir=args.log_dir,
        latent_dim=args.latent_dim,
        g_feature_maps=args.g_feature_maps,
        d_feature_maps=args.d_feature_maps,
        lr=args.lr,
        beta1=args.beta1,
        beta2=args.beta2,
        batch_size=args.batch_size,
        image_size=64,
        label_smoothing=args.label_smoothing,
        gradient_clip=args.gradient_clip,
        device=args.device
    )
    
    # Train
    trainer.train(
        epochs=args.epochs,
        save_interval=args.save_interval,
        sample_interval=args.sample_interval,
        log_interval=args.log_interval
    )


if __name__ == "__main__":
    main()
