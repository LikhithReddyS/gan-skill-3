"""
Data Loader module for Cancer GAN Project.
Provides PyTorch Dataset and DataLoader for breast cancer histopathology images.
"""

import os
import random
from pathlib import Path
from typing import Tuple, Optional, List

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np


class CancerDataset(Dataset):
    """
    PyTorch Dataset for breast cancer histopathology images.
    
    Loads images from a directory structure and applies transformations.
    Images are normalized to [-1, 1] range for GAN training.
    """
    
    def __init__(
        self,
        root_dir: str,
        image_size: int = 64,
        transform: Optional[transforms.Compose] = None,
        target_class: Optional[int] = None
    ):
        """
        Initialize the CancerDataset.
        
        Args:
            root_dir: Root directory containing images.
            image_size: Target size for resizing images.
            transform: Optional custom transforms. If None, default transforms are applied.
            target_class: If specified, only load images of this class (0 or 1).
        """
        self.root_dir = Path(root_dir)
        self.image_size = image_size
        self.target_class = target_class
        
        # Default transform: resize, to tensor, normalize to [-1, 1]
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # Maps to [-1, 1]
            ])
        else:
            self.transform = transform
        
        # Collect all image paths
        self.image_paths = self._collect_images()
        
        if len(self.image_paths) == 0:
            print(f"Warning: No images found in {root_dir}")
    
    def _collect_images(self) -> List[Path]:
        """Collect all image paths from the root directory."""
        valid_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
        image_paths = []
        
        # Check if directory exists
        if not self.root_dir.exists():
            return image_paths
        
        # Walk through directory and collect images
        for path in self.root_dir.rglob('*'):
            if path.suffix.lower() in valid_extensions:
                # If target_class is specified, filter by class
                if self.target_class is not None:
                    # Assuming folder structure: root/class_label/images
                    # Or filename contains class label
                    parent_name = path.parent.name
                    if parent_name.isdigit() and int(parent_name) != self.target_class:
                        continue
                image_paths.append(path)
        
        return image_paths
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Get an image by index.
        
        Args:
            idx: Index of the image.
            
        Returns:
            Transformed image tensor of shape (C, H, W).
        """
        img_path = self.image_paths[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image


def get_dataloaders(
    data_dir: str = "data/breast_cancer",
    image_size: int = 64,
    batch_size: int = 64,
    num_workers: int = 4,
    target_class: Optional[int] = None
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create DataLoaders for train, validation, and test sets.
    
    Args:
        data_dir: Root directory containing train/val/test subdirectories.
        image_size: Target image size.
        batch_size: Batch size for DataLoader.
        num_workers: Number of workers for data loading.
        target_class: If specified, only load images of this class.
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    data_dir = Path(data_dir)
    
    # Create datasets
    train_dataset = CancerDataset(
        root_dir=data_dir / "train",
        image_size=image_size,
        target_class=target_class
    )
    
    val_dataset = CancerDataset(
        root_dir=data_dir / "val",
        image_size=image_size,
        target_class=target_class
    )
    
    test_dataset = CancerDataset(
        root_dir=data_dir / "test",
        image_size=image_size,
        target_class=target_class
    )
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


def create_sample_dataset(output_dir: str = "data/breast_cancer", num_samples: int = 1000):
    """
    Create a sample synthetic dataset for testing purposes.
    This generates random colored images to verify the pipeline works.
    
    Args:
        output_dir: Output directory for the dataset.
        num_samples: Number of samples to generate.
    """
    output_dir = Path(output_dir)
    
    # Create directories
    for split in ['train', 'val', 'test']:
        for label in ['0', '1']:
            (output_dir / split / label).mkdir(parents=True, exist_ok=True)
    
    # Split samples
    train_samples = int(num_samples * 0.8)
    val_samples = int(num_samples * 0.1)
    test_samples = num_samples - train_samples - val_samples
    
    def generate_samples(directory: Path, n_samples: int, label: str):
        """Generate random images for a split."""
        for i in range(n_samples):
            # Create random image with slight color bias based on label
            if label == '1':  # Cancer positive - slightly more pink/red
                img = np.random.randint(100, 256, (64, 64, 3), dtype=np.uint8)
                img[:, :, 0] = np.clip(img[:, :, 0] + 30, 0, 255)  # More red
            else:  # Cancer negative - slightly more purple/blue
                img = np.random.randint(100, 256, (64, 64, 3), dtype=np.uint8)
                img[:, :, 2] = np.clip(img[:, :, 2] + 30, 0, 255)  # More blue
            
            img = Image.fromarray(img)
            img.save(directory / label / f"sample_{i:04d}.png")
    
    # Generate samples for each split
    for label in ['0', '1']:
        generate_samples(output_dir / 'train', train_samples // 2, label)
        generate_samples(output_dir / 'val', val_samples // 2, label)
        generate_samples(output_dir / 'test', test_samples // 2, label)
    
    print(f"Created sample dataset at {output_dir}")
    print(f"  Train: {train_samples} samples")
    print(f"  Val: {val_samples} samples")
    print(f"  Test: {test_samples} samples")


if __name__ == "__main__":
    # Create sample dataset for testing
    create_sample_dataset("data/breast_cancer", num_samples=1000)
    
    # Test data loading
    train_loader, val_loader, test_loader = get_dataloaders(
        data_dir="data/breast_cancer",
        batch_size=64
    )
    
    print(f"\nDataLoader created:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    
    # Test batch
    batch = next(iter(train_loader))
    print(f"\nBatch shape: {batch.shape}")
    print(f"Batch range: [{batch.min():.2f}, {batch.max():.2f}]")
