"""
Logging utility module for training metrics and TensorBoard support.
"""

import os
import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False


class Logger:
    """
    Logger class for tracking training metrics.
    Supports CSV logging and TensorBoard.
    """
    
    def __init__(
        self,
        log_dir: str = "logs",
        experiment_name: Optional[str] = None,
        use_tensorboard: bool = True
    ):
        """
        Initialize the logger.
        
        Args:
            log_dir: Directory to save logs.
            experiment_name: Name of the experiment (defaults to timestamp).
            use_tensorboard: Whether to use TensorBoard logging.
        """
        if experiment_name is None:
            experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.log_dir = Path(log_dir) / experiment_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # CSV logging
        self.csv_path = self.log_dir / "training_log.csv"
        self.csv_file = None
        self.csv_writer = None
        self.csv_initialized = False
        
        # TensorBoard
        self.use_tensorboard = use_tensorboard and TENSORBOARD_AVAILABLE
        self.writer = None
        
        if self.use_tensorboard:
            self.writer = SummaryWriter(log_dir=str(self.log_dir))
        
        self.step = 0
    
    def _init_csv(self, fieldnames: list):
        """Initialize CSV file with fieldnames."""
        self.csv_file = open(self.csv_path, 'w', newline='')
        self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=fieldnames)
        self.csv_writer.writeheader()
        self.csv_initialized = True
    
    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """
        Log metrics to CSV and TensorBoard.
        
        Args:
            metrics: Dictionary of metric names and values.
            step: Training step (uses internal counter if not provided).
        """
        if step is None:
            step = self.step
            self.step += 1
        
        # Add step to metrics
        metrics_with_step = {"step": step, **metrics}
        
        # CSV logging
        if not self.csv_initialized:
            self._init_csv(list(metrics_with_step.keys()))
        self.csv_writer.writerow(metrics_with_step)
        self.csv_file.flush()
        
        # TensorBoard logging
        if self.use_tensorboard and self.writer is not None:
            for name, value in metrics.items():
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(name, value, step)
    
    def log_images(self, tag: str, images, step: Optional[int] = None):
        """
        Log images to TensorBoard.
        
        Args:
            tag: Tag for the images.
            images: Tensor of images (N, C, H, W).
            step: Training step.
        """
        if self.use_tensorboard and self.writer is not None:
            if step is None:
                step = self.step
            self.writer.add_images(tag, images, step)
    
    def log_text(self, message: str):
        """
        Log a text message to a separate text file.
        
        Args:
            message: Message to log.
        """
        text_path = self.log_dir / "log.txt"
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(text_path, 'a') as f:
            f.write(f"[{timestamp}] {message}\n")
    
    def close(self):
        """Close all logging resources."""
        if self.csv_file is not None:
            self.csv_file.close()
        if self.writer is not None:
            self.writer.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
