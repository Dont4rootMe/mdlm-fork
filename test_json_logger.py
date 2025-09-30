#!/usr/bin/env python3
"""
Test script for the LightningJSONLogger to verify functionality.
"""

import os
import time
import math
from lightning_json_logger import LightningJSONLogger

def test_json_logger():
    """Test the JSON logger functionality."""
    print("Testing LightningJSONLogger...")
    
    # Create logger
    logger = LightningJSONLogger(
        save_dir="./test_logs",
        experiment_name="test_experiment",
        version="0",
        update_freq=2  # Update plots every 2 steps
    )
    
    # Test config logging
    test_config = {
        'model': 'test_model',
        'learning_rate': 0.001,
        'batch_size': 32,
        'epochs': 10,
        'optimizer': 'adam'
    }
    logger.log_config(test_config)
    
    # Simulate training with metrics
    print("Simulating training metrics...")
    for step in range(50):
        # Simulate some training metrics
        train_loss = 2.0 * math.exp(-step * 0.05) + 0.1 * math.sin(step * 0.2)
        train_acc = 1.0 - math.exp(-step * 0.03) + 0.05 * math.sin(step * 0.1)
        
        # Simulate validation metrics (less frequent)
        metrics = {
            'train/loss': train_loss,
            'train/accuracy': train_acc,
            'train/learning_rate': 0.001 * (0.99 ** (step // 10))
        }
        
        if step % 5 == 0:  # Validation every 5 steps
            val_loss = train_loss + 0.1
            val_acc = train_acc - 0.05
            metrics.update({
                'val/loss': val_loss,
                'val/accuracy': val_acc
            })
        
        logger.log_metrics(metrics, step=step)
        
        # Small delay to simulate training time
        time.sleep(0.1)
        
        if step % 10 == 0:
            print(f"Step {step}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}")
    
    print("Finalizing logger...")
    logger.finalize()
    
    print(f"Test completed! Check results in: {logger.log_dir}")
    print("Files created:")
    for root, dirs, files in os.walk(logger.log_dir):
        for file in files:
            filepath = os.path.join(root, file)
            print(f"  {filepath}")

if __name__ == "__main__":
    test_json_logger()
