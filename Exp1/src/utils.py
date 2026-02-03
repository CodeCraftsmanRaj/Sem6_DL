"""
Utility functions and helpers
Author: Raj Kalpesh Mathuria
UID: 2023300139
"""

import os
import json
import numpy as np
from datetime import datetime


def create_experiment_summary(history, test_metrics, class_accuracy, output_dir):
    """
    Create and save experiment summary as JSON
    
    Args:
        history: Training history
        test_metrics: Test evaluation metrics
        class_accuracy: Class-wise accuracy dictionary
        output_dir: Output directory path
    """
    summary = {
        'experiment': 'Experiment 2 - CNN Image Classification',
        'dataset': 'Fashion MNIST',
        'author': 'Raj Kalpesh Mathuria',
        'uid': '2023300139',
        'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'training': {
            'epochs': len(history.history['loss']),
            'final_train_accuracy': float(history.history['accuracy'][-1]),
            'final_train_loss': float(history.history['loss'][-1]),
            'final_val_accuracy': float(history.history['val_accuracy'][-1]),
            'final_val_loss': float(history.history['val_loss'][-1]),
            'best_val_accuracy': float(max(history.history['val_accuracy']))
        },
        'testing': {
            'test_accuracy': float(test_metrics['test_accuracy']),
            'test_loss': float(test_metrics['test_loss'])
        },
        'class_wise_accuracy': {k: float(v) for k, v in class_accuracy.items()}
    }
    
    filepath = os.path.join(output_dir, 'experiment_summary.json')
    with open(filepath, 'w') as f:
        json.dump(summary, f, indent=4)
    
    print(f"✓ Experiment summary saved to: {filepath}")
    
    return summary


def format_time(seconds):
    """
    Format seconds into human-readable time
    
    Args:
        seconds: Time in seconds
    
    Returns:
        str: Formatted time string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def ensure_dir(directory):
    """
    Create directory if it doesn't exist
    
    Args:
        directory: Directory path
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")


if __name__ == "__main__":
    print("Utilities module loaded successfully!")
