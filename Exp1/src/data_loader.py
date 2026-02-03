"""
Data loading and preprocessing module for Fashion MNIST
Author: Raj Kalpesh Mathuria
UID: 2023300139
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
# Access ImageDataGenerator via keras to avoid Pylance errors
ImageDataGenerator = keras.preprocessing.image.ImageDataGenerator
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def load_fashion_mnist():
    """
    Load Fashion MNIST dataset
    
    Returns:
        tuple: (x_train, y_train), (x_test, y_test)
    """
    print("Loading Fashion MNIST dataset...")
    (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
    
    print(f"Training data shape: {x_train.shape}")
    print(f"Training labels shape: {y_train.shape}")
    print(f"Test data shape: {x_test.shape}")
    print(f"Test labels shape: {y_test.shape}")
    
    return (x_train, y_train), (x_test, y_test)


def preprocess_data(x_train, y_train, x_test, y_test):
    """
    Preprocess and normalize image data
    
    Args:
        x_train: Training images
        y_train: Training labels
        x_test: Test images
        y_test: Test labels
    
    Returns:
        tuple: Preprocessed (x_train, y_train), (x_test, y_test)
    """
    print("\nPreprocessing data...")
    
    # Reshape data to add channel dimension (for CNN)
    x_train = x_train.reshape(-1, config.IMAGE_SIZE[0], config.IMAGE_SIZE[1], 1)
    x_test = x_test.reshape(-1, config.IMAGE_SIZE[0], config.IMAGE_SIZE[1], 1)
    
    # Normalize pixel values to [0, 1]
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    print(f"Preprocessed training data shape: {x_train.shape}")
    print(f"Preprocessed test data shape: {x_test.shape}")
    print(f"Pixel value range: [{x_train.min():.2f}, {x_train.max():.2f}]")
    
    return (x_train, y_train), (x_test, y_test)


def create_data_augmentation():
    """
    Create data augmentation generator
    
    Returns:
        ImageDataGenerator: Configured data augmentation generator
    """
    if not config.USE_AUGMENTATION:
        return None
    
    print("\nCreating data augmentation generator...")
    datagen = ImageDataGenerator(
        rotation_range=config.ROTATION_RANGE,
        width_shift_range=config.WIDTH_SHIFT_RANGE,
        height_shift_range=config.HEIGHT_SHIFT_RANGE,
        horizontal_flip=config.HORIZONTAL_FLIP,
        fill_mode='nearest'
    )
    
    print(f"Augmentation parameters:")
    print(f"  - Rotation range: {config.ROTATION_RANGE}°")
    print(f"  - Width shift: {config.WIDTH_SHIFT_RANGE}")
    print(f"  - Height shift: {config.HEIGHT_SHIFT_RANGE}")
    print(f"  - Horizontal flip: {config.HORIZONTAL_FLIP}")
    
    return datagen


def split_validation_data(x_train, y_train):
    """
    Split training data into train and validation sets
    
    Args:
        x_train: Training images
        y_train: Training labels
    
    Returns:
        tuple: (x_train, y_train), (x_val, y_val)
    """
    split_idx = int(len(x_train) * (1 - config.VALIDATION_SPLIT))
    
    x_val = x_train[split_idx:]
    y_val = y_train[split_idx:]
    x_train = x_train[:split_idx]
    y_train = y_train[:split_idx]
    
    print(f"\nData split:")
    print(f"  - Training samples: {len(x_train)}")
    print(f"  - Validation samples: {len(x_val)}")
    
    return (x_train, y_train), (x_val, y_val)


def get_class_distribution(labels):
    """
    Get class distribution in the dataset
    
    Args:
        labels: Array of class labels
    
    Returns:
        dict: Class distribution
    """
    unique, counts = np.unique(labels, return_counts=True)
    distribution = dict(zip(unique, counts))
    
    print("\nClass distribution:")
    for class_idx, count in distribution.items():
        print(f"  - {config.CLASS_NAMES[class_idx]}: {count} samples")
    
    return distribution


def prepare_data():
    """
    Complete data preparation pipeline
    
    Returns:
        tuple: ((x_train, y_train), (x_val, y_val), (x_test, y_test), datagen)
    """
    # Set random seed for reproducibility
    np.random.seed(config.RANDOM_SEED)
    tf.random.set_seed(config.RANDOM_SEED)
    
    # Load data
    (x_train, y_train), (x_test, y_test) = load_fashion_mnist()
    
    # Get class distribution
    get_class_distribution(y_train)
    
    # Preprocess data
    (x_train, y_train), (x_test, y_test) = preprocess_data(
        x_train, y_train, x_test, y_test
    )
    
    # Split validation data
    (x_train, y_train), (x_val, y_val) = split_validation_data(x_train, y_train)
    
    # Create data augmentation generator
    datagen = create_data_augmentation()
    if datagen is not None:
        datagen.fit(x_train)
    
    return (x_train, y_train), (x_val, y_val), (x_test, y_test), datagen


if __name__ == "__main__":
    # Test data loading
    (x_train, y_train), (x_val, y_val), (x_test, y_test), datagen = prepare_data()
    print("\n✓ Data loading successful!")
