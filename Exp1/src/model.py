"""
CNN Model Architecture for Fashion MNIST Classification
Author: Raj Kalpesh Mathuria
UID: 2023300139
"""

import tensorflow as tf
from tensorflow import keras
# Access layers and models from keras directly to avoid Pylance errors
layers = keras.layers
models = keras.models
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def create_cnn_model():
    """
    Create a Convolutional Neural Network model for image classification
    
    Architecture:
        - Multiple Convolutional layers with ReLU activation
        - MaxPooling layers for downsampling
        - Dropout layers for regularization
        - Dense layers for classification
        - Softmax activation for multi-class output
    
    Returns:
        keras.Model: Compiled CNN model
    """
    print("\nBuilding CNN model...")
    
    model = models.Sequential([
        # Input layer
        layers.Input(shape=(config.IMAGE_SIZE[0], config.IMAGE_SIZE[1], 1)),
        
        # First Convolutional Block
        layers.Conv2D(
            config.CONV_FILTERS[0], 
            config.KERNEL_SIZE, 
            activation='relu',
            padding='same',
            name='conv1'
        ),
        layers.BatchNormalization(name='bn1'),
        layers.Conv2D(
            config.CONV_FILTERS[0], 
            config.KERNEL_SIZE, 
            activation='relu',
            padding='same',
            name='conv2'
        ),
        layers.BatchNormalization(name='bn2'),
        layers.MaxPooling2D(config.POOL_SIZE, name='pool1'),
        layers.Dropout(0.25, name='dropout1'),
        
        # Second Convolutional Block
        layers.Conv2D(
            config.CONV_FILTERS[1], 
            config.KERNEL_SIZE, 
            activation='relu',
            padding='same',
            name='conv3'
        ),
        layers.BatchNormalization(name='bn3'),
        layers.Conv2D(
            config.CONV_FILTERS[1], 
            config.KERNEL_SIZE, 
            activation='relu',
            padding='same',
            name='conv4'
        ),
        layers.BatchNormalization(name='bn4'),
        layers.MaxPooling2D(config.POOL_SIZE, name='pool2'),
        layers.Dropout(0.25, name='dropout2'),
        
        # Third Convolutional Block
        layers.Conv2D(
            config.CONV_FILTERS[2], 
            config.KERNEL_SIZE, 
            activation='relu',
            padding='same',
            name='conv5'
        ),
        layers.BatchNormalization(name='bn5'),
        layers.MaxPooling2D(config.POOL_SIZE, name='pool3'),
        layers.Dropout(0.4, name='dropout3'),
        
        # Flatten and Dense layers
        layers.Flatten(name='flatten'),
        
        # First Dense Block
        layers.Dense(config.DENSE_UNITS[0], activation='relu', name='dense1'),
        layers.BatchNormalization(name='bn6'),
        layers.Dropout(config.DROPOUT_RATE, name='dropout4'),
        
        # Output layer with Softmax activation
        layers.Dense(config.NUM_CLASSES, activation='softmax', name='output')
    ])
    
    return model


def compile_model(model):
    """
    Compile the CNN model with optimizer, loss function, and metrics
    
    Args:
        model: Keras model to compile
    
    Returns:
        keras.Model: Compiled model
    """
    print("\nCompiling model...")
    
    optimizer = keras.optimizers.Adam(learning_rate=config.LEARNING_RATE)
    
    model.compile(
        optimizer=optimizer,
        loss=config.LOSS_FUNCTION,
        metrics=config.METRICS
    )
    
    print(f"Optimizer: {config.OPTIMIZER} (lr={config.LEARNING_RATE})")
    print(f"Loss function: {config.LOSS_FUNCTION}")
    print(f"Metrics: {config.METRICS}")
    
    return model


def display_model_summary(model):
    """
    Display model architecture summary
    
    Args:
        model: Keras model
    """
    print("\n" + "="*70)
    print("MODEL ARCHITECTURE SUMMARY")
    print("="*70)
    model.summary()
    print("="*70)
    
    # Count parameters
    total_params = model.count_params()
    print(f"\nTotal parameters: {total_params:,}")
    
    # Calculate model size (approximate)
    model_size_mb = (total_params * 4) / (1024 * 1024)  # 4 bytes per parameter
    print(f"Approximate model size: {model_size_mb:.2f} MB")


def create_callbacks():
    """
    Create training callbacks for model optimization
    
    Returns:
        list: List of Keras callbacks
    """
    callbacks = []
    
    # Model checkpoint - save best model
    checkpoint_path = os.path.join(config.MODEL_DIR, 'best_model.h5')
    model_checkpoint = keras.callbacks.ModelCheckpoint(
        checkpoint_path,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    callbacks.append(model_checkpoint)
    
    # Early stopping - stop if no improvement
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
    callbacks.append(early_stopping)
    
    # Reduce learning rate on plateau
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1
    )
    callbacks.append(reduce_lr)
    
    # TensorBoard logging
    log_dir = os.path.join(config.OUTPUT_DIR, 'logs')
    tensorboard = keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=1
    )
    callbacks.append(tensorboard)
    
    print("\nCallbacks configured:")
    print("  ✓ Model checkpoint (save best model)")
    print("  ✓ Early stopping (patience=5)")
    print("  ✓ Learning rate reduction (factor=0.5, patience=3)")
    print("  ✓ TensorBoard logging")
    
    return callbacks


def build_and_compile_model():
    """
    Complete model building and compilation pipeline
    
    Returns:
        tuple: (model, callbacks)
    """
    # Set random seed
    tf.random.set_seed(config.RANDOM_SEED)
    
    # Create model
    model = create_cnn_model()
    
    # Compile model
    model = compile_model(model)
    
    # Display summary
    display_model_summary(model)
    
    # Create callbacks
    callbacks = create_callbacks()
    
    return model, callbacks


if __name__ == "__main__":
    # Test model creation
    model, callbacks = build_and_compile_model()
    print("\n✓ Model creation successful!")
