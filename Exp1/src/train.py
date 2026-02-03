"""
Training module for CNN model
Author: Raj Kalpesh Mathuria
UID: 2023300139
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
import time
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def train_model(model, train_data, val_data, callbacks, datagen=None):
    """
    Train the CNN model
    
    Args:
        model: Compiled Keras model
        train_data: Tuple of (x_train, y_train)
        val_data: Tuple of (x_val, y_val)
        callbacks: List of Keras callbacks
        datagen: Optional ImageDataGenerator for data augmentation
    
    Returns:
        History: Training history object
    """
    x_train, y_train = train_data
    x_val, y_val = val_data
    
    print("\n" + "="*70)
    print("TRAINING CNN MODEL")
    print("="*70)
    print(f"Training samples: {len(x_train)}")
    print(f"Validation samples: {len(x_val)}")
    print(f"Batch size: {config.BATCH_SIZE}")
    print(f"Epochs: {config.EPOCHS}")
    print(f"Data augmentation: {'Enabled' if datagen is not None else 'Disabled'}")
    print("="*70 + "\n")
    
    start_time = time.time()
    
    if datagen is not None:
        # Training with data augmentation
        print("Training with data augmentation...\n")
        history = model.fit(
            datagen.flow(x_train, y_train, batch_size=config.BATCH_SIZE),
            epochs=config.EPOCHS,
            validation_data=(x_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
    else:
        # Training without data augmentation
        print("Training without data augmentation...\n")
        history = model.fit(
            x_train, y_train,
            batch_size=config.BATCH_SIZE,
            epochs=config.EPOCHS,
            validation_data=(x_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
    
    training_time = time.time() - start_time
    
    print("\n" + "="*70)
    print("TRAINING COMPLETED")
    print("="*70)
    print(f"Total training time: {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
    print(f"Average time per epoch: {training_time/len(history.history['loss']):.2f} seconds")
    print("="*70 + "\n")
    
    return history


def evaluate_model(model, test_data):
    """
    Evaluate the trained model on test data
    
    Args:
        model: Trained Keras model
        test_data: Tuple of (x_test, y_test)
    
    Returns:
        dict: Evaluation metrics
    """
    x_test, y_test = test_data
    
    print("\n" + "="*70)
    print("EVALUATING MODEL ON TEST DATA")
    print("="*70)
    print(f"Test samples: {len(x_test)}\n")
    
    # Evaluate on test set
    test_loss, test_accuracy = model.evaluate(
        x_test, y_test, 
        batch_size=config.BATCH_SIZE,
        verbose=1
    )
    
    print("\n" + "="*70)
    print("TEST RESULTS")
    print("="*70)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy*100:.2f}%")
    print("="*70 + "\n")
    
    return {
        'test_loss': test_loss,
        'test_accuracy': test_accuracy
    }


def get_predictions(model, x_data):
    """
    Get model predictions
    
    Args:
        model: Trained Keras model
        x_data: Input data
    
    Returns:
        tuple: (predictions, predicted_classes)
    """
    predictions = model.predict(x_data, verbose=0)
    predicted_classes = np.argmax(predictions, axis=1)
    
    return predictions, predicted_classes


def calculate_class_wise_accuracy(model, test_data):
    """
    Calculate accuracy for each class
    
    Args:
        model: Trained Keras model
        test_data: Tuple of (x_test, y_test)
    
    Returns:
        dict: Class-wise accuracy
    """
    x_test, y_test = test_data
    
    print("\nCalculating class-wise accuracy...")
    
    # Get predictions
    _, predicted_classes = get_predictions(model, x_test)
    
    # Calculate accuracy for each class
    class_accuracy = {}
    for class_idx in range(config.NUM_CLASSES):
        class_mask = (y_test == class_idx)
        class_predictions = predicted_classes[class_mask]
        class_labels = y_test[class_mask]
        
        accuracy = np.mean(class_predictions == class_labels)
        class_accuracy[config.CLASS_NAMES[class_idx]] = accuracy
    
    print("\n" + "="*70)
    print("CLASS-WISE ACCURACY")
    print("="*70)
    for class_name, accuracy in class_accuracy.items():
        print(f"{class_name:20s}: {accuracy*100:6.2f}%")
    print("="*70 + "\n")
    
    return class_accuracy


def save_model(model, filename='final_model.h5'):
    """
    Save the trained model
    
    Args:
        model: Trained Keras model
        filename: Name of the file to save
    """
    filepath = os.path.join(config.MODEL_DIR, filename)
    model.save(filepath)
    print(f"\n✓ Model saved to: {filepath}")


def load_saved_model(filename='final_model.h5'):
    """
    Load a saved model
    
    Args:
        filename: Name of the file to load
    
    Returns:
        keras.Model: Loaded model
    """
    filepath = os.path.join(config.MODEL_DIR, filename)
    model = keras.models.load_model(filepath)
    print(f"✓ Model loaded from: {filepath}")
    
    return model


def get_training_summary(history, test_metrics):
    """
    Get a summary of training results
    
    Args:
        history: Training history object
        test_metrics: Dictionary of test metrics
    
    Returns:
        dict: Training summary
    """
    summary = {
        'epochs_trained': len(history.history['loss']),
        'final_train_loss': history.history['loss'][-1],
        'final_train_accuracy': history.history['accuracy'][-1],
        'final_val_loss': history.history['val_loss'][-1],
        'final_val_accuracy': history.history['val_accuracy'][-1],
        'best_val_accuracy': max(history.history['val_accuracy']),
        'test_loss': test_metrics['test_loss'],
        'test_accuracy': test_metrics['test_accuracy']
    }
    
    print("\n" + "="*70)
    print("TRAINING SUMMARY")
    print("="*70)
    print(f"Epochs trained: {summary['epochs_trained']}")
    print(f"\nFinal Training Metrics:")
    print(f"  Loss: {summary['final_train_loss']:.4f}")
    print(f"  Accuracy: {summary['final_train_accuracy']*100:.2f}%")
    print(f"\nFinal Validation Metrics:")
    print(f"  Loss: {summary['final_val_loss']:.4f}")
    print(f"  Accuracy: {summary['final_val_accuracy']*100:.2f}%")
    print(f"  Best Val Accuracy: {summary['best_val_accuracy']*100:.2f}%")
    print(f"\nTest Metrics:")
    print(f"  Loss: {summary['test_loss']:.4f}")
    print(f"  Accuracy: {summary['test_accuracy']*100:.2f}%")
    print("="*70 + "\n")
    
    return summary


if __name__ == "__main__":
    print("Training module loaded successfully!")
