"""
Visualization and plotting module for CNN training results
Author: Raj Kalpesh Mathuria
UID: 2023300139
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# Set plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def plot_training_history(history, save=True):
    """
    Plot training and validation accuracy and loss curves
    
    Args:
        history: Keras History object
        save: Whether to save the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot accuracy
    axes[0].plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
    axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    axes[0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].legend(loc='lower right', fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Plot loss
    axes[1].plot(history.history['loss'], label='Training Loss', linewidth=2)
    axes[1].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    axes[1].set_title('Model Loss', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Loss', fontsize=12)
    axes[1].legend(loc='upper right', fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save:
        filepath = os.path.join(config.OUTPUT_DIR, 'training_history.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"✓ Training history plot saved to: {filepath}")
    
    plt.show()
    plt.close()


def plot_sample_images(x_data, y_data, num_samples=25, predictions=None, save=True):
    """
    Plot sample images from the dataset
    
    Args:
        x_data: Image data
        y_data: Labels
        num_samples: Number of samples to plot
        predictions: Optional predicted labels
        save: Whether to save the plot
    """
    num_samples = min(num_samples, len(x_data))
    grid_size = int(np.ceil(np.sqrt(num_samples)))
    
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
    axes = axes.ravel()
    
    indices = np.random.choice(len(x_data), num_samples, replace=False)
    
    for i, idx in enumerate(indices):
        axes[i].imshow(x_data[idx].reshape(28, 28), cmap='gray')
        
        if predictions is not None:
            true_label = config.CLASS_NAMES[y_data[idx]]
            pred_label = config.CLASS_NAMES[predictions[idx]]
            color = 'green' if y_data[idx] == predictions[idx] else 'red'
            axes[i].set_title(f'True: {true_label}\nPred: {pred_label}', 
                            fontsize=9, color=color, fontweight='bold')
        else:
            axes[i].set_title(config.CLASS_NAMES[y_data[idx]], 
                            fontsize=10, fontweight='bold')
        
        axes[i].axis('off')
    
    # Hide unused subplots
    for i in range(num_samples, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save:
        filename = 'sample_predictions.png' if predictions is not None else 'sample_images.png'
        filepath = os.path.join(config.OUTPUT_DIR, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"✓ Sample images plot saved to: {filepath}")
    
    plt.show()
    plt.close()


def plot_confusion_matrix(y_true, y_pred, save=True):
    """
    Plot confusion matrix
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        save: Whether to save the plot
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=config.CLASS_NAMES,
                yticklabels=config.CLASS_NAMES,
                cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if save:
        filepath = os.path.join(config.OUTPUT_DIR, 'confusion_matrix.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"✓ Confusion matrix plot saved to: {filepath}")
    
    plt.show()
    plt.close()


def plot_class_wise_accuracy(class_accuracy, save=True):
    """
    Plot bar chart of class-wise accuracy
    
    Args:
        class_accuracy: Dictionary of class-wise accuracy
        save: Whether to save the plot
    """
    classes = list(class_accuracy.keys())
    accuracies = [acc * 100 for acc in class_accuracy.values()]
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(classes, accuracies, color=sns.color_palette("husl", len(classes)))
    
    plt.title('Class-wise Accuracy', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Class', fontsize=12, fontweight='bold')
    plt.ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 105)
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    
    if save:
        filepath = os.path.join(config.OUTPUT_DIR, 'class_wise_accuracy.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"✓ Class-wise accuracy plot saved to: {filepath}")
    
    plt.show()
    plt.close()


def plot_misclassified_samples(x_test, y_test, predictions, num_samples=16, save=True):
    """
    Plot misclassified samples
    
    Args:
        x_test: Test images
        y_test: True labels
        predictions: Predicted labels
        num_samples: Number of misclassified samples to plot
        save: Whether to save the plot
    """
    # Find misclassified samples
    misclassified_indices = np.where(y_test != predictions)[0]
    
    if len(misclassified_indices) == 0:
        print("No misclassified samples found!")
        return
    
    num_samples = min(num_samples, len(misclassified_indices))
    sample_indices = np.random.choice(misclassified_indices, num_samples, replace=False)
    
    grid_size = int(np.ceil(np.sqrt(num_samples)))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
    axes = axes.ravel()
    
    for i, idx in enumerate(sample_indices):
        axes[i].imshow(x_test[idx].reshape(28, 28), cmap='gray')
        true_label = config.CLASS_NAMES[y_test[idx]]
        pred_label = config.CLASS_NAMES[predictions[idx]]
        axes[i].set_title(f'True: {true_label}\nPred: {pred_label}', 
                         fontsize=9, color='red', fontweight='bold')
        axes[i].axis('off')
    
    # Hide unused subplots
    for i in range(num_samples, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle('Misclassified Samples', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    if save:
        filepath = os.path.join(config.OUTPUT_DIR, 'misclassified_samples.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"✓ Misclassified samples plot saved to: {filepath}")
    
    plt.show()
    plt.close()


def print_classification_report(y_true, y_pred):
    """
    Print detailed classification report
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
    """
    report = classification_report(y_true, y_pred, 
                                   target_names=config.CLASS_NAMES,
                                   digits=4)
    
    print("\n" + "="*70)
    print("CLASSIFICATION REPORT")
    print("="*70)
    print(report)
    print("="*70 + "\n")


def plot_learning_rate_schedule(history, save=True):
    """
    Plot learning rate schedule if available
    
    Args:
        history: Keras History object
        save: Whether to save the plot
    """
    if 'lr' not in history.history:
        print("Learning rate information not available in history.")
        return
    
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['lr'], linewidth=2, color='purple')
    plt.title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Learning Rate', fontsize=12)
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save:
        filepath = os.path.join(config.OUTPUT_DIR, 'learning_rate_schedule.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"✓ Learning rate schedule plot saved to: {filepath}")
    
    plt.show()
    plt.close()


def visualize_all_results(history, test_data, model, class_accuracy):
    """
    Generate all visualization plots
    
    Args:
        history: Training history object
        test_data: Tuple of (x_test, y_test)
        model: Trained model
        class_accuracy: Dictionary of class-wise accuracy
    """
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70 + "\n")
    
    x_test, y_test = test_data
    
    # Get predictions
    predictions_probs = model.predict(x_test, verbose=0)
    predictions = np.argmax(predictions_probs, axis=1)
    
    # Plot training history
    print("Plotting training history...")
    plot_training_history(history)
    
    # Plot sample predictions
    print("Plotting sample predictions...")
    plot_sample_images(x_test, y_test, num_samples=25, predictions=predictions)
    
    # Plot confusion matrix
    print("Plotting confusion matrix...")
    plot_confusion_matrix(y_test, predictions)
    
    # Plot class-wise accuracy
    print("Plotting class-wise accuracy...")
    plot_class_wise_accuracy(class_accuracy)
    
    # Plot misclassified samples
    print("Plotting misclassified samples...")
    plot_misclassified_samples(x_test, y_test, predictions)
    
    # Plot learning rate schedule
    print("Plotting learning rate schedule...")
    plot_learning_rate_schedule(history)
    
    # Print classification report
    print_classification_report(y_test, predictions)
    
    print("="*70)
    print("VISUALIZATION COMPLETED")
    print("="*70 + "\n")


if __name__ == "__main__":
    print("Plotting module loaded successfully!")
