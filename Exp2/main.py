"""
Main execution script for Fashion MNIST CNN Classification
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
from src.data_loader import prepare_data
from src.model import build_and_compile_model
from src.train import (train_model, evaluate_model, calculate_class_wise_accuracy, 
                       save_model, get_training_summary)
from src.plotting import visualize_all_results


def print_header():
    print("\n" + "="*70)
    print("EXPERIMENT 2: CNN FOR IMAGE CLASSIFICATION")
    print("Fashion MNIST Dataset")
    print("="*70)
    print("Author: Raj Kalpesh Mathuria")
    print("UID: 2023300139")
    print("Division: PE-C")
    print("="*70 + "\n")


def print_system_info():
    import tensorflow as tf
    
    print("="*70)
    print("SYSTEM INFORMATION")
    print("="*70)
    print(f"Python version: {sys.version.split()[0]}")
    print(f"TensorFlow version: {tf.__version__}")
    print(f"GPU Available: {tf.config.list_physical_devices('GPU')}")
    print("="*70 + "\n")


def main():
    print_header()
    print_system_info()
    
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    print("\n" + "="*70)
    print("STEP 1: DATA PREPARATION")
    print("="*70)
    
    (x_train, y_train), (x_val, y_val), (x_test, y_test), datagen = prepare_data()
    
    print("\n" + "="*70)
    print("STEP 2: MODEL BUILDING")
    print("="*70)
    
    model, callbacks = build_and_compile_model()
    
    print("\n" + "="*70)
    print("STEP 3: MODEL TRAINING")
    print("="*70)
    
    history = train_model(
        model=model,
        train_data=(x_train, y_train),
        val_data=(x_val, y_val),
        callbacks=callbacks,
        datagen=datagen
    )
    
    print("\n" + "="*70)
    print("STEP 4: MODEL EVALUATION")
    print("="*70)
    
    test_metrics = evaluate_model(model, (x_test, y_test))
    
    class_accuracy = calculate_class_wise_accuracy(model, (x_test, y_test))
    
    summary = get_training_summary(history, test_metrics)
    
    print("\n" + "="*70)
    print("STEP 5: VISUALIZATION")
    print("="*70)
    
    visualize_all_results(history, (x_test, y_test), model, class_accuracy)
    
    print("\n" + "="*70)
    print("STEP 6: SAVING MODEL")
    print("="*70)
    
    save_model(model, 'final_model.keras')
    
    print("\n" + "="*70)
    print("EXPERIMENT COMPLETED SUCCESSFULLY!")
    print("="*70)
    print("\nResults saved in:")
    print(f"  - Models: {config.MODEL_DIR}")
    print(f"  - Outputs: {config.OUTPUT_DIR}")
    print("\n" + "="*70 + "\n")
    
    return model, history, test_metrics, class_accuracy


if __name__ == "__main__":
    try:
        model, history, test_metrics, class_accuracy = main()
        print("✓ All tasks completed successfully!\n")
    except Exception as e:
        print(f"\n✗ Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
