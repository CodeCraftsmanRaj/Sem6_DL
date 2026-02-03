"""
Configuration file for Fashion MNIST CNN experiment
Author: Raj Kalpesh Mathuria
UID: 2023300139
"""

import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')

# Dataset parameters
DATASET_NAME = 'fashion_mnist'
IMAGE_SIZE = (28, 28)
NUM_CLASSES = 10
CLASS_NAMES = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Model hyperparameters
BATCH_SIZE = 128
EPOCHS = 20
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.2

# Model architecture
CONV_FILTERS = [32, 64, 128]
KERNEL_SIZE = (3, 3)
POOL_SIZE = (2, 2)
DROPOUT_RATE = 0.5
DENSE_UNITS = [128]

# Training parameters
OPTIMIZER = 'adam'
LOSS_FUNCTION = 'sparse_categorical_crossentropy'
METRICS = ['accuracy']

# Data augmentation
USE_AUGMENTATION = True
ROTATION_RANGE = 10
WIDTH_SHIFT_RANGE = 0.1
HEIGHT_SHIFT_RANGE = 0.1
HORIZONTAL_FLIP = True

# Visualization
PLOT_HISTORY = True
PLOT_SAMPLE_PREDICTIONS = True
NUM_SAMPLE_PREDICTIONS = 10

# Reproducibility
RANDOM_SEED = 42
