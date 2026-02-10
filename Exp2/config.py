import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')

DATASET_NAME = 'fashion_mnist'
IMAGE_SIZE = (28, 28)
NUM_CLASSES = 9
CLASS_NAMES = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Sneaker', 'Bag', 'Ankle boot']

BATCH_SIZE = 128
EPOCHS = 30
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.2

CONV_FILTERS = [32, 64, 128]
KERNEL_SIZE = (3, 3)
POOL_SIZE = (2, 2)
DROPOUT_RATE = 0.4
DENSE_UNITS = [256]

OPTIMIZER = 'adam'
LOSS_FUNCTION = 'sparse_categorical_crossentropy'
METRICS = ['accuracy']

USE_AUGMENTATION = True
ROTATION_RANGE = 12
WIDTH_SHIFT_RANGE = 0.12
HEIGHT_SHIFT_RANGE = 0.12
HORIZONTAL_FLIP = True

PLOT_HISTORY = True
PLOT_SAMPLE_PREDICTIONS = True
NUM_SAMPLE_PREDICTIONS = 10

RANDOM_SEED = 42
