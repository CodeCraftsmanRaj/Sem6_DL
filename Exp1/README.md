# Experiment 2: CNN for Image Classification
## Fashion MNIST Dataset

**Author:** Raj Kalpesh Mathuria  
**UID:** 2023300139  
**Division:** PE-C  
**Course:** Deep Learning (CE312)  
**Institute:** Sardar Patel Institute of Technology, Mumbai  

---

## Objective
To design, implement, and evaluate a Convolutional Neural Network (CNN) using Python for image classification using accuracy and visualization techniques.

## Prerequisites
- Basic knowledge of Python programming
- Linear algebra, probability, and statistics
- Fundamentals of neural networks and deep learning
- Basic understanding of images and pixels

---

## Project Structure

```
Exp1/
├── main.py                 # Main execution script
├── config.py              # Configuration file with hyperparameters
├── src/
│   ├── data_loader.py     # Data loading and preprocessing
│   ├── model.py           # CNN model architecture
│   ├── train.py           # Training logic
│   └── plotting.py        # Visualization functions
├── data/                  # Dataset storage (auto-downloaded)
├── models/                # Saved models
│   ├── best_model.h5      # Best model during training
│   └── final_model.h5     # Final trained model
└── outputs/               # Plots and visualizations
    ├── training_history.png
    ├── confusion_matrix.png
    ├── class_wise_accuracy.png
    ├── sample_predictions.png
    ├── misclassified_samples.png
    └── learning_rate_schedule.png
```

---

## Algorithm Steps

1. **Import required Python libraries** - TensorFlow, NumPy, Matplotlib, etc.
2. **Load the image dataset** - Fashion MNIST dataset
3. **Split data into training and testing sets** - 80% train, 20% validation
4. **Preprocess and normalize image data** - Scale pixels to [0, 1]
5. **Define CNN model architecture** - Conv layers, pooling, dropout, dense layers
6. **Compile the model** - Adam optimizer, categorical crossentropy loss
7. **Train the CNN model** - With data augmentation and callbacks
8. **Validate the model during training** - Track validation metrics
9. **Evaluate the trained model on test data** - Calculate final accuracy
10. **Calculate accuracy and loss** - Overall and class-wise metrics
11. **Visualize training and validation performance** - Generate plots

---

## Model Architecture

The CNN model consists of:

### Convolutional Blocks:
- **Block 1:** 2x Conv2D (32 filters) + BatchNorm + MaxPooling + Dropout (0.25)
- **Block 2:** 2x Conv2D (64 filters) + BatchNorm + MaxPooling + Dropout (0.25)
- **Block 3:** Conv2D (128 filters) + BatchNorm + MaxPooling + Dropout (0.4)

### Fully Connected Layers:
- **Flatten layer**
- **Dense layer** (128 units) + BatchNorm + Dropout (0.5)
- **Output layer** (10 units with Softmax activation)

### Key Features:
- **Filters:** Convolution layers with 3x3 kernels
- **Pooling:** Max pooling (2x2) for downsampling
- **Regularization:** Dropout to prevent overfitting
- **Normalization:** Batch normalization for stable training
- **Activation:** ReLU for hidden layers, Softmax for output

---

## Configuration

Key hyperparameters (can be modified in `config.py`):

```python
BATCH_SIZE = 128
EPOCHS = 20
LEARNING_RATE = 0.001
DROPOUT_RATE = 0.5
VALIDATION_SPLIT = 0.2
```

Data Augmentation:
```python
USE_AUGMENTATION = True
ROTATION_RANGE = 10
WIDTH_SHIFT_RANGE = 0.1
HEIGHT_SHIFT_RANGE = 0.1
HORIZONTAL_FLIP = True
```

---

## Installation & Setup

### 1. Install Dependencies

```bash
pip install tensorflow numpy matplotlib seaborn scikit-learn
```

### 2. Run the Experiment

```bash
cd Exp1
python main.py
```

---

## Pre-Lab Questions & Answers

### 1. What is a Convolutional Neural Network (CNN) and how does it differ from a traditional neural network?

**Answer:** A CNN is a specialized neural network designed for processing grid-like data such as images. Unlike traditional neural networks that use fully connected layers, CNNs use:
- **Convolutional layers** that apply filters to detect local patterns (edges, textures)
- **Pooling layers** to reduce spatial dimensions
- **Parameter sharing** where the same filter is applied across the entire image, reducing parameters
- **Spatial hierarchy** to learn increasingly complex features

Traditional neural networks don't preserve spatial structure and require many more parameters for image data.

### 2. Why are convolution and pooling layers important in CNNs?

**Answer:** 
- **Convolution layers** detect local patterns and features (edges, corners, textures) by sliding filters across the input. They preserve spatial relationships and use parameter sharing for efficiency.
- **Pooling layers** reduce spatial dimensions, making the network:
  - More computationally efficient
  - Translation-invariant (recognizes patterns regardless of position)
  - Less prone to overfitting
  - Able to capture broader features in deeper layers

### 3. What is the role of filters in convolution layers?

**Answer:** Filters (kernels) are small matrices that slide over the input image to detect specific patterns:
- Early layers learn simple features (edges, colors)
- Deeper layers learn complex patterns (shapes, objects)
- Each filter produces a feature map highlighting where its pattern appears
- Multiple filters at each layer detect different features
- Filters are learned automatically during training through backpropagation

### 4. What is overfitting and how does dropout help in reducing it?

**Answer:** 
**Overfitting** occurs when a model learns training data too well, including noise, causing poor performance on new data.

**Dropout** prevents overfitting by:
- Randomly deactivating neurons during training (e.g., 50% with dropout=0.5)
- Forcing the network to learn redundant representations
- Preventing co-adaptation of neurons
- Creating an ensemble effect by training multiple sub-networks
- During inference, all neurons are active but outputs are scaled

### 5. What is Softmax activation and why is it used in classification?

**Answer:** Softmax is an activation function that converts raw model outputs (logits) into probability distributions:
- Outputs sum to 1.0
- Each output represents the probability of a class
- Uses exponential function: $\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_{j} e^{x_j}}$
- Ideal for multi-class classification
- Works with cross-entropy loss for training
- Provides interpretable confidence scores

### 6. What is data augmentation and how does it improve CNN performance?

**Answer:** Data augmentation artificially increases training data by applying random transformations:
- **Transformations:** rotation, shifting, flipping, zooming, brightness changes
- **Benefits:**
  - Increases dataset size without collecting more data
  - Improves model generalization
  - Reduces overfitting
  - Makes model robust to variations
  - Helps with class imbalance
- Creates variations that the model might encounter in real-world scenarios

---

## Expected Output

The experiment will generate:

1. **Model Summary** - Architecture details and parameter count
2. **Training Progress** - Epoch-by-epoch metrics
3. **Test Accuracy** - Final performance on test set
4. **Class-wise Accuracy** - Performance per fashion category
5. **Visualizations:**
   - Training/validation accuracy and loss curves
   - Confusion matrix
   - Sample predictions
   - Misclassified examples
   - Class-wise accuracy bar chart

---

## Fashion MNIST Classes

0. T-shirt/top
1. Trouser
2. Pullover
3. Dress
4. Coat
5. Sandal
6. Shirt
7. Sneaker
8. Bag
9. Ankle boot

---

## Conclusion

This experiment demonstrates:
- Implementation of a CNN from scratch using TensorFlow/Keras
- Proper data preprocessing and augmentation techniques
- Model training with regularization (dropout, batch normalization)
- Comprehensive evaluation using multiple metrics
- Visualization of results for analysis

The CNN successfully classifies Fashion MNIST images with high accuracy, showcasing the power of deep learning for computer vision tasks.

---

## References

- TensorFlow Documentation: https://www.tensorflow.org/
- Fashion MNIST Dataset: https://github.com/zalandoresearch/fashion-mnist
- Deep Learning by Ian Goodfellow, Yoshua Bengio, and Aaron Courville

---

## Troubleshooting

If you encounter issues:

1. **Memory Error:** Reduce `BATCH_SIZE` in config.py
2. **Slow Training:** Disable data augmentation or reduce epochs
3. **Import Errors:** Ensure all dependencies are installed
4. **GPU Issues:** TensorFlow will automatically use CPU if GPU is unavailable

---

**Date:** February 3, 2026  
**Status:** Completed ✓
