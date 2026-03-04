# train.py

import tensorflow as tf
import config
from dataset import load_data
from model import build_model
import matplotlib.pyplot as plt


# Load Data
X_train, X_test, y_train, y_test = load_data()

# Build Model
model = build_model()

# Compile Model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train Model
history = model.fit(
    X_train,
    y_train,
    epochs=config.EPOCHS,
    batch_size=config.BATCH_SIZE,
    validation_split=config.TEST_SPLIT
)

# Save Model
model.save("rnn_model.h5")

# Plot Accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(["Train", "Validation"])
plt.show()

# Plot Loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(["Train", "Validation"])
plt.show()