# train.py

import os
import json
import matplotlib.pyplot as plt
import config
from dataset import load_data
from model import build_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Create Output Directories
os.makedirs(config.MODEL_DIR, exist_ok=True)
os.makedirs(config.PLOT_DIR, exist_ok=True)
os.makedirs(config.METRIC_DIR, exist_ok=True)
os.makedirs(config.HISTORY_DIR, exist_ok=True)

# Load Data
X_train, X_test, y_train, y_test = load_data()

# Build Model
model = build_model()

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Callbacks
early_stop = EarlyStopping(patience=2, restore_best_weights=True)

checkpoint = ModelCheckpoint(
    filepath=os.path.join(config.MODEL_DIR, "best_model.keras"),
    save_best_only=True
)

# Train
history = model.fit(
    X_train,
    y_train,
    epochs=config.EPOCHS,
    batch_size=config.BATCH_SIZE,
    validation_split=config.TEST_SPLIT,
    callbacks=[early_stop, checkpoint]
)

# Save Final Model
model.save(os.path.join(config.MODEL_DIR, "final_model.keras"))

# Save History
with open(os.path.join(config.HISTORY_DIR, "history.json"), "w") as f:
    json.dump(history.history, f)

# ---- Save Accuracy Plot ----
plt.figure()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(["Train", "Validation"])
plt.savefig(os.path.join(config.PLOT_DIR, "accuracy.png"))
plt.close()

# ---- Save Loss Plot ----
plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(["Train", "Validation"])
plt.savefig(os.path.join(config.PLOT_DIR, "loss.png"))
plt.close()

print("Training complete. Outputs saved in 'outputs/' directory.")