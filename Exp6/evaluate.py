# evaluate.py

import os
import json
import config
from dataset import load_data
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix

# Load Data
X_train, X_test, y_train, y_test = load_data()

# Load Best Model
model = load_model(os.path.join(config.MODEL_DIR, "best_model.keras"))

loss, accuracy = model.evaluate(X_test, y_test)
print("Test Accuracy:", accuracy)

# Predictions
y_pred = (model.predict(X_test) > 0.5).astype("int32")

report = classification_report(y_test, y_pred, output_dict=True)
conf_matrix = confusion_matrix(y_test, y_pred).tolist()

# Save Metrics
metrics = {
    "test_accuracy": float(accuracy),
    "classification_report": report,
    "confusion_matrix": conf_matrix
}

with open(os.path.join(config.METRIC_DIR, "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=4)

print("Evaluation metrics saved.")