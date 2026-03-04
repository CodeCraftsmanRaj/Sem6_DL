# evaluate.py

from dataset import load_data
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix

# Load Data
X_train, X_test, y_train, y_test = load_data()

# Load Trained Model
model = load_model("rnn_model.h5")

# Evaluate
loss, accuracy = model.evaluate(X_test, y_test)
print("Test Accuracy:", accuracy)

# Predictions
y_pred = (model.predict(X_test) > 0.5).astype("int32")

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))