# =========================================
# EXPERIMENT 10: ANOMALY DETECTION (FINAL SUBMISSION)
# =========================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from sklearn.datasets import fetch_kddcup99
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.callbacks import EarlyStopping

# =========================================
# SETUP
# =========================================

np.random.seed(42)
os.makedirs("plots", exist_ok=True)

# =========================================
# LOAD DATA
# =========================================

print("Loading dataset...")
data = fetch_kddcup99(percent10=True)

X = pd.DataFrame(data.data)

# Convert categorical → numeric
for col in X.columns:
    try:
        X[col] = X[col].astype(float)
    except:
        X[col] = pd.factorize(X[col])[0]

y = pd.Series(data.target)
y = np.where(y == b'normal.', 0, 1)

# Sample dataset
X = X.sample(80000, random_state=42)
y = y[X.index]

print("Dataset shape:", X.shape)

# =========================================
# NORMALIZATION
# =========================================

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# =========================================
# SPLIT DATA
# =========================================

X_normal = X_scaled[y == 0]

X_train, X_val = train_test_split(X_normal, test_size=0.2, random_state=42)

X_test = X_scaled
y_test = y

# =========================================
# MODEL
# =========================================

input_dim = X_train.shape[1]

input_layer = Input(shape=(input_dim,))

encoded = Dense(64, activation='relu')(input_layer)
encoded = Dense(32, activation='relu')(encoded)
encoded = Dense(16, activation='relu')(encoded)

decoded = Dense(32, activation='relu')(encoded)
decoded = Dense(64, activation='relu')(decoded)
decoded = Dense(input_dim, activation='sigmoid')(decoded)

autoencoder = Model(input_layer, decoded)
autoencoder.compile(optimizer='adam', loss='mae')

# =========================================
# DENOISING
# =========================================

noise_factor = 0.05

X_train_noisy = X_train + noise_factor * np.random.normal(size=X_train.shape)
X_val_noisy = X_val + noise_factor * np.random.normal(size=X_val.shape)

X_train_noisy = np.clip(X_train_noisy, 0., 1.)
X_val_noisy = np.clip(X_val_noisy, 0., 1.)

# =========================================
# TRAIN
# =========================================

early_stop = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)

history = autoencoder.fit(
    X_train_noisy, X_train,
    epochs=20,
    batch_size=256,
    validation_data=(X_val_noisy, X_val),
    callbacks=[early_stop],
    verbose=1
)

# =========================================
# THRESHOLD (FINAL)
# =========================================

recon_val = autoencoder.predict(X_val)
val_mae = np.mean(np.abs(X_val - recon_val), axis=1)

threshold = np.percentile(val_mae, 90)

print("Final Threshold:", threshold)

# =========================================
# TEST
# =========================================

recon_test = autoencoder.predict(X_test)
test_mae = np.mean(np.abs(X_test - recon_test), axis=1)

y_pred = (test_mae > threshold).astype(int)

# =========================================
# RESULTS
# =========================================

print("\nFinal Classification Report:\n")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))

# =========================================
# PLOTS
# =========================================

plt.figure()
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Val')
plt.legend()
plt.title("Loss Curve")
plt.savefig("plots/loss_curve.png")
plt.show()

plt.figure()
plt.hist(test_mae, bins=50)
plt.axvline(threshold, color='r', linestyle='--')
plt.title("Error Distribution")
plt.savefig("plots/error_distribution.png")
plt.show()

plt.figure()
plt.plot(test_mae[:2000])
plt.axhline(threshold, color='r', linestyle='--')
plt.title("Error vs Samples")
plt.savefig("plots/error_vs_samples.png")
plt.show()