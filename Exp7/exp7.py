import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.losses import Huber
from tensorflow.keras.callbacks import CSVLogger

# ===============================
# Create output directory
# ===============================

output_dir = "experiment_outputs"
os.makedirs(output_dir, exist_ok=True)

# ===============================
# STEP 1: Generate Synthetic Dataset
# ===============================

np.random.seed(42)

time_steps = 500
time = np.arange(time_steps)

trend = 0.05 * time
seasonality = 10 * np.sin(0.1 * time)
noise = np.random.normal(0, 2, time_steps)

series = trend + seasonality + noise

df = pd.DataFrame({
    "Time": time,
    "Value": series
})

df.to_csv(f"{output_dir}/synthetic_dataset.csv", index=False)

# ===============================
# STEP 2: Plot dataset
# ===============================

plt.figure()
plt.plot(df["Time"], df["Value"])
plt.title("Synthetic Time Series Dataset")
plt.xlabel("Time")
plt.ylabel("Value")
plt.savefig(f"{output_dir}/dataset_plot.png")
plt.close()

# ===============================
# STEP 3: Normalize dataset
# ===============================

scaler = MinMaxScaler()

scaled_data = scaler.fit_transform(df[["Value"]])

scaled_df = pd.DataFrame(scaled_data, columns=["Scaled_Value"])
scaled_df.to_csv(f"{output_dir}/normalized_dataset.csv", index=False)

# ===============================
# STEP 4: Sliding Window
# ===============================

def create_dataset(data, window_size):

    X = []
    y = []

    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size])

    return np.array(X), np.array(y)


window_size = 8

X, y = create_dataset(scaled_data, window_size)

np.save(f"{output_dir}/X.npy", X)
np.save(f"{output_dir}/y.npy", y)

# ===============================
# STEP 5: Train Test Split
# ===============================

split = int(len(X) * 0.8)

X_train = X[:split]
X_test = X[split:]

y_train = y[:split]
y_test = y[split:]

np.save(f"{output_dir}/X_train.npy", X_train)
np.save(f"{output_dir}/X_test.npy", X_test)

# ===============================
# STEP 6: Reshape for LSTM
# ===============================

X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# ===============================
# STEP 7: LSTM Model
# ===============================

model = Sequential()

model.add(LSTM(128, return_sequences=True, input_shape=(window_size,1)))
model.add(Dropout(0.2))

model.add(LSTM(64, return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(32))

model.add(Dense(16, activation="relu"))
model.add(Dense(1))

model.compile(
    optimizer="adam",
    loss=Huber()
)

model.summary()

# ===============================
# STEP 8: Train Model
# ===============================

logger = CSVLogger(f"{output_dir}/training_log.csv")

history = model.fit(
    X_train,
    y_train,
    epochs=80,
    batch_size=4,
    validation_data=(X_test, y_test),
    callbacks=[logger],
    verbose=1
)

# ===============================
# Save model
# ===============================

model.save(f"{output_dir}/lstm_model.h5")

# ===============================
# STEP 9: Predictions
# ===============================

train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

train_pred = scaler.inverse_transform(train_pred)
test_pred = scaler.inverse_transform(test_pred)

y_train_inv = scaler.inverse_transform(y_train)
y_test_inv = scaler.inverse_transform(y_test)

pd.DataFrame(train_pred).to_csv(f"{output_dir}/train_predictions.csv", index=False)
pd.DataFrame(test_pred).to_csv(f"{output_dir}/test_predictions.csv", index=False)

# ===============================
# STEP 10: Metrics
# ===============================

def MAPE(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def SMAPE(y_true, y_pred):
    return 100/len(y_true) * np.sum(
        2*np.abs(y_pred-y_true)/(np.abs(y_true)+np.abs(y_pred))
    )

mse = mean_squared_error(y_test_inv, test_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test_inv, test_pred)
r2 = r2_score(y_test_inv, test_pred)

mape = MAPE(y_test_inv, test_pred)
smape = SMAPE(y_test_inv, test_pred)

accuracy = 100 - mape

train_loss = history.history['loss'][-1]
val_loss = history.history['val_loss'][-1]

metrics = pd.DataFrame({

    "Metric":[
        "MSE",
        "RMSE",
        "MAE",
        "MAPE (%)",
        "SMAPE (%)",
        "R2 Score",
        "Forecast Accuracy (%)",
        "Training Loss",
        "Validation Loss"
    ],

    "Value":[
        mse,
        rmse,
        mae,
        mape,
        smape,
        r2,
        accuracy,
        train_loss,
        val_loss
    ]

})

metrics.to_csv(f"{output_dir}/metrics.csv", index=False)

print("\nMODEL PERFORMANCE\n")
print(metrics)

# ===============================
# STEP 11: Actual vs Predicted Plot
# ===============================

plt.figure()

plt.plot(y_test_inv, label="Actual")
plt.plot(test_pred, label="Predicted")

plt.title("Actual vs Predicted")
plt.legend()

plt.savefig(f"{output_dir}/prediction_plot.png")
plt.close()

# ===============================
# STEP 12: Loss Plot
# ===============================

plt.figure()

plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")

plt.title("Training Loss Curve")
plt.legend()

plt.savefig(f"{output_dir}/loss_plot.png")
plt.close()

# ===============================
# STEP 13: Percentage Error Plot
# ===============================

percentage_error = ((y_test_inv - test_pred)/y_test_inv)*100

plt.figure()

plt.plot(percentage_error)

plt.title("Percentage Forecast Error")
plt.xlabel("Time Step")
plt.ylabel("Error (%)")

plt.savefig(f"{output_dir}/percentage_error_plot.png")
plt.close()

print("\nExperiment Completed Successfully")
print("All outputs saved in folder:", output_dir)
