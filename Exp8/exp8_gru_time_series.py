import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import CSVLogger, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import GRU, Dense, Dropout, Input
from tensorflow.keras.losses import Huber
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import set_random_seed

sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["axes.labelsize"] = 11

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "Monthly_Modal_Time_Series.csv"
OUTPUT_DIR = BASE_DIR / "experiment_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

SEED = 42
WINDOW_SIZE = 18
TEST_MONTHS = 24
EPOCHS = 300
BATCH_SIZE = 8
TARGET_COLUMN = "Target"
FILTER_AGENCY = "City and County of Honolulu"
FILTER_MODE = "MB"

set_random_seed(SEED)
np.random.seed(SEED)


def create_supervised_sequences(values: np.ndarray, months, window_size: int):
    features, targets = [], []
    for index in range(window_size, len(values)):
        window_rows = []
        for month_index in range(index - window_size, index):
            month_number = months[month_index].month
            window_rows.append(
                [
                    values[month_index],
                    np.sin(2 * np.pi * month_number / 12),
                    np.cos(2 * np.pi * month_number / 12),
                ]
            )
        features.append(window_rows)
        targets.append(values[index])
    return np.array(features), np.array(targets)


def invert_scale(scaler: MinMaxScaler, array_2d: np.ndarray) -> np.ndarray:
    return scaler.inverse_transform(array_2d.reshape(-1, 1)).reshape(-1)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    r2 = r2_score(y_true, y_pred)
    return {
        "MSE": float(mse),
        "RMSE": rmse,
        "MAE": float(mae),
        "MAPE_percent": float(mape),
        "R2_score": float(r2),
    }


def save_line_plot(x, y, title, xlabel, ylabel, path, label=None, second_series=None, second_label=None):
    plt.figure()
    plt.plot(x, y, label=label, linewidth=2)
    if second_series is not None:
        plt.plot(x, second_series, label=second_label, linewidth=2)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if label or second_series is not None:
        plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


# Step 1 and 2: load the real-world government dataset and create a forecasting target
raw_source_df = pd.read_csv(DATA_PATH)
working_df = raw_source_df[["Agency", "Mode", "MoYr", "Unlinked Passenger Trips"]].copy()
working_df = working_df[
    (working_df["Agency"] == FILTER_AGENCY) & (working_df["Mode"] == FILTER_MODE)
].copy()
working_df["Month"] = pd.to_datetime(working_df["MoYr"], format="%Y%m%d", errors="coerce")
working_df["Unlinked Passenger Trips"] = pd.to_numeric(
    working_df["Unlinked Passenger Trips"], errors="coerce"
)
working_df = working_df.dropna(subset=["Month", "Unlinked Passenger Trips"])

# Build a single monthly target series from one complete agency-mode combination.
time_series_df = (
    working_df.groupby("Month", as_index=False)["Unlinked Passenger Trips"]
    .sum()
    .sort_values("Month")
    .reset_index(drop=True)
)
time_series_df = time_series_df.rename(columns={"Unlinked Passenger Trips": TARGET_COLUMN})
time_series_df["Target_Next_Month"] = time_series_df[TARGET_COLUMN].shift(-1)
time_series_df.to_csv(OUTPUT_DIR / "loaded_dataset.csv", index=False)

save_line_plot(
    time_series_df["Month"],
    time_series_df[TARGET_COLUMN],
    "Monthly Modal Transit Demand",
    "Month",
    TARGET_COLUMN,
    OUTPUT_DIR / "dataset_plot.png",
    label=TARGET_COLUMN,
)

# Step 3 and 4: normalize the target series and create supervised sequences
time_series_df["Log_Target"] = np.log1p(time_series_df[TARGET_COLUMN])
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_values = scaler.fit_transform(time_series_df[["Log_Target"]]).reshape(-1)
normalized_df = pd.DataFrame(
    {
        "Month": time_series_df["Month"],
        TARGET_COLUMN: time_series_df[TARGET_COLUMN],
        "Log_Target": time_series_df["Log_Target"],
        f"Scaled_{TARGET_COLUMN}": scaled_values,
        "Target_Next_Month": time_series_df["Target_Next_Month"],
    }
)
normalized_df.to_csv(OUTPUT_DIR / "normalized_dataset.csv", index=False)

X, y = create_supervised_sequences(scaled_values, time_series_df["Month"].tolist(), WINDOW_SIZE)
sequence_months = time_series_df["Month"].iloc[WINDOW_SIZE:].reset_index(drop=True)

split_index = len(X) - TEST_MONTHS
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]
train_months, test_months = sequence_months[:split_index], sequence_months[split_index:]

np.save(OUTPUT_DIR / "X_train.npy", X_train)
np.save(OUTPUT_DIR / "X_test.npy", X_test)
np.save(OUTPUT_DIR / "y_train.npy", y_train)
np.save(OUTPUT_DIR / "y_test.npy", y_test)

# Step 5 to 8: define and train the GRU model
model = Sequential(
    [
        Input(shape=(WINDOW_SIZE, X_train.shape[2])),
        GRU(128, return_sequences=True),
        Dropout(0.15),
        GRU(64),
        Dropout(0.10),
        Dense(16, activation="relu"),
        Dense(1),
    ]
)
model.compile(optimizer=Adam(learning_rate=0.001), loss=Huber(), metrics=["mae"])
print("\nGRU MODEL ARCHITECTURE\n")
model.summary()

callbacks = [
    EarlyStopping(monitor="val_loss", patience=25, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=8, min_lr=1e-5, verbose=1),
    CSVLogger(OUTPUT_DIR / "training_log.csv"),
]

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=1,
    callbacks=callbacks,
)

model.save(OUTPUT_DIR / "gru_model.keras")

# Step 9 to 11: evaluate predictions
train_predictions_scaled = model.predict(X_train, verbose=0).reshape(-1, 1)
test_predictions_scaled = model.predict(X_test, verbose=0).reshape(-1, 1)

train_predictions = np.expm1(invert_scale(scaler, train_predictions_scaled))
test_predictions = np.expm1(invert_scale(scaler, test_predictions_scaled))
y_train_actual = np.expm1(invert_scale(scaler, y_train))
y_test_actual = np.expm1(invert_scale(scaler, y_test))

train_results = pd.DataFrame(
    {
        "Month": train_months,
        "Actual": y_train_actual,
        "Predicted": train_predictions,
        "Residual": y_train_actual - train_predictions,
    }
)
test_results = pd.DataFrame(
    {
        "Month": test_months,
        "Actual": y_test_actual,
        "Predicted": test_predictions,
        "Residual": y_test_actual - test_predictions,
    }
)

train_results.to_csv(OUTPUT_DIR / "train_predictions.csv", index=False)
test_results.to_csv(OUTPUT_DIR / "test_predictions.csv", index=False)

metrics = regression_metrics(y_test_actual, test_predictions)
metrics["Training_loss"] = float(history.history["loss"][-1])
metrics["Validation_loss"] = float(history.history["val_loss"][-1])

# Convert regression output into directional classes for the confusion matrix.
actual_direction = np.where(np.diff(y_test_actual, prepend=y_train_actual[-1]) >= 0, "Increase", "Decrease")
predicted_direction = np.where(
    np.diff(test_predictions, prepend=y_train_actual[-1]) >= 0, "Increase", "Decrease"
)
labels = ["Decrease", "Increase"]
cm = confusion_matrix(actual_direction, predicted_direction, labels=labels)

cm_df = pd.DataFrame(
    cm,
    index=[f"Actual_{label}" for label in labels],
    columns=[f"Predicted_{label}" for label in labels],
)
cm_df.to_csv(OUTPUT_DIR / "confusion_matrix.csv")

metrics_df = pd.DataFrame(
    {
        "Metric": list(metrics.keys()) + ["Directional_accuracy_percent"],
        "Value": list(metrics.values()) + [float((actual_direction == predicted_direction).mean() * 100)],
    }
)
metrics_df.to_csv(OUTPUT_DIR / "metrics.csv", index=False)
with open(OUTPUT_DIR / "metrics.json", "w", encoding="utf-8") as file:
    json.dump({row["Metric"]: row["Value"] for _, row in metrics_df.iterrows()}, file, indent=2)

# Step 12: plots
save_line_plot(
    history.epoch,
    history.history["loss"],
    "GRU Training Loss",
    "Epoch",
    "Loss",
    OUTPUT_DIR / "training_loss_plot.png",
    label="Training Loss",
    second_series=history.history["val_loss"],
    second_label="Validation Loss",
)

plt.figure()
plt.plot(train_results["Month"], train_results["Actual"], label="Train Actual", linewidth=2)
plt.plot(train_results["Month"], train_results["Predicted"], label="Train Predicted", linewidth=2)
plt.plot(test_results["Month"], test_results["Actual"], label="Test Actual", linewidth=2)
plt.plot(test_results["Month"], test_results["Predicted"], label="Test Predicted", linewidth=2)
plt.axvline(test_results["Month"].iloc[0], color="black", linestyle="--", label="Test Split")
plt.title("Actual vs Predicted Target Values")
plt.xlabel("Month")
plt.ylabel(TARGET_COLUMN)
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "actual_vs_predicted_plot.png", dpi=300)
plt.close()

plt.figure()
plt.plot(test_results["Month"], test_results["Residual"], marker="o")
plt.axhline(0, color="black", linestyle="--")
plt.title("Residual Errors on Test Set")
plt.xlabel("Month")
plt.ylabel("Residual")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "residual_plot.png", dpi=300)
plt.close()

percentage_error = ((test_results["Actual"] - test_results["Predicted"]) / test_results["Actual"]) * 100
plt.figure()
plt.plot(test_results["Month"], percentage_error, marker="o", color="darkorange")
plt.axhline(0, color="black", linestyle="--")
plt.title("Percentage Error on Test Set")
plt.xlabel("Month")
plt.ylabel("Percentage Error (%)")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "percentage_error_plot.png", dpi=300)
plt.close()

plt.figure(figsize=(6, 5))
sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues")
plt.title("Directional Confusion Matrix")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "confusion_matrix_plot.png", dpi=300)
plt.close()

# Step 13: forecast the next month
last_window_rows = []
for month_value, scaled_value in zip(time_series_df["Month"].tolist()[-WINDOW_SIZE:], scaled_values[-WINDOW_SIZE:]):
    month_number = month_value.month
    last_window_rows.append(
        [
            scaled_value,
            np.sin(2 * np.pi * month_number / 12),
            np.cos(2 * np.pi * month_number / 12),
        ]
    )
last_window = np.array(last_window_rows, dtype=float).reshape(1, WINDOW_SIZE, 3)
next_month_scaled = model.predict(last_window, verbose=0)
next_month_forecast = np.expm1(invert_scale(scaler, next_month_scaled))[0]
next_month_date = (time_series_df["Month"].iloc[-1] + pd.offsets.MonthBegin(1)).strftime("%Y-%m")

forecast_df = pd.DataFrame(
    [{"Forecast_Month": next_month_date, f"Forecasted_{TARGET_COLUMN}": round(float(next_month_forecast), 2)}]
)
forecast_df.to_csv(OUTPUT_DIR / "next_month_forecast.csv", index=False)

with open(OUTPUT_DIR / "summary.txt", "w", encoding="utf-8") as file:
    file.write("Experiment 8: GRU Time-Series Forecasting\n")
    file.write(f"Dataset: {DATA_PATH.name}\n")
    file.write("Source columns used: Agency, Mode, MoYr, Unlinked Passenger Trips\n")
    file.write(f"Filtered Agency: {FILTER_AGENCY}\n")
    file.write(f"Filtered Mode: {FILTER_MODE}\n")
    file.write(f"Target column created: {TARGET_COLUMN}\n")
    file.write(f"Date range: {time_series_df['Month'].iloc[0]:%Y-%m} to {time_series_df['Month'].iloc[-1]:%Y-%m}\n")
    file.write(f"Window size: {WINDOW_SIZE}\n")
    file.write(f"Train samples: {len(X_train)}\n")
    file.write(f"Test samples: {len(X_test)}\n")
    file.write("\nMetrics\n")
    for metric_name, metric_value in metrics_df.itertuples(index=False):
        file.write(f"{metric_name}: {metric_value:.4f}\n")
    file.write("\nNext Month Forecast\n")
    file.write(f"{next_month_date}: {next_month_forecast:.2f}\n")

print("Experiment 8 completed successfully.")
print(metrics_df)
print(f"Next month forecast ({next_month_date}): {next_month_forecast:.2f}")
