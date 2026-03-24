# Experiment 8: GRU Time-Series Forecasting

This experiment implements a Gated Recurrent Unit (GRU) model for monthly airline passenger forecasting.

## Files

- `exp8_gru_time_series.py`: Main GRU training and evaluation script
- `airline_passengers_custom.csv`: Dataset used in the experiment
- `experiment_outputs/`: Saved metrics, predictions, plots, and trained model

## Run

From the project root:

```bash
uv run python Exp8/exp8_gru_time_series.py
```

## Saved Outputs

- Dataset and normalization CSV files
- `gru_model.keras`
- `metrics.csv` and `metrics.json`
- `train_predictions.csv` and `test_predictions.csv`
- `next_month_forecast.csv`
- Training loss, prediction, residual, percentage error, and confusion matrix plots

## Note on Confusion Matrix

Because forecasting is a regression task, the confusion matrix is computed on movement direction:

- `Increase`: passenger count went up compared with the previous month
- `Decrease`: passenger count went down compared with the previous month
