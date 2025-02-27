# brkb_svm_volatility_forecast

- This project predicts future volatility of **$BRKB** (Berkshire Hathaway) using a PyTorch neural network and classifies volatility as good or bad with an SVM model.
- It includes data preprocessing, model training, and visualization of predictions.
- Good Volatility: Positive stock return over 30 days, labeled as 1.
- Bad Volatility: Flat or negative stock return over 30 days, labeled as 0.
- Accuracy: Average confidence in predicting bad volatility.

## Files
- `brkb_svm_volatility_forecast.py`: Main script for training and predicting volatility.
- `BRKB.csv`: Dataset containing historical stock prices of $BRKB.
- No output image file is generated; visualization is displayed interactively via Plotly.

## Libraries Used
- `numpy`
- `pandas`
- `torch` (PyTorch)
- `sklearn` (scikit-learn)
- `plotly`

## Timeframe
- **Input**: Data ranges from **2023-05-01** to the latest date in `BRKB.csv`.
- **Output**: Predicts the next 10 days of volatility and classifies volatility periods.
