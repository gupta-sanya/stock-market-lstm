# Stock Market Analysis & LSTM Price Prediction

Exploratory analysis of four large tech stocks (AAPL, GOOG, MSFT, AMZN) and an LSTM neural network that predicts the next-day closing price of one stock.

## What the notebook covers
1. Price and trading volume history (last 12 months, via `yfinance`)
2. Moving averages (10 / 20 / 50 days) and Bollinger Bands
3. Daily returns and their distribution
4. Correlation between stocks (returns and closing prices)
5. Risk vs expected return, relative performance and rolling volatility
6. LSTM model (2 layers, 60-day window) trained on closing prices since 2012
7. Evaluation against a naive "tomorrow = today" baseline (RMSE, MAE, MAPE, directional accuracy)

## Results
| Model | RMSE | MAE | MAPE |
|-------|------|-----|------|
| LSTM | 14.28 | 11.96 | 4.00% |
| Naive baseline | 4.95 | 3.57 | 1.23% |

Directional accuracy of the LSTM was about 52%. In this run the LSTM did not beat the naive baseline: predicting price from past prices alone is hard and the model was trained for only 5 epochs. Possible improvements are longer training with early stopping, extra features (volume, indicators) and walk-forward validation. This is a learning project, not a trading signal.

## Tech stack
Python 3.11, pandas, NumPy, Matplotlib, Seaborn, scikit-learn, yfinance, TensorFlow/Keras

## How to run
```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```
Open `stock_market_lstm.ipynb` in VS Code or Jupyter, select the `venv` kernel and run all cells. Stock data is downloaded from Yahoo Finance, so an internet connection is needed. Change `TICKER` in section 6 to model a different stock.

## Files
- `stock_market_lstm.ipynb`: main notebook with outputs
- `lstm_model.keras`: trained model
- `requirements.txt`: library versions used
