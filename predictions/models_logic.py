import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score
from statsmodels.tsa.arima.model import ARIMA
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    
from sklearn.preprocessing import MinMaxScaler

def run_linear_regression(df):
    """
    Performs Linear Regression to predict next 2 days.
    """
    X = df[['Day']].values
    y = df['Close'].values
    
    model = LinearRegression()
    model.fit(X, y)
    
    # Predict next 2 days
    last_day = X[-1][0]
    next_days = np.array([[last_day + 1], [last_day + 2]])
    predictions = model.predict(next_days)
    
    mse = mean_squared_error(y, model.predict(X))
    rmse = np.sqrt(mse)
    
    return predictions.tolist(), {"MSE": round(mse, 4), "RMSE": round(rmse, 4)}

def run_logistic_regression(df):
    """
    Performs Logistic Regression to predict price movement (Up/Down).
    """
    df['Movement'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    df = df.dropna()
    
    X = df[['Day', 'Close']].values
    y = df['Movement'].values
    
    model = LogisticRegression()
    model.fit(X, y)
    
    # Predict next 2 days movement
    last_day = X[-1][0]
    last_close = df['Close'].iloc[-1]
    next_days = np.array([[last_day + 1, last_close], [last_day + 2, last_close]])
    predictions = model.predict(next_days)
    
    acc = accuracy_score(y, model.predict(X))
    
    return predictions.tolist(), {"Accuracy": f"{round(acc * 100, 2)}%"}

def run_arima_model(df, forecast_days=7):
    """
    Performs ARIMA modeling for next 7 days.
    """
    series = df['Close'].values
    
    # p, d, q parameters (simplified)
    model = ARIMA(series, order=(5, 1, 0))
    model_fit = model.fit()
    
    forecast = model_fit.forecast(steps=forecast_days)
    
    # Simplified evaluation: AIC
    return forecast.tolist(), {"AIC": round(model_fit.aic, 2)}

def run_lstm_model(df, forecast_days=7):
    """
    Performs LSTM RNN for next 7 days.
    """
    if not TENSORFLOW_AVAILABLE:
        return [0] * forecast_days, {"Error": "TensorFlow not available on this Python version (3.14). Please use Python 3.11/3.12 for LSTM."}
        
    data = df[['Close']].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    X, y = [], []
    for i in range(60, len(scaled_data)):
        X.append(scaled_data[i-60:i, 0])
        y.append(scaled_data[i, 0])
        
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
        LSTM(50),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, epochs=1, batch_size=1, verbose=0)
    
    # Predict next 7 days
    last_60_days = scaled_data[-60:]
    current_batch = last_60_days.reshape((1, 60, 1))
    
    predictions = []
    for _ in range(forecast_days):
        pred = model.predict(current_batch, verbose=0)
        predictions.append(pred[0, 0])
        current_batch = np.append(current_batch[:, 1:, :], [[pred[0]]], axis=1)
        
    res = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    
    return res.flatten().tolist(), {"Loss": "Optimized via Adam"}
