import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def fetch_stock_data(symbol, period="1y"):
    """
    Fetches historical stock data from yfinance and cleans it.
    """
    # Append .NS for Indian stocks if not present and if it's not a crypto symbol
    if not symbol.endswith(".NS") and not symbol.endswith("-USD"):
        ticker_symbol = f"{symbol.upper()}.NS"
    else:
        ticker_symbol = symbol.upper()

    try:
        ticker = yf.Ticker(ticker_symbol)
        df = ticker.history(period=period)
        
        if df.empty:
            # Fallback for symbols that might not be in the Indian market or require different suffix
            ticker = yf.Ticker(symbol.upper())
            df = ticker.history(period=period)
            
        if df.empty:
            return None, "No data found for symbol."

        # Data Cleaning
        df = df.dropna()
        df.reset_index(inplace=True)
        
        return df, None
    except Exception as e:
        return None, str(e)

def prepare_data_for_regression(df):
    """
    Prepares data for Linear/Logistic regression.
    """
    # Create features like Day Number, Moving Averages
    df['Day'] = df.index
    df['MA_5'] = df['Close'].rolling(window=5).mean()
    df['MA_20'] = df['Close'].rolling(window=20).mean()
    df = df.dropna()
    
    return df
