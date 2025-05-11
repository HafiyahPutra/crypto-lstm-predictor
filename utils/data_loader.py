# utils/data_loader.py

import requests
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

COINS = {
    "BTC": "bitcoin",
    "ETH": "ethereum",
    "BNB": "binancecoin",
    "XRP": "ripple",
    "SOL": "solana"
}

API_URL = "https://api.coingecko.com/api/v3/coins/{coin_id}/ohlc?vs_currency=usd&days=30"
API_KEY = "CG-7pi9DCcf6E6PmCFBLrwvGtZT"

def fetch_ohlc_data(coin_symbol):
    coin_id = COINS.get(coin_symbol.upper())
    if not coin_id:
        raise ValueError(f"Koin {coin_symbol} tidak dikenali.")
    
    url = API_URL.format(coin_id=coin_id)
    headers = {
        "x-cg-demo-api-key": API_KEY
    }
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        raise Exception(f"Gagal mengambil data untuk {coin_symbol}: {response.text}")
    
    data = response.json()
    df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit='ms')
    df.set_index("timestamp", inplace=True)
    return df

def prepare_lstm_data(df_close, window_size=30, forecast_days=7):
    """
    df_close: Series of close prices
    """
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df_close.values.reshape(-1, 1))
    
    X, y = [], []
    for i in range(len(scaled_data) - window_size - forecast_days + 1):
        X.append(scaled_data[i:i+window_size])
        y.append(scaled_data[i+window_size:i+window_size+forecast_days])
    
    return np.array(X), np.array(y), scaler
