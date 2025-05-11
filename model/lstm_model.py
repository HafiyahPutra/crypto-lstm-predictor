# model/lstm_model.py

import os
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

def build_lstm_model(input_shape, output_size):
    model = Sequential()
    model.add(LSTM(64, return_sequences=False, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(Dense(output_size))
    model.compile(optimizer='adam', loss='mse')
    return model

def train_and_save_model(X, y, coin_symbol):
    input_shape = (X.shape[1], X.shape[2])
    output_size = y.shape[1]
    model = build_lstm_model(input_shape, output_size)

    early_stop = EarlyStopping(monitor='loss', patience=10)
    model.fit(X, y, epochs=100, batch_size=8, verbose=1, callbacks=[early_stop])

    model_path = os.path.join(MODEL_DIR, f"{coin_symbol.upper()}_lstm_model.h5")
    model.save(model_path)
    return model_path

def load_lstm_model(coin_symbol):
    model_path = os.path.join(MODEL_DIR, f"{coin_symbol.upper()}_lstm_model.h5")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model untuk {coin_symbol} tidak ditemukan. Harap latih dulu.")
    return load_model(model_path)

def predict_future(model, recent_data, forecast_days, scaler):
    """
    recent_data: (1, time_steps, 1) — contoh: 1 sample input [30 timesteps]
    """
    prediction = model.predict(recent_data)
    prediction = scaler.inverse_transform(prediction.reshape(-1, 1)).flatten()
    return prediction
