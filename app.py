# app.py

from flask import Flask, render_template, request
from utils.data_loader import fetch_ohlc_data, prepare_lstm_data
from model.lstm_model import load_lstm_model, predict_future
import joblib
import numpy as np
import matplotlib.pyplot as plt
import os
import io
import base64

app = Flask(__name__)
MODEL_DIR = "models"
COINS = ["BTC", "ETH", "BNB", "XRP", "SOL"]

def plot_prediction(predicted, coin_symbol):
    days = list(range(1, len(predicted) + 1))
    plt.figure(figsize=(8,4))
    plt.plot(days, predicted, marker='o', label="Prediksi Harga")
    plt.title(f"Prediksi Harga 7 Hari Kedepan untuk {coin_symbol}")
    plt.xlabel("Hari ke-")
    plt.ylabel("Harga (USD)")
    plt.grid(True)
    plt.legend()

    # Simpan ke buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode()
    buf.close()
    plt.close()
    return img_base64

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    chart = None

    if request.method == "POST":
        coin = request.form["coin"]
        mode = request.form["mode"]

        try:
            # Ambil data Close terbaru (30 hari)
            df = fetch_ohlc_data(coin)
            close_prices = df["close"]

            # Load scaler
            scaler = joblib.load(f"{MODEL_DIR}/{coin}_scaler.gz")

            # Siapkan input 30 hari terakhir
            last_30 = close_prices.values[-30:].reshape(-1, 1)
            scaled_input = scaler.transform(last_30)
            X_input = np.expand_dims(scaled_input, axis=0)  # shape (1, 30, 1)

            # Load model
            model = load_lstm_model(coin)

            # Prediksi
            predicted = predict_future(model, X_input, forecast_days=7, scaler=scaler)
            prediction = [round(p, 2) for p in predicted]
            chart = plot_prediction(predicted, coin)

        except Exception as e:
            prediction = f"Terjadi kesalahan: {e}"

    return render_template("index.html", coins=COINS, prediction=prediction, chart=chart)

if __name__ == "__main__":
    app.run(debug=True)
