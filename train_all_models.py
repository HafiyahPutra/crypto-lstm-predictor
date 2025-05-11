# train_all_models.py

from utils.data_loader import fetch_ohlc_data, prepare_lstm_data
from model.lstm_model import train_and_save_model
import joblib
import os

COINS = ["BTC", "ETH", "BNB", "XRP", "SOL"]
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

def train_all():
    for coin in COINS:
        print(f"\n🔁 Melatih model untuk: {coin}")
        try:
            df = fetch_ohlc_data(coin)
            close_prices = df["close"]
            X, y, scaler = prepare_lstm_data(close_prices)

            # Simpan scaler
            scaler_path = os.path.join(MODEL_DIR, f"{coin}_scaler.gz")
            joblib.dump(scaler, scaler_path)

            # Latih dan simpan model
            model_path = train_and_save_model(X, y, coin)
            print(f"✅ Model {coin} disimpan di: {model_path}")
            print(f"✅ Scaler {coin} disimpan di: {scaler_path}")

        except Exception as e:
            print(f"❌ Gagal melatih model untuk {coin}: {e}")

if __name__ == "__main__":
    train_all()
