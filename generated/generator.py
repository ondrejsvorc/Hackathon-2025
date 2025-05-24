import os
import numpy as np
import joblib
import h5py
import tensorflow as tf
from tensorflow.keras.models import load_model
import time
import uuid

# === CESTY ===
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
MODEL_DIR = os.path.join(BASE_DIR, 'usablemodels', 'Vae500')
OUTPUT_HDF5 = os.path.join(SCRIPT_DIR, "generated_structured_signals")

# === MODELY & SCALERY ===
DECODERS = {
    "art": "vae_decoder_epoch100art.keras",
    "icp": "vae_decoder_epoch090icp.keras"
}
SCALERS = {
    "art": "rfft_scaler_art.pkl",
    "icp": "rfft_scaler_icp.pkl"
}

NUM_SIGNALS = 1         # Jeden dlouhý signál (např. 1x 360000 vzorků)
WINDOW_SIZE = 1000
NUM_WINDOWS = 360       # 360 * 1000 = 360000
FREQUENCY = 100.0       # Hz
START_TIME = int(time.time() * 1e6)  # aktuální čas v µs

def generate_signal(decoder_path, scaler_path, latent_dim):
    z = np.random.randn(NUM_WINDOWS, latent_dim)
    decoder = load_model(decoder_path, compile=False)
    fft = decoder.predict(z)
    scaler = joblib.load(scaler_path)

    # Inverzní škálování
    fft_scaled = scaler.inverse_transform(fft.reshape(-1, 2)).reshape(fft.shape)

    # IFFT
    full_signal = []
    for spectrum in fft_scaled:
        complex_spectrum = spectrum[:, 0] + 1j * spectrum[:, 1]
        signal = np.fft.irfft(complex_spectrum, n=WINDOW_SIZE)
        full_signal.extend(signal)

    return np.array(full_signal, dtype=np.float32)

def save_to_hdf5(icp_signal, art_signal):
    with h5py.File(f"{OUTPUT_HDF5}_{uuid.uuid1()}.hdf5", 'w') as f:
        waves = f.create_group("waves")

        for name, signal in [("icp", icp_signal), ("art", art_signal)]:
            waves.create_dataset(name, data=signal)

            index_dtype = np.dtype([("startidx", "<i8"), ("starttime", "<u8"), ("length", "<i8"), ("frequency", "<f8")])
            quality_dtype = np.dtype([("time", "<u8"), ("value", "<u4")])

            index_data = np.array([(0, START_TIME, len(signal), FREQUENCY)], dtype=index_dtype)
            quality_data = np.array([(START_TIME, 0)], dtype=quality_dtype)

            waves.create_dataset(f"{name}.index", data=index_data)
            waves.create_dataset(f"{name}.quality", data=quality_data)

    print(f"💾 Uloženo do souboru: {OUTPUT_HDF5}")

def main():
    print("➡️ Generuji signály s přesnou strukturou HDF5...")

    # === Načti modely a scalery
    signals = {}
    for label in ["icp", "art"]:
        decoder_path = os.path.join(MODEL_DIR, DECODERS[label])
        scaler_path = os.path.join(MODEL_DIR, SCALERS[label])
        if not os.path.exists(decoder_path) or not os.path.exists(scaler_path):
            print(f"❌ Chybí model nebo scaler: {label.upper()}")
            return
        decoder = load_model(decoder_path, compile=False)
        latent_dim = decoder.input_shape[1]
        signals[label] = generate_signal(decoder_path, scaler_path, latent_dim)
        print(f"✅ {label.upper()} vygenerován: {signals[label].shape}")

    save_to_hdf5(signals["icp"], signals["art"])

if __name__ == "__main__":
    for i in range(10):
        main()
