import os
import numpy as np
import matplotlib.pyplot as plt
import joblib
import h5py
import tensorflow as tf
from tensorflow.keras.models import load_model

# --- Paths ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
MODEL_DIR = os.path.join(BASE_DIR, 'usablemodels', 'Vae500')
OUTPUT_DIR = SCRIPT_DIR

# --- Modely & Scaler ---
DECODER_MODEL_FILES = [
    'vae_decoder_epoch090art.keras',
    'vae_decoder_epoch100art.keras',
    'vae_decoder_epoch070icp.keras',
    'vae_decoder_epoch080icp.keras',
    'vae_decoder_epoch090icp.keras'
]
SCALERS = {
    "art": "rfft_scaler_art.pkl",
    "icp": "rfft_scaler_icp.pkl"
}

NUM_SIGNALS_PER_MODEL = 500
WINDOW_SIZE = 1000

# --- Generování ---
def generate_signals():
    print(f"TensorFlow {tf.__version__} | NumPy {np.__version__} | H5py {h5py.__version__}")
    print(f"Používám modely z: {MODEL_DIR}")
    print(f"Výstup ukládám do: {OUTPUT_DIR}")

    for decoder_file in DECODER_MODEL_FILES:
        model_path = os.path.join(MODEL_DIR, decoder_file)
        if not os.path.exists(model_path):
            print(f"⚠️ Soubor {decoder_file} nenalezen, přeskakuji.")
            continue

        # === Načti model ===
        try:
            decoder = load_model(model_path, compile=False)
            print(f"\n✅ Načten model: {decoder_file}")
            latent_dim = decoder.input_shape[1]
        except Exception as e:
            print(f"❌ Chyba při načítání modelu {decoder_file}: {e}")
            continue

        # === Vyber správný scaler ===
        scale_key = "icp" if "icp" in decoder_file.lower() else "art"
        scaler_path = os.path.join(MODEL_DIR, SCALERS[scale_key])
        if not os.path.exists(scaler_path):
            print(f"❌ Scaler '{scale_key}' nenalezen ({scaler_path})")
            continue

        try:
            scaler = joblib.load(scaler_path)
            print(f"✅ Načten scaler: {SCALERS[scale_key]}")
        except Exception as e:
            print(f"❌ Chyba při načtení scaleru: {e}")
            continue

        # === Generuj latentní vektory ===
        latent_vectors = np.random.randn(NUM_SIGNALS_PER_MODEL, latent_dim)

        try:
            decoded_fft = decoder.predict(latent_vectors)
            print(f"Vygenerován tvar spektra: {decoded_fft.shape}")
        except Exception as e:
            print(f"❌ Chyba při predikci dekodéru: {e}")
            continue

        try:
            reshaped = decoded_fft.reshape(-1, 2)
            inv_scaled = scaler.inverse_transform(reshaped).reshape(decoded_fft.shape)
        except Exception as e:
            print(f"❌ Chyba při škálování výstupu: {e}")
            continue

        # === Konverze do časové domény ===
        signals = []
        for spectrum in inv_scaled:
            complex_fft = spectrum[:, 0] + 1j * spectrum[:, 1]
            signal = np.fft.irfft(complex_fft, n=WINDOW_SIZE)
            signals.append(signal)

        signals = np.array(signals)
        print(f"✅ Zrekonstruováno {len(signals)} signálů.")

        # === Ulož do HDF5 ===
        output_file = f"generated_signals_{os.path.splitext(decoder_file)[0]}.hdf5"
        output_path = os.path.join(OUTPUT_DIR, output_file)
        try:
            with h5py.File(output_path, 'w') as f:
                f.create_dataset("signals", data=signals)
                f.create_dataset("fft/re", data=inv_scaled[:, :, 0])
                f.create_dataset("fft/im", data=inv_scaled[:, :, 1])
                f.attrs['source_model'] = decoder_file
                f.attrs['latent_dim'] = latent_dim
                f.attrs['scaler'] = scaler_path
                f.attrs['num_generated'] = NUM_SIGNALS_PER_MODEL
            print(f"💾 Uloženo do: {output_path}")
        except Exception as e:
            print(f"❌ Chyba při ukládání: {e}")

if __name__ == "__main__":
    generate_signals()
