import numpy as np
import h5py
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

ENCODER_PATH = "models/vae_encoder.keras"
DECODER_PATH = "models/vae_decoder.keras"
FFT_HDF5_PATH = "output/all_fft_signals.hdf5"
LATENT_DIM = 4

encoder = load_model(ENCODER_PATH, compile=False)
decoder = load_model(DECODER_PATH, compile=False)

with h5py.File(FFT_HDF5_PATH, "r") as f:
    fft_real = f["fft/re"][:]
    fft_imag = f["fft/im"][:]
    fft_data = np.stack([fft_real, fft_imag], axis=-1)

z_mean, z_log_var, _ = encoder.predict(fft_data)
reconstructed = decoder.predict(z_mean)

mse_val = mean_squared_error(fft_data.reshape(-1, 2), reconstructed.reshape(-1, 2))
mae_val = mean_absolute_error(fft_data.reshape(-1, 2), reconstructed.reshape(-1, 2))

# KL divergence
kl = -0.5 * np.mean(1 + z_log_var - np.square(z_mean) - np.exp(z_log_var))

print("\n=== VAE MODEL EVALUATION ===")
print(f"FFT vzorků celkem: {fft_data.shape[0]}")
print(f"Latent dimenze: {LATENT_DIM}")
print(f"\n--- Rekonstrukce ---")
print(f"Reconstruction MSE: {mse_val:.6f}")
print(f"Reconstruction MAE: {mae_val:.6f}")
print(f"\n--- KL Divergence ---")
print(f"KL Divergence: {kl:.6f}")
print(f"\n--- Latentní prostor (z_mean) ---")
print(f"Průměr: {np.mean(z_mean, axis=0)}")
print(f"Směrodatná odchylka: {np.std(z_mean, axis=0)}")
print(f"Minimum: {np.min(z_mean, axis=0)}")
print(f"Maximum: {np.max(z_mean, axis=0)}")

if LATENT_DIM == 2:
    plt.figure(figsize=(6, 6))
    plt.scatter(z_mean[:, 0], z_mean[:, 1], alpha=0.5, s=2)
    plt.title("Latentní prostor (2D)")
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
elif LATENT_DIM == 3:
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(z_mean[:, 0], z_mean[:, 1], z_mean[:, 2], alpha=0.5, s=2)
    ax.set_title("Latentní prostor (3D)")
    ax.set_xlabel("z[0]")
    ax.set_ylabel("z[1]")
    ax.set_zlabel("z[2]")
    plt.tight_layout()
    plt.show()
else:
    print(f"\nVizualizace není dostupná pro latentní prostor > 3 dimenze.")