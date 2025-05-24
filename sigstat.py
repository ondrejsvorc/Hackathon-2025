from dataclasses import asdict, dataclass
import json
import h5py
import numpy as np
from datetime import timedelta
import os
from lib.loader import FolderExtractor

SAMPLING_RATE = 100
OUTPUT_FILE = "statistics.json"
SIGNAL_NAME = "art"

@dataclass
class SignalStats:
    file: str
    signal_type: str
    sampling_rate: int
    length_samples: int
    duration: str
    nan_percentage: float
    min_val: float
    max_val: float
    in_threshold_percentage: float
    mean: float
    std: float

def compute_stats(signal, signal_type, threshold_min, threshold_max, fs, file_stem):
    length_samples = len(signal)
    time_seconds = length_samples / fs
    time_duration = str(timedelta(seconds=int(time_seconds)))

    nan_count = np.sum(np.isnan(signal))
    nan_percentage = (nan_count / length_samples) * 100

    signal_clean = signal[~np.isnan(signal)]

    min_val = float(np.min(signal_clean)) if len(signal_clean) > 0 else None
    max_val = float(np.max(signal_clean)) if len(signal_clean) > 0 else None

    in_threshold = np.sum((signal_clean >= threshold_min) & (signal_clean <= threshold_max))
    in_threshold_percentage = (in_threshold / len(signal_clean) * 100) if len(signal_clean) > 0 else None

    mean_val = float(np.mean(signal_clean)) if len(signal_clean) > 0 else None
    std_val = float(np.std(signal_clean)) if len(signal_clean) > 0 else None

    return SignalStats(
        file=file_stem,
        signal_type=signal_type,
        sampling_rate=fs,
        length_samples=length_samples,
        duration=time_duration,
        nan_percentage=nan_percentage,
        min_val=min_val,
        max_val=max_val,
        in_threshold_percentage=in_threshold_percentage,
        mean=mean_val,
        std=std_val
    )

extractor = FolderExtractor("data")
raw_data_dict = extractor.get_raw_data(SIGNAL_NAME)

stats_list = []
for file_stem, signal in raw_data_dict.items():
    print(f"Zpracovávám {file_stem}...")
    stats = compute_stats(signal, SIGNAL_NAME.upper(), 0, 250, SAMPLING_RATE, file_stem)
    stats_list.append(asdict(stats))

with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    json.dump(stats_list, f, ensure_ascii=False, indent=2)

print(f"\nVýsledky uloženy do {OUTPUT_FILE}")