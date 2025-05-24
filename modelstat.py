from dataclasses import asdict, dataclass
import json
import os
import numpy as np

from lib.loader import FolderExtractor

SIGNAL_NAME = "art"
GENERATED_DIR = "generated"
OUTPUT_FILE = "model-statistics.json"

if not os.path.isdir(GENERATED_DIR):
    os.makedirs(GENERATED_DIR)

@dataclass
class ModelStats:
    file: str
    min: float
    max: float
    mean: float
    std: float
    nan_count: int

def compute_stats(data, file_name):
    data_flat = data.flatten()
    return ModelStats(
        file=file_name,
        min=float(np.nanmin(data_flat)) if data_flat.size > 0 else None,
        max=float(np.nanmax(data_flat)) if data_flat.size > 0 else None,
        mean=float(np.nanmean(data_flat)) if data_flat.size > 0 else None,
        std=float(np.nanstd(data_flat)) if data_flat.size > 0 else None,
        nan_count=int(np.isnan(data_flat).sum()),
    )

extractor = FolderExtractor(GENERATED_DIR)
raw_data_dict = extractor.get_raw_data(SIGNAL_NAME)

stats_list = []
for file_stem, signal in raw_data_dict.items():
    print(f"Zpracovávám {file_stem}...")
    stats = compute_stats(signal, file_stem)
    stats_list.append(asdict(stats))

with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    json.dump(stats_list, f, ensure_ascii=False, indent=2)

print(f"\nVýsledky uloženy do {OUTPUT_FILE}")