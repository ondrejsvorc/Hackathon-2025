from dataclasses import asdict, dataclass
import json
import os
import numpy as np
from lib.loader import FolderExtractor

SIGNAL_NAME_ART = "art"
SIGNAL_NAME_ICP = "icp"
GENERATED_DIR = "generated"
OUTPUT_FILE_ART = "model-statistics-art.json"
OUTPUT_FILE_ICP = "model-statistics-icp.json"

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
raw_data_dict_art = extractor.get_raw_data(SIGNAL_NAME_ART)
raw_data_dict_icp = extractor.get_raw_data(SIGNAL_NAME_ICP)

stats_list = []
for file_stem, signal in raw_data_dict_art.items():
    print(f"Zpracovávám {file_stem}...")
    stats = compute_stats(signal, file_stem)
    stats_list.append(asdict(stats))

with open(OUTPUT_FILE_ART, 'w', encoding='utf-8') as f:
    json.dump(stats_list, f, ensure_ascii=False, indent=2)

print(f"\nVýsledky uloženy do {OUTPUT_FILE_ART}")

stats_list = []
for file_stem, signal in raw_data_dict_icp.items():
    print(f"Zpracovávám {file_stem}...")
    stats = compute_stats(signal, file_stem)
    stats_list.append(asdict(stats))

with open(OUTPUT_FILE_ICP, 'w', encoding='utf-8') as f:
    json.dump(stats_list, f, ensure_ascii=False, indent=2)

print(f"\nVýsledky uloženy do {OUTPUT_FILE_ICP}")
