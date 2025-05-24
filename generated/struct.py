import h5py

# Zadej cestu k HDF5 souboru
HDF5_PATH = "data/signals_2024-03-04\dataset_0\TBI_001_v2_1_2_20.hdf5"

def print_structure(name, obj):
    if isinstance(obj, h5py.Dataset):
        print(f"📄 {obj.name}  (shape: {obj.shape}, dtype: {obj.dtype})")
    elif isinstance(obj, h5py.Group):
        print(f"📁 {obj.name}/")

with h5py.File(HDF5_PATH, 'r') as f:
    print(f"\n📦 Struktura HDF5 souboru: {HDF5_PATH}")
    f.visititems(print_structure)
