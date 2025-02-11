"""
Script to stack npy files into a single (X, 2, 1024) npy file.

Outputs a (X, 2, 1024) where (total number of samples, iq, frames),
and a label npy which contains the modulation type and Z-parameter value.
I.e. a 4qam with doppler offet of 10Hz.
"""


import numpy as np
import glob, os, re

# Change modulation types to ones we collected
# Change Z-parameter values we also collected
modulations_in_order = [
    "bpsk", "qpsk", "4qam", "16qam", "apsk"
]
doppler_in_order = [10, 12, 14, 16, 18, 20, 22, 24, 26]

# A helper function to parse filename: e.g. "8qam_doppler_60.npy"
def parse_filename(fname):
    """
    Extracts the modulation string and doppler offset from a filename
    like '8qam_doppler_60.npy'.
    Returns (mod_str, doppler_int).
    """
    base = os.path.basename(fname)
    # This assumes the filename format "<mod>_doppler_<offset>.npy"
    # e.g. "8qam_doppler_60.npy" -> mod="8qam", offset="60"
    match = re.match(r"^([a-zA-Z0-9]+)_snr_(\d+)\.npy$", base)
    if not match:
        raise ValueError(f"Filename {base} does not match expected pattern.")

    mod_str = match.group(1)         # e.g. "8qam"
    doppler_str = match.group(2)     # e.g. "60"
    doppler_int = int(doppler_str)
    return mod_str.lower(), doppler_int

# Map each modulation string to a numeric ID, based on the order above:
mod_to_id = {m: i for i, m in enumerate(modulations_in_order)}

# We'll load all npy files from some directory (modify path as needed)
all_files = glob.glob("/home/ash/raw_npy/snr_npy/*.npy")

# First, parse the filenames and store (mod, doppler, filepath)
info_list = []
for f in all_files:
    mod_str, doppler_val = parse_filename(f)
    info_list.append((mod_str, doppler_val, f))

# Sort them: first by modulation according to 'modulations_in_order',
# then by doppler according to 'doppler_in_order'.
def sort_key(item):
    # item = (mod_str, doppler_val, filepath)
    mod_str, doppler_val, _ = item
    # index in the known mod list
    mod_rank = modulations_in_order.index(mod_str)
    # index in the known doppler list
    dop_rank = doppler_in_order.index(doppler_val)
    return (mod_rank, dop_rank)

info_list.sort(key=sort_key)

# Now load the arrays in the sorted order, stacking them.
data_list = []
label_list = []

for (mod_str, doppler_val, fpath) in info_list:
    # Load array: shape = (4096, 2, 1024)
    arr = np.load(fpath) 
    
    # Map the modulation string to an integer ID
    mod_id = mod_to_id[mod_str]  # e.g. "8qam" -> 2 if that's its index

    # Create a label array for each row in this .npy file
    # We want each of the 4096 samples to carry the same (mod_id, doppler_val).
    # shape -> (4096, 2)
    n_samples = arr.shape[0]
    label_arr = np.tile([mod_id, doppler_val], (n_samples, 1))

    data_list.append(arr)         # shape (4096, 2, 1024)
    label_list.append(label_arr)  # shape (4096, 2)

# Concatenate along axis=0 to stack everything
# final_data: (X, 2, 1024) where X = sum of all 4096's across files
# final_label: (X, 2)
final_data = np.concatenate(data_list, axis=0)
final_label = np.concatenate(label_list, axis=0)