import pickle
import numpy as np
from tqdm import tqdm
import sys
sys.path.append('../')
from src import const

# 1. Load the .pkl file
with open(sys.argv[1], 'rb') as f:
    data = pickle.load(f)  # List of dictionaries

n1 = const.N_RESIDUE_TYPES
n2 = const.N_ATOM_TYPES
n = n1 + n2

# 2. Process each entry with progress bar
for entry in tqdm(data, desc="Processing entries", unit="entry"):
    # Handle molecule_one_hot (B,9) → (B,48)
    if 'molecule_one_hot' in entry:
        original = entry['molecule_one_hot']
        if original.shape[1] != n:  # Only pad if not already 48
            padded = np.pad(original, 
                          pad_width=((0,0), (n1,0)),
                          mode='constant',
                          constant_values=0)
            entry['molecule_one_hot'] = padded
    
    # Handle pocket_one_hot (N,39) → (N,48)
    if 'pocket_one_hot' in entry:
        original = entry['pocket_one_hot']
        if original.shape[1] != n:  # Only pad if not already 48
            padded = np.pad(original,
                          pad_width=((0,0), (0,n2)),
                          mode='constant',
                          constant_values=0)
            entry['pocket_one_hot'] = padded

# 3. Save the modified data
with open(sys.argv[2], 'wb') as f:
    pickle.dump(data, f)
