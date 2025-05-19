import random
import sys, os
import pickle

train_ratio = 0.8
val_size = 50
test_size = 1000

shuffle = True
seed = None
input_data_path = sys.argv[1]

# load data from a pickle file
with open(input_data_path, 'rb') as f:
    data = pickle.load(f)

if seed is not None:
    random.seed(seed)

if shuffle:
    random.shuffle(data)

total = len(data)
test_size = min(test_size, int(total*(1-train_ratio)))
train_size = total - val_size - test_size

val_end = train_size + val_size

train_data = data[:train_size]
val_data = data[train_size:val_end]
test_data = data[val_end:]

# Save the split data into separate pickle files
prefix = input_data_path.replace('.pkl', '')
train_data_path = f'{prefix}_train.pkl'
val_data_path = f'{prefix}_val.pkl'
test_data_path = f'{prefix}_test.pkl'

with open(train_data_path, 'wb') as f:
    pickle.dump(train_data, f)
with open(val_data_path, 'wb') as f:
    pickle.dump(val_data, f)
with open(test_data_path, 'wb') as f:
    pickle.dump(test_data, f)

print(f"Train data saved to {train_data_path}")
print(f"Validation data saved to {val_data_path}")
print(f"Test data saved to {test_data_path}")
print(f"Total data: {total}")
print(f"Train data size: {len(train_data)}")
print(f"Validation data size: {len(val_data)}")
print(f"Test data size: {len(test_data)}")

