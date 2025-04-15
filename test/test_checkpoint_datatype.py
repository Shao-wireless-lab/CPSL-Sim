import torch
import numpy as np
import pickle

checkpoint_path = (r"/home/ece213/CPSL-Sim/results/ppo/CPSL_2025-02-27-13-10/debug/"
             r"MyTrainer_CPSL_0v0o5_f06a8_00000_0_observation_type=local,use_safe_action=False,"
             r"custom_model=DeepsetModel_2025-02-27_13-10-50/checkpoint_000200/checkpoint-200")

# Load the checkpoint
with open(checkpoint_path, "rb") as f:
    checkpoint_data = pickle.load(f)

# Function to recursively print types of objects
def inspect_data(obj, level=0):
    indent = "  " * level
    if isinstance(obj, dict):
        print(f"{indent}Dict with {len(obj)} keys:")
        for key, value in obj.items():
            print(f"{indent}  Key: {key}, Type: {type(value)}")
            inspect_data(value, level + 1)
    elif isinstance(obj, list):
        print(f"{indent}List with {len(obj)} elements, First Element Type: {type(obj[0]) if obj else 'Empty'}")
        for i, value in enumerate(obj[:5]):  # Show first 5 elements for brevity
            print(f"{indent}  Index {i}: {type(value)}")
            inspect_data(value, level + 1)
    elif isinstance(obj, np.ndarray):
        print(f"{indent}Numpy array, dtype: {obj.dtype}, shape: {obj.shape}")
    else:
        print(f"{indent}Type: {type(obj)}")

# Inspect the checkpoint data
inspect_data(checkpoint_data)

# Attempt to unpickle the 'worker' key
worker_data = pickle.loads(checkpoint_data["worker"])

# Check the type of the unpickled data
print(type(worker_data))

inspect_data(worker_data)