import pickle
import numpy as np

def inspect_checkpoint(checkpoint_path):
    """ Load and inspect checkpoint for numpy.object_ issues """
    with open(checkpoint_path, "rb") as f:
        checkpoint_data = pickle.load(f)

    print("Checkpoint keys:", checkpoint_data.keys())

    if "worker" in checkpoint_data:
        worker_state = checkpoint_data["worker"]
        print("Worker state type:", type(worker_state))

        if isinstance(worker_state, dict):
            for k, v in worker_state.items():
                print(f"Key: {k}, Type: {type(v)}")
                if isinstance(v, np.ndarray) and v.dtype == np.object_:
                    print(f"⚠️ Found numpy.object_ at key {k}, converting to float32")

checkpoint= ("/home/ece213/CPSL-Sim/results/ppo/CPSL_2025-02-27-13-10/debug/"
             "MyTrainer_CPSL_0v0o5_f06a8_00000_0_observation_type=local,use_safe_action=False,"
             "custom_model=DeepsetModel_2025-02-27_13-10-50/params.pkl")
inspect_checkpoint(checkpoint)
