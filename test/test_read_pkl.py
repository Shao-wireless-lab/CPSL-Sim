import pickle


file_path = (r"/home/ece213/CPSL-Sim-2/results/ppo/CPSL_2025-02-27-13-10/debug/"
             r"MyTrainer_CPSL_0v0o5_f06a8_00000_0_observation_type=local,use_safe_action=False,"
             r"custom_model=DeepsetModel_2025-02-27_13-10-50/params.pkl")

# Open the file in read-binary mode
with open(file_path, 'rb') as file:
    data = pickle.load(file)

# Print the loaded data
print(data)

# Print the loaded data type
print(type(data))

