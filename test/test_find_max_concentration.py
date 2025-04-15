import scipy.io
import numpy as np

# Load the .mat files
mat_data1 = scipy.io.loadmat("/home/ece213/CPSL-Sim/cuas/plume_data/Plume-C-Data-Height-2.mat")
mat_data2 = scipy.io.loadmat("/home/ece213/CPSL-Sim/cuas/plume_data/Plume-C-Data-Height-5.mat")

# Extract concentration data
C1 = mat_data1['C']  # Replace 'C' with the actual variable name if different
C2 = mat_data2['C']  # Replace 'C' with the actual variable name if different


print(C1.shape)

# Find the top 10 maximum values and their positions
top_10_indices_C1 = np.argpartition(C1.flatten(), -10)[-10:]  # Get the indices of the top 10 values
top_10_indices_C2 = np.argpartition(C2.flatten(), -10)[-10:]

# Sort the top 10 values in descending order
top_10_indices_C1 = top_10_indices_C1[np.argsort(C1.flatten()[top_10_indices_C1])[::-1]]
top_10_indices_C2 = top_10_indices_C2[np.argsort(C2.flatten()[top_10_indices_C2])[::-1]]

# Get the values and their corresponding positions
top_10_values_C1 = C1.flatten()[top_10_indices_C1]
top_10_positions_C1 = [np.unravel_index(idx, C1.shape) for idx in top_10_indices_C1]

top_10_values_C2 = C2.flatten()[top_10_indices_C2]
top_10_positions_C2 = [np.unravel_index(idx, C2.shape) for idx in top_10_indices_C2]

# Print results
print("Top 10 max concentrations for height 2:")
for i in range(10):
    print(f"Value: {top_10_values_C1[i]} at position {top_10_positions_C1[i]}")

print("\nTop 10 max concentrations for height 5:")
for i in range(10):
    print(f"Value: {top_10_values_C2[i]} at position {top_10_positions_C2[i]}")
