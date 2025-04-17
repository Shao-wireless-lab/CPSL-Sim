import json
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')

# Load the simulation config file (JSON format)
with open("/home/ece213/CPSL-Sim_2/configs/sim_config.cfg", "r") as f:
    config = json.load(f)

# Get the baseline value
baseline_value = config["env_config"]["lat_flux_threshold_5"]


# Change it to your local path
# Load the JSON file
with open("/home/ece213/CPSL-Sim_2/results/test_results/test_2025-03-28-13-29/trajectory_764.json", "r") as f:
    data = json.load(f)

#with open("/home/ece213/CPSL-Sim_2/results/test_results/test_2025-03-28-13-43/trajectory_3199.json", "r") as f:
    #data = json.load(f)

flux_over_time = []
#first_detection_timestep = None

# Iterate over timesteps
for timestep, entry in enumerate(data):
    if "infos" in entry:
        # Just pick one agent to extract flux (since all are the same)
        first_agent_info = next(iter(entry['infos'].values()), {})

        # Get sum_flux
        flux = first_agent_info.get("sum_flux", 0)
        flux_over_time.append(flux)

        # Check for detection
        #if first_detection_timestep is None and first_agent_info.get("no_detection", 1) == 0:
            #first_detection_timestep = timestep
            #print(f"Gas first detected at timestep {timestep}!")





# Plotting
plt.figure(figsize=(10, 5))
plt.plot(flux_over_time, label="Sum Flux")

plt.axhline(y=baseline_value, color='red', linestyle='--', label=f"Flux threshold value({baseline_value})")
# Add this line to label the baseline value on the y-axis
xlim = plt.xlim()
plt.text(xlim[0], baseline_value, f"{baseline_value}", color='red', va='bottom', ha='left', fontsize=9)


#if first_detection_timestep is not None:
    #plt.axvline(x=first_detection_timestep, color='r', linestyle='--', label="First Detection")

plt.xlabel("Timestep")
plt.ylabel("Flux")
plt.title("Sum Flux over Time")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
