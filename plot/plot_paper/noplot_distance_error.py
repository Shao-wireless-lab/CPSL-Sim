import json
import os
import numpy as np
import csv

# === Base result folder ===
base_dir = "/home/ece213/CPSL-Sim_2/results/test_results/"

# === Folder: label mappings ===
folder_map = {
    "test_easy_80_60_2025-04-11-09-23":      ("No Meander", "Emitter [80, 60]"),
    "test_medium_80_60_2025-04-11-10-42":   ("Small Meander", "Emitter [80, 60]"),
    "test_hard_80_60_2025-04-11-14-16":  ("Medium Meander", "Emitter [80, 60]"),
    "test_easy_60_120_2025-04-11-10-11":     ("No Meander", "Emitter [60, 120]"),
    "test_medium_60_120_2025-04-11-10-25":  ("Small Meander", "Emitter [60, 120]"),
    "test_hard_60_120_2025-04-11-10-26": ("Medium Meander", "Emitter [60, 120]"),
}

# === Load tolerance from config ===
sim_config_path = "/home/ece213/CPSL-Sim_2/configs/sim_config.cfg"
with open(sim_config_path, "r") as f:
    sim_config = json.load(f)
tolerance = sim_config['env_config']["tolerance"]

# === Collect data ===
grouped_data = {}

for folder, (meander_label, emitter_loc) in folder_map.items():
    folder_path = os.path.join(base_dir, folder, "episodes_traj")
    if not os.path.exists(folder_path):
        print(f"⚠️ Missing folder: {folder_path}")
        continue

    key = (meander_label, emitter_loc)
    if key not in grouped_data:
        grouped_data[key] = []

    for file in os.listdir(folder_path):
        if file.endswith(".json") and "trajectory" in file:
            json_path = os.path.join(folder_path, file)
            with open(json_path, "r") as f:
                data = json.load(f)

            last_info = None
            for entry in data:
                if "infos" in entry:
                    first_info = next(iter(entry["infos"].values()), {})
                    last_info = first_info  # keep updating to the last available

            if last_info:
                grouped_data[key].append({
                    "success": bool(last_info.get("target_found", 0)),
                    "distance": last_info.get("final_cent_dist_to_emitter", None),
                    "offset": last_info.get("final_cent_dist_to_emitter_offset", None)
                })

# === Compute stats and save ===
output = []

for (meander, location), records in grouped_data.items():
    total = len(records)
    successes = [r for r in records if r["success"]]
    all_distances = [r["distance"] for r in records if r["distance"] is not None]
    all_offsets = [r["offset"] for r in records if r["offset"] is not None]
    succ_distances = [r["distance"] for r in successes if r["distance"] is not None]
    succ_offsets = [r["offset"] for r in successes if r["offset"] is not None]

    row = {
        "meander": meander,
        "location": location,
        "num_total": total,
        "num_success": len(successes),
        "success_rate": round(len(successes) / total, 3) if total > 0 else None,
        "mean_distance_all": round(np.mean(all_distances), 3) if all_distances else None,
        "mean_distance_success_only": round(np.mean(succ_distances), 3) if succ_distances else None,
        "mean_offset_all": round(np.mean(all_offsets), 3) if all_offsets else None,
        "mean_offset_success_only": round(np.mean(succ_offsets), 3) if succ_offsets else None,
        "tolerance": tolerance
    }
    output.append(row)

# === Save to CSV ===
save_path = "/home/ece213/CPSL-Sim_2/results/processed/summary_results.csv"
os.makedirs(os.path.dirname(save_path), exist_ok=True)

with open(save_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=output[0].keys())
    writer.writeheader()
    for row in output:
        writer.writerow(row)

print(f"✅ Summary saved to: {save_path}")
