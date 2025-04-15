import json
import os
import plotly.graph_objects as go
import configparser

# === Set the folder path ===
root_dir = "/home/ece213/CPSL-Sim_2/results/test_results/test_medium_80_60_2025-04-11-09-35/episodes_traj/"
parent_dir = os.path.dirname(os.path.dirname(root_dir))

# === Collect data from each episode ===
final_distance_offset_data = []


# === Load the sim_config.cfg (actually JSON) for tolerance ===
sim_config_path = "/home/ece213/CPSL-Sim_2/configs/sim_config.cfg"
tolerance = None

if os.path.exists(sim_config_path):
    try:
        with open(sim_config_path, "r") as f:
            sim_config = json.load(f)
        tolerance = sim_config['env_config']["tolerance"]
    except Exception as e:
        raise ValueError(f"❌ Failed to read 'tolerance' from config file: {e}")
else:
    raise FileNotFoundError(f"❌ Config file not found at: {sim_config_path}")


for file in os.listdir(root_dir):
    if file.endswith(".json") and "trajectory" in file:
        json_path = os.path.join(root_dir, file)
        print(f"📂 Processing: {json_path}")

        with open(json_path, "r") as f:
            data = json.load(f)

        last_info = None
        recording = False
        target_found_flag = 1  # Default to 1, update if found

        for entry in data:
            if "infos" in entry:
                first_agent_info = next(iter(entry["infos"].values()), {})

                if "target_found" in first_agent_info:
                    target_found_flag = first_agent_info["target_found"]

                detection = first_agent_info.get("detection", 0)
                if not recording and detection == 1:
                    recording = True

                if recording:
                    last_info = first_agent_info  # Last relevant entry

        if last_info is not None and "final_cent_dist_to_emitter_offset" in last_info:
            final_distance_offset_data.append({
                "episode": file.replace(".json", ""),
                "offset": last_info["final_cent_dist_to_emitter_offset"],
                "target_found": target_found_flag
            })
        else:
            print(f"⚠️ Skipping {file}: Missing 'final_cent_dist_to_emitter_offset'.")

# === Generate final_distance_offset plot ===
if final_distance_offset_data:
    episode_names = [d["episode"] for d in final_distance_offset_data]
    dist_values = [d["offset"] for d in final_distance_offset_data]
    bar_colors = ["red" if d["target_found"] == 0 else "steelblue" for d in final_distance_offset_data]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=episode_names,
        y=dist_values,
        marker_color=bar_colors,
        name="Final Offset",
        text=[f"{v:.2f}" for v in dist_values],
        textposition="outside"
    ))

    # === Add tolerance line if available ===
    if tolerance is not None:
        fig.add_shape(
            type="line",
            x0=-0.5,
            x1=len(episode_names) - 0.5,
            y0=tolerance,
            y1=tolerance,
            line=dict(color="green", width=2, dash="dash"),
        )
        fig.add_annotation(
            x=len(episode_names) - 1,
            y=tolerance,
            text=f"Tolerance distance: {tolerance:.2f} m",
            showarrow=False,
            yshift=10,
            font=dict(color="green", size=14)
        )

    fig.update_layout(
        title="Final Centroid Distance to Emitter Offset per Episode",
        xaxis_title="Episode",
        yaxis_title="Distance Offset",
        height=1000,
        width=2000,
        showlegend=False
    )

    # === Save plot ===
    output_dir = os.path.join(parent_dir, "plot_results/final_distance_offset")
    os.makedirs(output_dir, exist_ok=True)
    final_distance_offset_path = os.path.join(output_dir, "final_distance_centroid_to_emitter_offset.html")
    fig.write_html(final_distance_offset_path)
    print(f"✅ final_distance_offset plot saved to: {final_distance_offset_path}")
else:
    print("⚠️ No valid data found to plot.")
