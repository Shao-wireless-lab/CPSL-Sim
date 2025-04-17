import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import os

matplotlib.use('TkAgg')  # For matplotlib GUI
# Change it to your local path
# === Set the folder path ===
root_dir = "/home/ece213/CPSL-Sim/results/test_results/test_hard_60_120_2025-04-11-10-26/episodes_traj/"

# === Loop through only files with "trajectory" in the name ===
for file in os.listdir(root_dir):
    if file.endswith(".json") and "trajectory" in file:
        json_path = os.path.join(root_dir, file)
        print(f"📂 Processing: {json_path}")

        # === Load JSON ===
        with open(json_path, "r") as f:
            data = json.load(f)

        centroid_traj = []
        anchor_traj = []
        valid_timesteps = []
        emitter_loc = None
        recording = False

        # === Extract emitter_loc once ===
        for entry in data:
            if "infos" in entry:
                first_agent_info = next(iter(entry["infos"].values()), {})
                if "target_loc" in first_agent_info:
                    emitter_loc = first_agent_info["target_loc"]
                    break

        # === Extract trajectory data ===
        for timestep, entry in enumerate(data):
            if "infos" in entry:
                first_agent_info = next(iter(entry["infos"].values()), {})
                detection = first_agent_info.get("detection", 0)

                if not recording and detection == 1:
                    recording = True

                if recording:
                    centroid_loc = first_agent_info.get("centroid_loc")
                    anchor_loc = first_agent_info.get("anchor_loc")

                    if centroid_loc is not None and anchor_loc is not None:
                        centroid_traj.append(centroid_loc)
                        anchor_traj.append(anchor_loc)
                        valid_timesteps.append(timestep)

        if not centroid_traj or not anchor_traj:
            print(f"⚠️ Skipping {file}: No valid trajectory data.")
            continue

        # Convert to arrays
        centroid_array = np.array(centroid_traj)
        anchor_array = np.array(anchor_traj)
        timesteps_array = np.array(valid_timesteps)

        # === Plotly Interactive Plot ===
        fig = go.Figure()

        # Centroid Trace
        fig.add_trace(go.Scatter(
            x=centroid_array[:, 0],
            y=centroid_array[:, 1],
            mode='lines+markers+text',
            name='Centroid Trajectory',
            marker=dict(size=4),
            line=dict(width=1),
            text=[str(t) if i % 50 == 0 else '' for i, t in enumerate(timesteps_array)],
            textposition="top right",
            textfont=dict(size=10)
        ))

        # Anchor Trace
        fig.add_trace(go.Scatter(
            x=anchor_array[:, 0],
            y=anchor_array[:, 1],
            mode='lines+markers+text',
            name='Anchor Trajectory',
            marker=dict(symbol='x', size=5),
            line=dict(dash='dash', width=1),
            text=[
                f"A@{t}" if (i == 0 or not np.allclose(anchor_array[i], anchor_array[i - 1])) else ''
                for i, t in enumerate(timesteps_array)
            ],
            textposition="bottom left",
            textfont=dict(size=10, color='blue')
        ))

        # Emitter point
        if emitter_loc is not None:
            fig.add_trace(go.Scatter(
                x=[emitter_loc[0]],
                y=[emitter_loc[1]],
                mode='markers+text',
                name='Emitter',
                marker=dict(size=10, color='green'),
                text=["Emitter"],
                textposition="top right",
                textfont=dict(size=12, color='green')
            ))

        fig.update_layout(
            title=f"Centroid & Anchor Trajectories: {file}",
            xaxis_title="X",
            yaxis_title="Y",
            width=800,
            height=600,
            showlegend=True
        )

        # === Save to plot_results subfolder ===
        parent_dir = os.path.dirname(os.path.dirname(root_dir))
        output_dir = os.path.join(parent_dir, "plot_results/centroid&anchor")
        os.makedirs(output_dir, exist_ok=True)

        save_path = os.path.join(output_dir, file.replace(".json", ".html"))
        fig.write_html(save_path)
        print(f"✅ Saved to: {save_path}")
