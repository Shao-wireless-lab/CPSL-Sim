import matplotlib.pyplot as plt
import os
import numpy as np
from matplotlib.patches import Patch
import matplotlib
matplotlib.use('TkAgg')

# === Success rates grouped by difficulty, then by emitter location ===
scenarios_80_60 = [0.99, 0.89, 0.83]
scenarios_60_120 = [1.00, 0.93, 0.76]

# === Convert to percentages ===
scenarios_80_60_pct = [r * 100 for r in scenarios_80_60]
scenarios_60_120_pct = [r * 100 for r in scenarios_60_120]

labels = ['No Meander', 'Small Meander', 'Medium Meander']
x = np.arange(len(labels))
bar_width = 0.3

# === Colors per meander type ===
meander_colors = ['black', 'blue', 'red']  # shared by both emitters

# === Create the plot ===
fig, ax = plt.subplots(figsize=(8, 6))

# === Bars for emitter location [80, 60] ===
bars1 = []
for i, val in enumerate(scenarios_80_60_pct):
    bar = ax.bar(x[i] - bar_width/2, val, width=bar_width,
                 color=meander_colors[i], edgecolor='black')
    bars1.append(bar)

# === Bars for emitter location [60, 120] ===
bars2 = []
for i, val in enumerate(scenarios_60_120_pct):
    bar = ax.bar(x[i] + bar_width/2, val, width=bar_width,
                 color=meander_colors[i], edgecolor='white', hatch='//')
    bars2.append(bar)

# === Customize plot ===
ax.set_ylim(0, 120)
ax.set_ylabel("Success Rate (%)")
ax.set_title("Success Rate by Wind Condition")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.grid(axis='y', linestyle='--', alpha=0.7)

# === Custom Legend ===

# 1. Legend for Meander Levels (colors only)
meander_legend = [
    Patch(facecolor='black', edgecolor='black', label='No Meander'),
    Patch(facecolor='blue', edgecolor='black', label='Small Meander'),
    Patch(facecolor='red', edgecolor='black', label='Medium Meander'),
]

# 2. Legend for Emitter Locations (bar style only)
emitter_legend = [
    Patch(facecolor='gray', edgecolor='black', label='Emitter [80, 60]'),  # solid
    Patch(facecolor='gray', edgecolor='white', hatch='//', label='Emitter [60, 120]'),  # hatched
]

# Combine and add to plot
ax.legend(handles=meander_legend + emitter_legend, title='Legend')

# === Annotate bars with values ===
for group in [bars1, bars2]:
    for bar_container in group:
        for bar in bar_container.patches:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2.0, yval + 2,
                    f"{yval:.0f}%", ha='center', va='bottom')

# === Save the figure as PNG ===
save_path = "/home/ece213/CPSL-Sim_2/results/plots/success_rate_by_wind_condition.png"
os.makedirs(os.path.dirname(save_path), exist_ok=True)
plt.tight_layout()
plt.savefig(save_path, dpi=300)
print(f"Plot saved to: {save_path}")

# === Show and close ===
plt.show()
plt.close()
