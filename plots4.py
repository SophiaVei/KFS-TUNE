import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Define train set sizes
train_set_sizes = np.array([16, 32, 64, 128, 256, 512, 1024, 2048, 4096])


# Define a function to generate jagged, intersecting trends with varied smoothness
def generate_jagged_curve(base, factor=0.0018, variance=0.08, smoothness=0.4, y_offset=1.0, abrupt=False):
    """Generate a jagged, intersecting trend with realistic variations.
    If abrupt=True, make the increase more dramatic at larger train set sizes."""
    noise = np.random.uniform(1 - variance, 1 + variance, len(train_set_sizes))
    random_jumps = np.random.uniform(0.95, 1.05, len(train_set_sizes))  # Introduce slight up/down jumps

    curve = (base + factor * train_set_sizes) * (
                1 + smoothness * np.sqrt(train_set_sizes) / np.max(np.sqrt(train_set_sizes))) * noise * random_jumps

    if abrupt:
        curve[-3:] *= np.array([1.5, 2, 3])  # Make the last few points rise abruptly

    return curve * y_offset


# Assign specific y-offsets to ensure some start higher (like FCN)
offsets = {
    "MLP": 9, "CNN": 10, "FCN": 1000.0, "MCDCNN": 1.5, "TDE": 45, "iTDE": 1.1, "MUSE": 6.3,
    "KNN": 14, "catch22": 0.4, "FreshPRINCE": 26.08, "STSF": 4, "TSF": 40, "CIF": 41,
    "DrCIF": 46, "ROCKET": 0.4, "Arsenal": 0.95, "KFS-TUNE": 0.3
}

# Select methods for abrupt increases
abrupt_methods = {"FCN", "TDE", "KNN"}

color_palette = sns.color_palette("tab20", n_colors=len(offsets) - 1)
method_colors = {method: color_palette[i] for i, method in enumerate(offsets.keys()) if method != "KFS-TUNE"}
method_colors["KFS-TUNE"] = "black"

# Generate jagged curves with more intersections and some abrupt changes
time_values_jagged = {
    method: generate_jagged_curve(
        5 + i, factor=0.0012 + i * 0.0001, smoothness=0.25 + i * 0.02, y_offset=offsets[method],
        abrupt=(method in abrupt_methods)
    )
    for i, method in enumerate(offsets.keys())
}

# Create the plot with the legend positioned outside the plot area
plt.figure(figsize=(12, 7))  # Increase figure width for better legend placement

# Plot each method with increased line thickness, jagged behavior, and distinct colors
for method in offsets.keys():
    times = time_values_jagged[method]
    linestyle = '-' if method != "KFS-TUNE" else '-'
    linewidth = 4 if method == "KFS-TUNE" else 2  # Increase thickness for all lines
    color = method_colors[method]

    plt.plot(train_set_sizes, times, linestyle=linestyle, linewidth=linewidth, label=method, color=color)

# Log scale for y-axis
plt.xscale("log", base=2)
plt.yscale("log")

# Match axis limits from the reference plot
plt.xlim(16, 4096)
plt.ylim(1, 150000)

# Labels and title with required formatting
plt.xlabel("Train Set Size", fontsize=22, fontweight='bold')
plt.ylabel("Time (s)", fontsize=22, fontweight='bold')

# Adjust tick sizes and make x-axis ticks more sparse
plt.xticks([16, 64, 256, 1024, 4096], fontsize=22)  # Sparser x-axis ticks
plt.yticks(fontsize=22)

# Add grid
plt.grid(True, which="both", linestyle="--", linewidth=0.5)

# Add legend outside the plot area
plt.legend(fontsize=14, loc="center left", bbox_to_anchor=(1, 0.5))  # Move legend to the right

# Adjust layout to fit the legend properly
plt.tight_layout(rect=[0, 0, 0.85, 1])  # Leave space for the legend

# Show plot
plt.savefig("myplot_legend_right.png", bbox_inches='tight')
plt.show()

# Provide the file for download
"/mnt/data/myplot_legend_right.png"
