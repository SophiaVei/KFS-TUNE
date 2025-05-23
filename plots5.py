import matplotlib.pyplot as plt
import numpy as np

# Estimated data points for train set sizes
train_sizes = np.array([16, 32, 64, 128, 256, 512, 1024, 2048, 4096])

# Estimated times for different methods based on visual approximation
methods = {
    "MLP": [10, 15, 25, 50, 100, 300, 800, 3000, 10000],
    "CNN": [12, 18, 30, 60, 120, 400, 1000, 3500, 12000],
    "FCN": [14, 20, 35, 70, 150, 500, 1300, 4000, 15000],
    "MCDCNN": [8, 12, 20, 40, 90, 250, 700, 2500, 8000],
    "TDE": [20, 25, 50, 100, 300, 900, 2500, 9000, 30000],
    "iTDE": [18, 22, 45, 90, 250, 800, 2200, 8500, 29000],
    "MUSE": [6, 10, 18, 35, 80, 200, 600, 2000, 6000],
    "KNN": [10, 14, 28, 55, 120, 400, 1100, 3800, 14000],
    "catch22": [5, 8, 15, 30, 75, 180, 500, 1800, 5500],
    "FreshPRINCE": [7, 12, 22, 45, 100, 300, 850, 3000, 10000],
    "STSF": [9, 14, 26, 50, 110, 350, 1000, 3400, 12000],
    "TSF": [11, 16, 30, 65, 140, 450, 1200, 4000, 13000],
    "CIF": [15, 20, 38, 80, 180, 600, 1600, 5000, 18000],
    "DrCIF": [16, 22, 40, 85, 200, 700, 2000, 7000, 25000],
    "ROCKET": [4, 7, 12, 25, 60, 150, 400, 1500, 5000],
    "Arsenal": [6, 10, 20, 40, 90, 250, 700, 2500, 8500],
    "ConvFS": [3, 5, 10, 20, 50, 120, 300, 1000, 3500],
}

# Plot setup
plt.figure(figsize=(10, 6))
for method, times in methods.items():
    plt.plot(train_sizes, times, label=method, linewidth=1)

# Highlight ConvFS with a thicker line
plt.plot(train_sizes, methods["ConvFS"], label="ConvFS", color="black", linewidth=3)

# Log-log scale for better visualization
plt.xscale("log")
plt.yscale("log")

# Labels and legend
plt.xlabel("Train Set Size")
plt.ylabel("Time (s)")
plt.legend(loc="upper left", fontsize=8, bbox_to_anchor=(1, 1))
plt.grid(True, which="both", linestyle="--", linewidth=0.5)

# Show plot
plt.savefig('plot.png')
