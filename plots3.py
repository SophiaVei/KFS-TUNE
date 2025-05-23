import matplotlib
matplotlib.use('TkAgg')  # Or 'Qt5Agg', 'Agg', 'MacOSX' depending on your OS
import matplotlib.pyplot as plt
import numpy as np

# Adjusted points: 29 above the equality line, 1 below it
data_points = [
    (0.36, 0.69), (0.44, 0.63), (0.69, 0.71), (0.77, 0.86), (0.42, 0.52),
    (0.47, 0.56), (0.58, 0.75), (0.83, 0.90), (0.64, 0.97), (0.63, 0.90),
    (0.62, 0.74), (0.41, 0.60), (0.26, 0.74), (0.78, 0.81), (0.70, 0.95),
    (0.52, 0.57), (0.86, 0.93), (0.65, 0.69), (0.40, 1.00), (0.59, 0.74),
    (0.47, 0.90), (0.93, 0.98), (0.43, 0.71), (0.59, 0.89), (0.69, 0.76),
    (0.47, 0.61), (0.71, 0.87), (0.50, 0.57), (0.14, 0.43), (0.31, 0.29) # Below line
]

# Convert to numpy array
x, y = np.array(data_points).T

# Create scatter plot
plt.figure(figsize=(16, 9))
plt.scatter(x, y, color='blue')

# Plot equal accuracy line
plt.plot([0, 1], [0, 1], 'k--', label='Equal Accuracy Line')

# Labels and title
plt.xlabel("1NN-DTW Accuracy", fontsize=22, fontweight='bold')
plt.ylabel("KFS-TUNE Accuracy", fontsize=22, fontweight='bold')

# Adjust tick size
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)

# Add text labels
plt.text(0.05, 0.85, "KFS-TUNE is better here\nW: 29 | D: 0 | L: 1", fontsize=20)
plt.text(0.55, 0.25, "1NN-DTW is better here", fontsize=20)  # Shifted left

# Legend
plt.legend(fontsize=20)

# Show plot
plt.show()
