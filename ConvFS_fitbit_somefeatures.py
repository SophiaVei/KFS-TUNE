import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeClassifierCV
from ConvFS_functions import generate_kernels, transform_and_select_features
import time
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE, RandomOverSampler
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

# Load the provided dataset
file_path = 'daily_fitbit_surveys_semas.pkl'
data = pd.read_pickle(file_path)

# Explore the dataset to understand its structure
print(data.head())
print(data.info())

# Select relevant numeric columns as features (you can adjust this selection)
feature_columns = [
    'nightly_temperature', 'nremhr', 'rmssd', 'spo2', 'full_sleep_breathing_rate',
    'deep_sleep_breathing_rate', 'light_sleep_breathing_rate', 'rem_sleep_breathing_rate',
    'stress_score', 'sleep_points', 'responsiveness_points', 'exertion_points',
    'wrist_temperature', 'altitude', 'calories', 'vo2max', 'distance', 'oxygen_variation',
    'lightly_active_minutes', 'moderately_active_minutes', 'resting_heart_rate',
    'sedentary_minutes', 'steps', 'very_active_minutes', 'minutes_below_zone_1',
    'minutes_in_zone_1', 'minutes_in_zone_2', 'minutes_in_zone_3', 'bpm'
]

# Select 'mood' as the label
label_column = 'mood'

# Filter out rows with missing labels
data = data.dropna(subset=[label_column])

# Extract features and labels
X = data[feature_columns]
y = data[label_column]

# Handle missing values in features
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

# Encode the labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Ensure that X is of the correct shape (samples, features)
print(f'Shape of X: {X.shape}')
print(f'Shape of y: {y.shape}')

# Create a DataFrame to handle time series segmentation
time_series_data = pd.DataFrame(X, columns=feature_columns)
time_series_data['label'] = y
time_series_data['id'] = data['id']
time_series_data['date'] = data['date']

# Sort by id and date
time_series_data = time_series_data.sort_values(by=['id', 'date'])

# Group by 'id' and create time series segments
grouped = time_series_data.groupby('id')

# Define a function to create overlapping windows of data
def create_windows(data, window_size=3, step_size=1):
    X_windows = []
    y_windows = []
    for _, group in data:
        group_len = len(group)
        if group_len < window_size:
            continue
        for start in range(0, group_len - window_size + 1, step_size):
            end = start + window_size
            window = group.iloc[start:end]
            if len(window['label'].unique()) > 1:
                continue
            X_windows.append(window[feature_columns].values)
            y_windows.append(window['label'].values[0])
    return np.array(X_windows), np.array(y_windows)

# Create windows
X_windows, y_windows = create_windows(grouped)

# Check the shape of the windows
print(f'Shape of X_windows: {X_windows.shape}')
print(f'Shape of y_windows: {y_windows.shape}')

# Check class distribution
class_distribution = Counter(y_windows)
print(f"Class distribution in y_windows: {class_distribution}")

# Reshape X_windows for resampling
n_samples, window_size, n_features = X_windows.shape
X_windows_reshaped = X_windows.reshape(n_samples, -1)

# Filter out classes with too few samples
min_samples = 2  # Minimum samples per class
valid_classes = [cls for cls, count in class_distribution.items() if count >= min_samples]
mask = np.isin(y_windows, valid_classes)
X_windows_reshaped = X_windows_reshaped[mask]
y_windows = y_windows[mask]

# Check the new class distribution after filtering
class_distribution_filtered = Counter(y_windows)
print(f"Class distribution after filtering: {class_distribution_filtered}")

# Use RandomOverSampler first to handle very imbalanced classes
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X_windows_reshaped, y_windows)

# Use SMOTE to balance the classes
smote = SMOTE(random_state=42)
X_balanced, y_balanced = smote.fit_resample(X_resampled, y_resampled)

# Reshape X_balanced back to the original window shape
X_balanced = X_balanced.reshape(-1, window_size, n_features)

# Check the new class distribution
class_distribution_balanced = Counter(y_balanced)
print(f"Class distribution after SMOTE in y_balanced: {class_distribution_balanced}")

# Split the balanced data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_balanced, y_balanced, test_size=0.3, random_state=42
)

# Calculate the average series length
avg_series_length = np.mean([len(x) for x in X_train])

# Initialize total start time
total_start_time = time.time()

# Start time measurement for train transformation
start_time = time.time()
kernels = generate_kernels(X_train.shape[2], 10000, int(avg_series_length))
X_train_reshaped = X_train.reshape(-1, X_train.shape[2])
X_train_transformed, selector, best_num_features, scaler = transform_and_select_features(
    X_train_reshaped, kernels, y_train.repeat(window_size), is_train=True)
train_transform_time = time.time() - start_time

# Ensure y_train and X_train_transformed have consistent lengths
y_train_corrected = y_train.repeat(X_train_transformed.shape[0] // y_train.shape[0])

# Train classifier
start_time = time.time()
classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
classifier.fit(X_train_transformed, y_train_corrected)
training_time = time.time() - start_time

# Start time measurement for test transformation
start_time = time.time()
X_test_reshaped = X_test.reshape(-1, X_test.shape[2])
X_test_transformed = transform_and_select_features(
    X_test_reshaped, kernels, selector=selector, scaler=scaler, is_train=False)
test_transform_time = time.time() - start_time

# Ensure y_test and X_test_transformed have consistent lengths
y_test_corrected = y_test.repeat(X_test_transformed.shape[0] // y_test.shape[0])

# Test classifier
start_time = time.time()
predictions = classifier.predict(X_test_transformed)
test_time = time.time() - start_time
accuracy = np.mean(predictions == y_test_corrected)

# Print the results
print(f'Accuracy: {accuracy}')
print(f'Number of Features: {best_num_features}')  # Print number of features used
print(f'Training Transformation Time: {train_transform_time}s')
print(f'Training Time: {training_time}s')
print(f'Test Transformation Time: {test_transform_time}s')
print(f'Test Time: {test_time}s')

# Calculate total time
total_time = time.time() - total_start_time
print(f'Total time: {total_time}s')


# 1. Plot Class Distribution Before and After Balancing

# Before balancing
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
sns.barplot(x=list(class_distribution_filtered.keys()), y=list(class_distribution_filtered.values()), ax=axes[0])
axes[0].set_title('Class Distribution Before Balancing')
axes[0].set_xlabel('Class')
axes[0].set_ylabel('Frequency')

# After SMOTE
sns.barplot(x=list(class_distribution_balanced.keys()), y=list(class_distribution_balanced.values()), ax=axes[1])
axes[1].set_title('Class Distribution After SMOTE Balancing')
axes[1].set_xlabel('Class')
axes[1].set_ylabel('Frequency')

plt.tight_layout()
plt.show()

# 2. Plot Time Spent on Each Step vs Accuracy

# Creating a bar plot for timing
timing_data = {
    'Stage': ['Train Transformation', 'Training', 'Test Transformation', 'Testing'],
    'Time (s)': [train_transform_time, training_time, test_transform_time, test_time]
}

timing_df = pd.DataFrame(timing_data)

fig, ax1 = plt.subplots(figsize=(10, 6))
sns.barplot(x='Stage', y='Time (s)', data=timing_df, ax=ax1)
ax1.set_title('Time Spent on Each Stage')

# Overlay accuracy as a separate plot
ax2 = ax1.twinx()
ax2.plot(timing_df['Stage'], [accuracy]*4, color='red', marker='o', label='Accuracy')
ax2.set_ylabel('Accuracy')
ax2.set_ylim(0, 1)
ax2.legend(loc='upper left')

plt.tight_layout()
plt.show()

# 3. Plot the Number of Features Used After Selection

plt.figure(figsize=(8, 6))
sns.barplot(x=['Selected Features'], y=[best_num_features])
plt.title('Number of Features Used After Selection')
plt.ylabel('Number of Features')
plt.show()
