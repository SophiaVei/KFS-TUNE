from pathlib import Path
import time
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler

from KFSTUNE_functions import generate_kernels, scorers, transform_and_select_features

DATA_PATH = Path(__file__).resolve().parent.parent / "daily_fitbit_surveys_semas.pkl"
data = pd.read_pickle(DATA_PATH)

feature_columns = [
    "nightly_temperature",
    "nremhr",
    "rmssd",
    "spo2",
    "full_sleep_breathing_rate",
    "deep_sleep_breathing_rate",
    "light_sleep_breathing_rate",
    "rem_sleep_breathing_rate",
    "stress_score",
    "sleep_points",
    "responsiveness_points",
    "exertion_points",
    "wrist_temperature",
    "altitude",
    "calories",
    "vo2max",
    "distance",
    "oxygen_variation",
    "lightly_active_minutes",
    "moderately_active_minutes",
    "resting_heart_rate",
    "sedentary_minutes",
    "steps",
    "very_active_minutes",
    "minutes_below_zone_1",
    "minutes_in_zone_1",
    "minutes_in_zone_2",
    "minutes_in_zone_3",
    "bpm",
]

label_column = "mood"
required_columns = feature_columns + [label_column, "id", "date"]
data = data.dropna(subset=[label_column])[required_columns].copy()

label_encoder = LabelEncoder()
data["label"] = label_encoder.fit_transform(data[label_column])
data = data.sort_values(by=["id", "date"])


def create_windows(frame, window_size=3, step_size=1):
    X_windows = []
    y_windows = []
    groups = []

    for group_id, group in frame.groupby("id"):
        group = group.reset_index(drop=True)
        if len(group) < window_size:
            continue

        values = group[feature_columns].to_numpy(dtype=float)
        labels = group["label"].to_numpy()

        for start in range(0, len(group) - window_size + 1, step_size):
            end = start + window_size
            window_labels = labels[start:end]
            if len(np.unique(window_labels)) > 1:
                continue
            X_windows.append(values[start:end].reshape(-1))
            y_windows.append(window_labels[0])
            groups.append(group_id)

    return np.asarray(X_windows), np.asarray(y_windows), np.asarray(groups)


X_windows, y_windows, window_groups = create_windows(data)
print(f"Shape of X_windows: {X_windows.shape}")
print(f"Shape of y_windows: {y_windows.shape}")
print(f"Class distribution in windows: {Counter(y_windows)}")

gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
train_idx, test_idx = next(gss.split(X_windows, y_windows, groups=window_groups))

X_train_raw = X_windows[train_idx]
X_test_raw = X_windows[test_idx]
y_train = y_windows[train_idx]
y_test = y_windows[test_idx]

imputer = SimpleImputer(strategy="mean")
scaler = StandardScaler()

X_train_imputed = imputer.fit_transform(X_train_raw)
X_test_imputed = imputer.transform(X_test_raw)

X_train_scaled = scaler.fit_transform(X_train_imputed)
X_test_scaled = scaler.transform(X_test_imputed)

smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)

class_distribution_filtered = Counter(y_train)
class_distribution_balanced = Counter(y_train_balanced)

total_start_time = time.time()

start_time = time.time()
kernels = generate_kernels(
    input_length=X_train_balanced.shape[1],
    num_kernels=10000,
    avg_series_length=int(X_train_balanced.shape[1]),
)
X_train_transformed, selector, best_num_features, feature_scaler = transform_and_select_features(
    X_train_balanced,
    kernels,
    y=y_train_balanced,
    num_features=500,
    score_func=scorers["mi"],
    is_train=True,
)
train_transform_time = time.time() - start_time

start_time = time.time()
classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
classifier.fit(X_train_transformed, y_train_balanced)
training_time = time.time() - start_time

start_time = time.time()
X_test_transformed = transform_and_select_features(
    X_test_scaled,
    kernels,
    selector=selector,
    scaler=feature_scaler,
    is_train=False,
)
test_transform_time = time.time() - start_time

start_time = time.time()
predictions = classifier.predict(X_test_transformed)
test_time = time.time() - start_time
accuracy = accuracy_score(y_test, predictions)

print(f"Accuracy: {accuracy}")
print(f"Number of Features: {best_num_features}")
print(f"Training Transformation Time: {train_transform_time}s")
print(f"Training Time: {training_time}s")
print(f"Test Transformation Time: {test_transform_time}s")
print(f"Test Time: {test_time}s")

total_time = time.time() - total_start_time
print(f"Total time: {total_time}s")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
sns.barplot(
    x=list(class_distribution_filtered.keys()),
    y=list(class_distribution_filtered.values()),
    ax=axes[0],
)
axes[0].set_title("Train Class Distribution Before Balancing")
axes[0].set_xlabel("Class")
axes[0].set_ylabel("Frequency")

sns.barplot(
    x=list(class_distribution_balanced.keys()),
    y=list(class_distribution_balanced.values()),
    ax=axes[1],
)
axes[1].set_title("Train Class Distribution After SMOTE")
axes[1].set_xlabel("Class")
axes[1].set_ylabel("Frequency")

plt.tight_layout()
plt.show()

timing_data = {
    "Stage": ["Train Transformation", "Training", "Test Transformation", "Testing"],
    "Time (s)": [train_transform_time, training_time, test_transform_time, test_time],
}

timing_df = pd.DataFrame(timing_data)
fig, ax1 = plt.subplots(figsize=(10, 6))
sns.barplot(x="Stage", y="Time (s)", data=timing_df, ax=ax1)
ax1.set_title("Time Spent on Each Stage")

ax2 = ax1.twinx()
ax2.plot(timing_df["Stage"], [accuracy] * 4, color="red", marker="o", label="Accuracy")
ax2.set_ylabel("Accuracy")
ax2.set_ylim(0, 1)
ax2.legend(loc="upper left")

plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 6))
sns.barplot(x=["Selected Features"], y=[best_num_features])
plt.title("Number of Features Used After Selection")
plt.ylabel("Number of Features")
plt.show()
