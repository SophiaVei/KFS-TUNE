from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent


def generate_complex_time_series(num_samples, label):
    """
    Generate a time series with complex behaviors and overlap between classes.
    """
    time = np.linspace(0, 2 * np.pi, num_samples)
    choice = np.random.choice(["a", "b", "c"], p=[0.3, 0.4, 0.3]) if label == 0 else (
        np.random.choice(["a", "b", "c"], p=[0.2, 0.6, 0.2]) if label == 1 else
        np.random.choice(["a", "b", "c"], p=[0.4, 0.2, 0.4])
    )

    minimal_noise = np.random.choice([True, False], p=[0.2, 0.8])

    if minimal_noise:
        noise_std_range = (0.01, 0.05)
    elif choice == "a":
        noise_std_range = (0.1, 1.0)
    elif choice == "b":
        noise_std_range = (0.05, 0.5)
    else:
        noise_std_range = (0.01, 0.3)

    noise_std = np.random.uniform(*noise_std_range)

    if choice == "a":
        series = np.sin(time) + np.random.normal(0, noise_std, num_samples)
    elif choice == "b":
        series = (
            np.sin(2 * time) * np.linspace(0.5, 1.5, num_samples)
            + np.random.normal(0, noise_std, num_samples)
        )
        if label != 0:
            spike_indices = np.random.choice(num_samples, int(num_samples * 0.05), replace=False)
            series[spike_indices] += np.random.normal(0, 3, len(spike_indices))
    else:
        series = (
            0.5 * np.sin(time)
            + 0.5 * np.sin(3 * time + np.pi / 4)
            + np.random.normal(0, noise_std, num_samples)
        )
        series += np.power(time, 2) / 50 if label == 2 else -np.power(time, 2) / 50

    return series


def generate_random_time_series(num_series_range, num_samples_range, num_classes):
    series_list = []
    num_series = np.random.randint(num_series_range[0], num_series_range[1] + 1)

    for _ in range(num_series):
        label = np.random.randint(0, num_classes)
        num_samples = np.random.randint(num_samples_range[0], num_samples_range[1] + 1)
        series = generate_complex_time_series(num_samples, label)
        series_list.append(pd.DataFrame({"series": [series.tolist()], "label": [label]}))

    return pd.concat(series_list, ignore_index=True)


def generate_and_save_datasets(num_datasets, num_series_range, num_samples_range, num_classes, base_filename):
    base_filename = Path(base_filename)
    for i in range(num_datasets):
        dataset = generate_random_time_series(num_series_range, num_samples_range, num_classes)
        filename = base_filename.parent / f"{base_filename.name}_{i}.csv"
        dataset.to_csv(filename, index=False)
        print(f"Saved {filename}")


def generate_large_dataset_v2(num_series, num_samples_range, num_classes, filename):
    series_list = []
    for _ in range(num_series):
        label = np.random.randint(0, num_classes)
        num_samples_per_series = np.random.randint(num_samples_range[0], num_samples_range[1] + 1)
        series = generate_complex_time_series(num_samples_per_series, label)
        series_list.append({"series": series.tolist(), "label": label})

    data = pd.DataFrame(series_list)
    data.to_csv(filename, index=False)
    print(f"Saved large dataset {filename}")


def generate_single_large_series(num_samples, num_classes, filename):
    """
    Generate a large labeled dataset as consecutive 1000-point segments.
    """
    segment_labels = np.random.randint(0, num_classes, size=num_samples // 1000)

    series = []
    for label in segment_labels:
        segment = generate_complex_time_series(1000, label)
        series.extend(segment)

    series = series[:num_samples]

    segments = []
    start = 0
    for label in segment_labels:
        end = start + 1000
        segment = series[start:end]
        if len(segment) == 1000:
            segments.append({"series": list(segment), "label": label})
        start = end

    data = pd.DataFrame(segments)
    data.to_csv(filename, index=False)
    print(f"Saved labeled segment dataset {filename}")


def main():
    generate_and_save_datasets(40, (100, 1000), (30, 1000), 3, ROOT / "dataset")
    generate_large_dataset_v2(100000, (30, 70), 3, ROOT / "large_dataset_series_v2.csv")
    generate_single_large_series(1000000, 3, ROOT / "single_large_series.csv")


if __name__ == "__main__":
    main()
