import numpy as np
import pandas as pd

def generate_complex_time_series(num_samples, label):
    """
    Generate a time series with more complex behaviors and increased overlap between classes.
    """
    time = np.linspace(0, 2 * np.pi, num_samples)
    choice = np.random.choice(['a', 'b', 'c'], p=[0.3, 0.4, 0.3]) if label == 0 else (
        np.random.choice(['a', 'b', 'c'], p=[0.2, 0.6, 0.2]) if label == 1 else
        np.random.choice(['a', 'b', 'c'], p=[0.4, 0.2, 0.4]))

    if choice == 'a':
        # Noise standard deviation ranges from 0.1 to 1.0
        noise_std = np.random.uniform(0.1, 1.0)
        series = np.sin(time) + np.random.normal(0, noise_std, num_samples)
    elif choice == 'b':
        # Noise standard deviation ranges from 0.05 to 0.5
        noise_std = np.random.uniform(0.05, 0.5)
        series = np.sin(2 * time) * np.linspace(0.5, 1.5, num_samples) + np.random.normal(0, noise_std, num_samples)
        if label != 0:
            spike_indices = np.random.choice(num_samples, int(num_samples * 0.05), replace=False)
            series[spike_indices] += np.random.normal(0, 3, len(spike_indices))
    else:
        # Noise standard deviation ranges from 0.01 to 0.3
        noise_std = np.random.uniform(0.01, 0.3)
        series = 0.5 * np.sin(time) + 0.5 * np.sin(3 * time + np.pi / 4) + np.random.normal(0, noise_std, num_samples)
        series += np.power(time, 2) / 50 if label == 2 else -np.power(time, 2) / 50

    return series
def generate_complex_time_series(num_samples, label):
    """
    Generate a time series with more complex behaviors and increased overlap between classes.
    """
    time = np.linspace(0, 2 * np.pi, num_samples)
    choice = np.random.choice(['a', 'b', 'c'], p=[0.3, 0.4, 0.3]) if label == 0 else (
        np.random.choice(['a', 'b', 'c'], p=[0.2, 0.6, 0.2]) if label == 1 else
        np.random.choice(['a', 'b', 'c'], p=[0.4, 0.2, 0.4]))

    # Decide if this series should have minimal noise
    minimal_noise = np.random.choice([True, False], p=[0.2, 0.8])  # 20% chance of minimal noise

    if minimal_noise:
        noise_std_range = (0.01, 0.05)  # Very low noise
    else:
        # Regular noise ranges based on pattern
        if choice == 'a':
            noise_std_range = (0.1, 1.0)
        elif choice == 'b':
            noise_std_range = (0.05, 0.5)
        else:
            noise_std_range = (0.01, 0.3)

    noise_std = np.random.uniform(*noise_std_range)

    if choice == 'a':
        series = np.sin(time) + np.random.normal(0, noise_std, num_samples)
    elif choice == 'b':
        series = np.sin(2 * time) * np.linspace(0.5, 1.5, num_samples) + np.random.normal(0, noise_std, num_samples)
        if label != 0:
            spike_indices = np.random.choice(num_samples, int(num_samples * 0.05), replace=False)
            series[spike_indices] += np.random.normal(0, 3, len(spike_indices))
    else:
        series = 0.5 * np.sin(time) + 0.5 * np.sin(3 * time + np.pi / 4) + np.random.normal(0, noise_std, num_samples)
        series += np.power(time, 2) / 50 if label == 2 else -np.power(time, 2) / 50

    return series

def generate_random_time_series(num_series_range, num_samples_range, num_classes):
    data = pd.DataFrame(columns=['series', 'label'])
    series_list = []
    num_series = np.random.randint(num_series_range[0], num_series_range[1] + 1)

    for _ in range(num_series):
        label = np.random.randint(0, num_classes)
        num_samples = np.random.randint(num_samples_range[0], num_samples_range[1] + 1)
        series = generate_complex_time_series(num_samples, label)
        series_list.append(pd.DataFrame({'series': [series.tolist()], 'label': [label]}))

    data = pd.concat(series_list, ignore_index=True)
    return data


def generate_and_save_datasets(num_datasets, num_series_range, num_samples_range, num_classes, base_filename='dataset'):
    for i in range(num_datasets):
        dataset = generate_random_time_series(num_series_range, num_samples_range, num_classes)
        filename = f"{base_filename}_{i}.csv"
        dataset.to_csv(filename, index=False)
        print(f"Saved {filename}")

def generate_large_dataset_v2(num_series, num_samples_range, num_classes, filename):
    series_list = []
    for _ in range(num_series):
        label = np.random.randint(0, num_classes)
        num_samples_per_series = np.random.randint(num_samples_range[0], num_samples_range[1] + 1)  # Variable length
        series = generate_complex_time_series(num_samples_per_series, label)
        series_list.append({'series': series.tolist(), 'label': label})

    data = pd.DataFrame(series_list)
    data.to_csv(filename, index=False)
    print(f"Saved large dataset {filename}")


# Generate datasets with a variable number of series between 100 and 500, and variable series length between 30 and 100 samples
generate_and_save_datasets(40, (100, 1000), (30, 1000), 3)

# Generate large datasets with the new function
# Example call with updated parameters
generate_large_dataset_v2(100000, (30, 70), 3, '../large_dataset_series_v2.csv')


def generate_single_large_series(num_samples, num_classes, filename):
    """
    Generate a single large time series with a specified number of samples.
    Now introduces more variability in the series based on the label.
    """
    # Generate a sequence of labels for segments within the series
    segment_labels = np.random.randint(0, num_classes, size=num_samples // 1000)

    series = []
    for label in segment_labels:
        segment_length = 1000  # Each segment will be 1000 samples long
        segment = generate_complex_time_series(segment_length, label)
        series.extend(segment)

    # Ensure the series length matches num_samples
    series = series[:num_samples]

    # Convert the series into a DataFrame and save
    data = pd.DataFrame(
        {'series': [series], 'label': [segment_labels[0]]})  # Use the first segment's label as the overall label
    data.to_csv(filename, index=False)
    print(f"Saved single large series dataset {filename}")


# Example usage to generate a single series with 1,000,000 samples
generate_single_large_series(1000000, 3, '../single_large_series.csv')

