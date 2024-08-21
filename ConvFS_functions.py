import numpy as np
from numba import njit, prange
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import RidgeClassifierCV


def transform_and_select_features(X, kernels, y=None, num_features=500, selector=None, scaler=None, is_train=True):
    """
    Transforms the dataset using kernels and performs feature selection.
    Args:
    - X: Dataset to be transformed.
    - kernels: Pre-generated kernels.
    - y: Target labels (required for training set).
    - num_features: Number of features to select (fixed, based on prior knowledge).
    - selector: The fitted SelectKBest object (required for test set).
    - scaler: The MinMaxScaler object (required for test set).
    - is_train: Boolean indicating if the dataset is a training set.
    Returns:
    - X_transformed: Transformed and feature-selected dataset.
    - Additional returns for training set: selector, best_num_features, scaler
    """

    X_transformed = apply_kernels(X, kernels)

    if is_train:
        # Scale the transformed data
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X_transformed)

        # Select the top 'num_features' features using SelectKBest
        selector = SelectKBest(chi2, k=num_features)
        X_selected = selector.fit_transform(X_scaled, y)

        return X_selected, selector, num_features, scaler
    else:
        if scaler is not None and selector is not None:
            # Scale and transform the test data using the trained scaler and selector
            X_scaled = scaler.transform(X_transformed)
            X_selected = selector.transform(X_scaled)
            return X_selected
        else:
            raise ValueError("Scaler and selector must be provided for test set.")


@njit
def generate_kernels(input_length, num_kernels, avg_series_length):
    # Dynamically select candidate lengths based on average series length
    candidate_lengths =  np.array((7, 9, 11), dtype=np.int32)
    #candidate_lengths =  np.array((10, 15, 20), dtype=np.int32)
    #candidate_lengths = np.array((7, 9, 11), dtype=np.int32) if avg_series_length < 200 else np.array((10, 15, 20), dtype=np.int32)
    lengths = np.random.choice(candidate_lengths, num_kernels)
    weights = np.zeros(lengths.sum(), dtype=np.float64)
    biases = np.zeros(num_kernels, dtype=np.float64)
    dilations = np.zeros(num_kernels, dtype=np.int32)
    paddings = np.zeros(num_kernels, dtype=np.int32)

    a1 = 0
    for i in range(num_kernels):
        _length = lengths[i]
        _weights = np.random.normal(0, 1, _length) # Weights sampled from a normal distribution
        b1 = a1 + _length
        weights[a1:b1] = _weights - _weights.mean() # Mean-centering the weights
        biases[i] = np.random.uniform(-1, 1) # Bias sampled from a uniform distribution
        dilation = 2 ** np.random.uniform(0, np.log2((input_length - 1) / (_length - 1)))
        dilations[i] = np.int32(dilation) # Dilation determined through exponential sampling
        padding = ((_length - 1) * dilation) // 2 if np.random.randint(2) == 1 else 0
        paddings[i] = padding # Padding applied based on a random decision
        a1 = b1

    return weights, lengths, biases, dilations, paddings


@njit(fastmath=True)
def apply_kernel(X, weights, length, bias, dilation, padding):
    input_length = len(X)
    output_length = (input_length + (2 * padding)) - ((length - 1) * dilation)
    _ppv = 0
    _max = np.NINF
    _sum_values = np.empty(output_length, dtype=np.float64)  # Array to hold the sum values for std dev calculation

    end = (input_length + padding) - ((length - 1) * dilation)
    index_counter = 0  # Counter to keep track of the index in the _sum_values array

    for i in range(-padding, end):
        _sum = bias
        index = i
        for j in range(length):
            if index > -1 and index < input_length:
                _sum = _sum + weights[j] * X[index]
            index = index + dilation
        if _sum > _max:
            _max = _sum
        if _sum > 0:
            _ppv += 1
        _sum_values[index_counter] = _sum
        index_counter += 1

    _std_dev = np.std(_sum_values)  # Calculate the standard deviation of the activation map
    return _ppv / output_length, _max, _std_dev


@njit("float64[:,:](float64[:,:], Tuple((float64[::1], int32[:], float64[:], int32[:], int32[:])))", parallel=True, fastmath=True)
def apply_kernels(X, kernels):
    weights, lengths, biases, dilations, paddings = kernels
    num_examples, _ = X.shape
    num_kernels = len(lengths)
    num_features_per_kernel = 3  # Adding std deviation feature
    transformed_X = np.zeros((num_examples, num_kernels * num_features_per_kernel), dtype=np.float64)

    for i in prange(num_examples):
        a1 = 0  # for weights
        a2 = 0  # for features

        for j in range(num_kernels):
            b1 = a1 + lengths[j]
            b2 = a2 + num_features_per_kernel

            ppv, max_val, std_dev = apply_kernel(X[i], weights[a1:b1], lengths[j], biases[j], dilations[j], paddings[j])
            transformed_X[i, a2:b2] = np.array([ppv, max_val, std_dev], dtype=np.float64)

            a1 = b1
            a2 = b2

    return transformed_X