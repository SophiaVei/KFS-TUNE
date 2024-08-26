import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeClassifierCV
from ConvFS_functions import generate_kernels, transform_and_select_features
import time
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE, RandomOverSampler
from itertools import combinations
import random
from scipy.sparse import issparse

# Load the provided dataset
file_path = 'daily_fitbit_surveys_semas.pkl'
data = pd.read_pickle(file_path)

# Drop columns that are unsuitable for model training (e.g., ID, dates)
drop_columns = ['id', 'date']
data = data.drop(columns=drop_columns)

# Select 'mood' as the label
label_column = 'mood'

# Filter out rows with missing labels
data = data.dropna(subset=[label_column])

# Extract features and labels
X = data.drop(columns=[label_column])
y = data[label_column]

# Encode the labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Identify categorical and numerical columns
categorical_columns = X.select_dtypes(include=['object', 'category']).columns.tolist()
numeric_columns = X.select_dtypes(include=['number']).columns.tolist()

# Ensure categorical columns are uniformly strings and numeric columns are floats
X[categorical_columns] = X[categorical_columns].astype(str)
X[numeric_columns] = X[numeric_columns].astype(float)

# Store the available columns (both categorical and numerical)
all_columns = categorical_columns + numeric_columns

# Define the preprocessing for numeric and categorical data
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# Function to preprocess and transform the dataset
def preprocess_and_transform(selected_columns):
    # Split selected columns into categorical and numerical subsets
    selected_categorical = [col for col in selected_columns if col in categorical_columns]
    selected_numerical = [col for col in selected_columns if col in numeric_columns]

    # Update preprocessing pipelines based on selected columns
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, selected_numerical),
            ('cat', categorical_transformer, selected_categorical)
        ]
    )

    # Apply preprocessing
    X_preprocessed = preprocessor.fit_transform(X[selected_columns])

    # Handle imbalance in the dataset using oversampling and SMOTE
    ros = RandomOverSampler(random_state=42)
    X_resampled, y_resampled = ros.fit_resample(X_preprocessed, y_encoded)

    smote = SMOTE(random_state=42)
    X_balanced, y_balanced = smote.fit_resample(X_resampled, y_resampled)

    # Split the balanced data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_balanced, y_balanced, test_size=0.3, random_state=42
    )

    return X_train, X_test, y_train, y_test

# Store the best results
best_accuracy = 0
best_columns = None

# Run grid search with random feature subsets
n_trials = 100
for i in range(n_trials):
    # Randomly select a subset of features
    selected_columns = random.sample(all_columns, k=random.randint(5, len(all_columns)))
    print(f"Trial {i + 1}: Testing with {len(selected_columns)} features")

    # Preprocess and transform the dataset
    X_train, X_test, y_train, y_test = preprocess_and_transform(selected_columns)

    # Handle sparse matrices properly for calculating the series length
    if issparse(X_train):
        avg_series_length = np.mean([x.getnnz() for x in X_train])
    else:
        avg_series_length = np.mean([len(x) for x in X_train])

    # Initialize total start time
    start_time = time.time()

    # Generate kernels and transform data
    kernels = generate_kernels(X_train.shape[1], 10000, int(avg_series_length))
    X_train_transformed, selector, best_num_features, scaler = transform_and_select_features(
        X_train, kernels, y_train, is_train=True)

    # Train classifier
    classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
    classifier.fit(X_train_transformed, y_train)

    # Transform test data
    X_test_transformed = transform_and_select_features(
        X_test, kernels, selector=selector, scaler=scaler, is_train=False)

    # Test classifier
    predictions = classifier.predict(X_test_transformed)
    accuracy = np.mean(predictions == y_test)

    # Track the best result
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_columns = selected_columns

    # Print result for this trial
    print(f"Trial {i + 1} Accuracy: {accuracy:.4f} (Best Accuracy so far: {best_accuracy:.4f})")

# Final results
print(f"Best Accuracy: {best_accuracy}")
print(f"Best Feature Subset: {best_columns}")

# Save the best results to a CSV file
result = pd.DataFrame({
    'Best Accuracy': [best_accuracy],
    'Best Feature Subset': [best_columns]
})
result.to_csv('best_feature_subset_results.csv', index=False)
