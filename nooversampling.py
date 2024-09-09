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
from collections import Counter

# Load the provided dataset
file_path = 'daily_fitbit_surveys_semas.pkl'
data = pd.read_pickle(file_path)

# Explore the dataset to understand its structure
print(data.info())

# Drop columns that are unsuitable for model training (e.g., ID, dates)
drop_columns = ['id', 'date']
data = data.drop(columns=drop_columns)

# Select the specific features from the best feature subset
feature_columns = [
    'bmi', 'minutes_awake', 'stimulus_control_category', 'wrist_temperature', 'scl_avg',
    'mindfulness_end_heart_rate', 'intellect', 'extraversion', 'age', 'exertion_points',
    'exercise_duration', 'stability', 'mindfulness_start_heart_rate', 'minutes_to_fall_asleep',
    'light', 'negative_affect_score', 'environmental_reevaluation_category',
    'helping_relationships_category', 'oxygen_variation', 'water_amount', 'mood',
    'heart_rate_alert', 'step_goal', 'ttm_stage', 'stress_score', 'nightly_temperature'
]

# Select 'gender' as the label
label_column = 'gender'

# Filter out rows with missing labels
data = data.dropna(subset=[label_column])

# Extract features and labels
X = data[feature_columns]
y = data[label_column]

# Print the data before preprocessing
print("Data before preprocessing:")
print(X.head())

# Encode the labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Identify categorical and numerical columns
categorical_columns = X.select_dtypes(include=['object', 'category']).columns.tolist()
numeric_columns = X.select_dtypes(include=['number']).columns.tolist()

# Convert categorical columns to strings to avoid mixed types
X[categorical_columns] = X[categorical_columns].astype(str)

# Convert numeric columns to float to ensure consistency
X[numeric_columns] = X[numeric_columns].astype(float)

# Define the preprocessing for numeric and categorical data
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_columns),
        ('cat', categorical_transformer, categorical_columns)
    ]
)

# Apply the preprocessing
X_preprocessed = preprocessor.fit_transform(X)

# Get feature names for both numeric and categorical columns
preprocessor.fit(X)

# Numeric feature names remain the same
numeric_feature_names = numeric_columns

# Categorical feature names are expanded by the one-hot encoder
categorical_feature_names = preprocessor.named_transformers_['cat'].named_steps['encoder'].get_feature_names_out(categorical_columns)

# Combine numeric and categorical feature names
all_feature_names = list(numeric_feature_names) + list(categorical_feature_names)

# Print the feature names
print(f"Features used in the model: {all_feature_names}")

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_preprocessed, y_encoded, test_size=0.3, random_state=42
)

# Calculate the average series length
avg_series_length = np.mean([len(x) for x in X_train])

# Initialize total start time
total_start_time = time.time()

# Start time measurement for train transformation
start_time = time.time()
kernels = generate_kernels(X_train.shape[1], 10000, int(avg_series_length))
X_train_transformed, selector, best_num_features, scaler = transform_and_select_features(
    X_train, kernels, y_train, is_train=True)
train_transform_time = time.time() - start_time

# Train classifier
start_time = time.time()
classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
classifier.fit(X_train_transformed, y_train)
training_time = time.time() - start_time

# Start time measurement for test transformation
start_time = time.time()
X_test_transformed = transform_and_select_features(
    X_test, kernels, selector=selector, scaler=scaler, is_train=False)
test_transform_time = time.time() - start_time

# Test classifier
start_time = time.time()
predictions = classifier.predict(X_test_transformed)
test_time = time.time() - start_time
accuracy = np.mean(predictions == y_test)

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

# Get the indices for females and males in the test set
female_indices = (y_test == label_encoder.transform(['FEMALE'])[0])
male_indices = (y_test == label_encoder.transform(['MALE'])[0])

# Calculate misclassification for females
female_misclassified_count = np.sum(predictions[female_indices] != y_test[female_indices])
female_total_count = np.sum(female_indices)
female_misclassification_percent = (female_misclassified_count / female_total_count) * 100

# Calculate misclassification for males
male_misclassified_count = np.sum(predictions[male_indices] != y_test[male_indices])
male_total_count = np.sum(male_indices)
male_misclassification_percent = (male_misclassified_count / male_total_count) * 100

# Print the misclassification percentages for females and males
print(f'Female Misclassification Percentage: {female_misclassification_percent:.2f}%')
print(f'Male Misclassification Percentage: {male_misclassification_percent:.2f}%')
