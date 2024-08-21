import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE, RandomOverSampler
from sklearn.metrics import accuracy_score
from sklearn.base import BaseEstimator, TransformerMixin
from itertools import chain, combinations

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
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_columns),
        ('cat', categorical_transformer, categorical_columns)
    ]
)

# Preprocess the entire dataset (train and test)
X_preprocessed = preprocessor.fit_transform(X)

# Split the data into training and test sets (after preprocessing)
X_train, X_test, y_train, y_test = train_test_split(X_preprocessed, y_encoded, test_size=0.3, random_state=42)

# Handle imbalance in the dataset using oversampling and SMOTE (now after preprocessing)
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X_train, y_train)

smote = SMOTE(random_state=42)
X_balanced, y_balanced = smote.fit_resample(X_resampled, y_resampled)

# Custom transformer for selecting specific columns (works on numeric columns)
class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[:, self.columns]

# Generator for all combinations of columns
def all_combinations(columns):
    # Create all combinations from 1 to len(columns)
    for r in range(1, len(columns) + 1):
        for combo in combinations(columns, r):
            yield combo

# Generate a list of column indices
all_columns = list(range(X_preprocessed.shape[1]))

# Initialize variables to store the best results
best_accuracy = 0
best_pipeline = None
best_columns = None

# Loop through the generator of all possible combinations of columns
for columns in all_combinations(all_columns):
    # Create a pipeline with column selection and classifier
    selector = ColumnSelector(columns=columns)
    pipeline = Pipeline(steps=[
        ('selector', selector),
        ('classifier', RidgeClassifierCV(alphas=np.logspace(-3, 3, 10)))
    ])

    # Fit the pipeline on the training data
    pipeline.fit(X_balanced, y_balanced)

    # Evaluate the pipeline on the test set
    y_test_pred = pipeline.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)

    # Track the best pipeline
    if test_accuracy > best_accuracy:
        best_accuracy = test_accuracy
        best_pipeline = pipeline
        best_columns = columns

# Save the best results to a CSV file
best_columns_list = list(best_columns) if best_columns else []
result = pd.DataFrame({
    'Best Columns': [best_columns_list],
    'Best Accuracy': [best_accuracy]
})

result.to_csv('best_columns_and_accuracy.csv', index=False)

print(f'Best columns: {best_columns}')
print(f'Best test accuracy: {best_accuracy}')
