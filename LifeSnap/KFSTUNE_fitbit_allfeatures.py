from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from imblearn.over_sampling import SMOTE

from KFSTUNE_functions import generate_kernels, transform_and_select_features

DATA_PATH = Path(__file__).resolve().parent.parent / "daily_fitbit_surveys_semas.pkl"
data = pd.read_pickle(DATA_PATH)

drop_columns = ["id", "date"]
label_column = "gender"

data = data.drop(columns=drop_columns)
data = data.dropna(subset=[label_column]).copy()

X = data.drop(columns=[label_column])
y = data[label_column]

categorical_columns = X.select_dtypes(include=["object", "category"]).columns.tolist()
numeric_columns = X.select_dtypes(include=["number"]).columns.tolist()

X[categorical_columns] = X[categorical_columns].astype(str)
X[numeric_columns] = X[numeric_columns].astype(float)

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.3,
    random_state=42,
    stratify=y_encoded,
)

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_columns),
        ("cat", categorical_transformer, categorical_columns),
    ]
)

X_train_preprocessed = preprocessor.fit_transform(X_train_raw)
X_test_preprocessed = preprocessor.transform(X_test_raw)

smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_preprocessed, y_train)

avg_series_length = X_train_balanced.shape[1]

total_start_time = time.time()

start_time = time.time()
kernels = generate_kernels(X_train_balanced.shape[1], 10000, int(avg_series_length))
X_train_transformed, selector, best_num_features, scaler = transform_and_select_features(
    X_train_balanced,
    kernels,
    y=y_train_balanced,
    is_train=True,
)
train_transform_time = time.time() - start_time

start_time = time.time()
classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
classifier.fit(X_train_transformed, y_train_balanced)
training_time = time.time() - start_time

start_time = time.time()
X_test_transformed = transform_and_select_features(
    X_test_preprocessed,
    kernels,
    selector=selector,
    scaler=scaler,
    is_train=False,
)
test_transform_time = time.time() - start_time

start_time = time.time()
predictions = classifier.predict(X_test_transformed)
test_time = time.time() - start_time
accuracy = np.mean(predictions == y_test)

print(f"Accuracy: {accuracy}")
print(f"Number of Features: {best_num_features}")
print(f"Training Transformation Time: {train_transform_time}s")
print(f"Training Time: {training_time}s")
print(f"Test Transformation Time: {test_transform_time}s")
print(f"Test Time: {test_time}s")
print(classification_report(y_test, predictions, target_names=label_encoder.classes_))

total_time = time.time() - total_start_time
print(f"Total time: {total_time}s")
