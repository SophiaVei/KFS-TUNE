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
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
import seaborn as sns
from scipy.stats import pointbiserialr, chi2_contingency
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import mutual_info_classif

# Load the provided dataset
file_path = 'daily_fitbit_surveys_semas.pkl'
data = pd.read_pickle(file_path)

# Explore the dataset to understand its structure
print(data.info())

# Drop columns that are unsuitable for model training (e.g., ID, dates)
drop_columns = ['id', 'date']
data = data.drop(columns=drop_columns)

# Replace NaN values in the 'gender' column with "Prefer not to say"
data['gender'] = data['gender'].fillna("Prefer not to say")

# Check unique values in the 'gender' column before encoding
print("Unique values in 'gender' column before encoding:")
print(data['gender'].unique())

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

# Extract features and labels
X = data[feature_columns]
y = data[label_column]

# Print the data before preprocessing
print("Data before preprocessing:")
print(X.head())

# Encode the labels (including the new "Prefer not to say" class)
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

# Check the length of all_feature_names and make sure it matches the transformed data
print(f"Number of features: {len(all_feature_names)}")
print(f"Shape of transformed data: {X_preprocessed.shape[1]}")

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

# Get the indices for females, males, and "Prefer not to say" in the test set
female_indices = (y_test == label_encoder.transform(['FEMALE'])[0])
male_indices = (y_test == label_encoder.transform(['MALE'])[0])
prefer_not_say_indices = (y_test == label_encoder.transform(['Prefer not to say'])[0])

# Calculate misclassification for females
female_misclassified_count = np.sum(predictions[female_indices] != y_test[female_indices])
female_total_count = np.sum(female_indices)
female_misclassification_percent = (female_misclassified_count / female_total_count) * 100

# Calculate misclassification for males
male_misclassified_count = np.sum(predictions[male_indices] != y_test[male_indices])
male_total_count = np.sum(male_indices)
male_misclassification_percent = (male_misclassified_count / male_total_count) * 100

# Calculate misclassification for "Prefer not to say"
prefer_not_say_misclassified_count = np.sum(predictions[prefer_not_say_indices] != y_test[prefer_not_say_indices])
prefer_not_say_total_count = np.sum(prefer_not_say_indices)
prefer_not_say_misclassification_percent = (prefer_not_say_misclassified_count / prefer_not_say_total_count) * 100

# Print the misclassification percentages for all categories
print(f'Female Misclassification Percentage: {female_misclassification_percent:.2f}%')
print(f'Male Misclassification Percentage: {male_misclassification_percent:.2f}%')
print(f'Prefer Not to Say Misclassification Percentage: {prefer_not_say_misclassification_percent:.2f}%')

# Plotting misclassification percentage for all categories
labels = ['Females', 'Males', 'Prefer Not to Say']
misclassification_percentages = [female_misclassification_percent, male_misclassification_percent, prefer_not_say_misclassification_percent]

plt.figure(figsize=(8, 6))
plt.bar(labels, misclassification_percentages, color=['#FF9999', '#66B3FF', '#A9A9A9'])
plt.ylabel('Misclassification Percentage (%)', fontweight='bold', fontsize=18)
#plt.title('Gender Misclassification Percentage')

# Adding value labels on the bars
for i, v in enumerate(misclassification_percentages):
    plt.text(i, v + 0.5, f"{v:.2f}%", ha='center', fontweight='bold')

plt.show()

# 2. Classification Report (include "Prefer not to say")
class_report = classification_report(y_test, predictions, target_names=['Female', 'Male', 'Prefer not to say'])
print("Classification Report:\n", class_report)

# Bar plot for precision, recall, and f1-score (include "Prefer not to say")
precision, recall, f1, _ = precision_recall_fscore_support(y_test, predictions, average=None)
metrics_df = pd.DataFrame({
    'Precision': precision,
    'Recall': recall,
    'F1-Score': f1
}, index=['Female', 'Male', 'Prefer not to say'])

metrics_df.plot(kind='bar', figsize=(8, 6), ylim=(0, 1), title='Precision, Recall, F1-Score for Gender')
plt.ylabel('Score')
plt.show()



# Function to style and format the plots for a modern look
def fancy_plot_style():
    # Function to style and format the plots
    plt.style.use('ggplot')  # 'ggplot' is widely available
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 12,  # Adjust tick size to match label size (not bold)
        'ytick.labelsize': 12,
        'legend.fontsize': 10,
        'figure.figsize': (10, 6),
        'axes.edgecolor': '#333333',  # Darker edge color for axes
        'grid.color': '#cccccc',  # Light grid lines
        'grid.linewidth': 0.5,  # Thinner grid lines
        'axes.spines.top': False,  # Remove top spine for a cleaner look
        'axes.spines.right': False,  # Remove right spine for a cleaner look
        'axes.spines.left': False,  # Modern look: no left spine
        'axes.spines.bottom': False  # Modern look: no bottom spine
    })

fancy_plot_style()

# Adjusting all plots
plt.figure(figsize=(8, 6))
plt.bar(labels, misclassification_percentages, color=['#6EC5E9', '#66C2A5', '#A9A9A9'], edgecolor='none', linewidth=1.2)
plt.ylabel('Misclassification Percentage (%)', fontweight='bold', fontsize=18)
plt.xticks(fontsize=20)  # Adjust tick size
plt.yticks(fontsize=20)  # Adjust tick size
plt.ylim(0, max(misclassification_percentages) + 5)
plt.tight_layout()
plt.show()


### Plot 2: Confusion Matrix (Modern Heatmap with 2 Shades)
conf_matrix = confusion_matrix(y_test, predictions)
# Confusion Matrix
plt.figure(figsize=(8, 7), facecolor='none')
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap=sns.light_palette("#66C2A5", as_cmap=True), linewidths=0.5, linecolor='white',
            cbar_kws={'label': 'Count'}, xticklabels=['Female', 'Male', 'Prefer not to say'],
            yticklabels=['Female', 'Male', 'Prefer not to say'])
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.ylabel('Actual Label', fontweight='bold')
plt.xlabel('Predicted Label', fontweight='bold')
plt.tight_layout()
plt.show()


### Plot 3: Precision, Recall, and F1-Score (Modern Bar Chart)
# Precision, Recall, and F1-Score
metrics_df.plot(kind='bar', figsize=(8, 6), ylim=(0, 1), color=['#FF6F61', '#6EC5E9', '#66C2A5'], edgecolor='none', linewidth=1.2)
plt.xticks(rotation=45, ha='right', fontsize=20)
plt.yticks(fontsize=20)
plt.ylabel('Score', fontweight='bold', fontsize=22)
plt.tight_layout()
plt.show()

### Plot 4: Feature Importance (Modern Bar Chart)
# Train a RandomForestClassifier to calculate feature importances
rf = RandomForestClassifier(random_state=42)
rf.fit(X_preprocessed, y_encoded)

# Get feature importances from the trained RandomForest model
importances = pd.Series(rf.feature_importances_, index=all_feature_names)
# Plot top 20 feature importance with modern style
plt.figure(figsize=(14, 8))
ax = importances.sort_values(ascending=False).head(20).plot(kind='bar', color='#6EC5E9', edgecolor='none', linewidth=1.2)
plt.ylabel('Importance Score', fontweight='bold', fontsize=22)
plt.xlabel('Feature', fontweight='bold', fontsize=22)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=20)
plt.yticks(fontsize=20)
plt.tight_layout()
plt.show()

# Rotate the x-axis labels for better readability
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
plt.tight_layout()
plt.show()

### Plot 5: Mutual Information (Modern Bar Chart, if applicable)
# Calculate mutual information
mi = mutual_info_classif(X_preprocessed, y_encoded)
# Mutual Information
if len(mi) == len(all_feature_names):
    plt.figure(figsize=(18, 18))
    mi_df = pd.DataFrame({'Feature': all_feature_names, 'Mutual Information': mi})
    mi_df = mi_df.sort_values(by='Mutual Information', ascending=False)
    ax = mi_df.head(20).plot(kind='bar', x='Feature', y='Mutual Information', color='#66C2A5', edgecolor='none', linewidth=1.2)
    plt.ylabel('Mutual Information', fontweight='bold', fontsize=22)
    plt.xlabel('Feature', fontweight='bold', fontsize=22)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=19)

    plt.yticks(fontsize=19)
    plt.tight_layout()

    plt.show()
else:
    print(f"Mismatch between mutual information length ({len(mi)}) and feature names length ({len(all_feature_names)})")


# Print the MI dataframe for review
print(mi_df)