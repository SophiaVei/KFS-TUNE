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
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score, precision_recall_fscore_support
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pointbiserialr, chi2_contingency
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import mutual_info_classif
import matplotlib.pyplot as plt
print(plt.style.available)


# Load the provided dataset
file_path = 'daily_fitbit_surveys_semas.pkl'
data = pd.read_pickle(file_path)

# Explore the dataset to understand its structure
print(data.info())

# Drop columns that are unsuitable for model training (e.g., ID, dates)
drop_columns = ['id', 'date']
data = data.drop(columns=drop_columns)
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


# Plotting misclassification percentage for females and males
labels = ['Females', 'Males']
misclassification_percentages = [female_misclassification_percent, male_misclassification_percent]

plt.figure(figsize=(8, 6))
plt.bar(labels, misclassification_percentages, color=['#FF9999', '#66B3FF'])
plt.ylabel('Misclassification Percentage (%)')
plt.title('Gender Misclassification Percentage')

# Adding value labels on the bars
for i, v in enumerate(misclassification_percentages):
    plt.text(i, v + 0.5, f"{v:.2f}%", ha='center', fontweight='bold')

plt.show()


# 1. Confusion Matrix
conf_matrix = confusion_matrix(y_test, predictions)
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=['Female', 'Male'], yticklabels=['Female', 'Male'])
plt.title('Confusion Matrix')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.show()

# 2. Classification Report (Precision, Recall, F1-Score)
class_report = classification_report(y_test, predictions, target_names=['Female', 'Male'])
print("Classification Report:\n", class_report)

# Bar plot for precision, recall, and f1-score
precision, recall, f1, _ = precision_recall_fscore_support(y_test, predictions, average=None)
metrics_df = pd.DataFrame({
    'Precision': precision,
    'Recall': recall,
    'F1-Score': f1
}, index=['Female', 'Male'])

metrics_df.plot(kind='bar', figsize=(8, 6), ylim=(0, 1), title='Precision, Recall, F1-Score for Gender')
plt.ylabel('Score')
plt.show()

# 3. ROC Curve and AUC Score
fpr, tpr, _ = roc_curve(y_test, predictions)
roc_auc = roc_auc_score(y_test, predictions)

plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()


# Continuous variables (Point Biserial Correlation)
print("\n=== Point Biserial Correlation (Continuous Features) ===")
continuous_features = pd.DataFrame(X_preprocessed, columns=all_feature_names)[numeric_feature_names]
for feature in continuous_features.columns:
    corr, p_value = pointbiserialr(continuous_features[feature], y_encoded)
    print(f"Feature: {feature}, Point Biserial Correlation: {corr:.4f}, p-value: {p_value:.4f}")

# Categorical variables (Chi-square test)
print("\n=== Chi-Square Test (Categorical Features) ===")
for feature in categorical_feature_names:
    contingency_table = pd.crosstab(pd.DataFrame(X_preprocessed, columns=all_feature_names)[feature], y_encoded)
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    print(f"Feature: {feature}, Chi-Square: {chi2:.4f}, p-value: {p:.4f}")


# Train RandomForest for feature importance
rf = RandomForestClassifier(random_state=42)
rf.fit(X_preprocessed, y_encoded)
importances = pd.Series(rf.feature_importances_, index=all_feature_names)

# Plot feature importance with rotated x-axis labels for better visibility
plt.figure(figsize=(12, 6))
ax = importances.sort_values(ascending=False).head(20).plot(kind='bar', title="Top 20 Feature Importance (Random Forest)")
plt.ylabel('Importance Score')

# Rotate the x-axis labels for better readability
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

plt.tight_layout()  # Adjusts the plot to ensure everything fits
plt.show()



# Calculate permutation importance
perm_importance = permutation_importance(classifier, X_test_transformed, y_test, n_repeats=10, random_state=42)

# Ensure indices are within bounds of feature names
sorted_idx = perm_importance.importances_mean.argsort()
valid_idx = sorted_idx[sorted_idx < len(all_feature_names)]  # Only keep valid indices



# Calculate mutual information before using the variable `mi`
mi = mutual_info_classif(X_preprocessed, y_encoded)

# Ensure mutual information values and feature names are aligned
if len(mi) == len(all_feature_names):
    # Create a dataframe to display the results
    mi_df = pd.DataFrame({'Feature': all_feature_names, 'Mutual Information': mi})
    mi_df = mi_df.sort_values(by='Mutual Information', ascending=False)

    # Plot mutual information
    ax = mi_df.head(20).plot(kind='bar', x='Feature', y='Mutual Information', figsize=(12, 6), title="Top 20 Mutual Information Scores")
    plt.ylabel('Mutual Information')

    # Rotate the x-axis labels for better readability
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

    plt.tight_layout()  # Adjust layout to make sure everything fits
    plt.show()
else:
    print(f"Mismatch between mutual information length ({len(mi)}) and feature names length ({len(all_feature_names)})")

# Print the MI dataframe for review
print(mi_df)






import matplotlib.pyplot as plt
print(plt.style.available)
# Function to style and format the plots for a modern look
def fancy_plot_style():
    plt.style.use('ggplot')  # 'ggplot' is widely available
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.figsize': (10, 6),
        'axes.edgecolor': '#333333',  # Darker edge color for axes
        'grid.color': '#cccccc',  # Light grid lines
        'grid.linewidth': 0.5,    # Thinner grid lines
        'axes.spines.top': False,  # Remove top spine for a cleaner look
        'axes.spines.right': False,  # Remove right spine for a cleaner look
        'axes.spines.left': False,   # Modern look: no left spine
        'axes.spines.bottom': False  # Modern look: no bottom spine
    })

# Apply the fancy plot styling
fancy_plot_style()

### Plot 1: Gender Misclassification Percentage (Modern Bar Chart)
labels = ['Females', 'Males']
misclassification_percentages = [female_misclassification_percent, male_misclassification_percent]

plt.figure(figsize=(8, 6))
plt.bar(labels, misclassification_percentages, color=['#6EC5E9', '#66C2A5'], edgecolor='none', linewidth=1.2)
plt.ylabel('Misclassification Percentage (%)', fontweight='bold')
plt.title('Gender Misclassification Percentage', fontsize=16, fontweight='bold')

# Adding value labels on the bars
for i, v in enumerate(misclassification_percentages):
    plt.text(i, v + 0.5, f"{v:.2f}%", ha='center', fontweight='bold', fontsize=12, color='black')

plt.ylim(0, max(misclassification_percentages) + 5)
plt.tight_layout()
plt.show()

### Plot 2: Confusion Matrix (Modern Heatmap with 2 Shades)
conf_matrix = confusion_matrix(y_test, predictions)

plt.figure(figsize=(6, 5))
# Use only two color shades (light blue and dark blue)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap=sns.light_palette("#66C2A5", as_cmap=True), linewidths=0.5, linecolor='white',
            cbar_kws={'label': 'Count'}, xticklabels=['Female', 'Male'], yticklabels=['Female', 'Male'])
plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
plt.ylabel('Actual Label', fontweight='bold')
plt.xlabel('Predicted Label', fontweight='bold')
plt.tight_layout()
plt.show()

### Plot 3: Precision, Recall, and F1-Score (Modern Bar Chart)
metrics_df.plot(kind='bar', figsize=(8, 6), ylim=(0, 1), color=['#FF6F61', '#6EC5E9', '#66C2A5'], edgecolor='none', linewidth=1.2)
plt.title('Precision, Recall, F1-Score for Gender', fontsize=16, fontweight='bold')
plt.ylabel('Score', fontweight='bold')

# Rotate x-axis labels slightly for readability
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

### Plot 4: ROC Curve (Modern with Minimalist Design)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='#6EC5E9', label=f'ROC Curve (AUC = {roc_auc:.2f})', lw=2)
plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=2)
plt.title('ROC Curve', fontsize=16, fontweight='bold')
plt.xlabel('False Positive Rate', fontweight='bold')
plt.ylabel('True Positive Rate', fontweight='bold')
plt.legend()
plt.tight_layout()
plt.show()

### Plot 5: Top 20 Feature Importance (Modern Bar Chart)
plt.figure(figsize=(12, 6))
ax = importances.sort_values(ascending=False).head(20).plot(kind='bar', color='#6EC5E9', edgecolor='none', linewidth=1.2)
plt.title("Top 20 Feature Importance", fontsize=16, fontweight='bold')
plt.ylabel('Importance Score', fontweight='bold')
plt.xlabel('Feature')  # Adding label for x-axis

# Rotate the x-axis labels for better readability
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
plt.tight_layout()
plt.show()

### Plot 6: Top 20 Mutual Information Scores (Modern Bar Chart)
plt.figure(figsize=(12, 6))
ax = mi_df.head(20).plot(kind='bar', x='Feature', y='Mutual Information', color='#66C2A5', edgecolor='none', linewidth=1.2)
plt.title("Top 20 Mutual Information Scores", fontsize=16, fontweight='bold')
plt.ylabel('Mutual Information', fontweight='bold')

# Rotate the x-axis labels for better readability
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
plt.tight_layout()
plt.show()
