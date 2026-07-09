import time
from collections import Counter
from itertools import cycle

import matplotlib.pyplot as plt
import numpy as np
from imblearn.over_sampling import RandomOverSampler
from memory_profiler import memory_usage
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize
from sktime.classification.deep_learning.mcdcnn import MCDCNNClassifier
from sktime.datasets import load_UCR_UEA_dataset
from tslearn.preprocessing import TimeSeriesScalerMinMax

from aeon.classification.convolution_based import Arsenal, RocketClassifier
from aeon.classification.deep_learning.cnn import CNNClassifier
from aeon.classification.deep_learning.fcn import FCNClassifier
from aeon.classification.deep_learning.mlp import MLPClassifier
from aeon.classification.dictionary_based import (
    BOSSEnsemble,
    ContractableBOSS,
    IndividualBOSS,
    IndividualTDE,
    MUSE,
    TemporalDictionaryEnsemble,
    WEASEL,
)
from aeon.classification.distance_based import (
    KNeighborsTimeSeriesClassifier,
    ShapeDTW,
)
from aeon.classification.feature_based import (
    Catch22Classifier,
    FreshPRINCEClassifier,
)
from aeon.classification.interval_based import (
    CanonicalIntervalForestClassifier,
    DrCIFClassifier,
    SupervisedTimeSeriesForest,
    TimeSeriesForestClassifier,
)


# Example-only comparison script for a single UCR dataset.
# The plots include only results computed in this run.
dataset_name = "Car"


def dataframe_to_2darray(df):
    num_samples = df.shape[0]
    num_timesteps = len(df.iloc[0, 0])
    array_2d = np.empty((num_samples, num_timesteps))

    for i in range(num_samples):
        array_2d[i, :] = df.iloc[i, 0]

    return array_2d


def evaluate_classifier(classifier, X_train, X_test, y_train, y_test):
    def fit_and_predict():
        classifier.fit(X_train, y_train)
        return classifier.predict(X_test)

    start_time = time.time()
    mem_usage = memory_usage((fit_and_predict,), interval=0.1, include_children=True, retval=True)
    execution_time = time.time() - start_time
    max_mem_usage = max(mem_usage[0]) - min(mem_usage[0])
    predicted_labels = mem_usage[1]

    precision = precision_score(y_test, predicted_labels, average="weighted")
    accuracy = accuracy_score(y_test, predicted_labels)
    f1_score_val = f1_score(y_test, predicted_labels, average="weighted")
    confusion = confusion_matrix(y_test, predicted_labels)

    roc_auc_macro = None
    roc_auc_micro = None
    if hasattr(classifier, "predict_proba"):
        y_prob = classifier.predict_proba(X_test)
        roc_auc_macro = roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro")
        roc_auc_micro = roc_auc_score(y_test, y_prob, multi_class="ovr", average="micro")

    return (
        execution_time,
        max_mem_usage,
        precision,
        accuracy,
        f1_score_val,
        roc_auc_macro,
        roc_auc_micro,
        confusion,
    )


X_train_raw, y_train = load_UCR_UEA_dataset(dataset_name, split="train", return_X_y=True)
X_test_raw, y_test = load_UCR_UEA_dataset(dataset_name, split="test", return_X_y=True)

print("Length of each time series:", X_train_raw.iloc[0, 0].size)
print("Train size:", len(y_train))
print("Test size:", len(y_test))
print("Training set class distribution:", Counter(y_train))
print("Test set class distribution:", Counter(y_test))

scaler = TimeSeriesScalerMinMax()
X_train_processed = scaler.fit_transform(dataframe_to_2darray(X_train_raw))
X_test_processed = scaler.transform(dataframe_to_2darray(X_test_raw))

X_train_flat = X_train_processed.reshape((X_train_processed.shape[0], -1))
X_test_flat = X_test_processed.reshape((X_test_processed.shape[0], -1))

class_distribution = Counter(y_train)
min_class_size = min(class_distribution.values())
max_class_size = max(class_distribution.values())
imbalance_ratio = min_class_size / max_class_size
imbalance_threshold = 0.5

X_train_flat_resampled, y_train_resampled = X_train_flat, y_train
resampling_done = False

if imbalance_ratio < imbalance_threshold:
    print("Class imbalance detected. Applying RandomOverSampler...")
    ros = RandomOverSampler(random_state=0)
    X_train_flat_resampled, y_train_resampled = ros.fit_resample(X_train_flat, y_train)
    resampling_done = True

classifiers = [
    MLPClassifier(),
    CNNClassifier(),
    FCNClassifier(),
    MCDCNNClassifier(),
    BOSSEnsemble(),
    ContractableBOSS(),
    IndividualBOSS(),
    TemporalDictionaryEnsemble(),
    IndividualTDE(),
    WEASEL(support_probabilities=True),
    MUSE(support_probabilities=True),
    ShapeDTW(),
    KNeighborsTimeSeriesClassifier(),
    Catch22Classifier(),
    FreshPRINCEClassifier(),
    SupervisedTimeSeriesForest(),
    TimeSeriesForestClassifier(),
    CanonicalIntervalForestClassifier(),
    DrCIFClassifier(),
    RocketClassifier(),
    Arsenal(),
]

results = {
    "Classifier": [],
    "Execution Time": [],
    "Memory Usage": [],
    "Precision": [],
    "Accuracy": [],
    "F1 Score": [],
    "ROC-AUC Score (Macro)": [],
    "ROC-AUC Score (Micro)": [],
    "Confusion Matrix": [],
}

fpr_dict = {}
tpr_dict = {}
roc_auc_dict = {}

for classifier in classifiers:
    classifier_name = type(classifier).__name__
    if resampling_done:
        train_X, train_y = X_train_flat_resampled, y_train_resampled
    else:
        train_X, train_y = X_train_flat, y_train

    (
        exec_time,
        max_mem_usage,
        precision,
        accuracy,
        f1_score_val,
        roc_auc_macro,
        roc_auc_micro,
        confusion,
    ) = evaluate_classifier(classifier, train_X, X_test_flat, train_y, y_test)

    results["Classifier"].append(classifier_name)
    results["Execution Time"].append(exec_time)
    results["Memory Usage"].append(max_mem_usage)
    results["Precision"].append(precision)
    results["Accuracy"].append(accuracy)
    results["F1 Score"].append(f1_score_val)
    results["ROC-AUC Score (Macro)"].append(roc_auc_macro)
    results["ROC-AUC Score (Micro)"].append(roc_auc_micro)
    results["Confusion Matrix"].append(confusion)

    print(f"{classifier_name} Execution Time: {exec_time:.2f}s")
    print(f"{classifier_name} Memory Usage: {max_mem_usage:.2f} MB")
    print(f"{classifier_name} Precision: {precision:.2f}")
    print(f"{classifier_name} Accuracy: {accuracy:.2f}")
    print(f"{classifier_name} F1 Score: {f1_score_val:.2f}")
    print(f"{classifier_name} ROC-AUC Score (Macro): {roc_auc_macro}")
    print(f"{classifier_name} ROC-AUC Score (Micro): {roc_auc_micro}")

    if hasattr(classifier, "predict_proba"):
        y_prob = classifier.predict_proba(X_test_flat)
        y_test_bin = label_binarize(y_test, classes=np.unique(y_train))
        n_classes = y_test_bin.shape[1]

        fpr = {}
        tpr = {}
        roc_auc = {}
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        fpr_dict[classifier_name] = fpr
        tpr_dict[classifier_name] = tpr
        roc_auc_dict[classifier_name] = roc_auc

if fpr_dict:
    num_cols = 4
    num_rows = 6
    subplot_size_width = 4
    subplot_size_height = 4
    fig_width = subplot_size_width * num_cols
    fig_height = subplot_size_height * num_rows

    plt.figure(figsize=(fig_width, fig_height))
    for i, classifier_name in enumerate(fpr_dict):
        ax = plt.subplot(num_rows, num_cols, i + 1)
        for j in range(n_classes):
            ax.plot(
                fpr_dict[classifier_name][j],
                tpr_dict[classifier_name][j],
                lw=2,
                label=f"Class {j} (AUC = {roc_auc_dict[classifier_name][j]:.2f})",
            )
        ax.plot([0, 1], [0, 1], "k--", lw=2)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"ROC AUC for {classifier_name}")
        ax.legend(loc="lower right")

    plt.subplots_adjust(hspace=0.3, wspace=0.3, top=0.95)
    plt.suptitle(f"{dataset_name} ROC AUC Curves", fontsize=16)
    plt.savefig(f"{dataset_name}_ROC_AUC_curves.png", bbox_inches="tight")
    plt.show()


def plot_roc_auc_curves_macro(fpr_dict, tpr_dict, roc_auc_dict, classifiers, n_classes, dataset_name=dataset_name):
    plt.figure(figsize=(10, 8))

    colors = cycle(
        [
            "midnightblue",
            "indianred",
            "green",
            "purple",
            "orange",
            "brown",
            "pink",
            "gray",
            "olive",
            "cyan",
            "mediumaquamarine",
            "chocolate",
            "palegreen",
            "antiquewhite",
            "tan",
            "darkseagreen",
            "aquamarine",
            "cadetblue",
            "powderblue",
            "thistle",
            "palevioletred",
        ]
    )

    for classifier_name, color in zip(classifiers, colors):
        fpr = fpr_dict[classifier_name]
        tpr = tpr_dict[classifier_name]
        roc_auc = roc_auc_dict[classifier_name]

        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
        mean_tpr /= n_classes

        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        plt.plot(
            fpr["macro"],
            tpr["macro"],
            label=f"macro-average ROC curve of {classifier_name} (area = {roc_auc['macro']:.2f})",
            color=color,
            linestyle="-",
            linewidth=2,
        )

    plt.plot([0, 1], [0, 1], "k--", lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{dataset_name} Macro-average ROC curve per classifier")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(f"{dataset_name}_macro_average_roc_curve.png")
    plt.show()
    plt.close()


if fpr_dict:
    plot_roc_auc_curves_macro(
        fpr_dict,
        tpr_dict,
        roc_auc_dict,
        list(fpr_dict.keys()),
        n_classes,
    )


def plot_results_improved(results, metric, dataset_name, color, ylabel=None):
    plt.figure(figsize=(15, 8))
    plt.bar(results["Classifier"], results[metric], color=color)
    plt.xlabel("Classifiers")
    if ylabel:
        plt.ylabel(ylabel)
    title = f"{dataset_name} {metric} Comparison"
    plt.title(title)
    if metric == "Execution Time":
        max_execution_time = max(results[metric])
        plt.ylim(0, max_execution_time * 1.1)
    else:
        plt.ylim(0, max(results[metric]) * 1.1)

    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(f"{dataset_name}_{metric}.png", bbox_inches="tight")
    plt.show()


plot_results_improved(results, "ROC-AUC Score (Macro)", dataset_name, "saddlebrown")
plot_results_improved(results, "Execution Time", dataset_name, "sandybrown", ylabel="Time (s)")
plot_results_improved(results, "Memory Usage", dataset_name, "peachpuff", ylabel="Space (MB)")
plot_results_improved(results, "Precision", dataset_name, "peru")
plot_results_improved(results, "F1 Score", dataset_name, "sienna")

num_classifiers = len(results["Classifier"])
num_cols = 7
num_rows = -(-num_classifiers // num_cols)

plt.figure(figsize=(20, 4 * num_rows))
for i, classifier_name in enumerate(results["Classifier"]):
    plt.subplot(num_rows, num_cols, i + 1)
    plt.imshow(results["Confusion Matrix"][i], interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(f"{classifier_name}")
    plt.colorbar()
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    tick_marks = np.arange(len(np.unique(y_train)))
    plt.xticks(tick_marks, tick_marks, rotation=45)
    plt.yticks(tick_marks, tick_marks)

plt.subplots_adjust(top=0.85)
plt.suptitle(f"{dataset_name} Confusion Matrices", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(f"{dataset_name}_Confusion_Matrices.png", bbox_inches="tight")
plt.show()
