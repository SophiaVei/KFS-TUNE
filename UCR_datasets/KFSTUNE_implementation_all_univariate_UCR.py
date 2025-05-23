import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeClassifierCV
from KFSTUNE_functions import generate_kernels, transform_and_select_features
from sktime.datasets import load_UCR_UEA_dataset
import time



# Define a list of dataset names
dataset_names = [
    "Adiac",
    "ArrowHead",
    "Beef",
    "BeetleFly",
    "BirdChicken",
    "Car",
    "CBF",
    "ChlorineConcentration",
    "CinCECGTorso",
    "Coffee",
    "Computers",
    "CricketX",
    "CricketY",
    "CricketZ",
    "DiatomSizeReduction",
    "DistalPhalanxOutlineCorrect",
    "DistalPhalanxOutlineAgeGroup",
    "DistalPhalanxTW",
    "Earthquakes",
    "ECG200",
    "ECG5000",
    "ECGFiveDays",
    "ElectricDevices",
    "FaceAll",
    "FaceFour",
    "FacesUCR",
    "FiftyWords",
    "Fish",
    "FordA",
    "FordB",
    "GunPoint",
    "Ham",
    "HandOutlines",
    "Haptics",
    "Herring",
    "InlineSkate",
    "InsectWingbeatSound",
    "ItalyPowerDemand",
    "LargeKitchenAppliances",
    "Lightning2",
    "Lightning7",
    "Mallat",
    "Meat",
    "MedicalImages",
    "MiddlePhalanxOutlineCorrect",
    "MiddlePhalanxOutlineAgeGroup",
    "MiddlePhalanxTW",
    "MoteStrain",
    "NonInvasiveFetalECGThorax1",
    "NonInvasiveFetalECGThorax2",
    "OliveOil",
    "OSULeaf",
    "PhalangesOutlinesCorrect",
    "Phoneme",
    "Plane",
    "ProximalPhalanxOutlineCorrect",
    "ProximalPhalanxOutlineAgeGroup",
    "ProximalPhalanxTW",
    "RefrigerationDevices",
    "ScreenType",
    "ShapeletSim",
    "ShapesAll",
    "SmallKitchenAppliances",
    "SonyAIBORobotSurface1",
    "SonyAIBORobotSurface2",
    "StarLightCurves",
    "Strawberry",
    "SwedishLeaf",
    "Symbols",
    "SyntheticControl",
    "ToeSegmentation1",
    "ToeSegmentation2",
    "Trace",
    "TwoLeadECG",
    "TwoPatterns",
    "UWaveGestureLibraryX",
    "UWaveGestureLibraryY",
    "UWaveGestureLibraryZ",
    "UWaveGestureLibraryAll",
    "Wafer",
    "Wine",
    "WordSynonyms",
    "Worms",
    "WormsTwoClass",
    "Yoga"
] # Add more if needed


total_start_time = time.time()
results = []
for dataset_name in dataset_names:
    print(f"Processing dataset: {dataset_name}")
    X_train, y_train = load_UCR_UEA_dataset(dataset_name, split="train", return_X_y=True)
    X_test, y_test = load_UCR_UEA_dataset(dataset_name, split="test", return_X_y=True)

    # Convert DataFrame to numpy array if necessary
    if isinstance(X_train, pd.DataFrame):
        X_train = np.stack(X_train.iloc[:, 0].apply(lambda x: x.to_numpy() if isinstance(x, pd.Series) else x))
    if isinstance(X_test, pd.DataFrame):
        X_test = np.stack(X_test.iloc[:, 0].apply(lambda x: x.to_numpy() if isinstance(x, pd.Series) else x))

    avg_series_length = np.mean([len(x) for x in X_train])

    # Start time measurement for train transformation
    start_time = time.time()
    kernels = generate_kernels(X_train.shape[1], 10000, int(avg_series_length))
    X_train_transformed, selector, best_num_features, scaler = transform_and_select_features(X_train, kernels, y_train,
                                                                                             is_train=True)
    train_transform_time = time.time() - start_time

    # Train classifier
    start_time = time.time()
    classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
    classifier.fit(X_train_transformed, y_train)
    training_time = time.time() - start_time

    # Start time measurement for test transformation
    start_time = time.time()
    X_test_transformed = transform_and_select_features(X_test, kernels, selector=selector, scaler=scaler,
                                                       is_train=False)
    test_transform_time = time.time() - start_time

    # Test classifier
    start_time = time.time()
    predictions = classifier.predict(X_test_transformed)
    test_time = time.time() - start_time
    accuracy = np.mean(predictions == y_test)

    results.append({
        "Dataset": dataset_name,
        "Accuracy": accuracy,
        "Num Features": best_num_features,  # Added number of features used
        "Training Transformation Time": train_transform_time,
        "Training Time": training_time,
        "Test Transformation Time": test_transform_time,
        "Test Time": test_time,
    })

    # Print the results
    print(f"Dataset: {dataset_name}")
    print(f"Accuracy: {accuracy}")
    print(f"Number of Features: {best_num_features}")  # Print number of features used
    print(f"Training Transformation Time: {train_transform_time}s")
    print(f"Training Time: {training_time}s")
    print(f"Test Transformation Time: {test_transform_time}s")
    print(f"Test Time: {test_time}s")
    print("=" * 50)  # Separator for different datasets

# After processing all datasets, calculate the average accuracy and average time
average_accuracy = np.mean([result['Accuracy'] for result in results])
average_total_time = np.mean([
    result['Training Transformation Time'] +
    result['Training Time'] +
    result['Test Transformation Time'] +
    result['Test Time']
    for result in results
])

# Print the results
print(f'Average Accuracy: {average_accuracy}')
print(f'Average Total Time (Training Transformation + Training + Test Transformation + Test): {average_total_time}')

total_time = time.time() - total_start_time
print(total_time)



