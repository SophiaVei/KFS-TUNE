import argparse
import csv
import importlib.util
import json
import os
import subprocess
import sys
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import chi2
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from KFSTUNE_functions import generate_kernels, transform_and_select_features


UCR_BAKEOFF_DATASETS = [
    "Adiac", "ArrowHead", "Beef", "BeetleFly", "BirdChicken", "Car", "CBF",
    "ChlorineConcentration", "CinCECGTorso", "Coffee", "Computers",
    "CricketX", "CricketY", "CricketZ", "DiatomSizeReduction",
    "DistalPhalanxOutlineCorrect", "DistalPhalanxOutlineAgeGroup",
    "DistalPhalanxTW", "Earthquakes", "ECG200", "ECG5000", "ECGFiveDays",
    "ElectricDevices", "FaceAll", "FaceFour", "FacesUCR", "FiftyWords",
    "Fish", "FordA", "FordB", "GunPoint", "Ham", "HandOutlines", "Haptics",
    "Herring", "InlineSkate", "InsectWingbeatSound", "ItalyPowerDemand",
    "LargeKitchenAppliances", "Lightning2", "Lightning7", "Mallat", "Meat",
    "MedicalImages", "MiddlePhalanxOutlineCorrect",
    "MiddlePhalanxOutlineAgeGroup", "MiddlePhalanxTW", "MoteStrain",
    "NonInvasiveFetalECGThorax1", "NonInvasiveFetalECGThorax2", "OliveOil",
    "OSULeaf", "PhalangesOutlinesCorrect", "Phoneme", "Plane",
    "ProximalPhalanxOutlineCorrect", "ProximalPhalanxOutlineAgeGroup",
    "ProximalPhalanxTW", "RefrigerationDevices", "ScreenType", "ShapeletSim",
    "ShapesAll", "SmallKitchenAppliances", "SonyAIBORobotSurface1",
    "SonyAIBORobotSurface2", "StarLightCurves", "Strawberry", "SwedishLeaf",
    "Symbols", "SyntheticControl", "ToeSegmentation1", "ToeSegmentation2",
    "Trace", "TwoLeadECG", "TwoPatterns", "UWaveGestureLibraryX",
    "UWaveGestureLibraryY", "UWaveGestureLibraryZ", "UWaveGestureLibraryAll",
    "Wafer", "Wine", "WordSynonyms", "Worms", "WormsTwoClass", "Yoga",
]

METHODS = [
    "kfstune",
    "rocket",
    "minirocket",
    "multirocket",
    "srocket",
    "pocket",
    "detach_rocket",
]

PAPER_30_DATASETS = [
    "ArrowHead",
    "Crop",
    "WordSynonyms",
    "FiftyWords",
    "ShapesAll",
    "MixedShapesRegularTrain",
    "Car",
    "FreezerRegularTrain",
    "FreezerSmallTrain",
    "PowerCons",
    "ACSF1",
    "HouseTwenty",
    "CricketX",
    "Handwriting",
    "GunPointAgeSpan",
    "GunPointMaleVersusFemale",
    "GunPointOldVersusYoung",
    "ERing",
    "Rock",
    "SemgHandGenderCh2",
    "BME",
    "HandMovementDirection",
    "EOGHorizontalSignal",
    "EOGVerticalSignal",
    "InsectEPGRegularTrain",
    "InsectEPGSmallTrain",
    "PigAirwayPressure",
    "PigArtPressure",
    "PigCVP",
    "Fungi",
]


@dataclass
class Result:
    dataset: str
    method: str
    accuracy: float | None = None
    train_transform_time_s: float | None = None
    feature_selection_time_s: float | None = None
    training_time_s: float | None = None
    test_transform_time_s: float | None = None
    inference_time_s: float | None = None
    total_time_s: float | None = None
    retained_features: int | None = None
    retained_kernels: int | None = None
    memory_delta_mb: float | None = None
    status: str = "ok"
    error: str = ""


@contextmanager
def timed_result(result: Result):
    start = time.perf_counter()
    try:
        yield
    finally:
        result.total_time_s = time.perf_counter() - start


def rss_mb():
    try:
        import psutil

        return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    except Exception:
        return None


def import_from_path(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_ucr_dataset(dataset: str, ucr_path: str | None):
    if ucr_path:
        folder = Path(ucr_path) / dataset
        train = np.loadtxt(folder / f"{dataset}_TRAIN.tsv")
        test = np.loadtxt(folder / f"{dataset}_TEST.tsv")
        y_train, x_train = train[:, 0], train[:, 1:]
        y_test, x_test = test[:, 0], test[:, 1:]
    else:
        from sktime.datasets import load_UCR_UEA_dataset

        x_train, y_train = load_UCR_UEA_dataset(dataset, split="train", return_X_y=True)
        x_test, y_test = load_UCR_UEA_dataset(dataset, split="test", return_X_y=True)
        x_train = nested_univariate_to_2d(x_train)
        x_test = nested_univariate_to_2d(x_test)

    x_train = np.asarray(x_train, dtype=np.float64)
    x_test = np.asarray(x_test, dtype=np.float64)
    x_train[np.isnan(x_train)] = 0
    x_test[np.isnan(x_test)] = 0

    encoder = LabelEncoder()
    y_train = encoder.fit_transform(y_train)
    y_test = encoder.transform(y_test)
    return x_train, y_train, x_test, y_test


def nested_univariate_to_2d(x):
    if isinstance(x, pd.DataFrame):
        return np.stack(
            x.iloc[:, 0].apply(lambda s: s.to_numpy() if hasattr(s, "to_numpy") else np.asarray(s))
        )
    arr = np.asarray(x)
    if arr.ndim == 3 and arr.shape[1] == 1:
        return arr[:, 0, :]
    if arr.ndim == 2:
        return arr
    raise ValueError(f"Expected univariate panel data, got shape {arr.shape}")


def to_panel_3d(x_2d):
    return np.asarray(x_2d, dtype=np.float64)[:, np.newaxis, :]


def fit_ridge(x_train, y_train):
    clf = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
    clf.fit(x_train, y_train)
    return clf


def run_kfstune(dataset, x_train, y_train, x_test, y_test, args):
    result = Result(dataset=dataset, method="kfstune")
    mem0 = rss_mb()
    with timed_result(result):
        avg_len = int(np.mean([len(x) for x in x_train]))

        start = time.perf_counter()
        kernels = generate_kernels(x_train.shape[1], args.num_kernels, avg_len)
        x_train_full = transform_and_select_features(
            x_train,
            kernels,
            y=y_train,
            num_features=min(args.num_features, args.num_kernels * 3),
            score_func=chi2,
            is_train=True,
        )
        x_train_sel, selector, retained, scaler = x_train_full
        result.train_transform_time_s = time.perf_counter() - start
        result.feature_selection_time_s = result.train_transform_time_s
        result.retained_features = int(retained)

        start = time.perf_counter()
        clf = fit_ridge(x_train_sel, y_train)
        result.training_time_s = time.perf_counter() - start

        start = time.perf_counter()
        x_test_sel = transform_and_select_features(
            x_test, kernels, selector=selector, scaler=scaler, is_train=False
        )
        result.test_transform_time_s = time.perf_counter() - start

        start = time.perf_counter()
        pred = clf.predict(x_test_sel)
        result.inference_time_s = time.perf_counter() - start
        result.accuracy = accuracy_score(y_test, pred)

    mem1 = rss_mb()
    result.memory_delta_mb = None if mem0 is None or mem1 is None else mem1 - mem0
    return result


def run_sktime_rocket(dataset, x_train, y_train, x_test, y_test, args, method):
    result = Result(dataset=dataset, method=method)
    mem0 = rss_mb()
    with timed_result(result):
        from sktime.transformations.panel.rocket import Rocket, MiniRocket, MultiRocket

        transformer_cls = {
            "rocket": Rocket,
            "minirocket": MiniRocket,
            "multirocket": MultiRocket,
        }[method]
        transformer = transformer_cls(num_kernels=args.num_kernels)
        x_train_panel = to_panel_3d(x_train)
        x_test_panel = to_panel_3d(x_test)

        start = time.perf_counter()
        z_train = np.asarray(transformer.fit_transform(x_train_panel))
        result.train_transform_time_s = time.perf_counter() - start
        result.retained_features = int(z_train.shape[1])
        if method == "rocket":
            result.retained_kernels = int(args.num_kernels)

        start = time.perf_counter()
        clf = fit_ridge(z_train, y_train)
        result.training_time_s = time.perf_counter() - start

        start = time.perf_counter()
        z_test = np.asarray(transformer.transform(x_test_panel))
        result.test_transform_time_s = time.perf_counter() - start

        start = time.perf_counter()
        pred = clf.predict(z_test)
        result.inference_time_s = time.perf_counter() - start
        result.accuracy = accuracy_score(y_test, pred)

    mem1 = rss_mb()
    result.memory_delta_mb = None if mem0 is None or mem1 is None else mem1 - mem0
    return result


def run_rocket_official(dataset, x_train, y_train, x_test, y_test, args):
    result = Result(dataset=dataset, method="rocket")
    mem0 = rss_mb()
    pocket_dir = ROOT / "external_baselines" / "POCKET" / "ROCKET-PPV-MAX"
    if not hasattr(np, "NINF"):
        np.NINF = -np.inf
    rocket_functions = import_from_path("rocket_official_functions", pocket_dir / "rocket_functions.py")

    with timed_result(result):
        start = time.perf_counter()
        kernels = rocket_functions.generate_kernels(x_train.shape[-1], args.num_kernels)
        kernels = kernels[:-1]
        z_train = rocket_functions.apply_kernels(x_train.copy(), kernels)
        result.train_transform_time_s = time.perf_counter() - start
        result.retained_features = int(z_train.shape[1])
        result.retained_kernels = int(args.num_kernels)

        start = time.perf_counter()
        clf = fit_ridge(z_train, y_train)
        result.training_time_s = time.perf_counter() - start

        start = time.perf_counter()
        z_test = rocket_functions.apply_kernels(x_test.copy(), kernels)
        result.test_transform_time_s = time.perf_counter() - start

        start = time.perf_counter()
        pred = clf.predict(z_test)
        result.inference_time_s = time.perf_counter() - start
        result.accuracy = accuracy_score(y_test, pred)

    mem1 = rss_mb()
    result.memory_delta_mb = None if mem0 is None or mem1 is None else mem1 - mem0
    return result


def run_minirocket_official(dataset, x_train, y_train, x_test, y_test, args):
    result = Result(dataset=dataset, method="minirocket")
    mem0 = rss_mb()
    minirocket_dir = ROOT / "external_baselines" / "POCKET" / "MiniROCKET"
    minirocket = import_from_path("official_minirocket", minirocket_dir / "minirocket.py")

    with timed_result(result):
        num_features = max(84, (args.num_kernels // 84) * 84)
        x_train32 = x_train.astype(np.float32, copy=True)
        x_test32 = x_test.astype(np.float32, copy=True)

        start = time.perf_counter()
        parameters = minirocket.fit(x_train32, num_features=num_features)
        z_train = minirocket.transform(x_train32, parameters)
        result.train_transform_time_s = time.perf_counter() - start
        result.retained_features = int(z_train.shape[1])
        result.retained_kernels = 84

        start = time.perf_counter()
        clf = fit_ridge(z_train, y_train)
        result.training_time_s = time.perf_counter() - start

        start = time.perf_counter()
        z_test = minirocket.transform(x_test32, parameters)
        result.test_transform_time_s = time.perf_counter() - start

        start = time.perf_counter()
        pred = clf.predict(z_test)
        result.inference_time_s = time.perf_counter() - start
        result.accuracy = accuracy_score(y_test, pred)

    mem1 = rss_mb()
    result.memory_delta_mb = None if mem0 is None or mem1 is None else mem1 - mem0
    return result


def run_detach_rocket(dataset, x_train, y_train, x_test, y_test, args):
    result = Result(dataset=dataset, method="detach_rocket")
    mem0 = rss_mb()
    with timed_result(result):
        from detach_rocket.detach_classes import DetachMatrix

        pocket_dir = ROOT / "external_baselines" / "POCKET" / "ROCKET-PPV-MAX"
        if not hasattr(np, "NINF"):
            np.NINF = -np.inf
        rocket_functions = import_from_path("detach_rocket_features", pocket_dir / "rocket_functions.py")

        start = time.perf_counter()
        kernels = rocket_functions.generate_kernels(x_train.shape[-1], args.num_kernels)
        kernels = kernels[:-1]
        z_train = rocket_functions.apply_kernels(x_train.copy(), kernels)
        result.train_transform_time_s = time.perf_counter() - start

        start = time.perf_counter()
        z_test = rocket_functions.apply_kernels(x_test.copy(), kernels)
        result.test_transform_time_s = time.perf_counter() - start

        model = DetachMatrix(
            trade_off=args.detach_tradeoff,
            verbose=False,
        )

        start = time.perf_counter()
        model.fit(z_train, y_train)
        fit_time = time.perf_counter() - start

        result.feature_selection_time_s = fit_time
        result.retained_features = int(np.sum(getattr(model, "_feature_mask", [])))
        result.retained_kernels = None if result.retained_features is None else int(np.ceil(result.retained_features / 2))

        start = time.perf_counter()
        pred = model.predict(z_test)
        result.inference_time_s = time.perf_counter() - start
        result.accuracy = accuracy_score(y_test, pred)

    mem1 = rss_mb()
    result.memory_delta_mb = None if mem0 is None or mem1 is None else mem1 - mem0
    return result


def normalize_features(z_train, z_test):
    mean = np.mean(z_train, axis=0)
    z_train = z_train - mean
    z_test = z_test - mean
    norm = np.linalg.norm(z_train, axis=0)
    norm[norm == 0] = 1
    return z_train / norm, z_test / norm


def run_pocket(dataset, x_train, y_train, x_test, y_test, args):
    result = Result(dataset=dataset, method="pocket")
    mem0 = rss_mb()
    pocket_dir = ROOT / "external_baselines" / "POCKET" / "ROCKET-PPV-MAX"
    sys.path.insert(0, str(pocket_dir))
    try:
        if not hasattr(np, "NINF"):
            np.NINF = -np.inf
        rocket_functions = import_from_path("pocket_rocket_functions", pocket_dir / "rocket_functions.py")
        pruner_mod = import_from_path("pocket_pruner", pocket_dir / "PROCKET_pruner.py")

        with timed_result(result):
            start = time.perf_counter()
            kernels = rocket_functions.generate_kernels(x_train.shape[-1], args.num_kernels)
            kernels = kernels[:-1]
            z_train = rocket_functions.apply_kernels(x_train.copy(), kernels)
            result.train_transform_time_s = time.perf_counter() - start

            start = time.perf_counter()
            z_test = rocket_functions.apply_kernels(x_test.copy(), kernels)
            result.test_transform_time_s = time.perf_counter() - start

            z_train_norm, z_test_norm = normalize_features(z_train.copy(), z_test.copy())
            remain_num = max(1, min(args.pocket_remain_kernels, args.num_kernels - 1))
            n_class = int(np.max(y_train)) + 1

            start = time.perf_counter()
            pruner = pruner_mod.PROCKETPruner(
                n_class,
                y_train,
                z_test_norm,
                y_test,
                k=args.pocket_k,
                remain_num=remain_num,
                epoch=args.pocket_epochs,
                stop_thr=args.pocket_stop_thr,
                if_print=False,
                _dataset_name=dataset,
            )
            pruner.fit(z_train_norm, y_train)
            result.feature_selection_time_s = time.perf_counter() - start

            theta_zero = np.atleast_1d(pruner.Theta_zero_index).astype(int)
            drop = np.hstack((2 * theta_zero, 2 * theta_zero + 1))
            drop = drop[(drop >= 0) & (drop < z_train.shape[1])]
            z_train_pruned = np.delete(z_train, drop, axis=1)
            z_test_pruned = np.delete(z_test, drop, axis=1)

            start = time.perf_counter()
            clf = fit_ridge(z_train_pruned, y_train)
            result.training_time_s = time.perf_counter() - start
            result.retained_features = int(z_train_pruned.shape[1])
            result.retained_kernels = int(result.retained_features // 2)

            start = time.perf_counter()
            pred = clf.predict(z_test_pruned)
            result.inference_time_s = time.perf_counter() - start
            result.accuracy = accuracy_score(y_test, pred)
    finally:
        if sys.path and sys.path[0] == str(pocket_dir):
            sys.path.pop(0)

    mem1 = rss_mb()
    result.memory_delta_mb = None if mem0 is None or mem1 is None else mem1 - mem0
    return result


def run_srocket(dataset, x_train, y_train, x_test, y_test, args):
    result = Result(dataset=dataset, method="srocket")
    mem0 = rss_mb()
    srocket_dir = ROOT / "external_baselines" / "srocket" / "utils"
    if not hasattr(np, "NINF"):
        np.NINF = -np.inf
    rocket_functions = import_from_path("srocket_rocket_functions", srocket_dir / "rocket_functions.py")
    optimization = import_from_path("srocket_optimization", srocket_dir / "optimization.py")

    with timed_result(result):
        start = time.perf_counter()
        kernels = rocket_functions.generate_kernels(x_train.shape[-1], args.num_kernels)
        kernels = kernels[:-2]
        z_train_all = rocket_functions.apply_kernels(x_train.copy(), kernels)
        z_train = z_train_all[:, 0::2]
        result.train_transform_time_s = time.perf_counter() - start

        start = time.perf_counter()
        z_test_all = rocket_functions.apply_kernels(x_test.copy(), kernels)
        z_test = z_test_all[:, 0::2]
        result.test_transform_time_s = time.perf_counter() - start

        start = time.perf_counter()
        full_clf = fit_ridge(z_train, y_train)
        result.training_time_s = time.perf_counter() - start

        n_samples = z_train.shape[0]
        pop_size = args.srocket_population
        s = np.ones((args.num_kernels, pop_size))
        half = pop_size // 2
        if half > 0:
            s[:, :half] = np.round(np.random.random((args.num_kernels, half)))
        s = np.tile(s[np.newaxis, :, :], (n_samples, 1, 1))
        costs = np.ones(pop_size) * np.inf
        scores = np.zeros(pop_size)

        def candidate_scores(states):
            out = np.zeros(states.shape[2])
            for i in range(states.shape[2]):
                mask = states[0, :, i] > 0
                if np.any(mask):
                    out[i] = full_clf.score(z_train * mask, y_train)
            return out

        start = time.perf_counter()
        scores = candidate_scores(s)
        costs = 1 - (scores - np.mean(s[0], axis=0))
        best_state = s[0, :, int(np.argmin(costs))].copy()
        best_cost = float(np.min(costs))

        for _ in range(args.srocket_epochs):
            candidates = optimization.evolution(s, args.srocket_mutation, args.srocket_crossover)
            candidate_score = candidate_scores(candidates)
            candidate_cost = 1 - (candidate_score - np.mean(candidates[0], axis=0))
            s, best_state_epoch, costs, best_cost_epoch, _, scores, _ = optimization.selection(
                s, candidates, costs, candidate_cost, scores, candidate_score
            )
            if best_cost_epoch < best_cost:
                best_cost = float(best_cost_epoch)
                best_state = best_state_epoch.copy()

        result.feature_selection_time_s = time.perf_counter() - start
        mask = best_state > 0
        if not np.any(mask):
            mask[np.argmax(np.var(z_train, axis=0))] = True
        result.retained_kernels = int(np.sum(mask))
        result.retained_features = result.retained_kernels

        start = time.perf_counter()
        clf = fit_ridge(z_train[:, mask], y_train)
        result.training_time_s += time.perf_counter() - start

        start = time.perf_counter()
        pred = clf.predict(z_test[:, mask])
        result.inference_time_s = time.perf_counter() - start
        result.accuracy = accuracy_score(y_test, pred)

    mem1 = rss_mb()
    result.memory_delta_mb = None if mem0 is None or mem1 is None else mem1 - mem0
    return result


def run_method(method, dataset, x_train, y_train, x_test, y_test, args):
    runners = {
        "kfstune": run_kfstune,
        "rocket": run_rocket_official,
        "minirocket": run_minirocket_official,
        "multirocket": lambda *a: run_sktime_rocket(*a, method="multirocket"),
        "detach_rocket": run_detach_rocket,
        "srocket": run_srocket,
        "pocket": run_pocket,
    }
    try:
        return runners[method](dataset, x_train, y_train, x_test, y_test, args)
    except Exception as exc:
        return Result(
            dataset=dataset,
            method=method,
            status="error",
            error=f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=4)}",
        )


def append_rows(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(asdict(rows[0]).keys())
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def write_summary(results_csv: Path, summary_csv: Path, kfstune_name="kfstune"):
    df = pd.read_csv(results_csv)
    ok = df[df["status"] == "ok"].copy()
    if ok.empty:
        return

    pivot = ok.pivot_table(index="dataset", columns="method", values="accuracy", aggfunc="mean")
    ranks = pivot.rank(axis=1, ascending=False, method="average")
    summary = []
    for method in sorted(ok["method"].unique()):
        wins = ties = losses = np.nan
        if kfstune_name in pivot.columns and method != kfstune_name:
            both = pivot[[kfstune_name, method]].dropna()
            diff = both[method] - both[kfstune_name]
            eps = 1e-12
            wins = int(np.sum(diff > eps))
            ties = int(np.sum(np.abs(diff) <= eps))
            losses = int(np.sum(diff < -eps))
        method_rows = ok[ok["method"] == method]
        summary.append(
            {
                "method": method,
                "datasets_completed": int(method_rows["dataset"].nunique()),
                "mean_accuracy": method_rows["accuracy"].mean(),
                "average_rank": ranks[method].mean() if method in ranks else np.nan,
                "wins_vs_kfstune": wins,
                "ties_vs_kfstune": ties,
                "losses_vs_kfstune": losses,
                "mean_training_time_s": method_rows["training_time_s"].mean(),
                "mean_feature_selection_time_s": method_rows["feature_selection_time_s"].mean(),
                "mean_inference_time_s": method_rows["inference_time_s"].mean(),
                "mean_retained_features": method_rows["retained_features"].mean(),
                "mean_retained_kernels": method_rows["retained_kernels"].mean(),
                "mean_memory_delta_mb": method_rows["memory_delta_mb"].mean(),
            }
        )
    pd.DataFrame(summary).to_csv(summary_csv, index=False)


def progress_counts(results_csv: Path):
    if not results_csv.exists():
        return "No result rows written yet."
    df = pd.read_csv(results_csv)
    if df.empty:
        return "No result rows written yet."
    table = (
        df.pivot_table(
            index="method",
            columns="status",
            values="dataset",
            aggfunc="nunique",
            fill_value=0,
        )
        .reset_index()
        .sort_values("method")
    )
    total = df.groupby("method")["dataset"].nunique().rename("total_datasets").reset_index()
    table = total.merge(table, on="method", how="left")
    return table.to_string(index=False)


def existing_pairs(results_csv: Path):
    if not results_csv.exists():
        return set()
    df = pd.read_csv(results_csv)
    if df.empty:
        return set()
    return set(zip(df["dataset"].astype(str), df["method"].astype(str)))


def existing_pairs_excluding_statuses(results_csv: Path, rerun_statuses):
    if not results_csv.exists():
        return set()
    df = pd.read_csv(results_csv)
    if df.empty:
        return set()
    if rerun_statuses:
        df = df[~df["status"].astype(str).isin(set(rerun_statuses))]
    return set(zip(df["dataset"].astype(str), df["method"].astype(str)))


def remove_rerun_rows(results_csv: Path, datasets, methods, statuses):
    if not results_csv.exists() or not statuses:
        return
    df = pd.read_csv(results_csv)
    if df.empty:
        return
    datasets = set(map(str, datasets))
    methods = set(map(str, methods))
    statuses = set(map(str, statuses))
    drop_mask = (
        df["dataset"].astype(str).isin(datasets)
        & df["method"].astype(str).isin(methods)
        & df["status"].astype(str).isin(statuses)
    )
    if drop_mask.any():
        df.loc[~drop_mask].to_csv(results_csv, index=False)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run KFS-TUNE against ROCKET-family feature-selection/pruning baselines."
    )
    parser.add_argument("--datasets", nargs="+", default=["Car"])
    parser.add_argument("--all-ucr-bakeoff", action="store_true")
    parser.add_argument("--paper-30", action="store_true")
    parser.add_argument("--methods", nargs="+", default=["kfstune", "rocket", "minirocket"])
    parser.add_argument("--ucr-path", default=None, help="Optional local UCRArchive_2018 root.")
    parser.add_argument("--output-dir", default=str(ROOT / "results" / "rocket_family_baselines"))
    parser.add_argument("--num-kernels", type=int, default=10000)
    parser.add_argument("--num-features", type=int, default=500)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--rerun-statuses",
        nargs="+",
        default=[],
        help="With --skip-existing, rerun rows whose status is in this list, e.g. error timeout.",
    )

    parser.add_argument("--srocket-epochs", type=int, default=50)
    parser.add_argument("--srocket-population", type=int, default=16)
    parser.add_argument("--srocket-mutation", type=float, default=0.9)
    parser.add_argument("--srocket-crossover", type=float, default=0.9)

    parser.add_argument("--pocket-epochs", type=int, default=50)
    parser.add_argument("--pocket-k", type=float, default=1.0)
    parser.add_argument("--pocket-remain-kernels", type=int, default=5000)
    parser.add_argument("--pocket-stop-thr", type=float, default=0.001)

    parser.add_argument("--detach-tradeoff", type=float, default=0.1)
    parser.add_argument("--method-timeout-s", type=int, default=1800)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker-dataset", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker-method", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker-output-json", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    unknown = sorted(set(args.methods) - set(METHODS))
    if unknown:
        raise ValueError(f"Unknown methods: {unknown}. Valid methods: {METHODS}")
    if args.all_ucr_bakeoff:
        args.datasets = UCR_BAKEOFF_DATASETS
    if args.paper_30:
        args.datasets = PAPER_30_DATASETS
    return args


def worker_command(args, dataset, method, output_json):
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--worker-dataset",
        dataset,
        "--worker-method",
        method,
        "--worker-output-json",
        str(output_json),
        "--num-kernels",
        str(args.num_kernels),
        "--num-features",
        str(args.num_features),
        "--srocket-epochs",
        str(args.srocket_epochs),
        "--srocket-population",
        str(args.srocket_population),
        "--srocket-mutation",
        str(args.srocket_mutation),
        "--srocket-crossover",
        str(args.srocket_crossover),
        "--pocket-epochs",
        str(args.pocket_epochs),
        "--pocket-k",
        str(args.pocket_k),
        "--pocket-remain-kernels",
        str(args.pocket_remain_kernels),
        "--pocket-stop-thr",
        str(args.pocket_stop_thr),
        "--detach-tradeoff",
        str(args.detach_tradeoff),
    ]
    if args.ucr_path:
        cmd.extend(["--ucr-path", args.ucr_path])
    return cmd


def run_method_in_worker(args, dataset, method, output_dir):
    worker_dir = output_dir / "_worker"
    worker_dir.mkdir(parents=True, exist_ok=True)
    output_json = worker_dir / f"{dataset}__{method}.json"
    if output_json.exists():
        output_json.unlink()

    cmd = worker_command(args, dataset, method, output_json)
    try:
        completed = subprocess.run(
            cmd,
            cwd=str(ROOT),
            timeout=args.method_timeout_s,
            text=True,
            capture_output=True,
        )
    except subprocess.TimeoutExpired:
        return Result(
            dataset=dataset,
            method=method,
            status="timeout",
            error=f"Timed out after {args.method_timeout_s} seconds",
        )

    if output_json.exists():
        with output_json.open("r", encoding="utf-8") as handle:
            return Result(**json.load(handle))

    return Result(
        dataset=dataset,
        method=method,
        status="error",
        error=(
            f"Worker exited with code {completed.returncode} without writing output.\n"
            f"STDOUT:\n{completed.stdout[-4000:]}\nSTDERR:\n{completed.stderr[-4000:]}"
        ),
    )


def run_worker(args):
    if not args.worker_dataset or not args.worker_method or not args.worker_output_json:
        raise ValueError("Worker mode requires dataset, method, and output JSON path.")

    x_train, y_train, x_test, y_test = load_ucr_dataset(args.worker_dataset, args.ucr_path)
    row = run_method(args.worker_method, args.worker_dataset, x_train, y_train, x_test, y_test, args)
    output_json = Path(args.worker_output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as handle:
        json.dump(asdict(row), handle, indent=2)


def main():
    args = parse_args()
    if args.worker:
        run_worker(args)
        return

    output_dir = Path(args.output_dir)
    results_csv = output_dir / "per_dataset_results.csv"
    summary_csv = output_dir / "summary_results.csv"
    config_json = output_dir / "run_config.json"

    output_dir.mkdir(parents=True, exist_ok=True)
    if args.overwrite and results_csv.exists():
        results_csv.unlink()

    with config_json.open("w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=2)

    if args.skip_existing and args.rerun_statuses:
        remove_rerun_rows(results_csv, args.datasets, args.methods, args.rerun_statuses)

    for dataset in args.datasets:
        print(f"\nLoading {dataset}")
        completed = (
            existing_pairs_excluding_statuses(results_csv, args.rerun_statuses)
            if args.skip_existing
            else set()
        )
        for method in args.methods:
            if (dataset, method) in completed:
                print(f"  skipping {method} (already has a row for {dataset})", flush=True)
                continue
            print(f"  running {method}", flush=True)
            row = run_method_in_worker(args, dataset, method, output_dir)
            append_rows(results_csv, [row])
            if row.status == "ok":
                print(f"    accuracy={row.accuracy:.4f}, total={row.total_time_s:.2f}s", flush=True)
            else:
                print(f"    ERROR: {row.error.splitlines()[0] if row.error else 'unknown'}", flush=True)
            print(progress_counts(results_csv), flush=True)
        write_summary(results_csv, summary_csv)

    print(f"\nWrote per-dataset results: {results_csv}")
    print(f"Wrote summary results: {summary_csv}")


if __name__ == "__main__":
    main()
