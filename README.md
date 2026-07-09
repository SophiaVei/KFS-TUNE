# KFS-TUNE

KFS-TUNE is a time-series classification approach that combines random
convolutional kernels with feature selection in order to balance predictive
performance and computational efficiency.

This repository contains the code related to the KFS-TUNE paper, including the
core method, UCR/UEA benchmark scripts, LifeSnaps/Fitbit analysis scripts, a
synthetic data generator, and an example comparison script.

## Paper overview

### KFS-TUNE pipeline

![KFS-TUNE pipeline](https://github.com/user-attachments/assets/dcd74a1f-e6ee-4be0-ad39-fed48ecde109)

### Evaluation overview

![Evaluation overview](https://github.com/user-attachments/assets/b610bc5b-a30c-4e1d-b51e-06350ff725ef)

## Repository structure

- `KFSTUNE_functions.py`
  Core KFS-TUNE utilities for kernel generation, kernel feature extraction, and
  feature selection.

- `UCR/`
  Scripts for running KFS-TUNE on groups of UCR/UEA datasets.

- `LifeSnap/`
  Scripts for the LifeSnaps/Fitbit analyses:
  - `KFSTUNE_fitbit_allfeatures.py`
  - `KSFTUNE_fitbit_numericalfeatures.py`

- `synthetic/`
  Synthetic time-series data generation code.

- `sota_algorithms.py`
  Example comparison script on a single UCR dataset.

## Datasets

### UCR / UEA

The UCR scripts use `load_UCR_UEA_dataset(...)` from `sktime` and rely on the
standard train/test splits provided there.

### LifeSnaps / Fitbit

The LifeSnap scripts expect a local pickle file named:

`daily_fitbit_surveys_semas.pkl`

Place it at:

- `KFS-TUNE/daily_fitbit_surveys_semas.pkl`

If you are not running the LifeSnap scripts, you do not need this file.

### Synthetic

Synthetic datasets can be generated with:

```bash
python synthetic/generate_synthetic_data.py
```

The generator writes synthetic CSV datasets to the repository root when run.

## Running the scripts

Examples:

```bash
python UCR/KFSTUNE_implementation_all_univariate_UCR.py
python UCR/KFSTUNE_implementation_all_multivariate_UCR.py
python UCR/KFSTUNE_implementation_additional_UCR.py
python LifeSnap/KFSTUNE_fitbit_allfeatures.py
python LifeSnap/KSFTUNE_fitbit_numericalfeatures.py
python synthetic/generate_synthetic_data.py
python sota_algorithms.py
```

## Dependencies

Install the packages listed in `requirements.txt`.
