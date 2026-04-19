# Data preprocessing
"""
Wind Turbine Anomaly Detection — Preprocessing
================================================
Data loading, cleaning, feature engineering, splitting, and scaling.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

from config import (
    DATA_FILES,
    CSV_SEPARATOR,
    TIMESTAMP_COLUMN,
    STATUS_COLUMN,
    STATUS_UNKNOWN,
    DROP_COLUMNS,
    LABEL_MAP,
    TRAIN_SPLIT_RATIO,
)


def load_data(file_paths):
    """
    Load and concatenate multiple CSV files (semicolon-delimited)
    with timestamps parsed as datetime objects.
    """
    dataframes = []
    for path in file_paths:
        df = pd.read_csv(path, sep=CSV_SEPARATOR, parse_dates=[TIMESTAMP_COLUMN])
        print(f"Loaded {path}: {df.shape[0]} rows, {df.shape[1]} columns")
        dataframes.append(df)
    return dataframes


def remove_zero_columns(dataframes):
    """
    Remove columns that are entirely zero across any dataset.
    These carry no discriminative information and would only add noise.
    """
    all_zero_cols = set()
    for df in dataframes:
        numeric_cols = df.select_dtypes(include="number").columns
        zero_cols = set(numeric_cols[(df[numeric_cols] == 0).all()])
        all_zero_cols |= zero_cols

    cleaned = []
    for df in dataframes:
        df = df.drop(columns=all_zero_cols, errors="ignore")
        cleaned.append(df)

    if all_zero_cols:
        print(f"Dropped {len(all_zero_cols)} all-zero column(s): {sorted(all_zero_cols)}")
    else:
        print("No all-zero columns found.")

    return cleaned


def engineer_features(df):
    """
    Extract numerical time features from the timestamp column.
    Models cannot perform math on a raw timestamp, so we break
    it down into cyclical components to capture seasonality and daily routines.
    """
    df = df.copy()
    df["hour"] = df[TIMESTAMP_COLUMN].dt.hour
    df["month"] = df[TIMESTAMP_COLUMN].dt.month
    df["day_of_week"] = df[TIMESTAMP_COLUMN].dt.dayofweek
    return df


def combine_datasets(dataframes):
    """
    Concatenate datasets vertically with reset index to avoid
    duplicate indices.
    """
    df_combined = pd.concat(dataframes, ignore_index=True)
    print(f"Combined dataset: {df_combined.shape[0]} rows, {df_combined.shape[1]} columns")
    print(f"\nStatus distribution:\n{df_combined[STATUS_COLUMN].value_counts(dropna=False)}")
    return df_combined


def split_labeled_unknown(df_combined):
    """
    Split the combined dataset into labeled (Normal + Anomalous)
    and unknown subsets.
    """
    df_labeled = df_combined[df_combined[STATUS_COLUMN] != STATUS_UNKNOWN].copy()
    df_unknown = df_combined[df_combined[STATUS_COLUMN] == STATUS_UNKNOWN].copy()

    n_normal = (df_labeled[STATUS_COLUMN] == "Normal").sum()
    n_anom = (df_labeled[STATUS_COLUMN] == "Anomalous").sum()
    print(f"Labeled samples: {len(df_labeled)} (Normal: {n_normal}, Anomalous: {n_anom})")
    print(f"Unknown samples to predict: {len(df_unknown)}")

    return df_labeled, df_unknown


def define_features(df_labeled):
    """
    Define the feature space by excluding metadata and target columns.
    """
    feature_cols = [c for c in df_labeled.columns if c not in DROP_COLUMNS]
    print(f"Number of predictive features: {len(feature_cols)}")
    return feature_cols


def chronological_split(df_labeled, feature_cols, split_ratio=TRAIN_SPLIT_RATIO):
    """
    Perform a chronological train/validation split.
    Sort by timestamp first, then use the first `split_ratio` fraction
    for training and the rest for validation.
    This prevents data leakage in time-series data.
    """
    df_labeled_sorted = df_labeled.sort_values(TIMESTAMP_COLUMN).reset_index(drop=True)
    y_sorted = df_labeled_sorted[STATUS_COLUMN].map(LABEL_MAP)
    X_sorted = df_labeled_sorted[feature_cols]

    split_idx = int(len(X_sorted) * split_ratio)

    X_train = X_sorted.iloc[:split_idx]
    X_val = X_sorted.iloc[split_idx:]
    y_train = y_sorted.iloc[:split_idx]
    y_val = y_sorted.iloc[split_idx:]

    print(f"Training set:   {len(X_train)} samples "
          f"(Anomalous: {y_train.sum()} = {y_train.mean()*100:.1f}%)")
    print(f"Validation set: {len(X_val)} samples "
          f"(Anomalous: {y_val.sum()} = {y_val.mean()*100:.1f}%)")
    print(f"\nTrain period: {df_labeled_sorted[TIMESTAMP_COLUMN].iloc[0]} "
          f"to {df_labeled_sorted[TIMESTAMP_COLUMN].iloc[split_idx-1]}")
    print(f"Val period:   {df_labeled_sorted[TIMESTAMP_COLUMN].iloc[split_idx]} "
          f"to {df_labeled_sorted[TIMESTAMP_COLUMN].iloc[-1]}")

    return X_train, X_val, y_train, y_val, df_labeled_sorted


def scale_features(X_train, X_val=None, X_unknown=None):
    """
    Fit a StandardScaler on the training data and transform all datasets.
    This ensures all features contribute equally to the optimization.
    CRITICAL: We fit ONLY on training data to prevent information leakage.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    print(f"Scaling complete. {X_train_scaled.shape[1]} features scaled successfully.")

    result = {"scaler": scaler, "X_train_scaled": X_train_scaled}

    if X_val is not None:
        result["X_val_scaled"] = scaler.transform(X_val)

    if X_unknown is not None:
        result["X_unknown_scaled"] = scaler.transform(X_unknown)

    return result


def run_preprocessing_pipeline():
    """
    Orchestrate the full preprocessing pipeline:
    load → clean → engineer features → combine → split → define features
    Returns all data artifacts needed by the model pipeline.
    """
    import warnings
    warnings.filterwarnings("ignore")

    print("=" * 60)
    print("LOADING DATA")
    print("=" * 60)
    dataframes = load_data(DATA_FILES)

    print("\n" + "=" * 60)
    print("REMOVING DEAD SENSORS")
    print("=" * 60)
    dataframes = remove_zero_columns(dataframes)

    print("\n" + "=" * 60)
    print("FEATURE ENGINEERING")
    print("=" * 60)
    dataframes = [engineer_features(df) for df in dataframes]

    print("\n" + "=" * 60)
    print("COMBINING DATASETS")
    print("=" * 60)
    df_combined = combine_datasets(dataframes)

    print("\n" + "=" * 60)
    print("SEPARATING LABELED AND UNKNOWN DATA")
    print("=" * 60)
    df_labeled, df_unknown = split_labeled_unknown(df_combined)
    feature_cols = define_features(df_labeled)

    print("\n" + "=" * 60)
    print("CHRONOLOGICAL TRAIN/VALIDATION SPLIT")
    print("=" * 60)
    X_train, X_val, y_train, y_val, df_labeled_sorted = chronological_split(
        df_labeled, feature_cols
    )

    X_unknown = df_unknown[feature_cols]

    return {
        "X_train": X_train,
        "X_val": X_val,
        "y_train": y_train,
        "y_val": y_val,
        "X_unknown": X_unknown,
        "df_labeled_sorted": df_labeled_sorted,
        "df_unknown": df_unknown,
        "feature_cols": feature_cols,
    }
