"""Data preprocessing module for loan approval analysis."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class PreprocessingArtifacts:
    """Container for the outputs produced during preprocessing."""

    raw_data: pd.DataFrame
    cleaned_data: pd.DataFrame
    numeric_columns: List[str]
    categorical_columns: List[str]
    column_transformer: ColumnTransformer
    outlier_summary: pd.DataFrame


def load_data(path: str) -> pd.DataFrame:
    """Load the loan approval dataset from a CSV file."""
    data = pd.read_csv(path)
    return data


def initial_eda(data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Perform simple EDA including head, description, and missing summary."""
    eda_results = {
        "head": data.head(),
        "describe": data.describe(include="all"),
        "missing_summary": data.isna().sum().to_frame("missing_count"),
    }
    return eda_results


def _build_preprocessor(
    numeric_cols: List[str], categorical_cols: List[str]
) -> ColumnTransformer:
    """Create preprocessing pipelines for numeric and categorical features."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_cols),
            ("categorical", categorical_pipeline, categorical_cols),
        ]
    )
    return preprocessor


def detect_outliers_iqr(data: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
    """Detect potential outliers using the IQR rule for each numeric column."""
    summary_rows: List[Tuple[str, float, float, int]] = []
    for col in numeric_cols:
        series = data[col].dropna()
        if series.empty:
            continue
        q1, q3 = series.quantile([0.25, 0.75])
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = series[(series < lower_bound) | (series > upper_bound)]
        summary_rows.append((col, lower_bound, upper_bound, outliers.size))

    outlier_summary = pd.DataFrame(
        summary_rows,
        columns=["feature", "lower_bound", "upper_bound", "outlier_count"],
    )
    return outlier_summary


def preprocess_data(path: str) -> PreprocessingArtifacts:
    """Run the end-to-end preprocessing workflow."""
    raw_data = load_data(path)

    # Identify column types
    numeric_cols = raw_data.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = raw_data.select_dtypes(exclude=[np.number]).columns.tolist()

    preprocessor = _build_preprocessor(numeric_cols, categorical_cols)

    # Fit the preprocessor to the data and transform
    transformed_array = preprocessor.fit_transform(raw_data)
    transformed_feature_names = []

    # Extract feature names from column transformer for interpretability
    if numeric_cols:
        transformed_feature_names.extend(numeric_cols)
    if categorical_cols:
        encoder: OneHotEncoder = preprocessor.named_transformers_["categorical"]["onehot"]
        ohe_feature_names = encoder.get_feature_names_out(categorical_cols)
        transformed_feature_names.extend(ohe_feature_names.tolist())

    cleaned_df = pd.DataFrame(transformed_array, columns=transformed_feature_names)

    outlier_summary = detect_outliers_iqr(raw_data, numeric_cols)

    return PreprocessingArtifacts(
        raw_data=raw_data,
        cleaned_data=cleaned_df,
        numeric_columns=numeric_cols,
        categorical_columns=categorical_cols,
        column_transformer=preprocessor,
        outlier_summary=outlier_summary,
    )


def main() -> None:
    """CLI entry-point for running preprocessing in isolation."""
    artifacts = preprocess_data("loan_approval.csv")
    print("Raw shape:", artifacts.raw_data.shape)
    print("Transformed shape:", artifacts.cleaned_data.shape)
    print("Outlier summary:\n", artifacts.outlier_summary)


if __name__ == "__main__":
    main()
