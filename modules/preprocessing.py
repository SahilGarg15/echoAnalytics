# import pandas as pd
# import numpy as np

# # Handle missing values
# def handle_missing(df, strategy="drop", fill_value=None):
#     if strategy == "drop":
#         return df.dropna()
#     elif strategy == "mean":
#         return df.fillna(df.mean(numeric_only=True))
#     elif strategy == "median":
#         return df.fillna(df.median(numeric_only=True))
#     elif strategy == "mode":
#         return df.fillna(df.mode().iloc[0])
#     elif strategy == "constant" and fill_value is not None:
#         return df.fillna(fill_value)
#     return df


# # Remove outliers
# def remove_outliers(df, method="zscore", threshold=3):
#     if method == "zscore":
#         from scipy.stats import zscore
#         return df[(np.abs(zscore(df.select_dtypes(include=[np.number]))) < threshold).all(axis=1)]
#     elif method == "iqr":
#         Q1 = df.quantile(0.25)
#         Q3 = df.quantile(0.75)
#         IQR = Q3 - Q1
#         return df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]
#     return df


# # Convert dtypes
# def convert_dtypes(df, conversions):
#     for col, dtype in conversions.items():
#         try:
#             df[col] = df[col].astype(dtype)
#         except Exception as e:
#             print(f"Could not convert {col} to {dtype}: {e}")
#     return df


# # Rename columns
# def rename_columns(df, renames):
#     return df.rename(columns=renames)


# # Normalize units (simple multiplication factor)
# def normalize_units(df, column, factor=1.0):
#     if column in df.columns and pd.api.types.is_numeric_dtype(df[column]):
#         df[column] = df[column] * factor
#     return df


# # Save cleaned dataset
# def save_cleaned(df, path="cleaned.csv", format="csv"):
#     if format == "csv":
#         df.to_csv(path, index=False)
#     elif format == "excel":
#         df.to_excel(path, index=False)
#     elif format == "json":
#         df.to_json(path, orient="records")
#     return path

# """Preprocessing functions for data cleaning (Phase 7)."""
# import pandas as pd
# from sklearn.preprocessing import MinMaxScaler

# # Missing values
# def handle_missing_values(df: pd.DataFrame, method="Drop Rows"):
#     if method == "Drop Rows":
#         return df.dropna()
#     elif method == "Fill with Mean":
#         return df.fillna(df.mean(numeric_only=True))
#     elif method == "Fill with Median":
#         return df.fillna(df.median(numeric_only=True))
#     elif method == "Fill with Mode":
#         return df.fillna(df.mode().iloc[0])
#     return df

# # Outliers
# def remove_outliers(df: pd.DataFrame, column: str):
#     if column not in df.columns: return df
#     q1, q3 = df[column].quantile([0.25, 0.75])
#     iqr = q3 - q1
#     lower, upper = q1 - 1.5*iqr, q3 + 1.5*iqr
#     return df[(df[column] >= lower) & (df[column] <= upper)]

# # Data type conversion
# def convert_dtype(df: pd.DataFrame, column: str, new_type: str):
#     try:
#         if new_type == "int":
#             df[column] = df[column].astype(int)
#         elif new_type == "float":
#             df[column] = df[column].astype(float)
#         elif new_type == "str":
#             df[column] = df[column].astype(str)
#     except Exception as e:
#         print(f"Type conversion error: {e}")
#     return df

# # Rename columns
# def rename_column(df: pd.DataFrame, old_name: str, new_name: str):
#     return df.rename(columns={old_name: new_name})

# # Normalization
# def normalize_column(df: pd.DataFrame, column: str):
#     try:
#         scaler = MinMaxScaler()
#         df[column] = scaler.fit_transform(df[[column]])
#     except Exception as e:
#         print(f"Normalization error: {e}")
#     return df

# """Preprocessing & cleaning functions."""
# import pandas as pd
# from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

# # -----------------------------
# # Missing values
# # -----------------------------
# def handle_missing_values(df: pd.DataFrame, method="Drop Rows"):
#     if method == "Drop Rows":
#         return df.dropna()
#     elif method == "Fill with Mean":
#         return df.fillna(df.mean(numeric_only=True))
#     elif method == "Fill with Median":
#         return df.fillna(df.median(numeric_only=True))
#     elif method == "Fill with Mode":
#         return df.fillna(df.mode().iloc[0])
#     return df

# # -----------------------------
# # Outliers
# # -----------------------------
# def remove_outliers(df: pd.DataFrame, column: str):
#     if column not in df.columns: return df
#     q1, q3 = df[column].quantile([0.25, 0.75])
#     iqr = q3 - q1
#     lower, upper = q1 - 1.5*iqr, q3 + 1.5*iqr
#     return df[(df[column] >= lower) & (df[column] <= upper)]

# # -----------------------------
# # Data type conversion
# # -----------------------------
# def convert_dtype(df: pd.DataFrame, column: str, new_type: str):
#     try:
#         if new_type == "int":
#             df[column] = df[column].astype(int)
#         elif new_type == "float":
#             df[column] = df[column].astype(float)
#         elif new_type == "str":
#             df[column] = df[column].astype(str)
#     except Exception as e:
#         print(f"Type conversion error: {e}")
#     return df

# # -----------------------------
# # Rename columns
# # -----------------------------
# def rename_column(df: pd.DataFrame, old_name: str, new_name: str):
#     return df.rename(columns={old_name: new_name})

# # -----------------------------
# # Normalization / Scaling
# # -----------------------------
# def normalize_column(df: pd.DataFrame, column: str, method="MinMax"):
#     try:
#         scaler = None
#         if method=="MinMax": scaler = MinMaxScaler()
#         elif method=="Standard": scaler = StandardScaler()
#         elif method=="Robust": scaler = RobustScaler()
#         if scaler:
#             df[column] = scaler.fit_transform(df[[column]])
#     except Exception as e:
#         print(f"Scaling error: {e}")
#     return df

# """Preprocessing & cleaning functions."""
# import pandas as pd
# from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, LabelEncoder

# # -----------------------------
# # Missing values
# # -----------------------------
# def handle_missing_values(df: pd.DataFrame, method="Drop Rows"):
#     if method == "Drop Rows":
#         return df.dropna()
#     elif method == "Fill with Mean":
#         return df.fillna(df.mean(numeric_only=True))
#     elif method == "Fill with Median":
#         return df.fillna(df.median(numeric_only=True))
#     elif method == "Fill with Mode":
#         return df.fillna(df.mode().iloc[0])
#     return df

# # -----------------------------
# # Outliers
# # -----------------------------
# def remove_outliers(df: pd.DataFrame, column: str):
#     if column not in df.columns: return df
#     q1, q3 = df[column].quantile([0.25, 0.75])
#     iqr = q3 - q1
#     lower, upper = q1 - 1.5*iqr, q3 + 1.5*iqr
#     return df[(df[column] >= lower) & (df[column] <= upper)]

# # -----------------------------
# # Data type conversion
# # -----------------------------
# def convert_dtype(df: pd.DataFrame, column: str, new_type: str):
#     try:
#         if new_type == "int":
#             df[column] = df[column].astype(int)
#         elif new_type == "float":
#             df[column] = df[column].astype(float)
#         elif new_type == "str":
#             df[column] = df[column].astype(str)
#     except Exception as e:
#         print(f"Type conversion error: {e}")
#     return df

# # -----------------------------
# # Rename columns
# # -----------------------------
# def rename_column(df: pd.DataFrame, old_name: str, new_name: str):
#     return df.rename(columns={old_name: new_name})

# # -----------------------------
# # Normalization / Scaling
# # -----------------------------
# def normalize_column(df: pd.DataFrame, column: str, method="MinMax"):
#     try:
#         scaler = None
#         if method == "MinMax": scaler = MinMaxScaler()
#         elif method == "Standard": scaler = StandardScaler()
#         elif method == "Robust": scaler = RobustScaler()
#         if scaler:
#             df[column] = scaler.fit_transform(df[[column]])
#     except Exception as e:
#         print(f"Scaling error: {e}")
#     return df

# # -----------------------------
# # Encode categorical columns
# # -----------------------------
# def encode_categorical(df: pd.DataFrame, column: str, method="One-Hot"):
#     if column not in df.columns: return df
#     try:
#         if method == "One-Hot":
#             df = pd.get_dummies(df, columns=[column], drop_first=True)
#         elif method == "Label":
#             le = LabelEncoder()
#             df[column] = le.fit_transform(df[column].astype(str))
#     except Exception as e:
#         print(f"Encoding error: {e}")
#     return df


"""
Preprocessing & cleaning functions (batch-capable).
Save as: echoAnalytics/modules/preprocessing.py
"""

from __future__ import annotations
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, LabelEncoder
from typing import List, Dict, Any


# -----------------------------
# Primitive operations
# -----------------------------
def handle_missing_values(df: pd.DataFrame, method: str = "drop") -> pd.DataFrame:
    method = method.lower()
    if method == "drop":
        return df.dropna()
    if method == "mean":
        return df.fillna(df.mean(numeric_only=True))
    if method == "median":
        return df.fillna(df.median(numeric_only=True))
    if method == "mode":
        # fill with mode per column (non-numeric allowed)
        modes = df.mode().iloc[0]
        return df.fillna(modes)
    return df


def remove_outliers(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """IQR-based removal for a single column."""
    if column not in df.columns:
        return df
    try:
        q1, q3 = df[column].quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        return df[(df[column] >= lower) & (df[column] <= upper)]
    except Exception:
        return df


def convert_dtype(df: pd.DataFrame, column: str, new_type: str) -> pd.DataFrame:
    if column not in df.columns:
        return df
    try:
        if new_type == "int":
            df[column] = df[column].astype("Int64")
        elif new_type == "float":
            df[column] = df[column].astype(float)
        elif new_type == "str":
            df[column] = df[column].astype(str)
    except Exception:
        # ignore conversion errors and return df unchanged
        pass
    return df


def rename_column(df: pd.DataFrame, old_name: str, new_name: str) -> pd.DataFrame:
    if old_name not in df.columns:
        return df
    return df.rename(columns={old_name: new_name})


def encode_categorical(df: pd.DataFrame, column: str, method: str = "onehot") -> pd.DataFrame:
    if column not in df.columns:
        return df
    method = method.lower()
    try:
        if method == "onehot":
            return pd.get_dummies(df, columns=[column], drop_first=True)
        elif method == "label":
            le = LabelEncoder()
            df[column] = le.fit_transform(df[column].astype(str))
            return df
    except Exception:
        return df
    return df


def normalize_column(df: pd.DataFrame, column: str, method: str = "minmax") -> pd.DataFrame:
    if column not in df.columns:
        return df
    method = method.lower()
    mapper = {
        "minmax": MinMaxScaler(),
        "standard": StandardScaler(),
        "robust": RobustScaler()
    }
    scaler = mapper.get(method)
    if scaler is None:
        return df
    try:
        df[column] = scaler.fit_transform(df[[column]])
    except Exception:
        # fallback: do nothing
        pass
    return df


# -----------------------------
# Batch executor
# -----------------------------
def apply_steps(df: pd.DataFrame, steps: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Apply a list of preprocessing steps sequentially.

    Each step is a dictionary with an "action" key and parameters, e.g.:
        {"action": "handle_missing", "method": "median"}
        {"action": "remove_outliers", "columns": ["age","salary"]}
        {"action": "convert_dtype", "columns": ["age"], "dtype":"int"}
        {"action": "rename", "old":"oldname", "new":"newname"}
        {"action": "encode", "columns": ["cat1","cat2"], "method":"onehot"}
        {"action": "scale", "columns": ["age","salary"], "method":"minmax"}

    Returns transformed df.
    """
    for step in steps:
        action = (step.get("action") or "").lower()
        if action in {"handle_missing", "missing", "fill_missing"}:
            method = step.get("method", "mean").lower()
            df = handle_missing_values(df, method=method)
        elif action in {"remove_outliers", "outliers"}:
            cols = step.get("columns") or step.get("column") or []
            if isinstance(cols, str):
                cols = [cols]
            for col in cols:
                df = remove_outliers(df, col)
        elif action in {"convert_dtype", "convert"}:
            cols = step.get("columns") or ( [step.get("column")] if step.get("column") else [] )
            dtype = step.get("dtype") or step.get("new_type")
            for col in cols:
                df = convert_dtype(df, col, dtype)
        elif action in {"rename", "rename_column"}:
            old = step.get("old")
            new = step.get("new")
            if old and new:
                df = rename_column(df, old, new)
        elif action in {"encode", "encode_categorical"}:
            cols = step.get("columns") or ( [step.get("column")] if step.get("column") else [] )
            method = step.get("method", "onehot")
            for col in cols:
                df = encode_categorical(df, col, method)
        elif action in {"scale", "normalize", "normalize_column"}:
            cols = step.get("columns") or ( [step.get("column")] if step.get("column") else [] )
            method = step.get("method", "minmax")
            for col in cols:
                df = normalize_column(df, col, method)
        elif action in {"drop_columns", "drop"}:
            cols = step.get("columns") or step.get("cols") or []
            if isinstance(cols, str):
                cols = [cols]
            df = df.drop(columns=cols)
        # else: unknown action -> ignore silently
    return df
