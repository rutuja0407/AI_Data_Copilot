import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype


def apply_step(df: pd.DataFrame, step: dict) -> pd.DataFrame:
    """
    Apply a single cleaning step to the dataframe.
    step format example:
    {
      "description": "Fill missing values in age with median",
      "column": "age",
      "action": "fill_missing",
      "parameters": {"method": "median"}
    }
    """
    action = step.get("action")
    col = step.get("column")
    params = step.get("parameters", {})

    # If action doesn't need a specific column
    if action == "drop_duplicates":
        df = df.drop_duplicates()
        return df

    if col is not None and col not in df.columns:
        # Ignore invalid columns
        return df

    if action == "fill_missing":
        method = params.get("method", "mean")
        if col is None:
            return df

        if method == "mean" and is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].mean())
        elif method == "median" and is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        elif method == "mode":
            if not df[col].mode().empty:
                df[col] = df[col].fillna(df[col].mode().iloc[0])
        elif method == "constant":
            value = params.get("value", 0)
            df[col] = df[col].fillna(value)

    elif action == "drop_rows_with_missing":
        subset = params.get("subset", [col] if col else None)
        df = df.dropna(subset=subset)

    elif action == "clip_outliers":
        # Simple IQR-based clipping for numeric columns
        if col is not None and is_numeric_dtype(df[col]):
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            df[col] = df[col].clip(lower, upper)

    return df


def apply_steps(df: pd.DataFrame, steps: dict) -> pd.DataFrame:
    """
    Apply multiple steps. steps format:
    {"steps": [step1, step2, ...]}
    """
    for step in steps.get("steps", []):
        df = apply_step(df, step)
    return df
