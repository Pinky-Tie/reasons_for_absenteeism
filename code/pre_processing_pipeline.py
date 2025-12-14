import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from pylab import rcParams
from sklearn.impute import KNNImputer
from sklearn.impute import SimpleImputer

from utils1 import secret_missing_values

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)




def pp_pipeline(
    data,
    standardize_binary: bool = True,
    change_datatypes: bool = True,
    replace_special_missing: bool = True,
    impute_missing: bool = True
):
    df = data.copy()

    # ------------------------------------------------------------------
    # Helper functions
    # ------------------------------------------------------------------

    def binary_standardize(column):
        """Standardize binary columns to 0 and 1."""
        column = column.replace({'Yes': 1, 'No': 0, 'Y': 1, 'N': 0})
        return column.astype(int)

    def impute_missing_values(dataframe):
        """
        Handle missing values:
        - Education: imputed with highest observed value
        - Other numeric: KNNImputer
        - Categorical: most frequent
        """

        df = dataframe.copy()

        # -------------------------
        # Special rule: Education
        # -------------------------
        if "Education" in df.columns:
            max_education = df["Education"].max(skipna=True)
            df["Education"] = df["Education"].fillna(max_education)

        # Feature detection (excluding Education)
        numeric_features = (
            df.select_dtypes(include=[np.number])
              .columns.drop("Education", errors="ignore")
              .tolist()
        )

        categorical_features = df.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()

        # -------------------------
        # Numeric imputation (KNN)
        # -------------------------
        if numeric_features:
            num_imputer = KNNImputer(n_neighbors=5)
            numeric_array = num_imputer.fit_transform(df[numeric_features])
            num_df = pd.DataFrame(
                numeric_array,
                columns=numeric_features,
                index=df.index
            )

            # Restore integer types where possible
            for col in numeric_features:
                if pd.api.types.is_integer_dtype(df[col]):
                    num_df[col] = pd.to_numeric(num_df[col], downcast="integer")

            df[numeric_features] = num_df

        # -------------------------
        # Categorical imputation
        # -------------------------
        if categorical_features:
            cat_imputer = SimpleImputer(strategy="most_frequent")
            df[categorical_features] = cat_imputer.fit_transform(
                df[categorical_features]
            )

        return df

    # ------------------------------------------------------------------
    # Actions (all parameter-driven)
    # ------------------------------------------------------------------

    # 1-  Standardize binary columns
    if standardize_binary:
        binary_values = {'Y', 'N', 'Yes', 'No'}
        transforms_cols = []
        for col in df.columns:
            unique_vals = set(df[col].dropna().unique())
            if unique_vals.issubset(binary_values):
                df[col] = binary_standardize(df[col])
                transforms_cols.append(col)
        print(f"Step 1: Standardized binary columns: {transforms_cols}")
        

    # 2️-  Change datatypes
    if change_datatypes:
        transforms_cols_date = []
        transforms_cols_num = []
        # 2.1 Try numeric conversion
        for col in df.select_dtypes(include=["object"]).columns:
            try:
                df[col] = pd.to_numeric(df[col], downcast="integer")
                transforms_cols_num.append(col)
            except ValueError:
                pass

        # 2.2 Try datetime conversion
        for col in df.select_dtypes(include=["object"]).columns:
            try:
                df[col] = pd.to_datetime(df[col])
                transforms_cols_date.append(col)
            except (ValueError, TypeError):
                pass
        print(f"Step 2.1: Changed to following columns to integer type : {transforms_cols_num}")
        print(f"Step 2.2: Changed to following columns to datetime type : {transforms_cols_date}")

    # 3️- Handle missing values
    # 3.1 - Standardize special missing indicators
    if replace_special_missing:
        # Replace -1 with NaN
        #count number of -1 before replacement
        num_neg_ones = (df == -1).sum().sum()
        df = df.replace(-1, np.nan)
        # Replace dashes in specific columns with NaN
        cols_with_dash, rows_with_dash, mask_dash = secret_missing_values(df)
        existing_cols = [c for c in cols_with_dash if c in df.columns]
        #count number of dashes before replacement
        num_dashes = mask_dash.sum().sum()
        df[existing_cols] = df[existing_cols].apply(
            lambda col: col.astype(str).str.strip().replace('-', np.nan)
        )
        print(f"Step 3.1: Replaced {num_neg_ones} occurrences of -1 with NaN.")
        print(f"Step 3.1: Replaced {num_dashes} occurrences of '-' with NaN.")

    # 3.2- Handle missing values
    if impute_missing:
        #count total missing values before imputation
        total_missing_before = df.isnull().sum().sum()
        df = impute_missing_values(df)
        print(f"Step 3.2: Imputed missing values. Total missing values before: {total_missing_before}, after: {df.isnull().sum().sum()}.")

    return df




def group_by_worker(
    dataframe,
    demographic_cols,
    absence_time_col,
    dummies_prefix="Reason_"
):
    """
    Creates a dataframe grouped by worker ID, aggregating:
    - Demographic columns using mode
    - One-hot encoded reason columns using sum
    - Absence time using count, sum, and mean

    Args:
        dataframe (pd.DataFrame): Input dataframe with worker ID as index.
        demographic_cols (list): List of demographic column names.
        absence_time_col (str): Absence time column name.
        dummies_prefix (str): Prefix for one-hot encoded absence reasons.

    Returns:
        pd.DataFrame: Grouped dataframe by worker ID.
    """

    reason_columns = [c for c in dataframe.columns if c.startswith(dummies_prefix)]

    named_aggs = {}

    # Demographics → mode
    for col in demographic_cols:
        named_aggs[col] = (col, lambda x: x.mode().iloc[0])

    # One-hot encoded reasons → sum
    for col in reason_columns:
        named_aggs[col] = (col, "sum")

    # Absence metrics → count, sum, mean
    named_aggs[f"{absence_time_col}_count"] = (absence_time_col, "count")
    named_aggs[f"{absence_time_col}_sum"] = (absence_time_col, "sum")
    named_aggs[f"{absence_time_col}_mean"] = (absence_time_col, "mean")

    # Group by worker ID (index)
    grouped = dataframe.groupby(level=0).agg(**named_aggs)

    return grouped

    

