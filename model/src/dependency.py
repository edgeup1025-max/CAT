import pandas as pd
import numpy as np
import psutil
import os
import gc


def folder_lookup(keyword: str):
    folder_path = "D:/WORKSPACE/OFIICE_WORKS/model/src"
    for root, dirs, files in os.walk(folder_path):
        for f in files:
            if keyword in f.lower():
                if keyword in f.lower():
                    return os.path.join(root, f)


def check_null_values(func):

    def wrapper(self, *args):
        if len(args) > 0:
            data = args[0]
        else:
            raise ValueError("give a data and type should be in DataFrame")

        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")

        self.has_nulls = data.isnull().values.any()
        self.data = data
        return func(self, *args)

    return wrapper


def remove_the_duplicates(func):

    def wrapper(self, *args):
        if self.has_nulls:
            self.data = self.data.drop_duplicates()
        return func(self, *args)

    return wrapper


def check_order(beta_df):
    disordered_items = []
    for i, row in beta_df.iterrows():
        if not all(np.diff(row.values) > 0):
            disordered_items.append(i)
    print(f"Number of disordered items: {len(disordered_items)}")
    print(f"Disordered item indices: {disordered_items}")


def check_cardanlity(func):

    def wraps(beta_df):
        cardinalities = {}
        for column in beta_df.columns:
            cardinalities[column] = beta_df[column].nunique()
        print("Cardinalities of each column:")
        for column, cardinality in cardinalities.items():
            print(f"{column}: {cardinality}")
        return func(beta_df)

    return wraps


def remove_the_cols(cols):

    def decorator(func):

        def wrapper(beta_df):
            beta_df = beta_df.drop(columns=cols, errors='ignore')
            return func(beta_df)

        return wrapper

    return decorator


def check_disordered(beta_df):
    disordered_items = []
    for i, row in beta_df.iterrows():
        if not all(np.diff(row.values) > 0):
            disordered_items.append(i)
    print(f"Number of disordered items: {len(disordered_items)}")
    print(f"Disordered item indices: {disordered_items}")


def memory_cleanup(label=""):
    process = psutil.Process(os.getpid())
    before = process.memory_info().rss / 1024**2
    gc.collect()
    after = process.memory_info().rss / 1024**2
    print(f"[{label}] Memory: {before:.2f} → {after:.2f} MB")


def categorical_detection_report(df: pd.DataFrame, verbose=True):
    summary = []
    dichotomous_items, polytomous_items = [], []

    for col in df.columns:
        values = df[col].dropna().unique()
        unique_count = len(values)

        # Determine type based on unique values
        if unique_count <= 2:
            dichotomous_items.append(col)
            item_type = "dichotomous"
        elif unique_count > 2:
            polytomous_items.append(col)
            item_type = "polytomous"
        else:
            item_type = "unknown"

        summary.append({
            "Item": col,
            "Unique_Count": unique_count,
            "Min": np.min(values) if len(values) > 0 else np.nan,
            "Max": np.max(values) if len(values) > 0 else np.nan,
            "Type": item_type
        })

    summary_df = pd.DataFrame(summary)
    n_dicho = len(dichotomous_items)
    n_poly = len(polytomous_items)

    if n_dicho == len(df.columns):
        dataset_type = "dichotomous"
    elif n_poly == len(df.columns):
        dataset_type = "polytomous"
    else:
        dataset_type = "mixed"

    report = {
        "type": dataset_type,
        "dichotomous_items": dichotomous_items,
        "polytomous_items": polytomous_items,
        "summary_df": summary_df
    }

    if verbose:
        print("=== Categorical Detection Report (CDR) ===")
        print(f"Total items: {len(df.columns)}")
        print(f"Dichotomous items: {n_dicho}")
        print(f"Polytomous items: {n_poly}")
        print(f"Dataset type: {dataset_type.upper()}")
        print("==========================================\n")
        print(summary_df)

    return report
