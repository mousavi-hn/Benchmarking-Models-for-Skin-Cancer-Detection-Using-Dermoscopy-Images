"""
Create reproducible training, validation, and test splits for an image dataset.

The collected images are divided using stratified sampling so that the binary
class distribution is preserved across the training, validation, and test
partitions.
"""
import os
import pandas as pd
from sklearn.model_selection import train_test_split

from dataset import collect_image_paths

from src.configs import (
    SEED,
    DATASET_DIR,
    SPLIT_DIR,
)

def make_splits():
    """
    Create stratified training, validation, and test partitions.

    The full dataset is first split into 70% training data and 30% temporary
    data. The temporary partition is then divided equally between validation
    and test data, yielding an overall 70/15/15 split while preserving the
    binary class distribution.

    Returns:
        tuple[pandas.DataFrame, pandas.DataFrame, pandas.DataFrame]:
            Training, validation, and test DataFrames, respectively.
    """
    full_df = collect_image_paths(DATASET_DIR)
    print("Total images:", len(full_df))
    print(full_df["class_name"].value_counts())

    df = full_df.copy()

    # Split non-IXI normally by label ( labels are no and yes, no says there is no cancer and yes the opposite )
    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        stratify=df["label"],
        random_state=SEED
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        stratify=temp_df["label"],
        random_state=SEED
    )

    return train_df, val_df, test_df