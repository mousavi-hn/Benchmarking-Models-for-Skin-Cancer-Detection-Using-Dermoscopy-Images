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