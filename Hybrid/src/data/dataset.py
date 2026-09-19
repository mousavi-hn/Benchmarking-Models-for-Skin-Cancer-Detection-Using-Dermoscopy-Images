"""
Collect image paths and binary class labels from a single-source image dataset.

Images are discovered recursively under the ``no`` and ``yes`` class
directories and represented as a pandas DataFrame for subsequent dataset
splitting and model training.
"""
import os
from pathlib import Path

import pandas as pd

from src.configs import VALID_EXTENSIONS

def collect_image_paths(dataset_dir):
    """
    Collect supported image files and assign binary class labels.

    The function recursively scans the ``no`` and ``yes`` class directories.
    Images in ``no`` receive label 0 and images in ``yes`` receive label 1.

    Args:
        dataset_dir: Root directory containing the class directories.

    Returns:
        pandas.DataFrame: One row per discovered image with filepath, label,
        class name, and subject identifier fields.

    Raises:
        FileNotFoundError: If an expected class directory is missing.
        ValueError: If no supported image files are found.
    """
    records = []

    class_map = {
        "no": 0,
        "yes": 1,
    }

    for class_name, label in class_map.items():
        class_dir = os.path.join(dataset_dir, class_name)
        if not os.path.exists(class_dir):
            raise FileNotFoundError(f"Folder not found: {class_dir}")

        for root, _, files in os.walk(class_dir):
            for file in files:
                ext = Path(file).suffix.lower()
                if ext in VALID_EXTENSIONS:
                    records.append({
                        "filepath": os.path.join(root, file),
                        "label": label,
                        "class_name": class_name,
                        "subject_id": None
                    })

    full_df = pd.DataFrame(records)
    if full_df.empty:
        raise ValueError("No images found. Check dataset path and file extensions.")

    return full_df