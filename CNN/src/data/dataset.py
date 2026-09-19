import os
from pathlib import Path

import pandas as pd

from src.configs import VALID_EXTENSIONS

def collect_image_paths(dataset_dir):
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