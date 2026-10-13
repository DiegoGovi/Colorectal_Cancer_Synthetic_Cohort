"""
pre_train.py
80/20 train-test split of the preprocessed cohort (fixed seed).

Input : data/preprocesada/final_dataset.parquet
Output: data_train/train_df_seed{SEED}.parquet|.xlsx
        data_test/test_df_seed{SEED}.parquet|.xlsx
"""

import os
import random
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT / "data" / "preprocesada" / "final_dataset.parquet"
TRAIN_DIR = ROOT / "data_train"
TEST_DIR = ROOT / "data_test"

SEEDS = [42]
TEST_SIZE = 0.2


## -----------------------------------------------------------------------
def set_seed(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def split_df(df: pd.DataFrame, *, test_size: float = 0.2, seed: int = 42):
    if not (0.0 < test_size < 1.0):
        raise ValueError("test_size debe estar entre 0 y 1.")
    train = df.sample(frac=1 - test_size, random_state=seed)
    test = df.drop(train.index)
    return train, test


## -----------------------------------------------------------------------
if __name__ == "__main__":
    df = pd.read_parquet(INPUT_PATH, engine="pyarrow")

    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    TEST_DIR.mkdir(parents=True, exist_ok=True)

    for seed in SEEDS:
        set_seed(seed)
        train_df, test_df = split_df(df, test_size=TEST_SIZE, seed=seed)

        train_df.to_parquet(TRAIN_DIR / f"train_df_seed{seed}.parquet", index=False, engine="pyarrow")
        test_df.to_parquet(TEST_DIR / f"test_df_seed{seed}.parquet", index=False, engine="pyarrow")

        train_df.to_excel(TRAIN_DIR / f"train_df_seed{seed}.xlsx", index=False, engine="openpyxl")
        test_df.to_excel(TEST_DIR / f"test_df_seed{seed}.xlsx", index=False, engine="openpyxl")

        print(f"Guardados train/test con semilla {seed}: {len(train_df)} / {len(test_df)} filas")
