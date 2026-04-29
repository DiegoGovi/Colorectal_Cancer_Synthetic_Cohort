import pandas as pd
import os
import numpy as np
import random

df = pd.read_parquet(
    r'.\data\preprocesada\final_dataset.parquet',
    engine='pyarrow'
)

## -----------------------------------------------------------------------
def set_seed(seed = 42):
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

SEED = [42]

for i in SEED:

    set_seed(i)
    train_df, test_df = split_df(df, test_size=0.2, seed=i)

    #Rutas de salida
    train_path = r'C:\Users\DiegoGonzalez\Desktop\CopulaGAN\data_train\train_df_seed' + str(i) + '.parquet'
    test_path  = r'C:\Users\DiegoGonzalez\Desktop\CopulaGAN\data_test\test_df_seed ' + str(i) + '.parquet'

    # Guardar DataFrames
    train_df.to_parquet(train_path, index=False, engine='pyarrow')
    test_df.to_parquet(test_path, index=False, engine='pyarrow')


    train_path_xlsx = r'C:\Users\DiegoGonzalez\Desktop\CopulaGAN\data_train\train_df_seed' + str(i) + '.xlsx'
    test_path_xlsx  = r'C:\Users\DiegoGonzalez\Desktop\CopulaGAN\data_test\test_df_seed'  + str(i) + '.xlsx'

    # Crear carpetas
    os.makedirs(os.path.dirname(train_path_xlsx), exist_ok=True)
    os.makedirs(os.path.dirname(test_path_xlsx), exist_ok=True)

    # Guardar Excel
    train_df.to_excel(train_path_xlsx, index=False, engine='openpyxl')
    test_df.to_excel(test_path_xlsx, index=False, engine='openpyxl')

    print('Guardada archivo con semilla,', i)