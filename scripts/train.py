"""
train.py
Training + Optuna hyperparameter optimisation for the GAN-based SDV synthesizers.

Usage (from the repository root):
    python scripts/train.py --model CopulaGAN
    python scripts/train.py --model CTGAN
    python scripts/train.py --model CTGAN --seed 42 --n-trials 100

Input : data_train/train_df_seed{SEED}.parquet
Output: optuna_results/<model>/
        ├── optuna_trials_<MODEL>_seed<SEED>.xlsx
        ├── synthetic_final_<MODEL>_seed<SEED>.parquet | .xlsx
        └── synthesizer_final_<MODEL>_seed<SEED>.pkl
"""

import argparse
import gc
import math
import os
import warnings
from pathlib import Path

import optuna
import pandas as pd
import torch
from sdv.cag import FixedCombinations
from sdv.evaluation.single_table import evaluate_quality
from sdv.metadata import SingleTableMetadata
from sdv.single_table import CopulaGANSynthesizer, CTGANSynthesizer

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ═══════════════════════════════════════════════════════════════════════════════
# 1. CONFIGURACIÓN
# ═══════════════════════════════════════════════════════════════════════════════

SYNTHESIZERS = {
    "CopulaGAN": CopulaGANSynthesizer,
    "CTGAN": CTGANSynthesizer,
}

parser = argparse.ArgumentParser(description="Optuna optimisation of SDV GAN synthesizers")
parser.add_argument("--model", choices=list(SYNTHESIZERS), default="CopulaGAN")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--n-trials", type=int, default=100, help="Número de trials de Optuna")
parser.add_argument("--n-startup", type=int, default=10, help="Trials antes de que actúe el pruner")
args = parser.parse_args()

MODEL_NAME = args.model
SYNTH_CLASS = SYNTHESIZERS[MODEL_NAME]
SEED = args.seed
N_TRIALS = args.n_trials
N_STARTUP = args.n_startup

ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "data_train" / f"train_df_seed{SEED}.parquet"
METADATA_PATH = ROOT / "data_train" / f"my_metadata_seed{SEED}.json"
OUTPUT_DIR = ROOT / "optuna_results" / MODEL_NAME.lower()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- GPU -------------------------------------------------------------------
cuda_available = torch.cuda.is_available()
print(f"GPU disponible: {cuda_available}")
torch.cuda.empty_cache()
gc.collect()

# ═══════════════════════════════════════════════════════════════════════════════
# 2. CARGA DE DATOS Y METADATA
# ═══════════════════════════════════════════════════════════════════════════════

df_train = pd.read_parquet(TRAIN_PATH, engine="pyarrow")

if METADATA_PATH.exists():
    metadata = SingleTableMetadata.load_from_json(filepath=str(METADATA_PATH))
else:
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(df_train)
    metadata.save_to_json(filepath=str(METADATA_PATH))

# Número de filas a generar (80/20 → re-escalar al dataset completo)
NUM_ROWS = math.ceil(len(df_train) / 0.8)

# ═══════════════════════════════════════════════════════════════════════════════
# 3. CONSTRAINTS
# ═══════════════════════════════════════════════════════════════════════════════

constraints = [
    # Cancer staging
    FixedCombinations(column_names=["T_stage", "N_stage", "M_stage", "TNM_stage"]),
    # Chemotherapy
    FixedCombinations(column_names=["Adjuvant_CT", "Early_end_CT", "CT_toxicity"]),
    # Complications
    FixedCombinations(column_names=[
        "Anastomosis", "Complications", "Leak", "SSI",
        "Pneumonia", "Paralytic_ileus", "Kidney_failure",
    ]),
    # Surgery type
    FixedCombinations(column_names=["Site", "Surgery_type"]),
]

# ═══════════════════════════════════════════════════════════════════════════════
# 4. MAPA DE DIMENSIONES
# ═══════════════════════════════════════════════════════════════════════════════

DIM_MAP = {
    "64_64": (64, 64),
    "128_128": (128, 128),
    "128_128_128": (128, 128, 128),
    "256_256": (256, 256),
    "256_256_256": (256, 256, 256),
    "256_128_64": (256, 128, 64),
    "512_256": (512, 256),
    "512_256_128": (512, 256, 128),
    "512_512": (512, 512),
}

DIM_KEYS_MEDIUM = ["128_128", "256_256", "256_256_256", "512_256", "512_256_128"]

# ═══════════════════════════════════════════════════════════════════════════════
# 5. ESPACIO DE HIPERPARÁMETROS
# ═══════════════════════════════════════════════════════════════════════════════

def _suggest_params(trial):
    params = {
        "epochs": trial.suggest_int("epochs", 100, 800, step=50),
        "batch_size": trial.suggest_categorical("batch_size", [50, 100, 200, 300, 500]),
        "embedding_dim": trial.suggest_categorical("embedding_dim", [32, 64, 128, 256]),
        "generator_dim": trial.suggest_categorical("generator_dim", DIM_KEYS_MEDIUM),
        "generator_lr": trial.suggest_float("generator_lr", 5e-5, 2e-3, log=True),
        "generator_decay": trial.suggest_float("generator_decay", 1e-7, 1e-4, log=True),
        "discriminator_dim": trial.suggest_categorical("discriminator_dim", DIM_KEYS_MEDIUM),
        "discriminator_lr": trial.suggest_float("discriminator_lr", 1e-4, 3e-3, log=True),
        "discriminator_decay": trial.suggest_float("discriminator_decay", 1e-7, 1e-4, log=True),
        "discriminator_steps": trial.suggest_int("discriminator_steps", 1, 8),
        "pac": trial.suggest_categorical("pac", [2, 5, 10]),
        "log_frequency": True,
    }
    # batch_size debe ser divisible por pac
    if params["batch_size"] % params["pac"] != 0:
        raise optuna.exceptions.TrialPruned()
    return params

# ═══════════════════════════════════════════════════════════════════════════════
# 6. SINTETIZADOR
# ═══════════════════════════════════════════════════════════════════════════════

def _build_synthesizer(params):
    # Resolver tuplas de dimensiones
    resolved = {
        k: DIM_MAP[v] if k in ("generator_dim", "discriminator_dim") else v
        for k, v in params.items()
    }

    synth = SYNTH_CLASS(
        metadata,
        enforce_rounding=True,
        enforce_min_max_values=True,
        verbose=True,
        enable_gpu=cuda_available,
        **resolved,
    )
    synth.add_constraints(constraints=constraints)
    return synth

# ═══════════════════════════════════════════════════════════════════════════════
# 7. FUNCIÓN OBJETIVO DE OPTUNA
# ═══════════════════════════════════════════════════════════════════════════════

def objective(trial):
    params = _suggest_params(trial)

    try:
        synth = _build_synthesizer(params)
        synth.fit(df_train)
        synthetic_data = synth.sample(num_rows=NUM_ROWS)

        quality_report = evaluate_quality(
            real_data=df_train,
            synthetic_data=synthetic_data,
            metadata=metadata,
            verbose=False,
        )
        score = quality_report.get_score()

        # Reportar para pruner
        trial.report(score, step=0)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        print(f"  Trial {trial.number:>3} | Score: {score:.4f} | {params}")
        return score

    except optuna.exceptions.TrialPruned:
        raise
    except Exception as e:
        print(f"  Trial {trial.number:>3} falló: {e}")
        return 0.0
    finally:
        torch.cuda.empty_cache()
        gc.collect()

# ═══════════════════════════════════════════════════════════════════════════════
# 8. ESTUDIO OPTUNA
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n{'=' * 70}")
print(f"  Modelo: {MODEL_NAME} | Seed: {SEED} | Trials: {N_TRIALS}")
print(f"{'=' * 70}\n")

study = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=SEED),
    pruner=optuna.pruners.MedianPruner(n_startup_trials=N_STARTUP, n_warmup_steps=0),
)
study.optimize(objective, n_trials=N_TRIALS)

# ═══════════════════════════════════════════════════════════════════════════════
# 9. RESULTADOS DE OPTUNA
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\nMejores parámetros ({MODEL_NAME}):")
for k, v in study.best_params.items():
    print(f"  {k:>22s} = {v}")
print(f"Mejor score: {study.best_value:.4f}")

trials_excel = OUTPUT_DIR / f"optuna_trials_{MODEL_NAME}_seed{SEED}.xlsx"
study.trials_dataframe().to_excel(trials_excel, index=False, engine="openpyxl")
print(f"Trials guardados en {trials_excel}")

# ═══════════════════════════════════════════════════════════════════════════════
# 10. RE-ENTRENAMIENTO FINAL CON LOS MEJORES HIPERPARÁMETROS
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\nRe-entrenando {MODEL_NAME} con los mejores hiperparámetros…")
synthesizer_final = _build_synthesizer(study.best_params)
synthesizer_final.fit(df_train)

data_synthetic_final = synthesizer_final.sample(num_rows=NUM_ROWS)

final_report = evaluate_quality(
    real_data=df_train,
    synthetic_data=data_synthetic_final,
    metadata=metadata,
    verbose=True,
)
print(f"\nScore final ({MODEL_NAME}): {final_report.get_score():.4f}")

# ═══════════════════════════════════════════════════════════════════════════════
# 11. GUARDAR ARTEFACTOS
# ═══════════════════════════════════════════════════════════════════════════════

parquet_path = OUTPUT_DIR / f"synthetic_final_{MODEL_NAME}_seed{SEED}.parquet"
xlsx_path = OUTPUT_DIR / f"synthetic_final_{MODEL_NAME}_seed{SEED}.xlsx"
pkl_path = OUTPUT_DIR / f"synthesizer_final_{MODEL_NAME}_seed{SEED}.pkl"

data_synthetic_final.to_parquet(parquet_path, index=False)
data_synthetic_final.to_excel(xlsx_path, index=False, engine="openpyxl")
synthesizer_final.save(str(pkl_path))

print(f"\nArtefactos guardados en {OUTPUT_DIR}/")
print(f"  ├── {parquet_path.name}")
print(f"  ├── {xlsx_path.name}")
print(f"  └── {pkl_path.name}")
print("\n¡Listo!")
