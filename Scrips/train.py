import os
import gc
import warnings
import pandas as pd
import torch
import optuna
import math

from sdv.metadata import SingleTableMetadata
from sdv.single_table import (
    CopulaGANSynthesizer
)
from sdv.cag import FixedCombinations
from sdv.evaluation.single_table import evaluate_quality

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ═══════════════════════════════════════════════════════════════════════════════
# 1. CONFIGURACIÓN 
# ═══════════════════════════════════════════════════════════════════════════════

MODEL_NAME = "CopulaGAN"        # "CopulaGAN"
SEED       = 42
N_TRIALS   = 100                # Número de trials de Optuna
N_STARTUP  = 10                 # Trials antes de que actúe el pruner

BASE_DIR      = r"C:\Users\DiegoGonzalez\Desktop\CopulaGAN"
TRAIN_PATH    = os.path.join(BASE_DIR, "data_train",  f"train_df_seed{SEED}.parquet")
METADATA_PATH = os.path.join(BASE_DIR, "data_train",  f"my_metadata_seed{SEED}.json")
OUTPUT_DIR    = os.path.join(BASE_DIR, "optuna_results", MODEL_NAME.lower())
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- GPU -------------------------------------------------------------------
cuda_available = torch.cuda.is_available()
print(f"GPU disponible: {cuda_available}")
torch.cuda.empty_cache()
gc.collect()


# ═══════════════════════════════════════════════════════════════════════════════
# 2. CARGA DE DATOS Y METADATA
# ═══════════════════════════════════════════════════════════════════════════════

df_train = pd.read_parquet(TRAIN_PATH, engine="pyarrow")

if os.path.exists(METADATA_PATH):
    metadata = SingleTableMetadata.load_from_json(filepath=METADATA_PATH)
else:
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(df_train)
    metadata.save_to_json(filepath=METADATA_PATH)

# Número de filas a generar (80/20 → re-escalar al dataset completo)
NUM_ROWS = math.ceil(len(df_train) / 0.8)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. CONSTRAINTS
# ═══════════════════════════════════════════════════════════════════════════════
constraints = []

# Cancer staging
constraints.append(FixedCombinations(
    column_names=['T_stage', 'N_stage', 'M_stage', 'TNM_stage']
))
 
# Chemotherapy
constraints.append(FixedCombinations(column_names=[
        'Adjuvant_CT', 'Early_end_CT', 'CT_toxicity'
    ]))
 
# Complications
constraints.append(FixedCombinations(column_names=[
    'Anastomosis',
    'Complications',
    'Leak',
    'SSI',
    'Pneumonia',
    'Paralytic_ileus',
    'Kidney_failure'
]))
 
# Surgery type
constraints.append(FixedCombinations(column_names=[
        'Site', 'Surgery_type'
    ]))


# ═══════════════════════════════════════════════════════════════════════════════
# 5. MAPA DE DIMENSIONES 
# ═══════════════════════════════════════════════════════════════════════════════

DIM_MAP = {
    "64_64":       (64, 64),
    "128_128":     (128, 128),
    "128_128_128": (128, 128, 128),
    "256_256":     (256, 256),
    "256_256_256": (256, 256, 256),
    "256_128_64":  (256, 128, 64),
    "512_256":     (512, 256),
    "512_256_128": (512, 256, 128),
    "512_512":     (512, 512),
}

# Sub-conjuntos por modelo 
DIM_KEYS_SMALL  = ["64_64", "128_128", "128_128_128", "256_256"]
DIM_KEYS_MEDIUM = ["128_128", "256_256", "256_256_256", "512_256", "512_256_128"]
DIM_KEYS_LARGE  = list(DIM_MAP.keys())


# ═══════════════════════════════════════════════════════════════════════════════
# 6. FUNCIONES DE ESPACIO DE HIPERPARÁMETROS POR MODELO
# ═══════════════════════════════════════════════════════════════════════════════

def _suggest_params(trial):
    params = {
        "epochs":              trial.suggest_int("epochs", 100, 800, step=50),
        "batch_size":          trial.suggest_categorical("batch_size", [50, 100, 200, 300, 500]),
        "embedding_dim":       trial.suggest_categorical("embedding_dim", [32, 64, 128, 256]),
        "generator_dim":       trial.suggest_categorical("generator_dim", DIM_KEYS_MEDIUM),
        "generator_lr":        trial.suggest_float("generator_lr", 5e-5, 2e-3, log=True),
        "generator_decay":     trial.suggest_float("generator_decay", 1e-7, 1e-4, log=True),
        "discriminator_dim":   trial.suggest_categorical("discriminator_dim", DIM_KEYS_MEDIUM),
        "discriminator_lr":    trial.suggest_float("discriminator_lr", 1e-4, 3e-3, log=True),
        "discriminator_decay": trial.suggest_float("discriminator_decay", 1e-7, 1e-4, log=True),
        "discriminator_steps": trial.suggest_int("discriminator_steps", 1, 8),
        "pac":                 trial.suggest_categorical("pac", [2, 5, 10]),
        "log_frequency":       True,
    }
    # batch_size debe ser divisible por pac
    if params["batch_size"] % params["pac"] != 0:
        raise optuna.exceptions.TrialPruned()
    return params


# ═══════════════════════════════════════════════════════════════════════════════
# 7. SINTETIZADOR
# ═══════════════════════════════════════════════════════════════════════════════

def _build_synthesizer(params):

    # Resolver tuplas de dimensiones (GANs / VAE)
    resolved = {}
    for k, v in params.items():
        if k in ("generator_dim", "discriminator_dim",
                 "compress_dims", "decompress_dims"):
            resolved[k] = DIM_MAP[v]
        else:
            resolved[k] = v

    synth = CopulaGANSynthesizer(
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
# 8. FUNCIÓN OBJETIVO DE OPTUNA
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
# 9. ESTUDIO OPTUNA
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print(f"  Modelo: {MODEL_NAME}  |  Seed: {SEED}  |  Trials: {N_TRIALS}")
print(f"{'='*70}\n")

study = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=SEED),
    pruner=optuna.pruners.MedianPruner(
        n_startup_trials=N_STARTUP,
        n_warmup_steps=0,
    ),
)

study.optimize(objective, n_trials=N_TRIALS)

# ═══════════════════════════════════════════════════════════════════════════════
# 10. RESULTADOS DE OPTUNA
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n✅ Mejores parámetros ({MODEL_NAME}):")
for k, v in study.best_params.items():
    print(f"   {k:>22s} = {v}")
print(f"✅ Mejor score: {study.best_value:.4f}")

trials_excel = os.path.join(OUTPUT_DIR, f"optuna_trials_{MODEL_NAME}_seed{SEED}.xlsx")

study.trials_dataframe().to_excel(trials_excel, index=False, engine='openpyxl')
print(f"📄 Trials guardados en {trials_excel}")


# ═══════════════════════════════════════════════════════════════════════════════
# 11. RE-ENTRENAMIENTO FINAL CON LOS MEJORES HIPERPARÁMETROS
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n🔄 Re-entrenando {MODEL_NAME} con los mejores hiperparámetros…")
best_params = study.best_params

synthesizer_final = _build_synthesizer(best_params)
synthesizer_final.fit(df_train)

data_synthetic_final = synthesizer_final.sample(num_rows=NUM_ROWS)

final_report = evaluate_quality(
    real_data=df_train,
    synthetic_data=data_synthetic_final,
    metadata=metadata,
    verbose=True,
)
print(f"\n✅ Score final ({MODEL_NAME}): {final_report.get_score():.4f}")


# ═══════════════════════════════════════════════════════════════════════════════
# 12. GUARDAR ARTEFACTOS
# ═══════════════════════════════════════════════════════════════════════════════

parquet_path = os.path.join(OUTPUT_DIR, f"synthetic_final_{MODEL_NAME}_seed{SEED}.parquet")
xlsx_path    = os.path.join(OUTPUT_DIR, f"synthetic_final_{MODEL_NAME}_seed{SEED}.xlsx")
pkl_path     = os.path.join(OUTPUT_DIR, f"synthesizer_final_{MODEL_NAME}_seed{SEED}.pkl")

data_synthetic_final.to_parquet(parquet_path, index=False)
data_synthetic_final.to_excel(xlsx_path, index=False, engine="openpyxl")
synthesizer_final.save(pkl_path)

print(f"\nArtefactos guardados en {OUTPUT_DIR}/")
print(f"   ├── {os.path.basename(parquet_path)}")
print(f"   ├── {os.path.basename(xlsx_path)}")
print(f"   └── {os.path.basename(pkl_path)}")
print("\n ¡Listo!")

