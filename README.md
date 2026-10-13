# Colorectal Cancer Synthetic Cohort

Code to build, train and evaluate **synthetic tabular cohorts** of colorectal cancer surgery patients, with a focus on whether synthetic data remains valid for **survival analysis** in a single-centre, small-sample setting.

Four generative models from the [Synthetic Data Vault (SDV)](https://github.com/sdv-dev/SDV) are compared:

| Label | Model | Implementation | Hyperparameter search |
|---|---|---|---|
| **GC** | Gaussian Copula | `Notebooks/gaussiancopula.ipynb` | Grid search over marginal distributions |
| **TVAE** | Tabular Variational Autoencoder | `Notebooks/tabular_vae.ipynb` | Grid search |
| **CTGAN** | Conditional Tabular GAN | `scripts/train.py --model CTGAN` | Optuna (TPE, 100 trials, median pruner) |
| **C-GAN** | Copula-GAN | `scripts/train.py --model CopulaGAN` | Optuna (TPE, 100 trials, median pruner) |

Synthetic cohorts are evaluated along four axes: **fidelity**, **computational utility**, **clinical utility (survival analysis)** and **privacy**.

> ⚠️ **Data availability.** The real patient data are **not** included in this repository and cannot be shared publicly. Only the code, the variable-name dictionaries and an example synthetic output are provided.

---

## Repository structure

```text
Colorectal_Cancer_Synthetic_Cohort/
├── Notebooks/
│   ├── data_preprocesing.ipynb       # Exploratory, step-by-step version of the preprocessing
│   ├── gaussiancopula.ipynb          # GC: marginal selection, constraints, grid search, sampling
│   ├── tabular_vae.ipynb             # TVAE: constraints, grid search, sampling
│   ├── metrics_final.ipynb           # Full evaluation: fidelity, utility, survival, privacy
│   ├── km_combined_bootstrap.ipynb   # KM median survival with bootstrap 95% CI + numbers at risk
│   ├── collinearity.ipynb            # VIF / GVIF and Age-spline sensitivity analysis for the Cox model
│   └── graphs.ipynb                  # Comparative figures for all models (uses Functions_graphs.py)
├── scripts/
│   ├── data_preprocesing.py          # Preprocessing pipeline → final_dataset.parquet / .xlsx
│   ├── pre_train.py                  # 80/20 train–test split (seed 42)
│   ├── train.py                      # CTGAN / Copula-GAN training + Optuna optimisation (--model)
│   └── Functions_graphs.py           # Plotting and evaluation helpers
├── data/
│   ├── original_name_data.txt        # Original column names of the source database (1,098 fields)
│   └── dict_names_changes.txt        # Normalised names, line-by-line mapping of the file above
├── results/
│   └── synthetic_final_CopulaGAN.xlsx  # Example synthetic cohort generated with Copula-GAN
├── requirements.txt
├── .gitignore
└── README.md
```

Folders created at runtime (not tracked):

```text
data/cruda/            # Raw source files (not shared)
data/preprocesada/     # final_dataset.parquet / .xlsx
data_train/            # train_df_seed42.parquet / .xlsx, SDV metadata JSON
data_test/             # test_df_seed42.parquet / .xlsx
optuna_results/        # Optuna trials, synthetic cohorts and fitted synthesizers (.pkl)
results/full_finals/  # Final synthetic cohorts read by graphs.ipynb
plots/                # Figures and tables exported by the evaluation notebooks
```

---

## Pipeline

```text
Raw database ──► 1. Preprocessing ──► 2. Train/test split ──► 3. Generative models ──► 4. Evaluation
 (not shared)     data_preprocesing     pre_train.py           GC · TVAE · CTGAN · C-GAN   metrics_final · km_bootstrap
                                                                                            collinearity · graphs
```

### 1. Preprocessing — `scripts/data_preprocesing.py`

Turns the raw clinical database (1,098 fields) into an analysis-ready table of **41 variables + patient ID**:

- Column renaming using `data/dict_names_changes.txt`.
- Clinical selection of variables (demographics, comorbidities, tumour, treatment, complications, survival).
- Derived variables: node ratio, length of stay, overall survival time and recurrence-free time (in days, from surgery date).
- Cross-check and imputation of dates, death, recurrence, T stage, global TNM stage and tumour site from a secondary reference database, plus logical consistency rules (e.g. a death date implies `Death = 1`).
- Rule-based imputation of immunosuppression; KNN imputation (k = 5, standardised predictors) of BMI.
- Removal of low-prevalence / high-missingness variables and manually validated corrections.
- Translation of column names and categorical values to English.

Output: `data/preprocesada/final_dataset.parquet` and `.xlsx`.

The notebook `Notebooks/data_preprocesing.ipynb` contains the same steps with descriptive statistics and missing-value plots.

### 2. Train/test split — `scripts/pre_train.py`

Random 80/20 split with a fixed seed (`42`). Generators are trained **only on the training set**; the number of synthetic rows is rescaled to the full cohort size (`ceil(n_train / 0.8)`).

### 3. Generative models

All models share the same **SDV constraints** (`FixedCombinations`) so that only clinically observed combinations are generated:

| Group | Columns |
|---|---|
| Cancer staging | `T_stage`, `N_stage`, `M_stage`, `TNM_stage` |
| Chemotherapy | `Adjuvant_CT`, `Early_end_CT`, `CT_toxicity` |
| Complications | `Anastomosis`, `Complications`, `Leak`, `SSI`, `Pneumonia`, `Paralytic_ileus`, `Kidney_failure` |
| Surgery | `Site`, `Surgery_type` |

- **GC** — marginal distributions chosen per continuous variable (e.g. `truncnorm`, `gamma`, `beta`, `gaussian_kde`), then grid search over distributions and `enforce_min_max_values` / `enforce_rounding`.
- **TVAE** — grid search over epochs, batch size, embedding size, encoder/decoder dimensions, `l2scale` and `loss_factor`.
- **CTGAN / C-GAN** — Optuna search over epochs, batch size, embedding size, generator/discriminator architecture, learning rates, weight decay, discriminator steps and `pac`. The objective is the SDV Quality Score; the best configuration is retrained and saved (`.parquet`, `.xlsx`, `.pkl`).

### 4. Evaluation — `Notebooks/metrics_final.ipynb`

Set `model` at the top of the notebook and point `df_synth` to the corresponding synthetic file.

| Axis | Metrics |
|---|---|
| **Fidelity** | Kolmogorov–Smirnov test and standardised mean difference (continuous); χ² test and Cramér's V (categorical); Benjamini–Hochberg FDR correction |
| **Computational utility** | Variable-by-variable prediction with Random Forest (F1 for categorical, R² for continuous), real-vs-synthetic utility ratio; TSTR / TRTS / TRTR AUC (`graphs.ipynb`) |
| **Clinical utility** | Kaplan–Meier curves and log-rank test; Cox proportional hazards models (Age as natural cubic spline, df = 4, knots fitted on real data and reused on synthetic data; ridge penalty 0.1); hazard-ratio comparison, sign agreement, CI overlap / width / coverage |
| **Privacy** | Distance to Closest Record (DCR) and per-variable match rate; Membership Inference Attack (MIA); Attribute Inference Attack (AIA) with real-data baseline |

Complementary notebooks:

- `km_combined_bootstrap.ipynb` — KM median survival with 2,000-sample bootstrap 95% CI, fraction of resamples where the median is not reached, and numbers at risk, for the real cohort and the four synthetic cohorts.
- `collinearity.ipynb` — VIF of Cox covariates, sensitivity analysis comparing linear Age, Age spline and NAT substitution (concordance, log-likelihood, partial AIC), and generalised VIF (Fox & Monette) for the Age-spline block.
- `graphs.ipynb` + `scripts/Functions_graphs.py` — PCA overlap, correlation heatmaps, continuous and categorical distributions, TSTR/TRTS AUC, KM curves and concordance by tumour site for all models.

---

## Variables

| Domain | Variables |
|---|---|
| Identifier | `Patient_ID` |
| Demographics / baseline | `Sex`, `Age`, `BMI`, `Smoking_Status`, `ASA` |
| Comorbidities | `T2DM`, `HTN`, `Dl`, `IHD`, `COPD`, `CKD`, `IS` |
| Tumour | `Site`, `T_stage`, `N_stage`, `M_stage`, `TNM_stage`, `Node_ratio`, `R_grade` |
| Treatment | `NAT`, `Approach`, `Procedure`, `Surgery_type`, `Anastomosis`, `Stoma_created`, `Adjuvant_CT`, `Early_end_CT`, `CT_toxicity` |
| Post-operative outcomes | `Complications`, `Leak`, `SSI`, `Pneumonia`, `Paralytic_ileus`, `Kidney_failure`, `Reintervention`, `Readmission`, `LOS_days` |
| Survival | `Survival_days`, `Recurrence_free_days`, `Death`, `Recurrence` |

Continuous variables: `Age`, `BMI`, `Node_ratio`, `LOS_days`, `Survival_days`, `Recurrence_free_days`. All others are treated as categorical.

---

## Installation

Python 3.10–3.12. A CUDA-capable GPU is optional (used by CTGAN / Copula-GAN if available).

```bash
git clone https://github.com/DiegoGovi/Colorectal_Cancer_Synthetic_Cohort.git
cd Colorectal_Cancer_Synthetic_Cohort
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Main dependencies: `pandas`, `numpy`, `pyarrow`, `openpyxl`, `scikit-learn`, `sdv` (with the `sdv.cag` module), `sdmetrics`, `torch`, `optuna`, `lifelines`, `statsmodels`, `patsy`, `scipy`, `matplotlib`, `seaborn`, `tqdm`.

## Usage

Run from the repository root:

```bash
# 1. Preprocess the raw database (requires the raw files in data/cruda/)
python scripts/data_preprocesing.py

# 2. Train/test split → data_train/, data_test/
python scripts/pre_train.py

# 3. Train the GAN-based models → optuna_results/<model>/
python scripts/train.py --model CTGAN
python scripts/train.py --model CopulaGAN      # optional: --seed 42 --n-trials 100

# 4. Run the notebooks for GC and TVAE, then the evaluation notebooks
```

All paths in the scripts are resolved relative to the repository root, so they can be run from any working directory. `data_preprocesing.ipynb` and `graphs.ipynb` are meant to be run from `Notebooks/`. The GC, TVAE and evaluation notebooks (`gaussiancopula`, `tabular_vae`, `metrics_final`, `km_combined_bootstrap`, `collinearity`) were run on Google Colab and read their inputs from `/content/`; upload the files there or adapt the paths when running locally.

---

## Reproducibility notes

- Fixed seed `42` for the split, Optuna's TPE sampler and the Random Forest models.
- Synthetic cohorts have the same size as the full real cohort.
- The SDV metadata is detected from the training set and saved to `data_train/my_metadata_seed42.json` on first run.

## Authors

Helena Vidaković Bunčić<sup>d,\*</sup>, Diego González Oviaño<sup>d,\*</sup>, Sara Rodríguez Abou-Salama<sup>d,\*</sup>, Pablo Pérez Ruiz<sup>c</sup>, Carlos Blanco Velázquez<sup>d</sup>, Laura Llorente Sanz<sup>d</sup>, Jana Dziakova<sup>a,d,e</sup>, María Luaces<sup>b,e</sup>, Julio Mayol<sup>a,d,e</sup>

<sup>\*</sup> These authors contributed equally and share first authorship.

<sup>a</sup> Servicio de Cirugía, Hospital Clínico San Carlos, Madrid, Spain  
<sup>b</sup> Servicio de Cardiología, Hospital Clínico San Carlos, Madrid, Spain  
<sup>c</sup> Servicio de Medicina Preventiva, Hospital Clínico San Carlos, Madrid, Spain  
<sup>d</sup> Instituto de Investigación Sanitaria San Carlos (IdISSC), Madrid, Spain  
<sup>e</sup> Universidad Complutense de Madrid, Spain

This repository is developed and maintained by the **Unidad de Innovación of the Instituto de Investigación Sanitaria San Carlos (IdISSC)**, Hospital Clínico San Carlos, Madrid.

Contact: Diego González Oviaño · ORCID [0009-0005-0227-7040](https://orcid.org/0009-0005-0227-7040)

## License

© 2026 The authors. All rights reserved. No license is granted for reuse, modification or redistribution of this code without the authors' permission.
