"""
data_preprocessing.py
Preprocesado completo del dataset de tesis (cáncer colorrectal).
Genera final_dataset.parquet y final_dataset.xlsx.
"""

import pandas as pd
import numpy as np
import warnings
from sklearn.impute import KNNImputer

warnings.simplefilter(action="ignore", category=FutureWarning)

# ──────────────────────────────────────────────
# 1. Carga de datos
# ──────────────────────────────────────────────
df = pd.read_excel(r".\data\cruda\tesis_21_07_2023_correccion_1.xlsx", engine="openpyxl")
print("Longitud de la base de datos:", df.shape)

# ──────────────────────────────────────────────
# 2. Renombrar variables
# ──────────────────────────────────────────────
lst_new_names = []
with open(r".\data\cruda\dict_names_changes.txt", "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        lst_new_names.append(line)

df.columns = lst_new_names

# ──────────────────────────────────────────────
# 3. Transformaciones iniciales
# ──────────────────────────────────────────────
# Ratio ganglios
df["Ratio_ganglios"] = np.where(
    df["Ganglios_Totales"] > 0,
    (df["Ganglios_Positivos"] / df["Ganglios_Totales"]).round(6),
    np.nan,
)

# Si/No → 1/0
df = df.replace({"no": 0, "si": 1, "No": 0, "Si": 1, "NO": 0, "SI": 1})
df["Sexo"] = df["Sexo"].replace({"hombre": 0, "mujer": 1, "Hombre": 0, "Mujer": 1})
df["Anastomosis_Realizada"] = df["Anastomosis_Realizada"].replace({0: "No"})

# Clasificación ASA
inf_cla_asa = {"I": 1, "II": 2, "III": 3, "IV": 4}
df["Clasificacion_ASA"] = (
    df["Clasificacion_ASA"]
    .astype("string")
    .str.strip()
    .str.upper()
    .map(inf_cla_asa)
    .astype("Int64")
)

# Estado fumador
inf_fum = {"no fuma": 0, "exfumador": 1, "si activo": 2}
df["Estado_Fumador"] = (
    df["Estado_Fumador"]
    .astype("string")
    .str.strip()
    .str.lower()
    .map(inf_fum)
    .astype("Int64")
)

# ──────────────────────────────────────────────
# 4. Selección de columnas por validez clínica
# ──────────────────────────────────────────────
col_sociodemograficas = [
    "ID_Paciente", "Sexo", "Edad_Cirugia", "IMC",
    "Estado_Fumador", "Dependencia_Funcional", "Clasificacion_ASA",
]

col_antecendentes = [
    "Dm2", "Hta", "Dl", "Cardiopatia_Isquemica",
    "Epoc", "Irc", "Inmunosupresion",
]

col_tumor = [
    "Localizacion_ascendente", "Localizacion_sigma", "Localizacion_recto",
    "Estadio_T_Patologico", "estadio_N", "Estadio_M",
    "Ratio_ganglios", "Invasion_Vascular", "Grado_Reseccion_R",
]

col_tratamiento = [
    "Neoadyuvancia", "Abordaje_Quirurgico", "Tipo_Intervencion",
    "Anastomosis_Realizada", "Estoma_Realizado",
    "Quimioterapia_Adyuvante_si", "Finalizacion_Prematura_QT_si",
    "Toxicidad_QT_si",
]

tipo_intervencion = [
    "Tipo_Intervencion_hemicolectomia_derecha",
    "Tipo_Intervencion_hemicolectomia_derecha_ampliada",
    "Tipo_Intervencion_hemicolectomia_izquierda",
    "Tipo_Intervencion_sigmoidectomia",
    "Tipo_Intervencion_reseccion_anterior",
    "Tipo_Intervencion_reseccion_anterior_baja",
    "Tipo_Intervencion_reseccion_anterior_ultrabaja",
    "Tipo_Intervencion_colectomia_subtotal",
    "Tipo_Intervencion_colectomia_total",
    "Tipo_Intervencion_reseccion_segmentaria",
    "Tipo_Intervencion_amputacion_abdominoperineal",
]

col_outcomes = [
    "Complicaciones_Ingreso_si", "Complicacion_fuga_anastomotica",
    "Complicacion_infeccion_sitio_quirurgico", "Complicacion_neumonia",
    "Complicacion_TEP", "Complicacion_ileo_paralitico",
    "Complicacion_insuficiencia_renal", "Reintervencion_si",
    "Reingreso_30d_si", "Fecha_Primera_Consulta_Cirugia",
    "Fecha_Ingreso_Cirugia", "Fecha_Alta_Hospitalaria",
]

col_supervivencia = [
    "Recidiva_si", "exitus", "Fecha_Exitus",
    "Fecha_Ultima_Visita", "Fecha_Diagnostico_Recidiva",
]

new_cols = (
    col_sociodemograficas + col_antecendentes + col_tumor
    + col_tratamiento + tipo_intervencion + col_outcomes
    + col_supervivencia
)

df_post = df.loc[:, new_cols]
print("Preprocesado:", df_post.shape, "| Antes:", df.shape)

# ──────────────────────────────────────────────
# 5. Formato de fechas
# ──────────────────────────────────────────────
cols_dates = [
    "Fecha_Ingreso_Cirugia", "Fecha_Alta_Hospitalaria",
    "Fecha_Exitus", "Fecha_Ultima_Visita",
    "Fecha_Diagnostico_Recidiva", "Fecha_Primera_Consulta_Cirugia",
]

for col in cols_dates:
    df_post[col] = pd.to_datetime(df_post[col], errors="coerce")

df_post.rename(columns={"estadio_N": "Estadio_N"}, inplace=True)

# ──────────────────────────────────────────────
# 6. Imputación con base mini
# ──────────────────────────────────────────────
df_mini = pd.read_excel(r".\data\cruda\base tesis mini.xlsx", engine="openpyxl")
df_mini["exitus"] = df_mini["exitus"].str.lower().replace({"si": 1, "no": 0})
df_mini["recidiva"] = df_mini["recidiva"].str.lower().replace({"si": 1, "no": 0})

map_cols = {
    "Fecha_Alta_Hospitalaria":    "Fecha alta",
    "Fecha_Ingreso_Cirugia":      "fecha de cirugia",
    "Fecha_Diagnostico_Recidiva": "fecha de dg de recidiva",
    "Fecha_Exitus":               "fecha exitus",
    "Fecha_Ultima_Visita":        "ultima visita",
    "exitus":                     "exitus",
    "Recidiva_si":                "recidiva",
}

map_id = df_mini.drop_duplicates(subset="Study ID").set_index("Study ID")

for col_post, col_mini in map_cols.items():
    if col_mini in map_id.columns:
        df_post[col_post] = (
            df_post["ID_Paciente"]
            .map(map_id[col_mini])
            .combine_first(df_post[col_post])
        )

# ──────────────────────────────────────────────
# 7. Creación de variables temporales
# ──────────────────────────────────────────────
df_post["Estancia_Dias"] = (
    df_post["Fecha_Alta_Hospitalaria"] - df_post["Fecha_Ingreso_Cirugia"]
).dt.days

fin_supervivencia = df_post["Fecha_Exitus"].fillna(df_post["Fecha_Ultima_Visita"])
df_post["Supervivencia_Dias"] = (
    fin_supervivencia - df_post["Fecha_Ingreso_Cirugia"]
).dt.days

fin_libre_recidiva = df_post["Fecha_Diagnostico_Recidiva"].fillna(
    df_post["Fecha_Ultima_Visita"]
)
mask_no_recidiva = df_post["Recidiva_si"].eq(0)
fin_libre_recidiva = fin_libre_recidiva.where(
    ~(mask_no_recidiva & fin_libre_recidiva.isna()),
    df_post["Fecha_Exitus"],
)
df_post["Libre_Recidiva_Dias"] = (
    fin_libre_recidiva - df_post["Fecha_Ingreso_Cirugia"]
).dt.days

for c in ["Estancia_Dias", "Supervivencia_Dias"]:
    df_post.loc[df_post[c] < 0, c] = pd.NA

# ──────────────────────────────────────────────
# 8. Detección de inconsistencias
# ──────────────────────────────────────────────
mask_alta_ingre = df_post["Fecha_Alta_Hospitalaria"] < df_post["Fecha_Ingreso_Cirugia"]
mask_1_sin_fecha = df_post["exitus"].eq(1) & df_post["Fecha_Exitus"].isna()
mask_0_con_fecha = df_post["exitus"].eq(0) & df_post["Fecha_Exitus"].notna()
mask_exitus_antes_ingreso = df_post["Fecha_Exitus"] < df_post["Fecha_Ingreso_Cirugia"]
mask_ult_visita_antes_ingreso = df_post["Fecha_Ultima_Visita"] < df_post["Fecha_Ingreso_Cirugia"]
mask_recidiva_antes_ingreso = df_post["Fecha_Diagnostico_Recidiva"] < df_post["Fecha_Ingreso_Cirugia"]
mask_1_sin_fecha_rec = df_post["Recidiva_si"].eq(1) & df_post["Fecha_Diagnostico_Recidiva"].isna()
mask_0_con_fecha_rec = df_post["Recidiva_si"].eq(0) & df_post["Fecha_Diagnostico_Recidiva"].notna()

df_post.rename(columns={"exitus": "Exitus"}, inplace=True)

# ──────────────────────────────────────────────
# 9. Imputación lógica de Exitus y Recidiva
# ──────────────────────────────────────────────
exitus_num = pd.to_numeric(df_post["Exitus"], errors="coerce")
recidiva_num = pd.to_numeric(df_post["Recidiva_si"], errors="coerce")

mask_fecha_exitus = df_post["Fecha_Exitus"].notna()
mask_fecha_recidiva = df_post["Fecha_Diagnostico_Recidiva"].notna()

df_post["Exitus_def"] = exitus_num.copy()
df_post.loc[mask_fecha_exitus, "Exitus_def"] = 1
df_post.loc[exitus_num.eq(1) & ~mask_fecha_exitus, "Exitus_def"] = 1
df_post.loc[exitus_num.isna() & ~mask_fecha_exitus, "Exitus_def"] = pd.NA
df_post["Exitus_def"] = df_post["Exitus_def"].astype("Int64")

df_post["Recidiva_def"] = recidiva_num
df_post.loc[mask_fecha_recidiva, "Recidiva_def"] = 1
df_post.loc[recidiva_num.eq(1) & ~mask_fecha_recidiva, "Recidiva_def"] = 1
df_post.loc[recidiva_num.isna() & ~mask_fecha_recidiva, "Recidiva_def"] = pd.NA
df_post["Recidiva_def"] = df_post["Recidiva_def"].astype("Int64")

df_post = df_post.drop(columns=["Exitus", "Recidiva_si"])

# Eliminar fechas
df_no_dates = df_post.drop(columns=cols_dates)
df = df_no_dates.copy()

# ──────────────────────────────────────────────
# 10. Conversión de formatos
# ──────────────────────────────────────────────
col_change_for_str_to_num = ["IMC"]

col_str_to_int = [
    "Estado_Fumador", "Clasificacion_ASA", "Inmunosupresion", "Hta", "Dl",
    "Invasion_Vascular", "Neoadyuvancia", "Estoma_Realizado",
    "Exitus_def", "Recidiva_def", "Estancia_Dias",
    "Supervivencia_Dias", "Libre_Recidiva_Dias",
]

for col in col_change_for_str_to_num:
    s = df[col].astype(str).str.strip().str.replace(",", ".", regex=False)
    s = s.replace(["", "NA", "N/A", "nan"], pd.NA)
    df[col] = pd.to_numeric(s, errors="coerce").astype("float64")

for col in col_str_to_int:
    s = df[col].astype(str).str.strip().str.replace(",", ".", regex=False)
    s = s.replace(["", "NA", "N/A", "nan"], pd.NA)
    df[col] = pd.to_numeric(s, errors="coerce").astype("Int64")

# ──────────────────────────────────────────────
# 11. Estadio T y TNM global
# ──────────────────────────────────────────────
map_cols_t = {"Estadio_T_Patologico": "estadio T"}
map_id_t = df_mini.drop_duplicates(subset="Study ID").set_index("Study ID")

for col_post, col_mini in map_cols_t.items():
    if col_mini in map_id_t.columns:
        df[col_post] = (
            df["ID_Paciente"]
            .map(map_id_t[col_mini])
            .combine_first(df[col_post])
        )

# Estadio Global TNM
df["Estadio_Global_TNM"] = None

t = df["Estadio_T_Patologico"].fillna("MISSING")
n = df["Estadio_N"].fillna("MISSING")
m = df["Estadio_M"].fillna("MISSING")

condiciones = [
    m.eq("M1"),
    m.eq("M0") & n.isin(["N1", "N2"]),
    m.eq("M0") & n.eq("N0") & t.isin(["T3", "T4"]),
    m.eq("M0") & n.eq("N0") & t.isin(["T1", "T2"]),
]
valores_tnm = ["IV", "III", "II", "I"]

df["Estadio_Global_TNM"] = np.select(condiciones, valores_tnm, default=pd.NA)
df["Estadio_Global_TNM"] = df["Estadio_Global_TNM"].astype("string")

# Imputación Estadio Global desde mini
map_cols_est = {"Estadio_Global_TNM": "Estadio_Cancer"}
map_id_est = df_mini.drop_duplicates(subset="Study ID").set_index("Study ID")

for col_post, col_mini in map_cols_est.items():
    if col_mini in map_id_est.columns:
        df[col_post] = (
            df["ID_Paciente"]
            .map(map_id_est[col_mini])
            .combine_first(df[col_post])
        )

df["Estadio_Global_TNM"] = df["Estadio_Global_TNM"].replace({None: np.nan})

# Correcciones M, N, T
mask_m_x = df["Estadio_M"].astype("string").str.contains("x", case=False, na=True)
mask_estadio_iv = df["Estadio_Global_TNM"].astype("string").str.contains("IV", case=False, na=False)
df.loc[mask_m_x & mask_estadio_iv, "Estadio_M"] = "M1"

mask_estadio_i_iii = df["Estadio_Global_TNM"].astype("string").str.match(r"^(I|II|III)", case=False, na=False)
df.loc[mask_m_x & mask_estadio_i_iii, "Estadio_M"] = "M0"

mask_nx = df["Estadio_N"].astype("string") == "x"
df.loc[mask_nx, "Estadio_N"] = "N0"

mask_tx = df["Estadio_T_Patologico"].astype("string") == "x"
df.loc[mask_tx, "Estadio_Global_TNM"] = "0"
df.loc[mask_tx, "Estadio_T_Patologico"] = "T0"

# Validación clínica manual
df.loc[df["ID_Paciente"] == 64, "Estadio_M"] = "M0"
df.loc[df["ID_Paciente"] == 64, "Estadio_Global_TNM"] = "I"
df = df[df["ID_Paciente"] != 66]

# Limpieza residual
df["Estadio_M"] = df["Estadio_M"].replace({"x": np.nan, "X": np.nan})

# ──────────────────────────────────────────────
# 12. Localización e imputaciones varias
# ──────────────────────────────────────────────
df["Localizacion"] = None
map_cols_loc = {"Localizacion": "localizacion"}
map_id_loc = df_mini.drop_duplicates(subset="Study ID").set_index("Study ID")

for col_post, col_mini in map_cols_loc.items():
    if col_mini in map_id_loc.columns:
        df[col_post] = (
            df["ID_Paciente"]
            .map(map_id_loc[col_mini])
            .combine_first(df[col_post])
        )

df["Localizacion"] = df["Localizacion"].replace({None: "recto"})
df["Neoadyuvancia"] = df["Neoadyuvancia"].fillna(2)

# ──────────────────────────────────────────────
# 13. Tipos string
# ──────────────────────────────────────────────
cols_str = [
    "Grado_Reseccion_R", "Estadio_T_Patologico", "Anastomosis_Realizada",
    "Tipo_Intervencion", "Abordaje_Quirurgico", "Estadio_M",
    "Estadio_N", "Estadio_Global_TNM", "Localizacion",
]
df[cols_str] = df[cols_str].astype("string")

# ──────────────────────────────────────────────
# 14. Inmunosupresión por lógica clínica
# ──────────────────────────────────────────────
mask_inmunosuprimido = df["Neoadyuvancia"].eq(1) | df["Quimioterapia_Adyuvante_si"].eq(1)
df.loc[df["Inmunosupresion"].isna() & mask_inmunosuprimido, "Inmunosupresion"] = 1
df.loc[df["Inmunosupresion"].isna() & ~mask_inmunosuprimido, "Inmunosupresion"] = 0

# ──────────────────────────────────────────────
# 15. Tipo de procedimiento
# ──────────────────────────────────────────────
mapeo_procedimiento = {
    "Tipo_Intervencion_hemicolectomia_derecha": "HCD",
    "Tipo_Intervencion_hemicolectomia_derecha_ampliada": "HCDA",
    "Tipo_Intervencion_hemicolectomia_izquierda": "HCI",
    "Tipo_Intervencion_sigmoidectomia": "Sigma",
    "Tipo_Intervencion_reseccion_anterior": "RA",
    "Tipo_Intervencion_reseccion_anterior_baja": "RAB",
    "Tipo_Intervencion_reseccion_anterior_ultrabaja": "RAUB",
    "Tipo_Intervencion_colectomia_subtotal": "CST",
    "Tipo_Intervencion_colectomia_total": "CT",
    "Tipo_Intervencion_reseccion_segmentaria": "ResSeg",
    "Tipo_Intervencion_amputacion_abdominoperineal": "AAP",
}

df["Tipo_Procedimiento"] = None
for col, abrev in mapeo_procedimiento.items():
    mask = df[col] == 1
    df.loc[mask, "Tipo_Procedimiento"] = abrev

df.drop(columns=list(mapeo_procedimiento.keys()), inplace=True)

correccion_procedimiento = {
    29: "RA", 39: "ResSeg", 65: "RAB", 82: "CT", 98: "RA",
    109: "RA", 197: "Sigma", 198: "Sigma", 220: "ResMultiple",
    268: "ResSeg", 309: "RAB",
}

for id_pac, proc in correccion_procedimiento.items():
    df.loc[df["ID_Paciente"] == id_pac, "Tipo_Procedimiento"] = proc

df = df[df["ID_Paciente"] != 220].reset_index(drop=True)
df["Tipo_Procedimiento"] = df["Tipo_Procedimiento"].fillna("ResLoc")
df["Tipo_Procedimiento"] = df["Tipo_Procedimiento"].astype("string")

# ──────────────────────────────────────────────
# 16. Eliminación de columnas poco representativas
# ──────────────────────────────────────────────
cols_drop = [
    "Dependencia_Funcional", "Localizacion_sigma",
    "Localizacion_ascendente", "Localizacion_recto",
    "Complicacion_TEP", "Invasion_Vascular",
]
df = df.drop(columns=cols_drop)

# ──────────────────────────────────────────────
# 17. Imputación de IMC con KNN
# ──────────────────────────────────────────────
predictoras_imc = [
    "Edad_Cirugia", "Clasificacion_ASA", "Hta", "Dm2", "Dl",
    "Epoc", "Sexo", "Estado_Fumador", "Cardiopatia_Isquemica",
]

cols_imputacion = ["IMC"] + predictoras_imc
df_sub = df[cols_imputacion].copy()

means, stds = {}, {}
for col in predictoras_imc:
    col_data = df_sub[col].dropna()
    means[col] = col_data.mean()
    stds[col] = col_data.std()
    df_sub[col] = (df_sub[col] - means[col]) / stds[col]

imputer = KNNImputer(n_neighbors=5)
imputed_scaled = imputer.fit_transform(df_sub)
df_imputed_scaled = pd.DataFrame(imputed_scaled, columns=cols_imputacion)

for col in predictoras_imc:
    df_imputed_scaled[col] = df_imputed_scaled[col] * stds[col] + means[col]

df_imputed_scaled.index = df.index
df["IMC_imputado"] = df["IMC"]
mask_nan = df["IMC"].isna()
df.loc[mask_nan, "IMC_imputado"] = df_imputed_scaled.loc[mask_nan, "IMC"].round(2)

df.rename(columns={"IMC_imputado": "IMC_def"}, inplace=True)
df = df.drop(columns=["IMC"])

# ──────────────────────────────────────────────
# 18. Reordenación final de columnas
# ──────────────────────────────────────────────
variables_ordenadas = [
    "ID_Paciente", "Sexo", "Edad_Cirugia", "IMC_def",
    "Estado_Fumador", "Clasificacion_ASA",
    "Dm2", "Hta", "Dl", "Cardiopatia_Isquemica", "Epoc", "Irc", "Inmunosupresion",
    "Localizacion", "Tipo_Procedimiento",
    "Estadio_T_Patologico", "Estadio_N", "Estadio_M", "Estadio_Global_TNM",
    "Ratio_ganglios", "Grado_Reseccion_R",
    "Neoadyuvancia", "Abordaje_Quirurgico", "Tipo_Intervencion",
    "Anastomosis_Realizada", "Estoma_Realizado",
    "Quimioterapia_Adyuvante_si", "Finalizacion_Prematura_QT_si", "Toxicidad_QT_si",
    "Complicaciones_Ingreso_si", "Complicacion_fuga_anastomotica",
    "Complicacion_infeccion_sitio_quirurgico", "Complicacion_neumonia",
    "Complicacion_ileo_paralitico", "Complicacion_insuficiencia_renal",
    "Reintervencion_si", "Reingreso_30d_si", "Estancia_Dias",
    "Supervivencia_Dias", "Libre_Recidiva_Dias", "Exitus_def", "Recidiva_def",
]

df = df.loc[:, variables_ordenadas].copy()

# ──────────────────────────────────────────────
# 19. Métricas generales
# ──────────────────────────────────────────────
print("\n===== MÉTRICAS GENERALES =====")
print(f"Filas: {df.shape[0]}")
print(f"Columnas: {df.shape[1]}")
print("Duplicados por fila completa:", df.duplicated().sum())
if "ID_Paciente" in df.columns:
    print("Duplicados en ID_Paciente:", df["ID_Paciente"].duplicated().sum())
total_cells = df.shape[0] * df.shape[1]
total_nulos = df.isna().sum().sum()
print(f"Nulos totales: {total_nulos} ({(total_nulos / total_cells) * 100:.2f}%)")

# ──────────────────────────────────────────────
# 20. Traducción a inglés
# ──────────────────────────────────────────────
translation = {
    "ID_Paciente": "Patient_ID", "Sexo": "Sex", "Edad_Cirugia": "Age",
    "IMC_def": "BMI", "Clasificacion_ASA": "ASA", "Dm2": "T2DM",
    "Hta": "HTN", "Cardiopatia_Isquemica": "IHD", "Epoc": "COPD",
    "Estado_Fumador": "Smoking_Status", "Irc": "CKD",
    "Ratio_ganglios": "Node_ratio", "Anastomosis_Realizada": "Anastomosis",
    "Inmunosupresion": "IS", "Localizacion": "Site",
    "Estadio_T_Patologico": "T_stage", "Estadio_N": "N_stage",
    "Estadio_M": "M_stage", "Estadio_Global_TNM": "TNM_stage",
    "Grado_Reseccion_R": "R_grade", "Neoadyuvancia": "NAT",
    "Abordaje_Quirurgico": "Approach", "Tipo_Intervencion": "Procedure",
    "Estoma_Realizado": "Stoma_created",
    "Quimioterapia_Adyuvante_si": "Adjuvant_CT",
    "Finalizacion_Prematura_QT_si": "Early_end_CT",
    "Toxicidad_QT_si": "CT_toxicity",
    "Complicaciones_Ingreso_si": "Complications",
    "Complicacion_fuga_anastomotica": "Leak",
    "Complicacion_infeccion_sitio_quirurgico": "SSI",
    "Complicacion_neumonia": "Pneumonia",
    "Complicacion_ileo_paralitico": "Paralytic_ileus",
    "Complicacion_insuficiencia_renal": "Kidney_failure",
    "Reintervencion_si": "Reintervention",
    "Reingreso_30d_si": "Readmission",
    "Estancia_Dias": "LOS_days",
    "Supervivencia_Dias": "Survival_days",
    "Libre_Recidiva_Dias": "Recurrence_free_days",
    "Exitus_def": "Death", "Recidiva_def": "Recurrence",
    "Dl": "Dl", "Tipo_Procedimiento": "Surgery_type",
    # Valores (no columnas)
    "HCD": "RHC", "HCDa": "ERHC", "HCl": "LHC", "sigma": "SigR",
    "RA": "AR", "RAB": "LAR", "RAUB": "ULAR", "CST": "STC",
    "CT": "TC", "ResSeg": "SegR", "ResLoc": "LE", "AAP": "APR",
    "recto": "Rectum", "izquierdo": "Left", "derecho": "Right",
    "abierto": "Open", "laparoscópico finalizado": "Finished Laparoscopic",
    "laparoscópico convertido": "Converted Laparoscopic",
    "programada": "Elective", "urgente": "Urgent",
    "LL mecánica": "SS Stapled", "LT mecánica": "SE Stapled",
    "TL mecánica": "ES Stapled", "TT mecánica": "EE Stapled",
    "LL manual": "SS Hand-Sewn", "TT manual": "EE Hand-Sewn",
    "si": "Yes", "multiple": "Multiple",
}

col_translation = {k: v for k, v in translation.items() if k in df.columns}
val_translation = {k: v for k, v in translation.items() if k not in df.columns}

df.rename(columns=col_translation, inplace=True)

for old, new in val_translation.items():
    df.replace(to_replace=rf"(?i)^{old}$", value=new, regex=True, inplace=True)

# ──────────────────────────────────────────────
# 21. Exportación
# ──────────────────────────────────────────────
df.to_parquet("./data/preprocesada/final_dataset.parquet", index=False)
df.to_excel("./data/preprocesada/final_dataset.xlsx", index=False, engine="openpyxl")

print("\n✔ Exportado: final_dataset.parquet y final_dataset.xlsx")