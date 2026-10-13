import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
from matplotlib.ticker import MultipleLocator


# PRELIMINAR + FIDELIDAD
from scipy.stats import chi2, chi2_contingency, spearmanr, ks_2samp
from sklearn.decomposition import PCA
from scipy.stats import zscore
from sdv.evaluation.single_table import get_column_plot
from sdv.metadata import SingleTableMetadata

# UTILIDAD
from sklearn.model_selection import train_test_split
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, recall_score
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test
import matplotlib.ticker as mticker
from sklearn.preprocessing import LabelEncoder
 
# PRIVACIDAD
from sklearn.dummy import DummyClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

from matplotlib.colors import to_rgb, to_hex
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter

from matplotlib.legend_handler import HandlerBase


class MultiColorHandler(HandlerBase):
        def __init__(self, colors, **kwargs):
            self.colors = colors
            super().__init__(**kwargs)

        def create_artists(self, legend, orig_handle,
                           xdescent, ydescent, width, height, fontsize, trans):
            n = len(self.colors)
            w = width / n
            patches = []
            for i, c in enumerate(self.colors):
                patches.append(plt.Rectangle(
                    (xdescent + i * w, ydescent - height * 0.15),
                    w, height,
                    facecolor=c, edgecolor="white", linewidth=0.5,
                    transform=trans,
                ))
            return patches
        


def plot_pca_overlap(DFR, DFS, algoritmo, colores):

    DFR = DFR.copy()
    DFS = DFS.copy()

    DFR['_source'] = 'real'
    DFS['_source'] = 'synth'
    
    DFR = DFR.drop(columns=["Patient_ID"], errors="ignore")
    DFS = DFS.drop(columns=["Patient_ID"], errors="ignore")

    combined_raw = pd.concat([DFR, DFS], ignore_index=True)
    combined_encoded = pd.get_dummies(combined_raw.drop(columns='_source'),
                                      dummy_na=True).fillna(0).apply(pd.to_numeric)
    n_real = len(DFR)
    scaler = StandardScaler()
    combined_scaled = scaler.fit_transform(combined_encoded)
    pca = PCA(n_components=2)
    components = pca.fit_transform(combined_scaled)
    var_explained = pca.explained_variance_ratio_ * 100

    xlim, ylim = (-8, 4), (-8, 19)
    color_real  = '#6B7280'       # gris neutro
    color_synth = colores[algoritmo]

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharey=True)
    #fig.suptitle(f"PCA — Real vs {algoritmo}", fontsize=15, fontweight='bold', y=1.02)

    scatter_kw_real  = dict(alpha=0.35, s=55, color=color_real,  edgecolors='none', rasterized=True)
    scatter_kw_synth = dict(alpha=0.35, s=55, color=color_synth, edgecolors='none', rasterized=True)

    titles   = ["Real", algoritmo, "Combined"]
    datasets = [
        [(components[:n_real], scatter_kw_real)],
        [(components[n_real:], scatter_kw_synth)],
        [(components[:n_real], {**scatter_kw_real,  'label': 'Real'}),
         (components[n_real:], {**scatter_kw_synth, 'label': algoritmo})],
    ]

    for ax, title, groups in zip(axes, titles, datasets):
        for data, kw in groups:
            ax.scatter(data[:, 0], data[:, 1], **kw)

        ax.set_title(title, fontsize=13, fontweight='bold', pad=10)
        ax.set_xlabel(f"PC1 ({var_explained[0]:.1f}% variance)", fontsize=11)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.tick_params(labelsize=9)
        ax.spines[['top', 'right']].set_visible(False)

        if title == "Combined":
            ax.legend(fontsize=11, framealpha=0.7, markerscale=1.8, frameon=True, facecolor='white', edgecolor='#E5E7EB')

    axes[0].set_ylabel(f"PC2 ({var_explained[1]:.1f}% variance)", fontsize=11)

    plt.tight_layout()
    plt.savefig('../plots/pca_'+str(algoritmo)+'.png', dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()


def corr_heatmap(DFR, DFS, algoritmo):
    DFR = DFR.copy()
    DFS = DFS.copy()

    DFR = DFR.drop(columns=["Patient_ID"], errors="ignore")
    DFS = DFS.drop(columns=["Patient_ID"], errors="ignore")

    DFR_numeric_all = DFR.select_dtypes(include=np.number)
    DFS_numeric_all = DFS.select_dtypes(include=np.number)
    common_numeric_cols = list(set(DFR_numeric_all.columns) & set(DFS_numeric_all.columns))
    DFR_numeric = DFR_numeric_all[common_numeric_cols]
    DFS_numeric = DFS_numeric_all[common_numeric_cols]

    real_corr  = DFR_numeric.corr(method='spearman').fillna(0)
    synth_corr = DFS_numeric.corr(method='spearman').fillna(0)
    diff_corr  = np.abs(real_corr - synth_corr)

    short_cols = {c: c[:10] for c in common_numeric_cols}
    diff_corr  = diff_corr.rename(columns=short_cols, index=short_cols)

    plt.style.use('seaborn-v0_8-white')
    fig, ax = plt.subplots(figsize=(8, 7))

    sns.heatmap(diff_corr, ax=ax,
                cmap='YlOrRd', vmin=0, vmax=0.5,
                linewidths=0.1, linecolor='#e5e7eb',
                cbar_kws={'shrink': 0.6, 'label': '|Δr|'},
                xticklabels=False)   # ← elimina eje X

    ax.tick_params(axis='y', rotation=0, labelsize=8)

    ax.set_title(f"{algoritmo} — Correlation diff  (mean: {diff_corr.values.mean():.4f})",
                 fontsize=13, fontweight='bold', pad=12)
    ax.tick_params(axis='x', rotation=90, labelsize=6)
    ax.tick_params(axis='y', rotation=0,  labelsize=9)
    ax.spines[['top', 'right', 'bottom', 'left']].set_visible(False)

    plt.tight_layout()
    plt.savefig(f"../plots/corr_{algoritmo}.png", dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()

    return fig, real_corr, synth_corr, diff_corr

def plot_continuos_all(DFR, dfs_synth, colores, n_cols, n_rows, columnas_continuas):

    dict_x_label = {
    "Age":                  "Years",
    "BMI":                  "kg/m²",
    "Node_ratio":           "Ratio",
    "LOS_days":             "Days",
    "Survival_days":        "Days",
    "Recurrence_free_days": "Days",
    }



    sns.set_theme(style="white")

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 3.5))
    axes = axes.flatten()

    for i, col in enumerate(columnas_continuas):
        ax = axes[i]
        real_data  = pd.to_numeric(DFR[col].astype(str).str.replace(',', '.'),  errors='coerce').dropna()

        sns.kdeplot(real_data,  ax=ax,
                    color='grey',             fill=True, alpha=0.6,
                    label='Real',
                    cut=0, clip=(real_data.min(),  real_data.max()))
        
        #ax.axvline(real_data.mean(),  color='grey',             linestyle='--', linewidth=1.2, alpha=0.8)

        for algoritmo, DFS in dfs_synth.items():
            synth_data = pd.to_numeric(DFS[col].astype(str).str.replace(',', '.'),errors='coerce').dropna()

            sns.kdeplot(synth_data, ax=ax,
                    color=colores[algoritmo], fill=False, linewidth=1.1,
                    label=algoritmo,
                    cut=0, clip=(synth_data.min(), synth_data.max()))
        
            #ax.axvline(synth_data.mean(), color=colores[algoritmo], linestyle='--', linewidth=1.2, alpha=0.8)

        title = str(col).split('_')

        if len(title) > 1:
            if title[1] == 'days':
                title = title[0]
                pass
            else:
                title = str(title[0] + str(' ') + title[1].capitalize())
        
        else:
            title = title[0]

        ax.set_title(title, fontsize=10, fontweight='bold', pad=6)
        ax.set_xlabel(dict_x_label.get(col, col), fontsize=8)
        ax.set_ylabel('Density', fontsize=8)
        ax.tick_params(labelsize=8)
        ax.grid(False)
        if i == 1:
            ax.legend(
                fontsize=10,
                loc='upper right',
                frameon=True,
                facecolor='white',
                edgecolor='#E5E7EB'
            )

        else:
            ax.legend().remove()
        ax.spines[['top', 'right']].set_visible(False)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    #fig.suptitle(f"Continuous distributions — Real vs Synthetic", fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(f"../plots/kde_subplot_all.png", dpi=150, bbox_inches='tight')
    plt.show()


def plot_hist_all(DFR, dfs_synth, colores, cols_hist, name_save, n_cols, n_legend, degree, position, font_size):

    tick_label_map = {
    "Sex":           {0: "Male", 1: "Female"},
    "Complications": {0: "No", 1: "Yes"},
    "Death":         {0: "No", 1: "Yes"},
    "Recurrence":    {0: "No", 1: "Yes"},
}

    
    n_rows = math.ceil(len(cols_hist) / n_cols)

    # Convertir floats a Int64 para categoricas
    for col in cols_hist:
        if pd.api.types.is_float_dtype(DFR[col]):
            DFR[col]  = DFR[col].astype(pd.Int64Dtype())
        
        for algoritmo, DFS in dfs_synth.items():
            if pd.api.types.is_float_dtype(DFS[col]):
                DFS[col] = DFS[col].astype(pd.Int64Dtype())

    sns.set_theme(style="white")
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 3.5))
    axes = axes.flatten()

    colrs_list = list(colores.values())
    color_list = ['grey'] + colrs_list

    for i, col in enumerate(cols_hist):
        ax = axes[i]

        # Calcular % por categoría para cada dataframe
        real_pct  = DFR[col].value_counts(normalize=True).mul(100).rename('Real')

        for algoritmo, DFS in dfs_synth.items():

            synth_pct = DFS[col].value_counts(normalize=True).mul(100).rename(algoritmo)

            # Unir para que tengan las mismas categorías
            pct_df = pd.concat([real_pct, synth_pct], axis=1).fillna(0).sort_index()
            real_pct = pct_df

    
        pct_df.plot(kind='bar', ax=ax,
                    color=color_list,
                    alpha=0.95, edgecolor='none', width=0.7)

        title = str(col).split('_')

        if len(title) > 1:
            title = str(title[0] + str(' ') + title[1].capitalize())
        
        else:
            title = title[0]
        
        if col in tick_label_map:
            ax.set_xticklabels([tick_label_map[col].get(v, v) for v in pct_df.index])

        ax.set_title(title, fontsize=10, fontweight='bold', pad=6)
        ax.set_ylabel('%', fontsize=8)
        ax.set_xlabel('')
        ax.tick_params(axis='x', rotation=degree, labelsize=8)
        ax.tick_params(axis='y', labelsize=8)
        if i == n_legend:
            ax.legend(
                fontsize=font_size,
                loc=f'upper {position}',
                frameon=True,
                facecolor='white',
                edgecolor='#E5E7EB'
            )

        else:
            ax.legend().remove()
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(True, axis='y', which='major', linestyle='--', linewidth=0.9, alpha=0.8, color="#D8D8D8")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    #fig.suptitle(f"Site & Surgery Type — Real vs Synthetic", fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(f"../plots/{name_save}.png", dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()


def plot_bin_all(DFR, dfs_synth, colores, columnas_categoricas, cols_hist):

    cols_bin = [col for col in columnas_categoricas if col not in cols_hist]
    cols_bin.remove('Patient_ID')

    n_cols = 4
    n_rows = math.ceil(len(cols_bin) / n_cols)

    sns.set_theme(style="darkgrid")
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3.5))
    axes = axes.flatten()

    colrs_list = list(colores.values())
    color_list = ['grey'] + colrs_list

    for i, col in enumerate(cols_bin):
        ax = axes[i]

        real_pct  = DFR[col].value_counts(normalize=True).mul(100).rename('Real')

        for algoritmo, DFS in dfs_synth.items():
            synth_pct = DFS[col].value_counts(normalize=True).mul(100).rename(algoritmo)

            pct_df = pd.concat([real_pct, synth_pct], axis=1).fillna(0).sort_index()
            real_pct = pct_df


        pct_df.plot(kind='bar', ax=ax,
                    color=color_list,
                    alpha=0.7, edgecolor='none', width=0.6)

        ax.set_title(col, fontsize=10, fontweight='bold', pad=6)
        ax.set_ylabel('%', fontsize=8)
        ax.set_xlabel('')
        ax.set_ylim(0, 100)
        ax.tick_params(axis='x', rotation=0, labelsize=9)
        ax.tick_params(axis='y', labelsize=8)
        if i == 0:
            ax.legend(fontsize=8)
        else:
            ax.legend().remove()
        ax.spines[['top', 'right']].set_visible(False)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f"Binary distributions — Real vs Synthetic", fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(f"../plots/bin_subplot_all.png", dpi=150, bbox_inches='tight')
    plt.show()
    plt.close()

#-------------------------------------------------
# UTILIDAD
#-------------------------------------------------

def load_and_clean(path, ID_COL, drop_id=True):
    df = pd.read_parquet(path)
    if drop_id and ID_COL in df.columns:
        df = df.drop(columns=[ID_COL])
    return df
 
def fit_target_encoder(series):
    le = LabelEncoder()
    le.fit(series.astype(str))
    return le

def transform_target(series, encoder):
    return encoder.transform(series.astype(str))

def one_hot_align_three(X_train, X_test, X_synth):
    X_all = pd.concat([X_train, X_test, X_synth], axis=0)
    X_all_ohe = pd.get_dummies(X_all)

    n_train = len(X_train)
    n_test = len(X_test)

    X_train_ohe = X_all_ohe.iloc[:n_train].reset_index(drop=True)
    X_test_ohe = X_all_ohe.iloc[n_train:n_train + n_test].reset_index(drop=True)
    X_synth_ohe = X_all_ohe.iloc[n_train + n_test:].reset_index(drop=True)

    return X_train_ohe, X_test_ohe, X_synth_ohe

 
 
def train_and_score(X_train, y_train, X_test, y_test):
    """Entrena RF y devuelve AUC."""
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=7,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)

    y_prob = rf.predict_proba(X_test)
    y_pred = rf.predict(X_test)

    if y_prob.shape[1] == 2:
        auc = roc_auc_score(y_test, y_prob[:, 1])
    else:
        auc = roc_auc_score(y_test, y_prob, multi_class="ovr", average="weighted")

    return {
        "AUC":    round(auc, 4),
        "F1":     round(f1_score(y_test, y_pred, average="weighted"), 4),
        "Recall": round(recall_score(y_test, y_pred, average="weighted"), 4),
    }

def darken_color(color, factor=0.75):
    """
    Oscurece un color manteniendo la misma gama.
    factor < 1 oscurece; cuanto menor, más oscuro.
    """
    rgb = np.array(to_rgb(color))
    darker = rgb * factor
    return to_hex(np.clip(darker, 0, 1))
 
 
def compute_utility(df_train, df_test, df_synth, name, TARGET_COL):
    """Calcula TRTR, TSTR y TRTS para un dataset sintético."""
    encoder = fit_target_encoder(df_train[TARGET_COL])

    y_train = transform_target(df_train[TARGET_COL], encoder)
    y_test = transform_target(df_test[TARGET_COL], encoder)
    y_synth = transform_target(df_synth[TARGET_COL], encoder)

    X_train_raw = df_train.drop(columns=[TARGET_COL])
    X_test_raw = df_test.drop(columns=[TARGET_COL])
    X_synth_raw = df_synth.drop(columns=[TARGET_COL])

    # TRTR: Train Real -> Test Real
    Xtr, Xte, Xs = one_hot_align_three(X_train_raw, X_test_raw, X_synth_raw)

    trtr = train_and_score(Xtr, y_train, Xte, y_test)
    tstr = train_and_score(Xs, y_synth, Xte, y_test)
    trts = train_and_score(Xtr, y_train, Xs, y_synth)

    row = {"Algorithm": name}
    for metric in trtr:
        row[f"TRTR_{metric}"] = trtr[metric]
        row[f"TSTR_{metric}"] = tstr[metric]
        row[f"TRTS_{metric}"] = trts[metric]

    return row
 
def plot_utility(results, colores, save_path="../plots/utility_auc.png"):
    sns.set_theme(style="white")
    df = pd.DataFrame(results).set_index("Algorithm")
    baseline = df["TRTR_AUC"].iloc[0]

    x = np.arange(len(df))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 7))

    for i, model_name in enumerate(df.index):
        base_color = colores.get(model_name)
        dark_color = darken_color(base_color, factor=0.45)

        ax.bar(
            x[i] - width / 2,
            df.loc[model_name, "TRTS_AUC"],
            width,
            color=base_color,
            alpha=0.95,
            edgecolor="white",
            linewidth=0.6,
            label="TRTS" if i == 0 else None,
        )

        ax.bar(
            x[i] + width / 2,
            df.loc[model_name, "TSTR_AUC"],
            width,
            color=dark_color,
            alpha=0.95,
            edgecolor="white",
            linewidth=0.6,
            label="TSTR" if i == 0 else None,
        )

    ax.axhline(
        baseline,
        color="#7A7676",
        linewidth=2,
        linestyle="--",
        label="TRTR baseline",
    )

    #ax.set_title("Utility Evaluation by Method", fontsize=14, fontweight="bold", pad=14)
    ax.set_xlabel("Model", fontsize=11)
    ax.set_ylabel("AUC Score", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(df.index, rotation=0, ha="center", fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))



    light_colors = [colores[m] for m in df.index]
    dark_colors  = [darken_color(colores[m], factor=0.45) for m in df.index]

    h_trts = Patch(label="TRTS")
    h_tstr = Patch(label="TSTR")
    h_base = Line2D([0], [0], color="#7A7676", linewidth=2, linestyle="--", label="TRTR baseline")

    ax.legend(
        handles=[h_trts, h_tstr, h_base],
        handler_map={
            h_trts: MultiColorHandler(light_colors),
            h_tstr: MultiColorHandler(dark_colors),
        },
        fontsize=10, frameon=True, framealpha=1,
        edgecolor="black", fancybox=False, loc="lower right",
        borderpad=0.8, labelspacing=0.6,
        handlelength=3.5,   # ancho total (default ~2)
        handleheight=1.2,   # alto (default ~0.7)
    )

    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, axis='y', which='major', linestyle='--', linewidth=0.9, alpha=0.8, color="#D8D8D8")



    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()

#---------------------------------------------------------------
# KM 
#--------------------------------------------------------------

def prepare_survival(df, time_col, event_col):
    tmp = df[[time_col, event_col]].copy()

    for col in (time_col, event_col):
        tmp[col] = pd.to_numeric(tmp[col], errors="coerce")

    tmp = tmp.dropna()
    tmp = tmp[tmp[time_col] >= 0]

    return tmp, [{
        "name": "survival",
        "time_col": time_col,
        "event_col": event_col
    }]


def km_analysis(DFR, dfs_synth, time_col, event_col, colores):
    sns.set_theme(style="white")
    real_surv, _ = prepare_survival(DFR, time_col, event_col)
    km_real = KaplanMeierFitter(label="Real").fit(
        real_surv[time_col],
        event_observed=real_surv[event_col])

    fig, ax = plt.subplots(figsize=(8.5, 5.2))

    km_real.plot_survival_function(
    ax=ax,
    ci_show=False,
    color="grey",
    linewidth=2.5,
    linestyle="-")

    for algoritmo, DFS in dfs_synth.items():
        syn_surv, _ = prepare_survival(DFS, time_col, event_col)
        km_syn = KaplanMeierFitter(label=algoritmo).fit(
            syn_surv[time_col],
            event_observed=syn_surv[event_col])

        km_syn.plot_survival_function(
            ax=ax,
            ci_show=False,
            color=colores.get(algoritmo, "0.6"),
            linewidth=1.5,
            linestyle="--",
            alpha=0.8)


    #ax.set_title("Kaplan-Meier: Real vs Synthetic", fontsize=14, pad=12)
    ax.set_xlabel('Days', fontsize=11)
    ax.set_ylabel('Estimated survival', fontsize=11)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)

    ax.tick_params(axis="both", labelsize=10, width=0.8)
    
    ax.grid(True, axis='x', which='major', linestyle='--', linewidth=0.9, alpha=0.8, color="#D8D8D8")



    ax.legend(
        fontsize=12,
        loc=f'upper right',
        frameon=True,
        facecolor='white',
        edgecolor='#E5E7EB'
    )
    plt.tight_layout()
    plt.savefig("../plots/km_all.png", dpi=300, bbox_inches="tight")
    plt.show()


# --------------------------------------
# Restricciones clinicas 



def concordance(df, label):
    
    valid_surg = {
    "Right":   {"RHC", "ERHC"},
    "Left": {"LHC", "SigR", "SegR"},
    "Rectum":     {"APR", "AR", "LAR", "LE", "ULAR"},
    "Multiple":  {"STC", "TC"}}
        
    df["_conc"] = df.apply(
        lambda r: r["Surgery_type"] in valid_surg.get(r["Site"], set()),
        axis=1
    )
    rates = df.groupby("Site")["_conc"].mean()
    df.drop(columns="_conc", inplace=True)

    return rates.rename(label)


def plot_concordance_by_site(DFR, dfs_synth, colores):
    frames = [concordance(DFR, 'Real')]

    for name, DFS in dfs_synth.items():
        frames.append(concordance(DFS, name))
    
    conc = pd.concat(frames, axis=1).fillna(0)
    color_list = ['grey'] + [colores[m] for m in dfs_synth.keys()]
    
    conc.plot.bar(figsize=(10, 5), ylim=(0, 1.05), color=color_list,
            alpha=0.95, edgecolor='none', width=0.7)
    plt.axhline(1.0, color="black", ls="--", lw=1)
    plt.tick_params(axis='x', rotation=0, labelsize=10)
    plt.ylabel("Concordance rate")
    
    plt.legend(fontsize=11, loc="lower right", frameon=True, facecolor='white', edgecolor='#E5E7EB', framealpha=1)
    plt.grid(True, axis='y', which='major', linestyle='--', linewidth=0.9, alpha=0.8, color="#D8D8D8")
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)

                        
    plt.tight_layout()
    plt.savefig(f"../plots/concordance_by_site.png", dpi=150, bbox_inches='tight')
    plt.show()

