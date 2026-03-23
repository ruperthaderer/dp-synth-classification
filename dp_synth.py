"""
dp_synth – Differentially Private Synthetic Data Generation & Evaluation

This module implements a marginal-based DP synthetic data pipeline:

    Phase 1  – Build noisy joint histograms P(attribute, class) via the
               Laplace mechanism.
    Phase 2  – Derive class probabilities, conditional attribute
               distributions, sample attribute vectors, and assemble a
               synthetic DataFrame.

It also provides:
    • A classification evaluation pipeline (RandomForest by default).
    • An epsilon-sweep study across multiple datasets.
    • A PrivBayes baseline via the DataSynthesizer library.
    • Plotting utilities for accuracy comparisons, epsilon curves,
      confusion matrices, and feature-distribution comparisons.
"""

from __future__ import annotations

import os
import ast
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# ====================================================================
# 1. DIFFERENTIAL PRIVACY PRIMITIVES
# ====================================================================

def laplace_mech(
    value: float | np.ndarray | pd.Series,
    sensitivity: float,
    epsilon: float,
) -> float | np.ndarray | pd.Series:
    """Add Laplace noise calibrated to (sensitivity / epsilon)."""
    return value + np.random.laplace(loc=0, scale=sensitivity / epsilon)


# ====================================================================
# 2. SYNTHETIC DATA GENERATION
# ====================================================================

def _build_noisy_histograms(
    df: pd.DataFrame,
    class_name: str,
    attribute_names: list[str],
    epsilon: float,
) -> dict[str, pd.Series]:
    """
    Phase 1: For each attribute, compute the joint histogram
    P(attribute, class) and perturb it with Laplace noise.

    The total privacy budget *epsilon* is split equally across all
    attributes (sequential composition).
    """
    epsilon_per_hist = epsilon / len(attribute_names)
    histograms: dict[str, pd.Series] = {}

    for attr in attribute_names:
        counts = df[[attr, class_name]].value_counts()
        noisy = laplace_mech(counts, sensitivity=1, epsilon=epsilon_per_hist)
        noisy.clip(lower=0.0, inplace=True)
        histograms[attr] = noisy

    return histograms


def _compute_class_distribution(
    histograms: dict[str, pd.Series],
    n_tuples: int,
) -> tuple[dict, dict, dict, dict]:
    """
    Aggregate noisy histograms to obtain:
      • class_totals  – total noisy mass per class
      • p             – P(C = c)
      • class_tuples  – number of synthetic rows per class
      • class_vector  – list of repeated class labels per class
    """
    class_totals: dict[str, float] = {}

    for hist in histograms.values():
        for (attr_val, class_val), count in hist.items():
            class_totals[class_val] = class_totals.get(class_val, 0.0) + count

    total = sum(class_totals.values())

    if total == 0:
        # Fallback: uniform if noise zeroed everything out
        uniform = 1.0 / len(class_totals)
        p = {c: uniform for c in class_totals}
    else:
        p = {c: class_totals[c] / total for c in class_totals}

    class_tuples = {c: round(n_tuples * p[c]) for c in class_totals}
    class_vector = {c: [c] * class_tuples[c] for c in class_tuples}

    return class_totals, p, class_tuples, class_vector


def _compute_conditional_attributes(
    histograms: dict[str, pd.Series],
    class_totals: dict,
) -> dict[str, dict]:
    """
    Compute P(attribute = a | class = c) for every attribute and class,
    using the noisy histogram counts from Phase 1.
    """
    cond_attr: dict[str, dict] = {}

    for attr_name, hist in histograms.items():
        cond_attr[attr_name] = {}

        for c in class_totals:
            attr_counts: dict[str, float] = {}

            for (attr_val, class_val), count in hist.items():
                if class_val == c:
                    attr_counts[attr_val] = attr_counts.get(attr_val, 0.0) + count

            total_c = sum(attr_counts.values())

            if total_c == 0:
                if attr_counts:
                    uniform = 1.0 / len(attr_counts)
                    probs = {a: uniform for a in attr_counts}
                else:
                    probs = {}
            else:
                probs = {a: attr_counts[a] / total_c for a in attr_counts}

            cond_attr[attr_name][c] = probs

    return cond_attr


def _sample_attribute_vectors(
    cond_attr: dict[str, dict],
    class_tuples: dict,
    random_state: Optional[int] = None,
) -> dict[str, dict]:
    """
    For each attribute and class, draw a vector of attribute values
    with length n_c = class_tuples[c] according to P(attr | class).

    Strategy: fill deterministically via rounded target counts, then
    fix length mismatches by probabilistic sampling / removal.
    """
    if random_state is not None:
        np.random.seed(random_state)

    attr_vectors: dict[str, dict] = {}

    for attr_name, class_dict in cond_attr.items():
        attr_vectors[attr_name] = {}

        for c, probs in class_dict.items():
            n_c = class_tuples[c]

            if not probs:
                attr_vectors[attr_name][c] = [None] * n_c
                continue

            # Deterministic fill
            target_count = {
                attr_val: round(p_val * n_c) for attr_val, p_val in probs.items()
            }
            vec: list = []
            for attr_val, cnt in target_count.items():
                vec.extend([attr_val] * cnt)

            # Fix length: add or remove to match n_c exactly
            diff = n_c - len(vec)
            if diff > 0:
                vals = list(probs.keys())
                extra = np.random.choice(vals, size=diff, p=list(probs.values()))
                vec.extend(extra.tolist())
            elif diff < 0:
                remove_idx = np.random.choice(len(vec), size=-diff, replace=False)
                for idx in sorted(remove_idx, reverse=True):
                    vec.pop(idx)

            np.random.shuffle(vec)
            attr_vectors[attr_name][c] = vec

    return attr_vectors


def _assemble_synthetic_dataframe(
    attr_vectors: dict[str, dict],
    class_vector: dict,
    attribute_names: list[str],
    class_name: str,
) -> pd.DataFrame:
    """
    Combine per-class attribute vectors and class vectors into a single
    synthetic DataFrame.
    """
    df_blocks: list[pd.DataFrame] = []

    for c, class_vec in class_vector.items():
        n_c = len(class_vec)
        block: dict[str, list] = {}

        for attr in attribute_names:
            values = attr_vectors[attr][c]
            # Safety: trim or pad to n_c
            if len(values) < n_c:
                values = values + [values[-1]] * (n_c - len(values))
            elif len(values) > n_c:
                values = values[:n_c]
            block[attr] = values

        block[class_name] = class_vec
        df_blocks.append(pd.DataFrame(block))

    return pd.concat(df_blocks, ignore_index=True)


def generate(
    df: pd.DataFrame,
    n_tuples: Optional[int] = None,
    class_col: Optional[int] = None,
    epsilon: float = 1.0,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Generate a differentially private synthetic dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Original dataset. The last column is treated as the class
        variable unless *class_col* is specified.
    n_tuples : int, optional
        Number of synthetic rows (default: same as df).
    class_col : int, optional
        Column index of the class variable (default: last column).
    epsilon : float
        Overall privacy budget.
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Synthetic dataset with the same schema as *df*.
    """
    if n_tuples is None:
        n_tuples = df.shape[0]
    if class_col is None:
        class_col = df.shape[1] - 1

    class_name = df.columns[class_col]
    attribute_names = [col for col in df.columns if col != class_name]

    # Phase 1: noisy histograms
    histograms = _build_noisy_histograms(df, class_name, attribute_names, epsilon)

    # Phase 2: derive distributions, sample, assemble
    class_totals, p, class_tuples, class_vector = _compute_class_distribution(
        histograms, n_tuples
    )
    cond_attr = _compute_conditional_attributes(histograms, class_totals)
    attr_vectors = _sample_attribute_vectors(
        cond_attr, class_tuples, random_state=random_state
    )
    synthetic_df = _assemble_synthetic_dataframe(
        attr_vectors, class_vector, attribute_names, class_name
    )

    # Final row shuffle
    synthetic_df = synthetic_df.sample(
        frac=1, random_state=random_state
    ).reset_index(drop=True)

    return synthetic_df


# ====================================================================
# 3. CLASSIFICATION EVALUATION
# ====================================================================

def build_pipeline(clf=None):
    """
    Return a factory function ``make_model(df) -> fitted Pipeline``
    that OneHotEncodes all feature columns and feeds into *clf*
    (default: RandomForestClassifier).
    """
    if clf is None:
        clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)

    def make_model(df: pd.DataFrame) -> Pipeline:
        feature_cols = df.columns[:-1]
        preprocessor = ColumnTransformer(
            transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), feature_cols)]
        )
        return Pipeline(steps=[("prep", preprocessor), ("clf", clf)])

    return make_model


def evaluate_df(
    train_df: pd.DataFrame,
    test_df: Optional[pd.DataFrame] = None,
    n_splits: int = 10,
    base_random_state: int = 0,
) -> dict:
    """
    Train a RandomForest on *train_df* and evaluate on *test_df*
    (or on *train_df* itself if test_df is None).

    Repeats *n_splits* times with different random seeds to capture
    variance from the classifier's randomness.

    Returns
    -------
    dict with keys: ``classes``, ``accuracies`` (np.ndarray),
    ``confusion_matrices`` (list of np.ndarray).
    """
    X_train = train_df.iloc[:, :-1]
    y_train = train_df.iloc[:, -1]

    if test_df is None:
        X_test, y_test = X_train, y_train
    else:
        X_test = test_df.iloc[:, :-1]
        y_test = test_df.iloc[:, -1]

    classes = np.unique(pd.concat([y_train, y_test], axis=0))
    accuracies: list[float] = []
    confusion_matrices: list[np.ndarray] = []

    for i in range(n_splits):
        rs = base_random_state + i
        clf = RandomForestClassifier(n_estimators=200, random_state=42 + rs, n_jobs=-1)
        make_model = build_pipeline(clf)
        model = make_model(train_df)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        accuracies.append(float((y_pred == y_test).mean()))
        confusion_matrices.append(confusion_matrix(y_test, y_pred, labels=classes))

    return {
        "classes": classes,
        "accuracies": np.array(accuracies),
        "confusion_matrices": confusion_matrices,
    }


# ====================================================================
# 4. EPSILON-SWEEP STUDY
# ====================================================================

def run_epsilon_study(
    datasets: dict[str, pd.DataFrame],
    epsilons: list[float],
    n_reps: int = 10,
    n_splits: int = 10,
) -> pd.DataFrame:
    """
    For each dataset × epsilon × repetition:
      1. generate synthetic data
      2. evaluate classifier accuracy (train on synthetic, test on original)
      3. evaluate baseline accuracy (train & test on original)

    Returns a long-format DataFrame with one row per
    (dataset, epsilon, rep).
    """
    rows: list[dict] = []

    for ds_name, df in datasets.items():
        print(f"  Dataset: {ds_name}")
        for eps in epsilons:
            for rep in range(n_reps):
                synth = generate(df, epsilon=eps, random_state=42 + rep)
                res_synth = evaluate_df(synth, df, n_splits=n_splits)
                res_orig = evaluate_df(df, n_splits=n_splits)

                rows.append({
                    "dataset": ds_name,
                    "epsilon": eps,
                    "rep": rep,
                    "accuracy_synth": res_synth,
                    "accuracy_orig": res_orig,
                })

    return pd.DataFrame(rows)


def summarize_study(study_results: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse the raw study results into a summary table with mean/std
    accuracy per (dataset, epsilon).
    """

    def _extract_mean_accuracy(x):
        if isinstance(x, dict):
            d = x
        else:
            try:
                d = ast.literal_eval(x)
            except Exception:
                return np.nan
        accs = d.get("accuracies", None)
        if accs is None or len(accs) == 0:
            return np.nan
        return float(np.mean(accs))

    df = study_results.copy()
    df["acc_synth_mean"] = df["accuracy_synth"].apply(_extract_mean_accuracy)
    df["acc_orig_mean"] = df["accuracy_orig"].apply(_extract_mean_accuracy)

    summary = (
        df.groupby(["dataset", "epsilon"])
        .agg(
            mean_acc_synth=("acc_synth_mean", "mean"),
            std_acc_synth=("acc_synth_mean", "std"),
            mean_acc_orig=("acc_orig_mean", "mean"),
            std_acc_orig=("acc_orig_mean", "std"),
            n_runs=("rep", "nunique"),
        )
        .reset_index()
    )
    summary["diff_mean"] = summary["mean_acc_synth"] - summary["mean_acc_orig"]
    return summary.sort_values(["dataset", "epsilon"]).reset_index(drop=True)


# ====================================================================
# 5. PRIVBAYES BASELINE  (requires DataSynthesizer)
# ====================================================================

def privbayes_synthesize(
    df: pd.DataFrame,
    epsilon: float,
    k: int = 2,
    category_threshold: int = 50,
    out_dir: str = "privbayes_out",
    dataset_name: str = "dataset",
) -> pd.DataFrame:
    """
    Generate DP synthetic data using the DataSynthesizer library
    (correlated-attribute mode ≈ PrivBayes).

    Requires: ``pip install DataSynthesizer``
    """
    # Lazy imports so the rest of the module works without DataSynthesizer
    from DataSynthesizer.DataDescriber import DataDescriber
    from DataSynthesizer.DataGenerator import DataGenerator
    import builtins

    builtins.np = np  # workaround for DataSynthesizer eval() bug

    os.makedirs(out_dir, exist_ok=True)

    # Prepare input CSV (DataSynthesizer needs a file on disk)
    df_in = df.copy()
    for col in df_in.columns:
        if df_in[col].isna().any():
            df_in[col] = df_in[col].astype("object").fillna("__MISSING__")

    in_csv = os.path.join(out_dir, f"{dataset_name}_input.csv")
    df_in.to_csv(in_csv, index=False)

    # Detect column types
    attr_is_categorical = {}
    attr_datatype = {}
    for col in df_in.columns:
        series = df_in[col]
        is_object = str(series.dtype) in ["object", "category", "string"]
        is_cat = is_object or (series.nunique(dropna=True) <= category_threshold)

        attr_is_categorical[col] = bool(is_cat)
        if is_object:
            attr_datatype[col] = "String"
        elif pd.api.types.is_integer_dtype(series.dtype):
            attr_datatype[col] = "Integer"
        elif pd.api.types.is_float_dtype(series.dtype):
            attr_datatype[col] = "Float"
        else:
            attr_datatype[col] = "String"
            attr_is_categorical[col] = True

    # Describe & generate
    describer = DataDescriber(category_threshold=category_threshold)
    describer.describe_dataset_in_correlated_attribute_mode(
        dataset_file=in_csv,
        epsilon=epsilon,
        k=k,
        attribute_to_datatype=attr_datatype,
        attribute_to_is_categorical=attr_is_categorical,
    )
    desc_file = os.path.join(out_dir, f"{dataset_name}_description.json")
    describer.save_dataset_description_to_file(desc_file)

    generator = DataGenerator()
    out_csv = os.path.join(out_dir, f"{dataset_name}_synthetic.csv")
    generator.generate_dataset_in_correlated_attribute_mode(
        n=df_in.shape[0],
        description_file=desc_file,
    )
    generator.save_synthetic_data(out_csv)

    return pd.read_csv(out_csv)


def run_privbayes_study(
    df: pd.DataFrame,
    epsilons: list[float],
    n_splits: int = 10,
    dataset_name: str = "dataset",
) -> pd.DataFrame:
    """
    Run PrivBayes for each epsilon and return a summary DataFrame
    with mean and std accuracy.
    """
    rows: list[dict] = []
    for eps in epsilons:
        pb = privbayes_synthesize(df, epsilon=eps, dataset_name=dataset_name)
        res = evaluate_df(pb, df, n_splits=n_splits)
        rows.append({
            "epsilon": eps,
            "privbayes_mean_acc": float(np.mean(res["accuracies"])),
            "privbayes_std_acc": float(np.std(res["accuracies"], ddof=1)),
        })
    return pd.DataFrame(rows).sort_values("epsilon").reset_index(drop=True)


# ====================================================================
# 6. PLOTTING UTILITIES
# ====================================================================

# Presentation-friendly defaults (suitable for LaTeX / Beamer)
PLOT_DEFAULTS = {
    "figure.figsize": (10, 5.5),
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "font.size": 16,
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "legend.fontsize": 14,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "axes.grid": True,
    "grid.alpha": 0.25,
}


def apply_plot_defaults():
    """Apply presentation-quality matplotlib defaults."""
    plt.rcParams.update(PLOT_DEFAULTS)


def savefig(name: str, out_dir: str = "plots"):
    """Save current figure as both PNG and PDF."""
    os.makedirs(out_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{name}.png"), bbox_inches="tight")
    plt.savefig(os.path.join(out_dir, f"{name}.pdf"), bbox_inches="tight")
    plt.show()


# ---- 6a. Accuracy bar chart -------------------------------------------

def plot_accuracy_bar(
    orig_results: dict,
    synth_results: dict,
    title: str = "Accuracy: Original vs Synthetic",
):
    """Side-by-side bar chart comparing original and synthetic accuracy."""
    orig_acc = np.array(orig_results["accuracies"], dtype=float)
    synth_acc = np.array(synth_results["accuracies"], dtype=float)

    means = [orig_acc.mean(), synth_acc.mean()]
    stds = [orig_acc.std(ddof=1), synth_acc.std(ddof=1)]

    fig, ax = plt.subplots()
    ax.bar(["Original", "Synthetic"], means, yerr=stds, capsize=8)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(i, m + 0.02, f"{m:.3f} ± {s:.3f}", ha="center", va="bottom")
    plt.tight_layout()
    return fig, ax


# ---- 6b. Accuracy box plot --------------------------------------------

def plot_accuracy_box(
    orig_results: dict,
    synth_results: dict,
    title: str = "Accuracy Distribution",
):
    """Box plot of accuracy distributions."""
    orig_acc = np.array(orig_results["accuracies"], dtype=float)
    synth_acc = np.array(synth_results["accuracies"], dtype=float)

    fig, ax = plt.subplots()
    ax.boxplot(
        [orig_acc, synth_acc],
        labels=["Original", "Synthetic"],
        showmeans=True,
    )
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    plt.tight_layout()
    return fig, ax


# ---- 6c. Confusion matrix heatmap -------------------------------------

def plot_confusion_matrix(
    cm: np.ndarray,
    classes: np.ndarray,
    title: str = "Confusion Matrix",
    normalize: bool = True,
):
    """Plot a (optionally row-normalized) confusion matrix."""
    cm = np.array(cm, dtype=float)
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums != 0)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    suffix = " (normalized)" if normalize else ""
    ax.set_title(title + suffix)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    if len(classes) <= 10:
        ax.set_xticks(range(len(classes)))
        ax.set_yticks(range(len(classes)))
        ax.set_xticklabels(classes, rotation=45, ha="right")
        ax.set_yticklabels(classes)

    plt.tight_layout()
    return fig, ax


# ---- 6d. Feature distribution comparison ------------------------------

def plot_feature_distribution(
    col: str,
    df_orig: pd.DataFrame,
    df_synth: pd.DataFrame,
    top_k: int = 12,
    title: Optional[str] = None,
):
    """Compare a single feature's distribution (original vs synthetic)."""
    vc_o = df_orig[col].value_counts(normalize=True)
    vc_s = df_synth[col].value_counts(normalize=True)

    cats = vc_o.index[:top_k]
    o = vc_o.reindex(cats).fillna(0.0)
    s = vc_s.reindex(cats).fillna(0.0)

    x = np.arange(len(cats))
    w = 0.42

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.bar(x - w / 2, o.values, width=w, label="Original")
    ax.bar(x + w / 2, s.values, width=w, label="Synthetic")
    ax.set_xticks(x)
    ax.set_xticklabels([str(c) for c in cats], rotation=0)
    ax.set_ylabel("Relative frequency")
    ax.set_title(title or f"Distribution: {col} (top {top_k})")
    ax.legend()
    plt.tight_layout()
    return fig, ax


def plot_multi_feature_distributions(
    cols: list[str],
    df_orig: pd.DataFrame,
    df_synth: pd.DataFrame,
    top_k: int = 10,
    ncols: int = 2,
    suptitle: str = "Original vs Synthetic distributions",
):
    """Grid of feature distribution comparisons."""
    n = len(cols)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 5.5 + 2.8 * (nrows - 1)))
    axes = np.array(axes).reshape(-1)

    for ax, col in zip(axes, cols):
        vc_o = df_orig[col].value_counts(normalize=True)
        vc_s = df_synth[col].value_counts(normalize=True)
        cats = vc_o.index[:top_k]
        o = vc_o.reindex(cats).fillna(0.0)
        s = vc_s.reindex(cats).fillna(0.0)

        x = np.arange(len(cats))
        w = 0.42
        ax.bar(x - w / 2, o.values, width=w, label="Original")
        ax.bar(x + w / 2, s.values, width=w, label="Synthetic")
        ax.set_title(str(col))
        ax.set_xticks(x)
        ax.set_xticklabels([str(c) for c in cats], rotation=0)
        ax.set_ylim(0, max(o.max(), s.max()) * 1.25)

    for j in range(len(cols), len(axes)):
        axes[j].axis("off")

    fig.suptitle(suptitle, y=1.02)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")
    plt.tight_layout()
    return fig


# ---- 6e. Epsilon curves -----------------------------------------------

def plot_epsilon_curves(
    summary: pd.DataFrame,
    title: str = "Utility vs Epsilon",
):
    """Accuracy vs epsilon for each dataset, with original baselines."""
    fig, ax = plt.subplots(figsize=(10.5, 6))

    for ds, g in summary.groupby("dataset"):
        g = g.sort_values("epsilon")
        ax.errorbar(
            g["epsilon"], g["mean_acc_synth"], yerr=g["std_acc_synth"],
            capsize=4, marker="o", linestyle="-", label=f"{ds} (synthetic)",
        )
        base = float(g["mean_acc_orig"].iloc[0])
        ax.hlines(
            base,
            xmin=g["epsilon"].min(), xmax=g["epsilon"].max(),
            linestyles="--", linewidth=2, label=f"{ds} (original baseline)",
        )

    ax.set_xscale("log")
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("Epsilon (log scale)")
    ax.set_ylabel("Accuracy (mean ± std)")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    return fig, ax


def plot_epsilon_diff(
    summary: pd.DataFrame,
    title: str = "Synthetic − Original (Accuracy)",
):
    """Delta accuracy vs epsilon for each dataset."""
    fig, ax = plt.subplots(figsize=(10.5, 5.5))

    for ds, g in summary.groupby("dataset"):
        g = g.sort_values("epsilon")
        ax.plot(g["epsilon"], g["diff_mean"], marker="o", label=ds)

    ax.axhline(0, linewidth=2, linestyle="--")
    ax.set_xscale("log")
    ax.set_xlabel("Epsilon (log scale)")
    ax.set_ylabel("Δ Accuracy (synthetic − original)")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    return fig, ax


def plot_epsilon_std(
    summary: pd.DataFrame,
    title: str = "Std. Deviation vs Epsilon",
):
    """Standard deviation of synthetic accuracy vs epsilon."""
    fig, ax = plt.subplots(figsize=(10.5, 5.5))

    for ds, g in summary.groupby("dataset"):
        g = g.sort_values("epsilon")
        ax.plot(g["epsilon"], g["std_acc_synth"], marker="o", label=ds)

    ax.set_xscale("log")
    ax.set_xlabel("Epsilon (log scale)")
    ax.set_ylabel("Std. deviation of accuracy (synthetic)")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    return fig, ax


def plot_synth_vs_privbayes(
    summary: pd.DataFrame,
    pb_summary: pd.DataFrame,
    dataset: str = "adult",
    title: str = "My Synth vs PrivBayes",
):
    """
    Compare the custom marginal-based approach with PrivBayes
    on a single dataset.
    """
    subset = summary[summary["dataset"] == dataset]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(
        subset["epsilon"], subset["mean_acc_synth"],
        marker="o", label="Marginal-based (ours)",
    )
    ax.plot(
        pb_summary["epsilon"], pb_summary["privbayes_mean_acc"],
        marker="o", label="PrivBayes",
    )

    orig_baseline = float(subset["mean_acc_orig"].iloc[0])
    ax.axhline(orig_baseline, linestyle="--", linewidth=2, label="Original (baseline)")

    ax.set_xscale("log")
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("Epsilon (log scale)")
    ax.set_ylabel("Accuracy (train synthetic → test original)")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig, ax


def plot_three_way_bar(
    orig_results: dict,
    synth_results: dict,
    priv_results: dict,
    title: str = "Accuracy comparison",
):
    """Bar chart: Original vs My Synth vs PrivBayes."""
    all_accs = [
        np.array(orig_results["accuracies"], dtype=float),
        np.array(synth_results["accuracies"], dtype=float),
        np.array(priv_results["accuracies"], dtype=float),
    ]
    labels = ["Original", "My Synth", "PrivBayes"]
    means = [a.mean() for a in all_accs]
    stds = [a.std(ddof=1) for a in all_accs]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(labels, means, yerr=stds, capsize=8)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    plt.tight_layout()
    return fig, ax
