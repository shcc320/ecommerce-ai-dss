from __future__ import annotations

import json
import math
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from lightgbm import LGBMClassifier
from scipy.stats import friedmanchisquare, spearmanr, wilcoxon
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier


ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"


DATA_URL = (
    "https://archive.ics.uci.edu/static/public/468/"
    "online+shoppers+purchasing+intention+dataset.zip"
)
DATA_ZIP = DATA_RAW / "online_shoppers_purchasing_intention.zip"
DATA_CSV = DATA_RAW / "online_shoppers_intention.csv"

SEEDS = [7, 17, 27, 37, 47, 57, 67, 77, 87, 97]
PRIMARY_SEED = 47
TEST_SIZE = 0.30
SHAP_STABILITY_SEEDS = [7, 17, 27, 37, 47]

CRITERIA_WEIGHTS = pd.Series(
    {
        "Expected conversion impact": 0.30,
        "Evidence strength": 0.20,
        "Implementation feasibility": 0.15,
        "Cost efficiency": 0.15,
        "User-experience safety": 0.10,
        "Automation readiness": 0.10,
    }
)

STRATEGY_RULES = {
    "PageValues": {
        "strategy": "Prioritize high-value landing-page optimization",
        "knowledge_rule": "High page value indicates commercial intent; reinforce paths that already create value.",
        "Implementation feasibility": 0.86,
        "Cost efficiency": 0.73,
        "User-experience safety": 0.92,
        "Automation readiness": 0.86,
    },
    "ExitRates": {
        "strategy": "Reduce exit friction in product and checkout journeys",
        "knowledge_rule": "High exit rates reveal journey breakpoints that suppress conversion.",
        "Implementation feasibility": 0.82,
        "Cost efficiency": 0.69,
        "User-experience safety": 0.88,
        "Automation readiness": 0.76,
    },
    "BounceRates": {
        "strategy": "Improve first-screen relevance and loading quality",
        "knowledge_rule": "High bounce rates indicate weak landing-page match or slow first interaction.",
        "Implementation feasibility": 0.80,
        "Cost efficiency": 0.72,
        "User-experience safety": 0.90,
        "Automation readiness": 0.70,
    },
    "ProductRelated_Duration": {
        "strategy": "Trigger product-page assistance for long comparison sessions",
        "knowledge_rule": "Long product-page duration suggests comparison effort and need for decision support.",
        "Implementation feasibility": 0.74,
        "Cost efficiency": 0.66,
        "User-experience safety": 0.80,
        "Automation readiness": 0.83,
    },
    "ProductRelated": {
        "strategy": "Personalize product discovery depth and recommendations",
        "knowledge_rule": "Product-page count captures browsing depth and product discovery intensity.",
        "Implementation feasibility": 0.78,
        "Cost efficiency": 0.68,
        "User-experience safety": 0.84,
        "Automation readiness": 0.85,
    },
    "Administrative_Duration": {
        "strategy": "Streamline account and administrative navigation",
        "knowledge_rule": "Administrative dwell time reflects account, policy, or support friction.",
        "Implementation feasibility": 0.76,
        "Cost efficiency": 0.70,
        "User-experience safety": 0.88,
        "Automation readiness": 0.67,
    },
    "Informational_Duration": {
        "strategy": "Surface trust-building information before purchase decisions",
        "knowledge_rule": "Informational dwell time reflects demand for trust, policy, or product evidence.",
        "Implementation feasibility": 0.79,
        "Cost efficiency": 0.74,
        "User-experience safety": 0.91,
        "Automation readiness": 0.72,
    },
    "VisitorType": {
        "strategy": "Personalize journeys by visitor recency type",
        "knowledge_rule": "Visitor type separates returning, new, and other sessions with different intent patterns.",
        "Implementation feasibility": 0.84,
        "Cost efficiency": 0.77,
        "User-experience safety": 0.86,
        "Automation readiness": 0.89,
    },
    "Month": {
        "strategy": "Schedule seasonal merchandising and promotion windows",
        "knowledge_rule": "Month effects reveal recurrent seasonality in purchase intention.",
        "Implementation feasibility": 0.88,
        "Cost efficiency": 0.81,
        "User-experience safety": 0.89,
        "Automation readiness": 0.78,
    },
    "TrafficType": {
        "strategy": "Reweight acquisition channels by conversion quality",
        "knowledge_rule": "Traffic source patterns indicate channel quality and targeting mismatch.",
        "Implementation feasibility": 0.81,
        "Cost efficiency": 0.75,
        "User-experience safety": 0.82,
        "Automation readiness": 0.84,
    },
    "SpecialDay": {
        "strategy": "Tune holiday-specific timing and promotion intensity",
        "knowledge_rule": "Special-day proximity changes intent and should alter campaign timing.",
        "Implementation feasibility": 0.83,
        "Cost efficiency": 0.79,
        "User-experience safety": 0.85,
        "Automation readiness": 0.76,
    },
    "Weekend": {
        "strategy": "Adapt weekend merchandising and support coverage",
        "knowledge_rule": "Weekend sessions capture a distinct behavioral and staffing pattern.",
        "Implementation feasibility": 0.86,
        "Cost efficiency": 0.80,
        "User-experience safety": 0.86,
        "Automation readiness": 0.74,
    },
}


@dataclass
class PrimaryRun:
    model_name: str
    pipeline: Pipeline
    x_train: pd.DataFrame
    x_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    y_prob: np.ndarray
    y_pred: np.ndarray


def ensure_dirs() -> None:
    for path in [DATA_RAW, DATA_PROCESSED, RESULTS]:
        path.mkdir(parents=True, exist_ok=True)


def ensure_dataset() -> None:
    if DATA_CSV.exists():
        return
    if not DATA_ZIP.exists():
        print(f"Downloading dataset from {DATA_URL}")
        urllib.request.urlretrieve(DATA_URL, DATA_ZIP)
    with zipfile.ZipFile(DATA_ZIP) as archive:
        archive.extractall(DATA_RAW)
    if not DATA_CSV.exists():
        matches = list(DATA_RAW.glob("**/*online*shoppers*.csv"))
        if matches:
            matches[0].replace(DATA_CSV)
    if not DATA_CSV.exists():
        raise FileNotFoundError("Could not locate online_shoppers_intention.csv after extraction.")


def load_data() -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(DATA_CSV)
    df.columns = [str(c).strip() for c in df.columns]
    df["Revenue"] = df["Revenue"].astype(int)
    if "Weekend" in df.columns:
        df["Weekend"] = df["Weekend"].astype(int)
    x = df.drop(columns=["Revenue"])
    y = df["Revenue"]
    df.to_csv(DATA_PROCESSED / "online_shoppers_clean.csv", index=False)
    return x, y


def make_preprocessor(x: pd.DataFrame) -> tuple[ColumnTransformer, list[str], list[str]]:
    categorical_cols = x.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    numeric_cols = [c for c in x.columns if c not in categorical_cols]
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )
    return preprocessor, numeric_cols, categorical_cols


def get_models(seed: int) -> dict[str, object]:
    return {
        "Logistic regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            solver="lbfgs",
            random_state=seed,
        ),
        "Random forest": RandomForestClassifier(
            n_estimators=180,
            min_samples_leaf=4,
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=-1,
        ),
        "Gradient boosting": GradientBoostingClassifier(
            n_estimators=120,
            learning_rate=0.055,
            max_depth=3,
            random_state=seed,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=120,
            learning_rate=0.045,
            max_depth=3,
            eval_metric="logloss",
            tree_method="hist",
            random_state=seed,
            n_jobs=1,
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=120,
            learning_rate=0.045,
            max_depth=3,
            class_weight="balanced",
            random_state=seed,
            verbose=-1,
            n_jobs=1,
        ),
    }


def metric_row(y_true: pd.Series, y_prob: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_true, y_prob),
        "PR-AUC": average_precision_score(y_true, y_prob),
        "Brier": brier_score_loss(y_true, y_prob),
    }


def run_statistical_tests(metrics: pd.DataFrame) -> None:
    """Friedman test + pairwise Wilcoxon with Bonferroni correction."""
    lines = ["statistical_test,metric,statistic,p_value"]
    model_names = sorted(metrics["Model"].unique())
    n_comparisons = len(model_names) * (len(model_names) - 1) // 2

    for metric_name in ["F1", "PR-AUC", "ROC-AUC"]:
        pivot = metrics.pivot(index="Seed", columns="Model", values=metric_name)
        stat, p = friedmanchisquare(*[pivot[col] for col in pivot.columns])
        lines.append(f"friedman,{metric_name},{stat:.4f},{p:.6f}")

        if p < 0.05 and n_comparisons > 0:
            bonferroni_alpha = 0.05 / n_comparisons
            for i, ma in enumerate(model_names):
                for j, mb in enumerate(model_names):
                    if i >= j:
                        continue
                    try:
                        w_stat, w_p = wilcoxon(pivot[ma], pivot[mb], zero_method="zsplit")
                        significant = "yes" if w_p < bonferroni_alpha else "no"
                        lines.append(
                            f"wilcoxon,{metric_name},{ma} vs {mb},{w_stat:.4f},{w_p:.6f},"
                            f"bonferroni_alpha={bonferroni_alpha:.6f},significant={significant}"
                        )
                    except Exception:
                        lines.append(
                            f"wilcoxon,{metric_name},{ma} vs {mb},NA,NA,error"
                        )

    (RESULTS / "statistical_tests.csv").write_text("\n".join(lines), encoding="utf-8")



def train_and_evaluate(x: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, PrimaryRun, dict[str, pd.DataFrame]]:
    preprocessor, _, _ = make_preprocessor(x)
    rows = []
    primary_runs: dict[str, PrimaryRun] = {}
    curve_data: dict[str, pd.DataFrame] = {}

    for seed in SEEDS:
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=TEST_SIZE, random_state=seed, stratify=y
        )
        for model_name, model in get_models(seed).items():
            pipeline = Pipeline(
                steps=[
                    ("preprocess", clone(preprocessor)),
                    ("model", model),
                ]
            )
            pipeline.fit(x_train, y_train)
            y_prob = pipeline.predict_proba(x_test)[:, 1]
            y_pred = (y_prob >= 0.5).astype(int)
            row = {"Seed": seed, "Model": model_name}
            row.update(metric_row(y_test, y_prob, y_pred))
            rows.append(row)

            if seed == PRIMARY_SEED:
                primary_runs[model_name] = PrimaryRun(
                    model_name=model_name,
                    pipeline=pipeline,
                    x_train=x_train,
                    x_test=x_test,
                    y_train=y_train,
                    y_test=y_test,
                    y_prob=y_prob,
                    y_pred=y_pred,
                )
                fpr, tpr, _ = roc_curve(y_test, y_prob)
                precision, recall, _ = precision_recall_curve(y_test, y_prob)
                roc_df = pd.DataFrame({"Curve": "ROC", "x": fpr, "y": tpr, "Model": model_name})
                pr_df = pd.DataFrame({"Curve": "PR", "x": recall, "y": precision, "Model": model_name})
                curve_data[model_name] = pd.concat([roc_df, pr_df], ignore_index=True)

    metrics = pd.DataFrame(rows)
    metrics.to_csv(RESULTS / "model_metrics_by_seed.csv", index=False)

    summary = (
        metrics.groupby("Model")
        .agg(
            **{
                "Accuracy mean": ("Accuracy", "mean"),
                "Accuracy std": ("Accuracy", "std"),
                "F1 mean": ("F1", "mean"),
                "F1 std": ("F1", "std"),
                "ROC-AUC mean": ("ROC-AUC", "mean"),
                "ROC-AUC std": ("ROC-AUC", "std"),
                "PR-AUC mean": ("PR-AUC", "mean"),
                "PR-AUC std": ("PR-AUC", "std"),
                "Brier mean": ("Brier", "mean"),
                "Brier std": ("Brier", "std"),
            }
        )
        .reset_index()
        .sort_values(["PR-AUC mean", "ROC-AUC mean"], ascending=False)
    )
    summary.to_csv(RESULTS / "model_metrics_summary.csv", index=False)

    run_statistical_tests(metrics)

    best_model_name = summary.iloc[0]["Model"]
    primary = primary_runs[best_model_name]
    pd.concat(curve_data.values(), ignore_index=True).to_csv(RESULTS / "primary_seed_curves.csv", index=False)
    return metrics, primary, curve_data


def fit_primary_model(x: pd.DataFrame, y: pd.Series, model_name: str, seed: int = PRIMARY_SEED) -> PrimaryRun:
    preprocessor, _, _ = make_preprocessor(x)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=TEST_SIZE, random_state=seed, stratify=y
    )
    model = get_models(seed)[model_name]
    pipeline = Pipeline(
        steps=[
            ("preprocess", clone(preprocessor)),
            ("model", model),
        ]
    )
    pipeline.fit(x_train, y_train)
    y_prob = pipeline.predict_proba(x_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    return PrimaryRun(
        model_name=model_name,
        pipeline=pipeline,
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
        y_prob=y_prob,
        y_pred=y_pred,
    )


def run_feature_ablation(x: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    feature_sets = {
        "All features": x.columns.tolist(),
        "Without PageValues": [c for c in x.columns if c != "PageValues"],
    }
    rows = []
    for feature_set, columns in feature_sets.items():
        x_subset = x[columns].copy()
        preprocessor, _, _ = make_preprocessor(x_subset)
        for seed in SEEDS:
            x_train, x_test, y_train, y_test = train_test_split(
                x_subset, y, test_size=TEST_SIZE, random_state=seed, stratify=y
            )
            for model_name, model in get_models(seed).items():
                pipeline = Pipeline(
                    steps=[
                        ("preprocess", clone(preprocessor)),
                        ("model", model),
                    ]
                )
                pipeline.fit(x_train, y_train)
                y_prob = pipeline.predict_proba(x_test)[:, 1]
                y_pred = (y_prob >= 0.5).astype(int)
                row = {"Feature set": feature_set, "Seed": seed, "Model": model_name}
                row.update(metric_row(y_test, y_prob, y_pred))
                rows.append(row)

    ablation = pd.DataFrame(rows)
    ablation.to_csv(RESULTS / "feature_ablation_by_seed.csv", index=False)
    summary = (
        ablation.groupby(["Feature set", "Model"])
        .agg(
            **{
                "F1 mean": ("F1", "mean"),
                "ROC-AUC mean": ("ROC-AUC", "mean"),
                "PR-AUC mean": ("PR-AUC", "mean"),
                "Brier mean": ("Brier", "mean"),
            }
        )
        .reset_index()
        .sort_values(["Feature set", "PR-AUC mean"], ascending=[True, False])
    )
    summary.to_csv(RESULTS / "feature_ablation_summary.csv", index=False)
    return summary


def clean_feature_name(name: str) -> str:
    name = name.replace("num__", "").replace("cat__", "")
    name = name.replace("_", " ")
    name = name.replace("VisitorType", "Visitor type")
    name = name.replace("TrafficType", "Traffic type")
    return name


def raw_feature_name(transformed: str, raw_columns: list[str]) -> str:
    name = transformed.replace("num__", "").replace("cat__", "")
    for column in sorted(raw_columns, key=len, reverse=True):
        if name == column or name.startswith(f"{column}_"):
            return column
    return name.split("_")[0]


def compute_shap(
    primary: PrimaryRun,
    raw_columns: list[str],
    output_prefix: str = "",
    make_figures: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pipeline = primary.pipeline
    preprocessor = pipeline.named_steps["preprocess"]
    model = pipeline.named_steps["model"]

    x_train_t = preprocessor.transform(primary.x_train)
    x_test_t = preprocessor.transform(primary.x_test)
    feature_names = preprocessor.get_feature_names_out()
    display_names = [clean_feature_name(n) for n in feature_names]

    rng = np.random.default_rng(20260702)
    sample_size = min(800, x_test_t.shape[0])
    sample_idx = rng.choice(x_test_t.shape[0], size=sample_size, replace=False)
    x_sample = x_test_t[sample_idx]
    x_sample_df = pd.DataFrame(x_sample, columns=display_names)

    if primary.model_name == "Logistic regression":
        explainer = shap.LinearExplainer(model, x_train_t)
        shap_values = explainer.shap_values(x_sample)
    else:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(x_sample)

    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    elif getattr(shap_values, "ndim", 0) == 3:
        shap_values = shap_values[:, :, 1]

    mean_abs = np.abs(shap_values).mean(axis=0)
    shap_df = pd.DataFrame(
        {
            "Feature": display_names,
            "Transformed feature": feature_names,
            "Raw feature": [raw_feature_name(n, raw_columns) for n in feature_names],
            "Mean absolute SHAP": mean_abs,
        }
    ).sort_values("Mean absolute SHAP", ascending=False)
    shap_df["Normalized importance"] = shap_df["Mean absolute SHAP"] / shap_df["Mean absolute SHAP"].max()
    shap_df.to_csv(RESULTS / f"{output_prefix}shap_feature_importance.csv", index=False)

    grouped = (
        shap_df.groupby("Raw feature", as_index=False)["Mean absolute SHAP"]
        .sum()
        .sort_values("Mean absolute SHAP", ascending=False)
    )
    grouped["Normalized importance"] = grouped["Mean absolute SHAP"] / grouped["Mean absolute SHAP"].max()
    grouped.to_csv(RESULTS / f"{output_prefix}shap_raw_feature_importance.csv", index=False)

    if make_figures:
        save_shap_figures(shap_values, x_sample_df, shap_df)
    return shap_df, grouped


def save_current_figure(name: str) -> None:
    return


def save_shap_figures(shap_values: np.ndarray, x_sample_df: pd.DataFrame, shap_df: pd.DataFrame) -> None:
    return


def build_strategy_matrix(grouped_importance: pd.DataFrame, output_prefix: str = "") -> tuple[pd.DataFrame, pd.DataFrame]:
    selected_rows = []
    used_strategies = set()
    max_rows = 8
    for _, row in grouped_importance.iterrows():
        raw = row["Raw feature"]
        rule = STRATEGY_RULES.get(raw)
        if rule is None:
            continue
        strategy = rule["strategy"]
        if strategy in used_strategies:
            continue
        importance = float(row["Normalized importance"])
        selected_rows.append(
            {
                "Raw feature": raw,
                "Strategy": strategy,
                "Knowledge rule": rule["knowledge_rule"],
                "Expected conversion impact": 0.35 + 0.65 * importance,
                "Evidence strength": 0.40 + 0.60 * importance,
                "Implementation feasibility": rule["Implementation feasibility"],
                "Cost efficiency": rule["Cost efficiency"],
                "User-experience safety": rule["User-experience safety"],
                "Automation readiness": rule["Automation readiness"],
            }
        )
        used_strategies.add(strategy)
        if len(selected_rows) >= max_rows:
            break

    if len(selected_rows) < 5:
        raise RuntimeError("Too few mapped strategies; check strategy rules.")

    matrix = pd.DataFrame(selected_rows)
    matrix.to_csv(RESULTS / f"{output_prefix}strategy_decision_matrix.csv", index=False)

    fuzzy_rows = []
    for _, row in matrix.iterrows():
        out = {"Strategy": row["Strategy"], "Raw feature": row["Raw feature"]}
        for criterion in CRITERIA_WEIGHTS.index:
            value = float(row[criterion])
            spread = 0.06 if criterion in ["Expected conversion impact", "Evidence strength"] else 0.04
            low = max(0.0, value - spread)
            mid = value
            high = min(1.0, value + spread)
            out[f"{criterion} L"] = low
            out[f"{criterion} M"] = mid
            out[f"{criterion} U"] = high
            out[criterion] = (low + mid + high) / 3.0
        fuzzy_rows.append(out)
    fuzzy = pd.DataFrame(fuzzy_rows)
    fuzzy.to_csv(RESULTS / f"{output_prefix}strategy_fuzzy_matrix_defuzzified.csv", index=False)
    return matrix, fuzzy


def normalize_scores(series: pd.Series) -> pd.Series:
    min_v, max_v = series.min(), series.max()
    if math.isclose(min_v, max_v):
        return pd.Series(np.ones(len(series)), index=series.index)
    return (series - min_v) / (max_v - min_v)


def topsis(matrix: pd.DataFrame, weights: pd.Series) -> pd.Series:
    x = matrix[weights.index].to_numpy(dtype=float)
    w = weights.to_numpy(dtype=float)
    denom = np.sqrt((x**2).sum(axis=0))
    denom[denom == 0] = 1
    weighted = (x / denom) * w
    ideal = weighted.max(axis=0)
    nadir = weighted.min(axis=0)
    d_pos = np.sqrt(((weighted - ideal) ** 2).sum(axis=1))
    d_neg = np.sqrt(((weighted - nadir) ** 2).sum(axis=1))
    return pd.Series(d_neg / (d_pos + d_neg), index=matrix.index)


def edas(matrix: pd.DataFrame, weights: pd.Series) -> pd.Series:
    x = matrix[weights.index].to_numpy(dtype=float)
    w = weights.to_numpy(dtype=float)
    avg = x.mean(axis=0)
    avg[avg == 0] = 1
    pda = np.maximum(0, (x - avg) / avg)
    nda = np.maximum(0, (avg - x) / avg)
    sp = pda @ w
    sn = nda @ w
    nsp = sp / sp.max() if sp.max() > 0 else np.ones_like(sp)
    nsn = 1 - (sn / sn.max() if sn.max() > 0 else np.zeros_like(sn))
    return pd.Series((nsp + nsn) / 2, index=matrix.index)


def mabac(matrix: pd.DataFrame, weights: pd.Series) -> pd.Series:
    x = matrix[weights.index].copy()
    normalized = x.apply(normalize_scores, axis=0).to_numpy(dtype=float)
    w = weights.to_numpy(dtype=float)
    weighted = w * (normalized + 1.0)
    border = np.exp(np.log(np.clip(weighted, 1e-12, None)).mean(axis=0))
    q = (weighted - border).sum(axis=1)
    return pd.Series(q, index=matrix.index)


def run_mcda(fuzzy: pd.DataFrame, weights: pd.Series = CRITERIA_WEIGHTS) -> pd.DataFrame:
    result = fuzzy[["Strategy", "Raw feature"]].copy()
    score_map = {
        "TOPSIS": topsis(fuzzy, weights),
        "EDAS": edas(fuzzy, weights),
        "MABAC": mabac(fuzzy, weights),
    }
    for method, scores in score_map.items():
        result[f"{method} score"] = scores
        result[f"{method} rank"] = scores.rank(ascending=False, method="min").astype(int)
        result[f"{method} normalized"] = normalize_scores(scores)
    result["Aggregate score"] = result[[f"{m} normalized" for m in score_map]].mean(axis=1)
    result["Aggregate rank"] = result["Aggregate score"].rank(ascending=False, method="min").astype(int)
    result = result.sort_values("Aggregate rank")
    return result


def run_sensitivity(fuzzy: pd.DataFrame, n: int = 600) -> pd.DataFrame:
    return _run_sensitivity_inner(fuzzy, n, "baseline")


def _run_sensitivity_inner(fuzzy: pd.DataFrame, n: int, label: str) -> pd.DataFrame:
    rng = np.random.default_rng(20260702)
    base = CRITERIA_WEIGHTS.to_numpy(dtype=float)

    if label == "uniform":
        concentration = np.ones(len(base))
    else:
        concentration = base * 80

    records = []
    for i in range(n):
        weights = pd.Series(rng.dirichlet(concentration), index=CRITERIA_WEIGHTS.index)
        ranking = run_mcda(fuzzy, weights)
        for _, row in ranking.iterrows():
            records.append(
                {
                    "Simulation": i,
                    "Strategy": row["Strategy"],
                    "Aggregate rank": row["Aggregate rank"],
                    "Aggregate score": row["Aggregate score"],
                    "Top ranked": int(row["Aggregate rank"] == 1),
                }
            )
    long = pd.DataFrame(records)
    suffix = "" if label == "baseline" else f"_{label}"
    long.to_csv(RESULTS / f"sensitivity_rank_samples{suffix}.csv", index=False)
    summary = (
        long.groupby("Strategy")
        .agg(
            **{
                "Mean rank": ("Aggregate rank", "mean"),
                "Rank std": ("Aggregate rank", "std"),
                "Top-1 frequency": ("Top ranked", "mean"),
                "Mean aggregate score": ("Aggregate score", "mean"),
            }
        )
        .reset_index()
        .sort_values(["Mean rank", "Rank std"])
    )
    summary.to_csv(RESULTS / f"sensitivity_summary{suffix}.csv", index=False)
    return summary


def run_month_holdout_validation(x: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Evaluate deployment robustness by holding out each calendar month.

    This validation is intentionally stricter than a random split because the held-out
    month category is unseen during training. The purpose is not to maximize the score,
    but to show whether the predictive layer remains usable when seasonal distribution
    shifts are present.
    """
    if "Month" not in x.columns:
        raise ValueError("Month column is required for month-holdout validation.")

    month_order = ["Feb", "Mar", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    months = [m for m in month_order if m in set(x["Month"].astype(str))]
    feature_sets = {
        "Full feature": x.columns.tolist(),
        "PageValues-free": [c for c in x.columns if c != "PageValues"],
    }

    rows = []
    for feature_set, columns in feature_sets.items():
        x_subset = x[columns].copy()
        for month in months:
            test_mask = x["Month"].astype(str) == month
            train_mask = ~test_mask
            x_train, x_test = x_subset.loc[train_mask], x_subset.loc[test_mask]
            y_train, y_test = y.loc[train_mask], y.loc[test_mask]
            if y_test.nunique() < 2 or len(y_test) < 50:
                continue
            preprocessor, _, _ = make_preprocessor(x_subset)
            model = get_models(PRIMARY_SEED)["XGBoost"]
            pipeline = Pipeline(
                steps=[
                    ("preprocess", clone(preprocessor)),
                    ("model", model),
                ]
            )
            pipeline.fit(x_train, y_train)
            y_prob = pipeline.predict_proba(x_test)[:, 1]
            y_pred = (y_prob >= 0.5).astype(int)
            row = {
                "Feature set": feature_set,
                "Held-out month": month,
                "Test sessions": int(len(y_test)),
                "Positive sessions": int(y_test.sum()),
                "Positive rate": float(y_test.mean()),
            }
            row.update(metric_row(y_test, y_prob, y_pred))
            rows.append(row)

    month_df = pd.DataFrame(rows)
    month_df.to_csv(RESULTS / "month_holdout_validation.csv", index=False)

    summary = (
        month_df.groupby("Feature set")
        .agg(
            **{
                "Months": ("Held-out month", "nunique"),
                "F1 mean": ("F1", "mean"),
                "F1 std": ("F1", "std"),
                "ROC-AUC mean": ("ROC-AUC", "mean"),
                "ROC-AUC std": ("ROC-AUC", "std"),
                "PR-AUC mean": ("PR-AUC", "mean"),
                "PR-AUC std": ("PR-AUC", "std"),
                "Brier mean": ("Brier", "mean"),
            }
        )
        .reset_index()
    )
    summary.to_csv(RESULTS / "month_holdout_summary.csv", index=False)

    return month_df


def run_shap_stability(x: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Measure whether raw-feature SHAP rankings are stable across random seeds."""
    grouped_by_seed: dict[int, pd.DataFrame] = {}
    all_features = list(x.columns)
    for seed in SHAP_STABILITY_SEEDS:
        primary = fit_primary_model(x, y, "XGBoost", seed=seed)
        _, grouped = compute_shap(
            primary,
            all_features,
            output_prefix=f"stability_seed_{seed}_",
            make_figures=False,
        )
        tmp = grouped[["Raw feature", "Mean absolute SHAP", "Normalized importance"]].copy()
        tmp["Rank"] = tmp["Mean absolute SHAP"].rank(ascending=False, method="min")
        grouped_by_seed[seed] = tmp
        tmp.to_csv(RESULTS / f"shap_stability_seed_{seed}.csv", index=False)

    pair_rows = []
    seeds = list(grouped_by_seed)
    for i, a in enumerate(seeds):
        for b in seeds[i + 1:]:
            merged = grouped_by_seed[a].merge(
                grouped_by_seed[b],
                on="Raw feature",
                suffixes=(f" seed {a}", f" seed {b}"),
            )
            rho = spearmanr(merged[f"Rank seed {a}"], merged[f"Rank seed {b}"]).correlation
            top5_a = set(grouped_by_seed[a].sort_values("Rank").head(5)["Raw feature"])
            top5_b = set(grouped_by_seed[b].sort_values("Rank").head(5)["Raw feature"])
            top8_a = set(grouped_by_seed[a].sort_values("Rank").head(8)["Raw feature"])
            top8_b = set(grouped_by_seed[b].sort_values("Rank").head(8)["Raw feature"])
            pair_rows.append(
                {
                    "Seed A": a,
                    "Seed B": b,
                    "Spearman rho": float(rho),
                    "Top-5 overlap": len(top5_a & top5_b) / 5.0,
                    "Top-8 overlap": len(top8_a & top8_b) / 8.0,
                }
            )
    pairwise = pd.DataFrame(pair_rows)
    pairwise.to_csv(RESULTS / "shap_stability_pairwise.csv", index=False)
    summary = pd.DataFrame(
        [
            {
                "Compared seeds": len(SHAP_STABILITY_SEEDS),
                "Pairwise comparisons": len(pairwise),
                "Mean Spearman rho": pairwise["Spearman rho"].mean(),
                "Min Spearman rho": pairwise["Spearman rho"].min(),
                "Mean top-5 overlap": pairwise["Top-5 overlap"].mean(),
                "Mean top-8 overlap": pairwise["Top-8 overlap"].mean(),
            }
        ]
    )
    summary.to_csv(RESULTS / "shap_stability_summary.csv", index=False)

    return pairwise




def write_run_metadata() -> None:
    metadata = {
        "dataset_url": DATA_URL,
        "seeds": SEEDS,
        "primary_seed": PRIMARY_SEED,
        "test_size": TEST_SIZE,
        "criteria_weights": CRITERIA_WEIGHTS.to_dict(),
    }
    (RESULTS / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    ensure_dataset()
    x, y = load_data()
    metrics, primary, curve_data = train_and_evaluate(x, y)
    ablation = run_feature_ablation(x, y)
    no_page = x[[c for c in x.columns if c != "PageValues"]].copy()
    no_page_best = (
        ablation[ablation["Feature set"] == "Without PageValues"]
        .sort_values("PR-AUC mean", ascending=False)
        .iloc[0]["Model"]
    )
    deployment_primary = fit_primary_model(no_page, y, str(no_page_best))
    _, grouped_importance = compute_shap(primary, list(x.columns), make_figures=False)
    _, deployment_grouped_importance = compute_shap(
        deployment_primary,
        list(no_page.columns),
        output_prefix="deployment_safe_",
        make_figures=False,
    )
    matrix, fuzzy = build_strategy_matrix(grouped_importance)
    deployment_matrix, deployment_fuzzy = build_strategy_matrix(
        deployment_grouped_importance,
        output_prefix="deployment_safe_",
    )
    mcda = run_mcda(fuzzy)
    mcda.to_csv(RESULTS / "mcda_ranking.csv", index=False)
    deployment_mcda = run_mcda(deployment_fuzzy)
    deployment_mcda.to_csv(RESULTS / "deployment_safe_mcda_ranking.csv", index=False)
    sensitivity = run_sensitivity(fuzzy)
    sensitivity_uniform = _run_sensitivity_inner(fuzzy, 600, "uniform")
    month_holdout = run_month_holdout_validation(x, y)
    shap_stability = run_shap_stability(x, y)
    write_run_metadata()
    print(
        f"Pipeline complete.\n"
        f"Best model: {primary.model_name}\n"
        f"Top strategy: {mcda.iloc[0]['Strategy']}\n"
        f"Results directory: {RESULTS}"
    )


if __name__ == "__main__":
    main()
