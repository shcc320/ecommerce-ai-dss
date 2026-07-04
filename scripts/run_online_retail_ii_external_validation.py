#!/usr/bin/env python3
"""
External validation for the Information e-commerce AI-DSS manuscript using UCI Online Retail II.

Purpose
-------
This script converts Online Retail II transaction logs into a customer-month
repurchase-prediction task and evaluates whether the manuscript's evidence-to-
strategy decision-support pipeline transfers from session-level purchase
intention to transaction-level customer repurchase support.

Example
-------
python scripts/run_online_retail_ii_external_validation.py \
    --input data/raw/online_retail_ii.xlsx \
    --output online_retail_ii_external_results \
    --next-days 30 \
    --test-months 3 \
    --val-months 3 \
    --shap-sample 1000

Outputs
-------
<output>/
  results_external/
    online_retail_ii_dataset_summary.csv
    online_retail_ii_dataset_summary.json
    online_retail_ii_customer_month_panel.csv.gz
    online_retail_ii_split_metadata.json
    online_retail_ii_model_metrics.csv
    online_retail_ii_best_model_metadata.json
    online_retail_ii_feature_importance.csv
    online_retail_ii_strategy_drivers.csv

"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# -----------------------------
# Optional model dependencies
# -----------------------------
try:
    from xgboost import XGBClassifier  # type: ignore
except Exception:  # pragma: no cover
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier  # type: ignore
except Exception:  # pragma: no cover
    LGBMClassifier = None

try:
    import shap  # type: ignore
except Exception:  # pragma: no cover
    shap = None


# -----------------------------
# Utilities
# -----------------------------

def _safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _clean_colname(name: object) -> str:
    """Normalize column names while preserving readable tokens."""
    s = str(name).strip()
    s = re.sub(r"\s+", "", s)
    s = s.replace("/", "").replace("-", "").replace("_", "")
    return s.lower()


def _first_existing(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    lookup = {_clean_colname(c): c for c in df.columns}
    for cand in candidates:
        key = _clean_colname(cand)
        if key in lookup:
            return lookup[key]
    return None


def _format_float(x: float, ndigits: int = 3) -> str:
    if pd.isna(x):
        return "--"
    return f"{float(x):.{ndigits}f}"


def _onehot_encoder() -> OneHotEncoder:
    """Create a version-compatible OneHotEncoder."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # scikit-learn < 1.2
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _get_feature_names(preprocessor: ColumnTransformer, numeric_features: List[str], categorical_features: List[str]) -> List[str]:
    """Return post-transform feature names from a ColumnTransformer."""
    try:
        names = list(preprocessor.get_feature_names_out())
        # Remove transformer prefixes for cleaner result files.
        return [n.replace("num__", "").replace("cat__", "") for n in names]
    except Exception:
        names = list(numeric_features)
        try:
            ohe = preprocessor.named_transformers_["cat"].named_steps["onehot"]
            cat_names = list(ohe.get_feature_names_out(categorical_features))
            names.extend(cat_names)
        except Exception:
            names.extend(categorical_features)
        return names


# -----------------------------
# Data loading and cleaning
# -----------------------------

@dataclass
class ColumnMap:
    invoice: str
    stock_code: str
    quantity: str
    invoice_date: str
    unit_price: str
    customer_id: str
    country: str
    description: Optional[str] = None


def load_online_retail_ii(input_path: Path) -> pd.DataFrame:
    """Load Online Retail II from Excel or CSV."""
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    suffix = input_path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        # The original UCI workbook commonly has two sheets.
        sheets = pd.read_excel(input_path, sheet_name=None, engine=None)
        frames = []
        for sheet_name, frame in sheets.items():
            frame = frame.copy()
            frame["__source_sheet"] = str(sheet_name)
            frames.append(frame)
        df = pd.concat(frames, ignore_index=True)
    elif suffix == ".csv":
        # Try common encodings.
        try:
            df = pd.read_csv(input_path)
        except UnicodeDecodeError:
            df = pd.read_csv(input_path, encoding="ISO-8859-1")
    else:
        raise ValueError("Input must be .xlsx, .xls, or .csv")

    # Drop completely empty rows sometimes introduced by Excel parsing.
    df = df.dropna(how="all").reset_index(drop=True)
    return df


def infer_columns(df: pd.DataFrame) -> ColumnMap:
    """Infer common Online Retail II column names."""
    invoice = _first_existing(df, ["Invoice", "InvoiceNo", "InvoiceNumber", "BillNo"])
    stock_code = _first_existing(df, ["StockCode", "Stock Code", "ItemCode", "ProductCode"])
    quantity = _first_existing(df, ["Quantity", "Qty"])
    invoice_date = _first_existing(df, ["InvoiceDate", "Invoice Date", "Date", "TransactionDate"])
    unit_price = _first_existing(df, ["Price", "UnitPrice", "Unit Price", "UnitCost"])
    customer_id = _first_existing(df, ["Customer ID", "CustomerID", "CustomerId", "Customer"])
    country = _first_existing(df, ["Country", "Market"])
    description = _first_existing(df, ["Description", "ProductDescription", "ItemDescription"])

    missing = [
        name
        for name, value in {
            "Invoice/InvoiceNo": invoice,
            "StockCode": stock_code,
            "Quantity": quantity,
            "InvoiceDate": invoice_date,
            "Price/UnitPrice": unit_price,
            "Customer ID/CustomerID": customer_id,
            "Country": country,
        }.items()
        if value is None
    ]
    if missing:
        raise ValueError(
            "Could not infer required columns: " + ", ".join(missing) +
            f". Available columns: {list(df.columns)}"
        )

    return ColumnMap(
        invoice=invoice,
        stock_code=stock_code,
        quantity=quantity,
        invoice_date=invoice_date,
        unit_price=unit_price,
        customer_id=customer_id,
        country=country,
        description=description,
    )


def clean_transactions(raw: pd.DataFrame, colmap: ColumnMap) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return valid purchase transactions and a raw customer-level line table for return/cancellation features."""
    df = pd.DataFrame({
        "Invoice": raw[colmap.invoice],
        "StockCode": raw[colmap.stock_code],
        "Quantity": raw[colmap.quantity],
        "InvoiceDate": raw[colmap.invoice_date],
        "UnitPrice": raw[colmap.unit_price],
        "CustomerID": raw[colmap.customer_id],
        "Country": raw[colmap.country],
    })
    if colmap.description:
        df["Description"] = raw[colmap.description]

    df = df.dropna(subset=["CustomerID", "InvoiceDate"]).copy()
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    df = df.dropna(subset=["InvoiceDate"])

    # Numeric conversion with coercion for robust CSV input.
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
    df["UnitPrice"] = pd.to_numeric(df["UnitPrice"], errors="coerce")
    df = df.dropna(subset=["Quantity", "UnitPrice"])

    # Normalize ID-like columns.
    # Original customer IDs are often floats in Excel; remove trailing .0 when possible.
    def normalize_customer(x: object) -> str:
        if pd.isna(x):
            return ""
        if isinstance(x, float) and x.is_integer():
            return str(int(x))
        s = str(x).strip()
        if s.endswith(".0"):
            return s[:-2]
        return s

    df["CustomerID"] = df["CustomerID"].map(normalize_customer)
    df = df[df["CustomerID"].astype(str).str.len() > 0]
    df["Invoice"] = df["Invoice"].astype(str).str.strip()
    df["StockCode"] = df["StockCode"].astype(str).str.strip()
    df["Country"] = df["Country"].astype(str).str.strip().replace({"nan": "Unknown"})

    df["CancelledLine"] = df["Invoice"].astype(str).str.upper().str.startswith("C") | (df["Quantity"] < 0)
    df["ValidPurchase"] = (~df["CancelledLine"]) & (df["Quantity"] > 0) & (df["UnitPrice"] > 0)
    df["Revenue"] = df["Quantity"] * df["UnitPrice"]

    valid = df[df["ValidPurchase"]].copy()
    valid["Revenue"] = valid["Quantity"] * valid["UnitPrice"]
    valid = valid.sort_values(["CustomerID", "InvoiceDate"])

    raw_lines = df.copy()
    raw_lines = raw_lines.sort_values(["CustomerID", "InvoiceDate"])
    return valid, raw_lines


# -----------------------------
# Customer-month panel construction
# -----------------------------

def _safe_div(num: pd.Series, denom: pd.Series) -> pd.Series:
    return num / denom.replace(0, np.nan)


def _aggregate_history(history: pd.DataFrame, raw_history: pd.DataFrame, obs_date: pd.Timestamp) -> pd.DataFrame:
    """Aggregate customer historical features up to obs_date."""
    g = history.groupby("CustomerID")
    out = g.agg(
        first_purchase_date=("InvoiceDate", "min"),
        last_purchase_date=("InvoiceDate", "max"),
        total_invoices=("Invoice", "nunique"),
        total_lines=("Invoice", "size"),
        total_items=("Quantity", "sum"),
        total_revenue=("Revenue", "sum"),
        unique_products=("StockCode", "nunique"),
        active_purchase_days=("InvoiceDate", lambda x: x.dt.date.nunique()),
        avg_unit_price=("UnitPrice", "mean"),
    ).reset_index()

    out["recency_days"] = (obs_date - out["last_purchase_date"]).dt.days.clip(lower=0)
    out["customer_tenure_days"] = (obs_date - out["first_purchase_date"]).dt.days.clip(lower=0)
    out["avg_order_value"] = _safe_div(out["total_revenue"], out["total_invoices"])
    out["avg_items_per_invoice"] = _safe_div(out["total_items"], out["total_invoices"])
    out["revenue_per_active_day"] = _safe_div(out["total_revenue"], out["active_purchase_days"])

    # Recent-window behavior.
    for days in [30, 60, 90]:
        start = obs_date - pd.Timedelta(days=days)
        recent = history[(history["InvoiceDate"] > start) & (history["InvoiceDate"] <= obs_date)]
        if len(recent) == 0:
            recent_agg = pd.DataFrame({"CustomerID": out["CustomerID"]})
            recent_agg[f"invoices_{days}d"] = 0
            recent_agg[f"items_{days}d"] = 0.0
            recent_agg[f"revenue_{days}d"] = 0.0
            recent_agg[f"unique_products_{days}d"] = 0
        else:
            recent_agg = recent.groupby("CustomerID").agg(
                **{
                    f"invoices_{days}d": ("Invoice", "nunique"),
                    f"items_{days}d": ("Quantity", "sum"),
                    f"revenue_{days}d": ("Revenue", "sum"),
                    f"unique_products_{days}d": ("StockCode", "nunique"),
                }
            ).reset_index()
        out = out.merge(recent_agg, on="CustomerID", how="left")
        for c in [f"invoices_{days}d", f"items_{days}d", f"revenue_{days}d", f"unique_products_{days}d"]:
            out[c] = out[c].fillna(0)

    # Cancellation/return-risk proxies from all raw lines observed so far.
    if len(raw_history) > 0:
        rg = raw_history.groupby("CustomerID").agg(
            raw_lines_total=("Invoice", "size"),
            cancelled_lines=("CancelledLine", "sum"),
            raw_invoices_total=("Invoice", "nunique"),
        ).reset_index()
        rg["cancelled_line_rate"] = _safe_div(rg["cancelled_lines"], rg["raw_lines_total"]).fillna(0)
        out = out.merge(rg[["CustomerID", "raw_lines_total", "cancelled_lines", "raw_invoices_total", "cancelled_line_rate"]], on="CustomerID", how="left")
    else:
        out["raw_lines_total"] = 0
        out["cancelled_lines"] = 0
        out["raw_invoices_total"] = 0
        out["cancelled_line_rate"] = 0.0

    for c in ["raw_lines_total", "cancelled_lines", "raw_invoices_total", "cancelled_line_rate"]:
        out[c] = out[c].fillna(0)

    # Dominant country up to cutoff.
    country = (
        history.groupby(["CustomerID", "Country"]).size().reset_index(name="n")
        .sort_values(["CustomerID", "n"], ascending=[True, False])
        .drop_duplicates("CustomerID")[["CustomerID", "Country"]]
    )
    out = out.merge(country, on="CustomerID", how="left")
    out["Country"] = out["Country"].fillna("Unknown")

    # Observation-time features.
    out["obs_date"] = obs_date
    out["obs_month"] = obs_date.month
    out["obs_quarter"] = obs_date.quarter
    out["obs_year"] = obs_date.year
    out["month_sin"] = np.sin(2 * np.pi * out["obs_month"] / 12)
    out["month_cos"] = np.cos(2 * np.pi * out["obs_month"] / 12)

    # Drop raw datetimes from model features but keep useful metadata separately.
    return out


def build_customer_month_panel(
    valid: pd.DataFrame,
    raw_lines: pd.DataFrame,
    next_days: int = 30,
    min_history_days: int = 0,
    max_months: Optional[int] = None,
) -> pd.DataFrame:
    """Construct a customer-month panel with future repurchase labels."""
    if valid.empty:
        raise ValueError("No valid purchase rows after cleaning. Check input file and column mapping.")

    min_date = valid["InvoiceDate"].min().normalize()
    max_date = valid["InvoiceDate"].max().normalize()
    first_cutoff = (min_date + pd.offsets.MonthEnd(0)).normalize()
    if first_cutoff < min_date:
        first_cutoff = (min_date + pd.offsets.MonthEnd(1)).normalize()
    last_cutoff = (max_date - pd.Timedelta(days=next_days)).normalize()
    last_cutoff = (last_cutoff + pd.offsets.MonthEnd(0)).normalize()
    if last_cutoff > max_date - pd.Timedelta(days=next_days):
        last_cutoff = (last_cutoff - pd.offsets.MonthEnd(1)).normalize()

    cutoffs = pd.date_range(first_cutoff, last_cutoff, freq="M")
    if max_months is not None and len(cutoffs) > max_months:
        cutoffs = cutoffs[-max_months:]
    if len(cutoffs) < 6:
        raise ValueError(f"Too few monthly cutoffs ({len(cutoffs)}) for temporal validation.")

    rows: List[pd.DataFrame] = []
    valid = valid.sort_values("InvoiceDate")
    raw_lines = raw_lines.sort_values("InvoiceDate")

    for i, obs_date in enumerate(cutoffs, start=1):
        history = valid[valid["InvoiceDate"] <= obs_date]
        if min_history_days > 0:
            first_purchase = history.groupby("CustomerID")["InvoiceDate"].min()
            eligible_ids = first_purchase[(obs_date - first_purchase).dt.days >= min_history_days].index
            history = history[history["CustomerID"].isin(eligible_ids)]
        if history.empty:
            continue
        raw_history = raw_lines[raw_lines["InvoiceDate"] <= obs_date]
        future = valid[(valid["InvoiceDate"] > obs_date) & (valid["InvoiceDate"] <= obs_date + pd.Timedelta(days=next_days))]
        future_customers = set(future["CustomerID"].unique())

        panel_i = _aggregate_history(history, raw_history, obs_date)
        panel_i["RepurchaseNextNDays"] = panel_i["CustomerID"].isin(future_customers).astype(int)
        panel_i["horizon_days"] = next_days
        rows.append(panel_i)
        print(f"[panel] {i:02d}/{len(cutoffs)} cutoff={obs_date.date()} instances={len(panel_i)} positive_rate={panel_i['RepurchaseNextNDays'].mean():.3f}")

    panel = pd.concat(rows, ignore_index=True)
    panel = panel.sort_values(["obs_date", "CustomerID"]).reset_index(drop=True)
    return panel


# -----------------------------
# Modeling
# -----------------------------

def temporal_split(panel: pd.DataFrame, val_months: int, test_months: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    months = sorted(panel["obs_date"].dt.to_period("M").astype(str).unique())
    if len(months) < val_months + test_months + 2:
        # Fallback: at least 60/20/20 month split.
        n = len(months)
        train_end = max(1, int(n * 0.6))
        val_end = max(train_end + 1, int(n * 0.8))
        train_months = months[:train_end]
        val_months_list = months[train_end:val_end]
        test_months_list = months[val_end:]
    else:
        test_months_list = months[-test_months:]
        val_months_list = months[-(test_months + val_months):-test_months]
        train_months = months[:-(test_months + val_months)]

    panel = panel.copy()
    panel["obs_month_str"] = panel["obs_date"].dt.to_period("M").astype(str)
    train = panel[panel["obs_month_str"].isin(train_months)].copy()
    val = panel[panel["obs_month_str"].isin(val_months_list)].copy()
    test = panel[panel["obs_month_str"].isin(test_months_list)].copy()

    meta = {
        "train_months": train_months,
        "val_months": val_months_list,
        "test_months": test_months_list,
        "train_instances": int(len(train)),
        "val_instances": int(len(val)),
        "test_instances": int(len(test)),
        "train_positive_rate": float(train["RepurchaseNextNDays"].mean()),
        "val_positive_rate": float(val["RepurchaseNextNDays"].mean()),
        "test_positive_rate": float(test["RepurchaseNextNDays"].mean()),
    }
    return train, val, test, meta


def prepare_features(panel: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str], List[str]]:
    drop_cols = {
        "RepurchaseNextNDays",
        "CustomerID",
        "obs_date",
        "obs_month_str",
        "first_purchase_date",
        "last_purchase_date",
        "horizon_days",
    }
    feature_cols = [c for c in panel.columns if c not in drop_cols]
    X = panel[feature_cols].copy()
    y = panel["RepurchaseNextNDays"].astype(int).copy()

    # Keep only reasonable categorical fields; obs_year/quarter/month are numeric.
    categorical_features = [c for c in X.columns if X[c].dtype == "object" or str(X[c].dtype).startswith("category")]
    numeric_features = [c for c in X.columns if c not in categorical_features]

    # Ensure numeric columns are numeric.
    for c in numeric_features:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    for c in categorical_features:
        X[c] = X[c].astype(str).fillna("Unknown")

    return X, y, numeric_features, categorical_features


def build_preprocessor(numeric_features: List[str], categorical_features: List[str]) -> ColumnTransformer:
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", _onehot_encoder()),
    ])
    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, numeric_features),
            ("cat", cat_pipe, categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )


def build_models(seed: int, y_train: pd.Series) -> Dict[str, object]:
    pos = max(1, int(y_train.sum()))
    neg = max(1, int(len(y_train) - y_train.sum()))
    scale_pos_weight = neg / pos

    models: Dict[str, object] = {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs"),
        "Random Forest": RandomForestClassifier(
            n_estimators=350,
            max_depth=None,
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=250,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.85,
            random_state=seed,
        ),
    }

    if XGBClassifier is not None:
        models["XGBoost"] = XGBClassifier(
            n_estimators=180,
            learning_rate=0.05,
            max_depth=4,
            min_child_weight=2,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=1.0,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=seed,
            n_jobs=1,
            verbosity=0,
            scale_pos_weight=scale_pos_weight,
        )
    if LGBMClassifier is not None:
        models["LightGBM"] = LGBMClassifier(
            n_estimators=180,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.85,
            colsample_bytree=0.85,
            class_weight="balanced",
            random_state=seed,
            n_jobs=1,
            verbose=-1,
        )
    return models


def tune_threshold(y_true: np.ndarray, prob: np.ndarray) -> float:
    """Choose threshold maximizing F1 on validation data."""
    precision, recall, thresholds = precision_recall_curve(y_true, prob)
    if len(thresholds) == 0:
        return 0.5
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    best_idx = int(np.nanargmax(f1))
    return float(thresholds[best_idx])


def compute_metrics(y_true: np.ndarray, prob: np.ndarray, threshold: float) -> Dict[str, float]:
    pred = (prob >= threshold).astype(int)
    # roc_auc requires both classes.
    if len(np.unique(y_true)) < 2:
        roc = np.nan
        pr = np.nan
    else:
        roc = roc_auc_score(y_true, prob)
        pr = average_precision_score(y_true, prob)
    return {
        "PR_AUC": float(pr),
        "ROC_AUC": float(roc),
        "F1": float(f1_score(y_true, pred, zero_division=0)),
        "Balanced_Accuracy": float(balanced_accuracy_score(y_true, pred)),
        "Precision": float(precision_score(y_true, pred, zero_division=0)),
        "Recall": float(recall_score(y_true, pred, zero_division=0)),
        "Brier": float(brier_score_loss(y_true, prob)),
        "Threshold": float(threshold),
    }


def fit_and_evaluate(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    seed: int,
) -> Tuple[pd.DataFrame, Dict[str, Pipeline], str, Dict[str, object], Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, List[str], List[str]]]:
    X_train, y_train, numeric_features, categorical_features = prepare_features(train)
    X_val, y_val, _, _ = prepare_features(val)
    X_test, y_test, _, _ = prepare_features(test)

    # Align columns defensively.
    X_val = X_val[X_train.columns]
    X_test = X_test[X_train.columns]

    models = build_models(seed, y_train)
    rows = []
    fitted: Dict[str, Pipeline] = {}

    for model_name, model in models.items():
        print(f"[model] fitting {model_name} ...")
        preprocessor = build_preprocessor(numeric_features, categorical_features)
        pipe = Pipeline([
            ("preprocess", preprocessor),
            ("model", model),
        ])
        pipe.fit(X_train, y_train)
        fitted[model_name] = pipe

        val_prob = pipe.predict_proba(X_val)[:, 1]
        threshold = tune_threshold(y_val.to_numpy(), val_prob)
        val_metrics = compute_metrics(y_val.to_numpy(), val_prob, threshold)

        test_prob = pipe.predict_proba(X_test)[:, 1]
        test_metrics = compute_metrics(y_test.to_numpy(), test_prob, threshold)

        row = {
            "Model": model_name,
            "Validation_PR_AUC": val_metrics["PR_AUC"],
            "Validation_ROC_AUC": val_metrics["ROC_AUC"],
            "Test_PR_AUC": test_metrics["PR_AUC"],
            "Test_ROC_AUC": test_metrics["ROC_AUC"],
            "Test_F1": test_metrics["F1"],
            "Test_Balanced_Accuracy": test_metrics["Balanced_Accuracy"],
            "Test_Precision": test_metrics["Precision"],
            "Test_Recall": test_metrics["Recall"],
            "Test_Brier": test_metrics["Brier"],
            "Threshold_from_validation": threshold,
        }
        rows.append(row)
        print(
            f"[model] {model_name}: val PR-AUC={val_metrics['PR_AUC']:.3f}, "
            f"test PR-AUC={test_metrics['PR_AUC']:.3f}, test F1={test_metrics['F1']:.3f}"
        )

    metrics = pd.DataFrame(rows).sort_values(["Validation_PR_AUC", "Test_PR_AUC"], ascending=False).reset_index(drop=True)
    best_model_name = str(metrics.iloc[0]["Model"])
    best_meta = {
        "best_model_selected_by": "Validation_PR_AUC",
        "best_model": best_model_name,
        "available_models": list(models.keys()),
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
    }
    return metrics, fitted, best_model_name, best_meta, (X_train, y_train, X_val, y_val, X_test, y_test, numeric_features, categorical_features)


# -----------------------------
# Explainability and DSS strategy mapping
# -----------------------------

def compute_feature_importance(
    pipe: Pipeline,
    model_name: str,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    numeric_features: List[str],
    categorical_features: List[str],
    shap_sample: int,
    seed: int,
    no_shap: bool,
) -> Tuple[pd.DataFrame, str]:
    """Compute SHAP mean absolute importance when possible, otherwise permutation/model importance."""
    rng = np.random.RandomState(seed)
    if len(X_test) > shap_sample:
        sample_idx = rng.choice(np.arange(len(X_test)), size=shap_sample, replace=False)
        X_sample = X_test.iloc[sample_idx].copy()
        y_sample = y_test.iloc[sample_idx].copy()
    else:
        X_sample = X_test.copy()
        y_sample = y_test.copy()

    pre = pipe.named_steps["preprocess"]
    model = pipe.named_steps["model"]
    feature_names = _get_feature_names(pre, numeric_features, categorical_features)

    # Try SHAP for tree models if available.
    if (not no_shap) and shap is not None and model_name not in {"Logistic Regression"}:
        try:
            X_trans = pre.transform(X_sample)
            explainer = shap.TreeExplainer(model)
            sv = explainer.shap_values(X_trans)
            if isinstance(sv, list):
                sv = sv[-1]
            sv_arr = np.asarray(sv)
            if sv_arr.ndim == 3:
                # Some explainers return (n, features, classes); take positive class.
                sv_arr = sv_arr[:, :, -1]
            vals = np.abs(sv_arr).mean(axis=0)
            imp = pd.DataFrame({"Feature": feature_names, "Importance": vals})
            imp["Method"] = "mean_abs_shap"
            return aggregate_onehot_importance(imp), "SHAP mean absolute value"
        except Exception as exc:
            print(f"[xai] SHAP failed for {model_name}: {exc}. Falling back to permutation importance.")

    # For models with native importances, use transformed feature importances.
    try:
        if hasattr(model, "feature_importances_"):
            vals = np.asarray(model.feature_importances_)
            imp = pd.DataFrame({"Feature": feature_names[: len(vals)], "Importance": vals})
            imp["Method"] = "model_feature_importance"
            return aggregate_onehot_importance(imp), "model feature importance"
        if hasattr(model, "coef_"):
            vals = np.abs(np.ravel(model.coef_))
            imp = pd.DataFrame({"Feature": feature_names[: len(vals)], "Importance": vals})
            imp["Method"] = "absolute_logistic_coefficient"
            return aggregate_onehot_importance(imp), "absolute logistic coefficient"
    except Exception as exc:
        print(f"[xai] native importance failed: {exc}. Falling back to permutation importance.")

    # Last-resort permutation importance on original feature columns.
    print("[xai] computing permutation importance; this may take several minutes.")
    result = permutation_importance(pipe, X_sample, y_sample, scoring="average_precision", n_repeats=5, random_state=seed, n_jobs=-1)
    imp = pd.DataFrame({"Feature": X_sample.columns, "Importance": result.importances_mean})
    imp["Method"] = "permutation_importance_pr_auc"
    imp = imp.sort_values("Importance", ascending=False).reset_index(drop=True)
    return imp, "permutation importance by PR-AUC"


def aggregate_onehot_importance(imp: pd.DataFrame) -> pd.DataFrame:
    """Aggregate one-hot country levels and clean feature names for readable result files."""
    def base_feature(f: str) -> str:
        # ColumnTransformer names may look like Country_United Kingdom.
        if f.startswith("Country_"):
            return "Country"
        return f

    imp = imp.copy()
    imp["RawFeature"] = imp["Feature"]
    imp["Feature"] = imp["Feature"].map(base_feature)
    agg = imp.groupby("Feature", as_index=False).agg(
        Importance=("Importance", "sum"),
        Method=("Method", "first"),
    )
    total = float(agg["Importance"].sum())
    agg["NormalizedImportance"] = agg["Importance"] / total if total > 0 else 0
    agg = agg.sort_values("Importance", ascending=False).reset_index(drop=True)
    agg.insert(0, "Rank", np.arange(1, len(agg) + 1))
    return agg


def map_feature_to_strategy(feature: str) -> Tuple[str, str, str]:
    f = feature.lower()
    if "recency" in f or "last_purchase" in f:
        return (
            "Reactivation targeting",
            "Customers with longer recency intervals require differentiated reactivation or reminder policies.",
            "Customer retention / lifecycle marketing",
        )
    if "revenue" in f or "monetary" in f or "avg_order" in f:
        return (
            "Customer-value segmentation",
            "High monetary value and order value support VIP retention, premium offers, or margin-aware prioritization.",
            "Customer value management",
        )
    if "invoice" in f or "frequency" in f or "active_purchase" in f:
        return (
            "Loyalty-frequency segmentation",
            "Purchase frequency indicates repeat engagement and can guide loyalty tiers or churn-prevention triggers.",
            "Loyalty and churn management",
        )
    if "unique_products" in f or "stock" in f or "product" in f:
        return (
            "Cross-selling and assortment recommendation",
            "Product diversity indicates cross-category interest and supports bundle, recommendation, or assortment actions.",
            "Merchandising / recommendation",
        )
    if "item" in f or "quantity" in f:
        return (
            "Bundle and volume-incentive design",
            "Quantity and item-volume behavior can guide bundle offers and quantity-sensitive incentives.",
            "Promotion design",
        )
    if "cancel" in f or "return" in f:
        return (
            "Return-risk monitoring",
            "Cancellation or return proxies identify customers/orders requiring service-quality or risk-control attention.",
            "Risk and service management",
        )
    if "month" in f or "quarter" in f or "year" in f or "sin" in f or "cos" in f:
        return (
            "Seasonal campaign planning",
            "Temporal patterns support seasonal promotion timing and campaign-calendar planning.",
            "Seasonal operations",
        )
    if "country" in f:
        return (
            "Localization and market segmentation",
            "Country effects indicate localization needs for communication, delivery, and market-specific promotion.",
            "Market localization",
        )
    if "unit_price" in f or "price" in f:
        return (
            "Price-band segmentation",
            "Price exposure supports price-band targeting and margin-aware promotion design.",
            "Pricing support",
        )
    if "tenure" in f or "first_purchase" in f:
        return (
            "Lifecycle-stage segmentation",
            "Customer tenure helps distinguish new, growing, and mature customer relationships.",
            "Customer lifecycle management",
        )
    return (
        "General customer analytics",
        "The feature provides additional evidence for customer-level decision support.",
        "General analytics",
    )


def build_strategy_drivers(feature_importance: pd.DataFrame, top_k_features: int = 15) -> pd.DataFrame:
    top = feature_importance.head(top_k_features).copy()
    mapped = []
    for _, row in top.iterrows():
        strategy, interpretation, domain = map_feature_to_strategy(str(row["Feature"]))
        mapped.append({
            "Feature": row["Feature"],
            "FeatureImportance": float(row["NormalizedImportance"]),
            "Strategy": strategy,
            "Interpretation": interpretation,
            "DecisionDomain": domain,
        })
    mdf = pd.DataFrame(mapped)
    grouped = mdf.groupby(["Strategy", "DecisionDomain", "Interpretation"], as_index=False).agg(
        EvidenceScore=("FeatureImportance", "sum"),
        SupportingFeatures=("Feature", lambda x: "; ".join(map(str, x))),
    )

    # Operational scores are transparent rubric values, not claimed as expert elicitation.
    rubric = {
        "Reactivation targeting": (0.88, 0.78, 0.82, 0.90),
        "Customer-value segmentation": (0.82, 0.76, 0.86, 0.84),
        "Loyalty-frequency segmentation": (0.86, 0.80, 0.88, 0.86),
        "Cross-selling and assortment recommendation": (0.72, 0.64, 0.78, 0.76),
        "Bundle and volume-incentive design": (0.76, 0.68, 0.80, 0.78),
        "Return-risk monitoring": (0.70, 0.74, 0.72, 0.82),
        "Seasonal campaign planning": (0.84, 0.82, 0.84, 0.88),
        "Localization and market segmentation": (0.74, 0.68, 0.78, 0.80),
        "Price-band segmentation": (0.78, 0.70, 0.78, 0.78),
        "Lifecycle-stage segmentation": (0.84, 0.78, 0.84, 0.86),
        "General customer analytics": (0.70, 0.70, 0.70, 0.70),
    }
    grouped["EvidenceScore"] = grouped["EvidenceScore"] / max(grouped["EvidenceScore"].max(), 1e-12)
    scores = grouped["Strategy"].map(lambda s: rubric.get(s, rubric["General customer analytics"]))
    grouped["Feasibility"] = [x[0] for x in scores]
    grouped["LowCost"] = [x[1] for x in scores]
    grouped["CustomerExperienceSafety"] = [x[2] for x in scores]
    grouped["Actionability"] = [x[3] for x in scores]
    weights = {
        "EvidenceScore": 0.40,
        "Feasibility": 0.20,
        "LowCost": 0.15,
        "CustomerExperienceSafety": 0.15,
        "Actionability": 0.10,
    }
    grouped["WeightedDSSScore"] = sum(grouped[k] * w for k, w in weights.items())
    grouped = grouped.sort_values("WeightedDSSScore", ascending=False).reset_index(drop=True)
    grouped.insert(0, "Rank", np.arange(1, len(grouped) + 1))
    return grouped


# -----------------------------
# Output helpers
# -----------------------------

def save_json(obj: Dict[str, object], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)


# -----------------------------
# Main
# -----------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Online Retail II external validation for e-commerce AI-DSS manuscript.")
    parser.add_argument("--input", type=str, required=True, help="Path to Online Retail II .xlsx/.xls/.csv file.")
    parser.add_argument("--output", type=str, default="online_retail_ii_external_results", help="Output directory.")
    parser.add_argument("--next-days", type=int, default=30, help="Repurchase horizon in days. Default: 30.")
    parser.add_argument("--test-months", type=int, default=3, help="Number of final months used as test. Default: 3.")
    parser.add_argument("--val-months", type=int, default=3, help="Number of months before test used as validation. Default: 3.")
    parser.add_argument("--min-history-days", type=int, default=0, help="Minimum customer history before an observation month. Default: 0.")
    parser.add_argument("--max-months", type=int, default=None, help="Use only the most recent N monthly cutoffs for faster testing.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed. Default: 42.")
    parser.add_argument("--shap-sample", type=int, default=1000, help="Test-sample size for SHAP/importance. Default: 1000.")
    parser.add_argument("--no-shap", action="store_true", help="Disable SHAP and use native/permutation importance.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    output_root = Path(args.output).expanduser().resolve()
    results_dir = output_root / "results_external"
    for d in [results_dir]:
        _safe_mkdir(d)

    print(f"[load] reading {input_path}")
    raw = load_online_retail_ii(input_path)
    colmap = infer_columns(raw)
    print(f"[load] inferred columns: {colmap}")
    valid, raw_lines = clean_transactions(raw, colmap)
    print(f"[clean] raw rows={len(raw)}, valid purchase rows={len(valid)}, valid customers={valid['CustomerID'].nunique()}")

    print("[panel] building customer-month panel")
    panel = build_customer_month_panel(
        valid,
        raw_lines,
        next_days=args.next_days,
        min_history_days=args.min_history_days,
        max_months=args.max_months,
    )
    panel_path = results_dir / "online_retail_ii_customer_month_panel.csv.gz"
    panel.to_csv(panel_path, index=False, compression="gzip")

    dataset_summary = {
        "raw_rows": int(len(raw)),
        "valid_purchase_rows": int(len(valid)),
        "valid_customers": int(valid["CustomerID"].nunique()),
        "date_min": str(valid["InvoiceDate"].min().date()),
        "date_max": str(valid["InvoiceDate"].max().date()),
        "panel_instances": int(len(panel)),
        "panel_customers": int(panel["CustomerID"].nunique()),
        "panel_months": int(panel["obs_date"].dt.to_period("M").nunique()),
        "panel_positive_rate": float(panel["RepurchaseNextNDays"].mean()),
        "next_days": int(args.next_days),
    }
    pd.Series(dataset_summary).to_csv(results_dir / "online_retail_ii_dataset_summary.csv")
    save_json(dataset_summary, results_dir / "online_retail_ii_dataset_summary.json")

    print("[split] temporal validation split")
    train, val, test, split_meta = temporal_split(panel, args.val_months, args.test_months)
    save_json(split_meta, results_dir / "online_retail_ii_split_metadata.json")
    print(f"[split] train={len(train)}, val={len(val)}, test={len(test)}")
    print(f"[split] test months={split_meta['test_months']}")

    print("[fit] fitting and evaluating models")
    metrics, fitted, best_model_name, best_meta, feat_data = fit_and_evaluate(train, val, test, args.seed)
    metrics.to_csv(results_dir / "online_retail_ii_model_metrics.csv", index=False)
    save_json(best_meta, results_dir / "online_retail_ii_best_model_metadata.json")

    X_train, y_train, X_val, y_val, X_test, y_test, numeric_features, categorical_features = feat_data
    print(f"[xai] computing feature importance for best model: {best_model_name}")
    feature_importance, importance_method = compute_feature_importance(
        fitted[best_model_name],
        best_model_name,
        X_test,
        y_test,
        numeric_features,
        categorical_features,
        shap_sample=args.shap_sample,
        seed=args.seed,
        no_shap=args.no_shap,
    )
    feature_importance.to_csv(results_dir / "online_retail_ii_feature_importance.csv", index=False)

    print("[dss] building strategy-level decision-support drivers")
    strategies = build_strategy_drivers(feature_importance)
    strategies.to_csv(results_dir / "online_retail_ii_strategy_drivers.csv", index=False)

    print("[output] result CSV/JSON files written")

    print("\n[DONE] External validation completed.")
    print(f"Output directory: {output_root}")
    print("Please send back the complete output folder or at least:")
    print(f"  - {results_dir / 'online_retail_ii_model_metrics.csv'}")
    print(f"  - {results_dir / 'online_retail_ii_feature_importance.csv'}")
    print(f"  - {results_dir / 'online_retail_ii_strategy_drivers.csv'}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\n[ERROR] {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
