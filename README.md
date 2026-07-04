# E-Commerce AI-DSS Reproducibility Materials

This repository contains the reproducibility materials for the e-commerce AI-DSS study. It contains Python pipeline code, processed dataset files, result files, and Online Retail II external-validation output. 

## Repository contents

```text
scripts/
  run_pipeline.py
  run_online_retail_ii_external_validation.py

data/processed/
  online_shoppers_clean.csv

results/
  CSV and JSON outputs from the Online Shoppers experiment, including model
  metrics, feature ablation, SHAP importance, MCDA rankings, sensitivity
  analysis, month-holdout validation, and SHAP-stability checks.

external_online_retail_ii/results_external/
  Online Retail II customer-month panel and external-validation outputs,
  including dataset summary, temporal split metadata, model metrics,
  feature importance, strategy-driver results, and best-model metadata.
```

## Environment setup

Use Python 3.10 or 3.11. Install the required packages from the repository root:

```bash
pip install -r requirements.txt
```

## Script 1: Online Shoppers main pipeline

`run_pipeline.py` reproduces the session-level Online Shoppers experiment used in the main study. It loads the Online Shoppers Purchasing Intention dataset, trains the benchmark models, evaluates the full-feature and PageValues-free settings, computes SHAP-based feature importance, builds MCDA strategy rankings, and runs robustness checks.

Run from the repository root:

```bash
python scripts/run_pipeline.py
```

Expected outputs are written to:

```text
results/
  model_metrics_by_seed.csv
  model_metrics_summary.csv
  feature_ablation_by_seed.csv
  feature_ablation_summary.csv
  shap_feature_importance.csv
  shap_raw_feature_importance.csv
  strategy_decision_matrix.csv
  strategy_fuzzy_matrix_defuzzified.csv
  mcda_ranking.csv
  deployment_safe_shap_feature_importance.csv
  deployment_safe_strategy_decision_matrix.csv
  deployment_safe_strategy_fuzzy_matrix_defuzzified.csv
  deployment_safe_mcda_ranking.csv
  sensitivity_rank_samples.csv
  sensitivity_summary.csv
  sensitivity_rank_samples_uniform.csv
  sensitivity_summary_uniform.csv
  ranking_spearman_correlation.csv
  statistical_tests.csv
  month_holdout_validation.csv
  month_holdout_summary.csv
  shap_stability_*.csv
  shap_stability_pairwise.csv
  shap_stability_summary.csv
  run_metadata.json

data/processed/
  online_shoppers_clean.csv
```

If the raw Online Shoppers CSV is not already available in `data/raw/`, the script attempts to download it from the UCI Machine Learning Repository and then writes the cleaned processed file to `data/processed/`.

## Script 2: Online Retail II external validation

`run_online_retail_ii_external_validation.py` reproduces the transaction-level external validation. It converts the Online Retail II transaction log into a customer-month repurchase-prediction panel, applies a temporal train/validation/test split, trains the same family of tabular models, selects the best model by validation PR-AUC, computes feature importance for the selected model, and outputs strategy-level decision-support drivers.

The raw Online Retail II file is not redistributed in this repository. Download `online_retail_II.xlsx` from the UCI Machine Learning Repository and place it at:

```text
data/raw/online_retail_ii.xlsx
```

Run from the repository root:

```bash
python scripts/run_online_retail_ii_external_validation.py \
  --input data/raw/online_retail_ii.xlsx \
  --output online_retail_ii_external_results \
  --next-days 30 \
  --test-months 3 \
  --val-months 3 \
  --shap-sample 1000
```

Expected outputs are written to:

```text
online_retail_ii_external_results/results_external/
  online_retail_ii_customer_month_panel.csv.gz
  online_retail_ii_dataset_summary.csv
  online_retail_ii_dataset_summary.json
  online_retail_ii_split_metadata.json
  online_retail_ii_model_metrics.csv
  online_retail_ii_best_model_metadata.json
  online_retail_ii_feature_importance.csv
  online_retail_ii_strategy_drivers.csv
```

For a faster test run, use fewer monthly cutoffs and a smaller SHAP sample:

```bash
python scripts/run_online_retail_ii_external_validation.py \
  --input data/raw/online_retail_ii.xlsx \
  --output online_retail_ii_external_results_smoke \
  --max-months 8 \
  --shap-sample 300
```

## Data availability note

The original raw datasets should be downloaded from the UCI Machine Learning Repository. This repository provides processed data and result files used for reproducibility. 
