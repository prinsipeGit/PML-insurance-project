# Project: Regularized Logistic Regression on Prudential Life Insurance Data

Class project (MATH 104.1). Final deliverable is `main.ipynb` (repo root), developed in VS Code and submitted through Google Colab.

## Research framing
- **Dataset:** Kaggle "Prudential Life Insurance Assessment", `train.csv` only (the competition's `test.csv` has no labels). About 59,000 rows and 128 columns (`Id`, `Response`, ~126 features). Stored locally at `data/train.csv` (git-ignored).
- **Priority area:** health (life insurance underwriting based on medical history and medical keyword flags).
- **Target:** binary. `1` if `Response == 8`, else `0`. The direction of the 1–8 scale is not documented by Prudential, so describe class 8 neutrally as "the most common rating class" and state the assumption in the report.
- **Question:** Which applicant characteristics drive assignment to rating class 8, and can a sparse, interpretable model match a dense one?

## Project rules (from the instructions)
- **20% feature rule:** we may drop at most 20% of the original features (25 columns at ~126 features). `Id` is an identifier, not a feature. Every dropped column must be logged with a reason in the notebook.
- **No replication:** prior work used ordinal LR, multinomial LR, and XGBoost on the full 8-level target. Our differentiation: binary target, L1 vs L2 vs Elastic Net comparison, regularization paths, stability selection, calibration, and cost-sensitive thresholding.

## Notebook conventions (match the class sample notebook)
- Structure: title and intro → one imports cell → data source note → `### Preliminaries` → `### Baseline Logistic Regression` → regularized models.
- Short markdown explanation before each code cell. Formal but readable.
- Explore with `X.head()`, `X.info()`, `y.value_counts()`; encode the target with an explicit `.map({...})`.
- Split with `train_test_split(test_size=0.2, random_state=42, stratify=y)`.
- Models are `Pipeline`s ending in `("logreg", LogisticRegression(...))`. Baseline uses `penalty=None`.
- Evaluation per model: `classification_report`, `ConfusionMatrixDisplay` (`cmap="Blues"`), `roc_auc_score`, `RocCurveDisplay`. Use the `evaluate()` helper defined in the notebook rather than repeating this code.
- Prefix model-specific variables: `l1_y_pred`, `l2_y_prob`, `en_search`, etc.

## Technical rules
- Fixed seed: `random_state=42` everywhere.
- All preprocessing (imputation, scaling, one-hot encoding) lives inside the pipeline to avoid data leakage.
- Tune `C` (and `l1_ratio`) with `GridSearchCV` + `StratifiedKFold(5, shuffle=True, random_state=42)`, scoring `roc_auc`. Tune on the training set only.
- L1 and Elastic Net use `solver="saga"` with a high `max_iter`.
- The notebook must run top to bottom in Colab (Python 3.13). Keep the data-loading cell working both locally (`data/train.csv`) and in Colab (uploaded `train.csv`).
- Keep `requirements.txt` aligned with Colab's package versions.
- Never commit files in `data/`.
