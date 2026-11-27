"""Shared training / evaluation utilities for FinGuard (both modules).

Design rules:
  * Everything that learns from data (imputer, scaler, SMOTE) lives INSIDE a
    pipeline, so it is fit on training folds only -> no data leakage.
  * The same model zoo and the same metrics are used for the UPI fraud module
    and the credit risk module, so results are directly comparable.
"""
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from imblearn.over_sampling import SMOTE  # noqa: E402
from imblearn.pipeline import Pipeline as ImbPipeline  # noqa: E402
from sklearn.ensemble import (AdaBoostClassifier, IsolationForest,  # noqa: E402
                              RandomForestClassifier)
from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (average_precision_score, confusion_matrix,  # noqa: E402
                             f1_score, log_loss, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_validate  # noqa: E402
from sklearn.naive_bayes import GaussianNB  # noqa: E402
from sklearn.neighbors import KNeighborsClassifier  # noqa: E402
from sklearn.neural_network import MLPClassifier  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # noqa: E402
from sklearn.svm import SVC  # noqa: E402
from sklearn.tree import DecisionTreeClassifier  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

RANDOM_STATE = 42

# identifiers / time stamps / answer-leaking columns are never used as model features
NON_FEATURES = {"txn_id", "txn_time", "applicant_id", "fraud_type"}


# --------------------------------------------------------------------------
# Preprocessing
# --------------------------------------------------------------------------
def make_preprocessor(df, target, drop_cols=()):
    """Return (ColumnTransformer, feature_columns). Impute + scale numerics, one-hot categoricals."""
    feats = [c for c in df.columns if c != target and c not in drop_cols and c not in NON_FEATURES]
    cat = [c for c in feats if not pd.api.types.is_numeric_dtype(df[c])]
    num = [c for c in feats if c not in cat]
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), num),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat),
    ])
    return pre, feats


def get_feature_names(pre):
    return list(pre.get_feature_names_out())


# --------------------------------------------------------------------------
# Model zoo
# --------------------------------------------------------------------------
def get_models(imbalance="class_weight", fast=True, pos_weight=1.0, boosting=False):
    """Return {name: estimator}. imbalance in {'class_weight', 'none', 'smote'}.

    With 'smote' the class_weight is switched off (SMOTE is added in the pipeline instead).
    SVC / MLP / kNN are slow on big data, so run them on a subsample if fast=True
    (handled in the notebook via `max_rows_slow`).
    """
    cw = "balanced" if imbalance == "class_weight" else None
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight=cw, random_state=RANDOM_STATE),
        "Naive Bayes": GaussianNB(),
        "Decision Tree": DecisionTreeClassifier(max_depth=8, min_samples_leaf=20, class_weight=cw,
                                                random_state=RANDOM_STATE),
        "kNN": KNeighborsClassifier(n_neighbors=15, n_jobs=-1),
        "SVM (RBF)": SVC(kernel="rbf", C=1.0, probability=True, class_weight=cw, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=200, min_samples_leaf=3, class_weight=cw,
                                                n_jobs=-1, random_state=RANDOM_STATE),
        "AdaBoost": AdaBoostClassifier(n_estimators=150, learning_rate=0.5, random_state=RANDOM_STATE),
        "MLP": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, early_stopping=True,
                             random_state=RANDOM_STATE),
    }
    if boosting:
        # gradient boosting; pos_weight = (#negatives / #positives) mirrors class_weight='balanced'
        from lightgbm import LGBMClassifier
        from xgboost import XGBClassifier
        spw = pos_weight if imbalance == "class_weight" else 1.0
        models["XGBoost"] = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.08, subsample=0.8,
                                          colsample_bytree=0.8, scale_pos_weight=spw, tree_method="hist",
                                          eval_metric="aucpr", n_jobs=-1, random_state=RANDOM_STATE)
        models["LightGBM"] = LGBMClassifier(n_estimators=300, num_leaves=31, learning_rate=0.08, subsample=0.8,
                                            colsample_bytree=0.8, scale_pos_weight=spw, n_jobs=-1,
                                            random_state=RANDOM_STATE, verbose=-1)
    return models


SLOW_MODELS = {"SVM (RBF)", "kNN", "MLP"}


def build_pipeline(pre, model, imbalance="class_weight"):
    """Preprocessor -> (optional SMOTE) -> model. SMOTE runs only on training folds."""
    if imbalance == "smote":
        return ImbPipeline([("pre", pre), ("smote", SMOTE(random_state=RANDOM_STATE)), ("clf", model)])
    return Pipeline([("pre", pre), ("clf", model)])


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------
def ks_statistic(y_true, y_score):
    """Kolmogorov-Smirnov statistic, the standard separation measure in credit scoring."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(np.max(tpr - fpr))


def best_f1_threshold(y_true, y_score):
    p, r, t = precision_recall_curve(y_true, y_score)
    f1 = 2 * p * r / np.clip(p + r, 1e-12, None)
    i = int(np.nanargmax(f1[:-1]))
    return float(t[i])


def cost_optimal_threshold(y_true, y_score, cost_fn=10.0, cost_fp=1.0):
    """Pick the threshold minimising cost_fn*FN + cost_fp*FP (a missed fraud costs more than a false alarm)."""
    ths = np.unique(np.quantile(y_score, np.linspace(0.01, 0.999, 300)))
    best_t, best_c = 0.5, np.inf
    y_true = np.asarray(y_true)
    for t in ths:
        pred = y_score >= t
        fn = np.sum((~pred) & (y_true == 1))
        fp = np.sum(pred & (y_true == 0))
        c = cost_fn * fn + cost_fp * fp
        if c < best_c:
            best_c, best_t = c, t
    return float(best_t), float(best_c)


def holdout_metrics(y_true, y_score, threshold=0.5):
    """Full metric set at a threshold (default 0.5) plus threshold-free metrics."""
    y_pred = (y_score >= threshold).astype(int)
    return {
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_true, y_score),
        "PR-AUC": average_precision_score(y_true, y_score),
        "Log-loss": log_loss(y_true, np.clip(y_score, 1e-6, 1 - 1e-6)),
        "KS": ks_statistic(y_true, y_score),
    }


def cross_validate_models(X, y, pre, models, imbalance="class_weight", n_splits=5,
                          max_rows_slow=8000, verbose=True):
    """Stratified k-fold CV for every model. Returns a DataFrame sorted by PR-AUC.

    Slow models (SVM, kNN, MLP) are cross-validated on a stratified subsample
    of at most `max_rows_slow` rows to keep run time reasonable.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    scoring = {"precision": "precision", "recall": "recall", "f1": "f1",
               "roc_auc": "roc_auc", "pr_auc": "average_precision",
               "neg_log_loss": "neg_log_loss"}
    rows = []
    for name, model in models.items():
        Xm, ym = X, y
        note = ""
        if name in SLOW_MODELS and len(X) > max_rows_slow:
            Xm, ym = _stratified_subsample(X, y, max_rows_slow)
            note = f"(subsample {max_rows_slow})"
        pipe = build_pipeline(pre, model, imbalance)
        res = cross_validate(pipe, Xm, ym, cv=skf, scoring=scoring, n_jobs=1, error_score="raise")
        row = {"Model": name + (" " + note if note else ""),
               "Precision": res["test_precision"].mean(), "Recall": res["test_recall"].mean(),
               "F1": res["test_f1"].mean(), "ROC-AUC": res["test_roc_auc"].mean(),
               "PR-AUC": res["test_pr_auc"].mean(), "Log-loss": -res["test_neg_log_loss"].mean(),
               "PR-AUC std": res["test_pr_auc"].std()}
        rows.append(row)
        if verbose:
            print(f"  {name:<22} PR-AUC={row['PR-AUC']:.3f}  ROC-AUC={row['ROC-AUC']:.3f}  F1={row['F1']:.3f}")
    return pd.DataFrame(rows).sort_values("PR-AUC", ascending=False).reset_index(drop=True)


def fit_all_holdout(X_train, y_train, X_test, pre, models, imbalance="class_weight", max_rows_slow=8000):
    """Fit every model on the training split and return ({name: fitted_pipe}, {name: test_score})."""
    from sklearn.base import clone
    fitted, scores = {}, {}
    for name, model in models.items():
        Xt, yt = X_train, y_train
        if name in SLOW_MODELS and len(X_train) > max_rows_slow:
            Xt, yt = _stratified_subsample(X_train, y_train, max_rows_slow)
        pipe = build_pipeline(clone(pre), clone(model), imbalance)
        pipe.fit(Xt, yt)
        fitted[name] = pipe
        scores[name] = pipe.predict_proba(X_test)[:, 1]
    return fitted, scores


def _stratified_subsample(X, y, n):
    from sklearn.model_selection import train_test_split
    _, Xs, _, ys = train_test_split(X, y, test_size=n, stratify=y, random_state=RANDOM_STATE)
    return Xs, ys


def isolation_forest_scores(X_train_proc, X_test_proc, contamination):
    """Unsupervised anomaly score (higher = more anomalous), scaled to 0-1 for comparison."""
    iso = IsolationForest(n_estimators=200, contamination=contamination, random_state=RANDOM_STATE, n_jobs=-1)
    iso.fit(X_train_proc)
    raw = -iso.score_samples(X_test_proc)
    return (raw - raw.min()) / (raw.max() - raw.min() + 1e-12), iso


def bias_variance_curve(pipe, X, y, sizes=(0.1, 0.2, 0.4, 0.6, 0.8, 1.0), scoring="average_precision"):
    """Learning curve: training vs validation score as training size grows (bias-variance diagnostic)."""
    from sklearn.model_selection import learning_curve
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    sz, tr, va = learning_curve(pipe, X, y, train_sizes=list(sizes), cv=skf, scoring=scoring, n_jobs=1)
    return sz, tr.mean(axis=1), va.mean(axis=1)


# --------------------------------------------------------------------------
# Plots (all saved to reports/)
# --------------------------------------------------------------------------
def _save(fig, name):
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORTS_DIR / name
    fig.savefig(path, dpi=130, bbox_inches="tight")
    return path


def plot_roc_pr(results, y_true, prefix):
    """results: {model_name: y_score}. Saves one ROC and one PR figure."""
    fig, ax = plt.subplots(figsize=(6, 5))
    for n, s in results.items():
        fpr, tpr, _ = roc_curve(y_true, s)
        ax.plot(fpr, tpr, label=f"{n} ({roc_auc_score(y_true, s):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title=f"{prefix}: ROC curves")
    ax.legend(fontsize=7)
    _save(fig, f"{prefix}_roc.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    for n, s in results.items():
        p, r, _ = precision_recall_curve(y_true, s)
        ax.plot(r, p, label=f"{n} ({average_precision_score(y_true, s):.3f})")
    ax.axhline(np.mean(y_true), color="k", ls="--", lw=0.8, label="no-skill")
    ax.set(xlabel="Recall", ylabel="Precision", title=f"{prefix}: Precision-Recall curves")
    ax.legend(fontsize=7)
    _save(fig, f"{prefix}_pr.png")
    plt.close(fig)


def plot_confusion(y_true, y_score, threshold, title, fname):
    cm = confusion_matrix(y_true, (y_score >= threshold).astype(int))
    fig, ax = plt.subplots(figsize=(4, 3.6))
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, f"{v}", ha="center", va="center",
                color="white" if v > cm.max() / 2 else "black", fontsize=12)
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Legit/Good", "Fraud/Default"],
           yticklabels=["Legit/Good", "Fraud/Default"], xlabel="Predicted", ylabel="Actual",
           title=f"{title}\nthreshold={threshold:.3f}")
    _save(fig, fname)
    plt.close(fig)
    return cm


def plot_learning_curve(sizes, tr, va, title, fname):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(sizes, tr, "o-", label="Training")
    ax.plot(sizes, va, "o-", label="Validation")
    ax.set(xlabel="Training examples", ylabel="PR-AUC", title=title)
    ax.legend()
    _save(fig, fname)
    plt.close(fig)


def plot_model_comparison(df_cv, title, fname, metric="PR-AUC"):
    d = df_cv.sort_values(metric)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(d["Model"], d[metric])
    ax.set(xlabel=metric, title=title)
    _save(fig, fname)
    plt.close(fig)


def shap_summary(fitted_pipe, X_sample, fname, title, max_display=12):
    """Global SHAP summary for a tree-based classifier inside a fitted pipeline."""
    import shap
    pre, clf = fitted_pipe.named_steps["pre"], fitted_pipe.named_steps["clf"]
    Xp = pre.transform(X_sample)
    names = get_feature_names(pre)
    explainer = shap.TreeExplainer(clf)
    sv = explainer.shap_values(Xp)
    if isinstance(sv, list):          # older shap: list per class
        sv = sv[1]
    elif getattr(sv, "ndim", 2) == 3:  # newer shap: (n, features, classes)
        sv = sv[:, :, 1]
    plt.figure()
    shap.summary_plot(sv, Xp, feature_names=names, max_display=max_display, show=False)
    plt.title(title)
    fig = plt.gcf()
    path = _save(fig, fname)
    plt.close(fig)
    return sv, names, path


def top_reasons(fitted_pipe, row_df, k=3, direction="up"):
    """Local explanation for one row (tree models).

    direction='up'   -> top-k features pushing risk UP (reasons for suspicion)
    direction='down' -> top-k features pushing risk DOWN (protective signals)
    """
    import shap
    pre, clf = fitted_pipe.named_steps["pre"], fitted_pipe.named_steps["clf"]
    Xp = pre.transform(row_df)
    names = get_feature_names(pre)
    sv = shap.TreeExplainer(clf).shap_values(Xp)
    if isinstance(sv, list):
        sv = sv[1]
    elif getattr(sv, "ndim", 2) == 3:
        sv = sv[:, :, 1]
    contrib = pd.Series(sv[0], index=names).sort_values(ascending=(direction == "down"))
    return contrib.head(k)


# --------------------------------------------------------------------------
# Three-tier decisions: Allow / Verify / Block
# --------------------------------------------------------------------------
def oof_scores(pipe, X, y, n_splits=5):
    """Out-of-fold probabilities on the TRAINING data, used to choose thresholds
    without touching the test set."""
    from sklearn.model_selection import cross_val_predict
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    return cross_val_predict(pipe, X, y, cv=skf, method="predict_proba")[:, 1]


def choose_tiers(y_true, y_score, block_precision=0.95, verify_recall=0.95, max_review_share=0.05):
    """Pick two thresholds.

    block  : lowest score at which precision >= block_precision (be very sure before blocking)
    verify : highest score at which recall >= verify_recall (do not let fraud slip past unchecked),
             but never so low that more than `max_review_share` of ALL traffic is asked to verify
             (a review-capacity limit: customers and call-centres cannot handle more).
    Between the two, the customer is asked to verify (OTP / biometric / callback).
    """
    y_true = np.asarray(y_true)
    p, r, t = precision_recall_curve(y_true, y_score)
    p, r = p[:-1], r[:-1]
    ok_block = np.where(p >= block_precision)[0]
    block = float(t[ok_block[0]]) if len(ok_block) else float(np.quantile(y_score, 0.995))
    ok_verify = np.where(r >= verify_recall)[0]
    verify = float(t[ok_verify[-1]]) if len(ok_verify) else float(t[0])
    verify = max(verify, float(np.quantile(y_score, 1 - max_review_share)))   # capacity cap
    verify = min(verify, block)
    return {"verify": verify, "block": block}


def apply_tiers(y_score, tiers):
    """Map scores to 'Allow' / 'Verify' / 'Block'."""
    s = np.asarray(y_score)
    return np.where(s >= tiers["block"], "Block", np.where(s >= tiers["verify"], "Verify", "Allow"))


def tier_report(tier, is_fraud, fraud_type):
    """Table: for each tier, how many transactions, how many are real fraud, how many are genuine
    emergencies, and what share of all traffic reaches that tier."""
    d = pd.DataFrame({"tier": tier, "fraud": np.asarray(is_fraud),
                      "emergency": (np.asarray(fraud_type) == "legit_emergency").astype(int)})
    g = d.groupby("tier").agg(transactions=("fraud", "size"), real_fraud=("fraud", "sum"),
                              genuine_emergencies=("emergency", "sum"))
    g = g.reindex(["Allow", "Verify", "Block"]).fillna(0).astype(int)
    g["share_of_traffic_%"] = (100 * g["transactions"] / g["transactions"].sum()).round(2)
    return g


# --------------------------------------------------------------------------
# Credit scoring helper
# --------------------------------------------------------------------------
def pd_to_score(p, base_score=600, pdo=50, base_odds=19):
    """Convert probability of default to a 300-900 style score (higher = safer).

    Standard scorecard scaling: `pdo` points double the good:bad odds.
    """
    p = np.clip(p, 1e-6, 1 - 1e-6)
    odds_good = (1 - p) / p
    factor = pdo / np.log(2)
    offset = base_score - factor * np.log(base_odds)
    return np.clip(offset + factor * np.log(odds_good), 300, 900)


def risk_band(score):
    bins = [0, 550, 650, 750, 901]
    labels = ["High risk", "Medium-high", "Medium-low", "Low risk"]
    return pd.cut(pd.Series(np.atleast_1d(score)), bins=bins, labels=labels, right=False)


# --------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------
def save_model(obj, name):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / f"{name}.joblib"
    joblib.dump(obj, path)
    return path


def load_model(name):
    return joblib.load(MODELS_DIR / f"{name}.joblib")


# --------------------------------------------------------------------------
# Hyper-parameter tuning, calibration, time-based split
# --------------------------------------------------------------------------
PARAM_DISTS = {
    "XGBoost": {"clf__n_estimators": [150, 300, 500], "clf__max_depth": [3, 4, 5, 6, 8],
                "clf__learning_rate": [0.03, 0.05, 0.08, 0.12], "clf__subsample": [0.7, 0.8, 1.0],
                "clf__colsample_bytree": [0.6, 0.8, 1.0], "clf__min_child_weight": [1, 3, 5]},
    "LightGBM": {"clf__n_estimators": [150, 300, 500], "clf__num_leaves": [15, 31, 63],
                 "clf__learning_rate": [0.03, 0.05, 0.08, 0.12], "clf__subsample": [0.7, 0.8, 1.0],
                 "clf__colsample_bytree": [0.6, 0.8, 1.0], "clf__min_child_samples": [10, 20, 40]},
    "Random Forest": {"clf__n_estimators": [150, 300, 500], "clf__max_depth": [None, 10, 20],
                      "clf__min_samples_leaf": [1, 3, 5, 10], "clf__max_features": ["sqrt", 0.5]},
}


def tune_model(pipe, X, y, name, n_iter=12, cv=3, scoring="average_precision"):
    """Randomised hyper-parameter search (PR-AUC) on the TRAINING data only.
    Returns (best_pipeline_refit_on_all_training_data, search_results_dataframe)."""
    from sklearn.model_selection import RandomizedSearchCV
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)
    rs = RandomizedSearchCV(pipe, PARAM_DISTS[name], n_iter=n_iter, scoring=scoring, cv=skf,
                            random_state=RANDOM_STATE, n_jobs=1, refit=True)
    rs.fit(X, y)
    res = pd.DataFrame(rs.cv_results_)[["mean_test_score", "std_test_score", "params"]]
    return rs.best_estimator_, res.sort_values("mean_test_score", ascending=False).reset_index(drop=True)


def calibrate_pipeline(pipe, X, y, method="isotonic", cv=3):
    """Wrap a pipeline so its probabilities are calibrated (a '30%' score then means ~30% in practice)."""
    from sklearn.calibration import CalibratedClassifierCV
    cal = CalibratedClassifierCV(pipe, method=method, cv=cv)
    cal.fit(X, y)
    return cal


def calibration_table(y_true, scores, bins=8):
    from sklearn.calibration import calibration_curve
    frac, mean_pred = calibration_curve(y_true, scores, n_bins=bins, strategy="quantile")
    return pd.DataFrame({"mean_predicted": mean_pred, "actual_fraction_positive": frac})


def plot_calibration(y_true, score_dict, fname, title="Calibration (reliability) curve"):
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import brier_score_loss
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="perfect")
    for n, s in score_dict.items():
        frac, mp = calibration_curve(y_true, s, n_bins=8, strategy="quantile")
        ax.plot(mp, frac, "o-", label=f"{n} (Brier {brier_score_loss(y_true, s):.4f})")
    ax.set(xlabel="Predicted probability", ylabel="Actual fraction of positives", title=title)
    ax.legend(fontsize=7)
    _save(fig, fname)
    plt.close(fig)


def time_split(df, time_col="txn_time", train_frac=0.7):
    """Chronological split: earliest `train_frac` of rows for training, the latest for testing."""
    order = df.sort_values(time_col).index
    cut = int(len(order) * train_frac)
    return order[:cut], order[cut:]


def live_eval_split(index):
    """Split the held-out test rows into a 'live' half (what analysts review in the alert queue and may
    label) and an 'evaluation' half (never shown to analysts, never used for training or retraining),
    so a retrained model can be compared fairly with the old one."""
    idx = pd.Index(index)
    return idx[idx % 2 == 0], idx[idx % 2 == 1]
