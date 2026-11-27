# FinGuard: Explainable ML for UPI Fraud Detection, Mule-Account Detection and Loan Credit Risk Scoring

AIML CBP project. One platform with shared preprocessing, a shared model zoo, shared evaluation and an explainability layer, covering three problems:

| Module | Question | Output |
|---|---|---|
| **A. UPI fraud** | Is this payment fraudulent? | Fraud probability, **Allow / Verify / Block**, reasons for *and against* risk |
| **B. Mule accounts** | Is this account passing stolen money along? | Account mule-risk from the transaction network, flagged rings |
| **C. Credit risk** | Will this applicant default? | 300-900 score, risk band, decision, reason codes |

> **Data note (state this in your report):** no real UPI data is public, so every dataset here is **synthetic**, produced by `src/generate_data.py` and `src/graph_module.py`. Fraud patterns: collect-request scams, fake QR / new-merchant VPAs, mule pass-through bursts, SIM-swap takeovers, plus **genuine emergency withdrawals** labelled legitimate so the model must use context. The results show that the *method* works, not real-world accuracy. The next step is validation on real labelled data (for example ULB `creditcard.csv`, Give Me Some Credit); the notebooks are structured so only the loading cell changes.

## Project structure
```
FinGuard/
  data/        upi_transactions.csv, credit_risk.csv, account_edges.csv, accounts.csv (generated)
  notebooks/   01_shared_training_evaluation.ipynb   core experiment (8 models, CV, SMOTE, tiers, SHAP, credit scorecard)
               02_advanced_features.ipynb            time split, boosting, tuning, calibration, mule graph,
                                                     fairness, drift, retraining
  src/         generate_data.py   train_utils.py   graph_module.py   fairness.py
               monitoring.py      retrain.py       app.py (Streamlit demo)
  tests/       20 unit tests (pytest)
  models/      saved models (.joblib)
  reports/     metric tables (.csv) and plots (.png)
  requirements.txt   README.md
```

## How to run
```bash
pip install -r requirements.txt
python src/generate_data.py             # UPI + credit data (seeded, reproducible)
python src/graph_module.py              # mule network, detector, plots
jupyter notebook notebooks/01_shared_training_evaluation.ipynb    # Run All (~5-8 min) - creates the models
jupyter notebook notebooks/02_advanced_features.ipynb             # Run All (~4 min)
streamlit run src/app.py                # demo (needs the models from notebook 01)
python -m pytest tests -q               # 20 tests
python src/retrain.py --dry-run         # compare a feedback-retrained model without changing anything
```

## What the system does
* **Allow / Verify / Block.** A high score triggers *step-up verification* (OTP / biometric / callback), not an automatic block, so a genuine customer in an emergency is only briefly inconvenienced. *Block* where precision is at least 95%; *Verify* where recall is at least 95% but capped so at most 5% of traffic is ever sent to verification. Thresholds are chosen on **out-of-fold predictions from training data**, never on the test set.
* **Context features:** payments previously made to the payee, payee category (medical, retail, utility...), usual location, amount vs the user's average and vs their largest past payment, device and SIM change, velocity.
* **Explanations (SHAP):** for each case, what raises the risk *and* what lowers it. Loan decisions come with plain-language reason codes.
* **Mule detection:** money that arrives and is passed on within minutes, down a chain or round a cycle, even when each payment alone looks normal.
* **Fairness audit, drift monitoring, probability calibration and retraining from analyst feedback** (details below).

## Demo app (`streamlit run src/app.py`)
| Tab | What it shows |
|---|---|
| UPI transaction check | Example scenarios (everyday, hospital emergency at 2 am, account takeover); decision plus reasons for and against |
| Analyst alert queue | Flagged transactions ranked by risk, open a case, mark *confirm fraud* / *it was the customer* (saved to `reports/feedback_log.csv`) |
| Loan check | Credit score, band, approve / review / reject, reason codes |
| Batch + drift | Upload a CSV, get a scored and ranked file back, plus a PSI drift report against the training data |
| Cost trade-off | Move the cost of a missed fraud and see the best threshold; table of operating points |
| Mule accounts | Cross-validated mule risk per account, flagged accounts, the detected ring |

## Results (seed 42)
**UPI fraud, time-based split** (train 1 Jul to 1 Sep, test 1 Sep to 28 Sep: 9,000 transactions, 154 frauds)

| Model | Precision | Recall | F1 | PR-AUC | ROC-AUC |
|---|---|---|---|---|---|
| Random Forest | 0.978 | 0.864 | 0.917 | 0.879 | 0.940 |
| LightGBM | 0.985 | 0.864 | 0.920 | 0.877 | 0.930 |
| XGBoost | 0.971 | 0.864 | 0.914 | 0.876 | 0.928 |
| AdaBoost | 0.973 | 0.708 | 0.820 | 0.874 | 0.931 |
| MLP | 0.884 | 0.740 | 0.806 | 0.835 | 0.925 |
| Decision Tree | 0.276 | 0.838 | 0.416 | 0.800 | 0.914 |
| kNN | 1.000 | 0.533 | 0.695 | 0.780 | 0.922 |
| Logistic Regression | 0.217 | 0.870 | 0.347 | 0.693 | 0.931 |
| SVM (RBF) | 0.786 | 0.526 | 0.630 | 0.690 | 0.929 |
| Naive Bayes | 0.366 | 0.364 | 0.365 | 0.323 | 0.928 |

**Three-tier outcome** (held-out test set of 6,000 transactions, Random Forest, thresholds from training data):

| Tier | Transactions | Real fraud | Genuine emergencies | Share of traffic |
|---|---|---|---|---|
| Allow | 5,701 | 15\* | 36 | 95.0% |
| Verify | 207 | 1 | 37 | 3.5% |
| Block | 92 | 88 | **0** | 1.5% |

\* The 0.2% of labels deliberately flipped as noise (legitimate-looking rows labelled fraud), which no model can catch. Every injected fraud type ended in Verify or Block.

The emergency scenario (new account, SIM change 7 days ago, Rs 25,000 at 2 am leaving 2% of the balance) scores about 34% and goes to **Verify**; the same payment with a new device, unknown payee and unusual location scores about 70% and is **blocked**.

**Mule-account detection** (3,000 accounts, 24,436 transfers, 77 mules, 5-fold CV, behavioural features only)

| Method | Precision | Recall | F1 | PR-AUC |
|---|---|---|---|---|
| Simple rule (3+ fast forwards, 80%+ passed on) | 0.961 | 0.636 | 0.766 | n/a |
| Random Forest on graph features | 0.884 | 0.987 | 0.933 | 0.993 |

Of 86 flagged accounts, 76 are real mules; the other 10 are legitimate family members who forward money quickly (the deliberate look-alikes). Raw money totals were removed from the features because mules happen to move larger sums in this synthetic network, which would be a shortcut, not a real signal.

**Credit risk, hold-out:** Logistic Regression ROC-AUC 0.801 (PR-AUC 0.362, KS 0.446), AdaBoost 0.794, MLP 0.785, Random Forest 0.776. The unweighted logistic scorecard separates risk well: actual default rate rises from 0% in "Low risk" to 22% in "High risk".

**Fairness (loan model, gender not an input):** approval rates by gender 87.4% (F) vs 88.2% (M), disparate impact 0.99; by age group 0.96 to 1.00; every group passes the 4/5 rule. Gender cannot be predicted from the other inputs (proxy AUC 0.501).

**Drift monitoring (PSI):** a normal batch scores 0.001 to 0.006 per feature (stable); a batch with shifted amounts, hours and payee types scores 0.76 to 1.25 (DRIFT), and the model's own score distribution jumps from PSI 0.02 to 0.93.

**Retraining from feedback:** with 154 simulated analyst labels, evaluation PR-AUC moves from 0.8415 to 0.8458 on a held-out half the analysts never see; the new model is promoted only if it is not worse, and the old one is kept as `upi_fraud_rf_prev.joblib`.

## Honest findings (worth saying in the viva)
* **Boosting and tuning did not help here.** XGBoost and LightGBM match Random Forest (PR-AUC about 0.88), and hyper-parameter tuning changed test PR-AUC by under 0.001. On clean synthetic data there is a ceiling; the gap is expected to appear on messier real data.
* **The time split showed no penalty** (PR-AUC 0.879 time split vs 0.872 random split) because the synthetic data has no change over time. The protocol is right; it just has nothing to catch on this data.
* **Calibration changed little** (Brier 0.0026 to 0.0025); the boosted model's probabilities were already reasonable.
* **Imbalance handling:** class weights and SMOTE did not beat the unweighted Random Forest on PR-AUC; they mainly trade precision for recall in simpler models.
* **Cost slider:** because fraud scores separate cleanly on this data, the single best threshold barely moves; the operating-points table shows the trade-off directly (for example a cut-off of 0.10 catches 89 of 104 frauds but flags 193 legitimate customers, 0.45 flags 6).

## Limitations
* Synthetic data: patterns are designed to be learnable, so scores are optimistic compared with real traffic. Context features (known payee, location, device, SIM) assume the bank or PSP has those signals.
* No real labelled fraud dataset is included yet. A file with transaction features but **no fraud label** was supplied, so it cannot be used for supervised training; a labelled version is needed.
* The mule network is small (3,000 accounts); real graphs have millions of nodes and need scalable graph processing.
* The feedback loop retrains on demand (`retrain.py`); it is not automated or scheduled.
* This is a prototype for learning, not a production lending or fraud system.

## Syllabus mapping
Unit II: logistic regression, preprocessing, bias-variance, all metrics, calibration | Unit III: decision tree, kNN, SVM, feature selection | Unit IV: MLP, Naive Bayes, stratified k-fold, model selection | Unit V: Random Forest, AdaBoost, boosting (XGBoost / LightGBM), Isolation Forest, clustering-style anomaly detection, graph-based detection.
