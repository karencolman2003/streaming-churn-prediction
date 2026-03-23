"""
Streaming Subscriber Churn Prediction
======================================
Author: Karen Colman Martinez | kcolmanmartinez@smith.edu
Date:   March 2026

Research Question:
Which subscriber behaviors and content engagement patterns best predict
churn on a streaming platform — and how can we use that to inform
retention strategy?

Dataset:
Simulated streaming subscriber data (10,000 subscribers, 18 months)
modeled on documented churn drivers from streaming industry research.
Variables mirror what platforms like Paramount+ track operationally.

Models:
  1. Logistic Regression   — interpretable baseline
  2. Random Forest         — captures non-linear interactions
  3. XGBoost               — gradient boosting, production-grade

Evaluation: ROC-AUC, Precision-Recall, Confusion Matrix, Feature Importance
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, roc_curve, precision_recall_curve,
                              average_precision_score, confusion_matrix,
                              classification_report)
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)

# ─────────────────────────────────────────────────────────────────
# 1. SIMULATE STREAMING SUBSCRIBER DATA
# ─────────────────────────────────────────────────────────────────

N = 10_000

# Demographics & plan
tenure_months      = np.random.gamma(shape=2.5, scale=6, size=N).clip(1, 36).astype(int)
plan_type          = np.random.choice(["basic", "standard", "premium"], N, p=[0.35, 0.40, 0.25])
plan_price         = {"basic": 5.99, "standard": 9.99, "premium": 14.99}
monthly_price      = np.array([plan_price[p] for p in plan_type])
has_ads            = (plan_type == "basic").astype(int)

# Content engagement
avg_hours_per_week = np.random.gamma(shape=2, scale=3, size=N).clip(0.5, 40)
num_shows_watched  = np.random.poisson(lam=4, size=N).clip(1, 25)
binge_sessions_mo  = np.random.poisson(lam=2, size=N).clip(0, 15)
days_since_last_watch = np.random.exponential(scale=7, size=N).clip(0, 90).astype(int)

# Content genre preferences (one-hot)
genres = ["drama", "comedy", "sports", "reality", "kids", "news"]
primary_genre      = np.random.choice(genres, N, p=[0.28, 0.22, 0.18, 0.14, 0.10, 0.08])

# Platform behavior
num_profiles       = np.random.choice([1, 2, 3, 4], N, p=[0.45, 0.30, 0.15, 0.10])
device_type        = np.random.choice(["mobile", "tv", "tablet", "desktop"], N, p=[0.30, 0.40, 0.15, 0.15])
support_contacts   = np.random.poisson(lam=0.4, size=N).clip(0, 8)
payment_failures   = np.random.choice([0, 1, 2], N, p=[0.82, 0.13, 0.05])

# Churn signal construction (logistic model with documented drivers)
log_odds = (
    -1.8                                           # base (~6.5% churn rate, realistic for streaming)
    - 0.06  * tenure_months                        # longer tenure → lower churn
    - 0.10  * avg_hours_per_week                   # more engagement → lower churn
    - 0.07  * num_shows_watched                    # breadth of viewing → lower churn
    - 0.09  * binge_sessions_mo                    # binge behavior → lower churn
    + 0.04  * days_since_last_watch                # recency → higher churn
    + 0.60  * has_ads                              # ad-supported → higher churn
    + 0.90  * payment_failures                     # payment issues → strong churn signal
    + 0.25  * support_contacts                     # friction → higher churn
    - 0.20  * (num_profiles > 1).astype(int)       # multi-profile → stickier
    + 0.50  * (primary_genre == "news").astype(int)  # news viewers less sticky
    - 0.35  * (primary_genre == "sports").astype(int) # sports viewers stickier
    - 0.15  * (device_type == "tv").astype(int)    # TV viewers stickier
    + np.random.normal(0, 0.5, N)                  # noise
)
churn_prob = 1 / (1 + np.exp(-log_odds))
churned    = (np.random.uniform(size=N) < churn_prob).astype(int)

print(f"Dataset: N={N:,} subscribers  |  Churn rate: {churned.mean():.1%}")

# Build DataFrame
df = pd.DataFrame({
    "tenure_months":       tenure_months,
    "monthly_price":       monthly_price,
    "has_ads":             has_ads,
    "avg_hours_per_week":  avg_hours_per_week,
    "num_shows_watched":   num_shows_watched,
    "binge_sessions_mo":   binge_sessions_mo,
    "days_since_last_watch": days_since_last_watch,
    "num_profiles":        num_profiles,
    "support_contacts":    support_contacts,
    "payment_failures":    payment_failures,
    "genre_drama":         (primary_genre == "drama").astype(int),
    "genre_comedy":        (primary_genre == "comedy").astype(int),
    "genre_sports":        (primary_genre == "sports").astype(int),
    "genre_reality":       (primary_genre == "reality").astype(int),
    "genre_kids":          (primary_genre == "kids").astype(int),
    "genre_news":          (primary_genre == "news").astype(int),
    "device_tv":           (device_type == "tv").astype(int),
    "device_mobile":       (device_type == "mobile").astype(int),
    "churned":             churned,
})

FEATURES = [c for c in df.columns if c != "churned"]
X = df[FEATURES]
y = df["churned"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {len(X_train):,}  |  Test: {len(X_test):,}")


# ─────────────────────────────────────────────────────────────────
# 2. MODELS
# ─────────────────────────────────────────────────────────────────

# Logistic Regression
lr_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("lr", LogisticRegression(C=1.0, max_iter=1000, random_state=42))
])
lr_pipe.fit(X_train, y_train)
lr_prob = lr_pipe.predict_proba(X_test)[:, 1]

# Random Forest
rf = RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=20,
                             random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
rf_prob = rf.predict_proba(X_test)[:, 1]

# XGBoost
xgb = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
                     subsample=0.8, colsample_bytree=0.8,
                     scale_pos_weight=(y_train==0).sum()/(y_train==1).sum(),
                     eval_metric="logloss", random_state=42, verbosity=0)
xgb.fit(X_train, y_train,
        eval_set=[(X_test, y_test)], verbose=False)
xgb_prob = xgb.predict_proba(X_test)[:, 1]


# ─────────────────────────────────────────────────────────────────
# 3. EVALUATION
# ─────────────────────────────────────────────────────────────────

models = {
    "Logistic Regression": lr_prob,
    "Random Forest":       rf_prob,
    "XGBoost":             xgb_prob,
}

print("\n" + "="*55)
print("MODEL PERFORMANCE SUMMARY")
print("="*55)
print(f"{'Model':<25} {'ROC-AUC':>9} {'Avg Precision':>14}")
print("-"*55)
for name, prob in models.items():
    auc = roc_auc_score(y_test, prob)
    ap  = average_precision_score(y_test, prob)
    print(f"{name:<25} {auc:>9.4f} {ap:>14.4f}")

print("\nXGBoost Classification Report (threshold=0.5):")
print(classification_report(y_test, (xgb_prob >= 0.5).astype(int),
                             target_names=["Retained", "Churned"]))


# ─────────────────────────────────────────────────────────────────
# 4. VISUALIZATIONS
# ─────────────────────────────────────────────────────────────────

plt.style.use("seaborn-v0_8-whitegrid")
COLORS = {"Logistic Regression": "#2471a3",
          "Random Forest":       "#1e8449",
          "XGBoost":             "#b03a2e"}

fig = plt.figure(figsize=(18, 14))
gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.42, wspace=0.35)
fig.suptitle("Streaming Subscriber Churn Prediction\nModel Performance & Feature Analysis",
             fontsize=15, fontweight="bold")

# ── Panel A: ROC Curves ───────────────────────────────────────────
ax = fig.add_subplot(gs[0, 0])
for name, prob in models.items():
    fpr, tpr, _ = roc_curve(y_test, prob)
    auc = roc_auc_score(y_test, prob)
    ax.plot(fpr, tpr, lw=2, color=COLORS[name], label=f"{name} (AUC={auc:.3f})")
ax.plot([0,1],[0,1],"k--",lw=0.8)
ax.set_title("A. ROC Curves", fontweight="bold")
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.legend(fontsize=7.5)

# ── Panel B: Precision-Recall Curves ─────────────────────────────
ax = fig.add_subplot(gs[0, 1])
for name, prob in models.items():
    prec, rec, _ = precision_recall_curve(y_test, prob)
    ap = average_precision_score(y_test, prob)
    ax.plot(rec, prec, lw=2, color=COLORS[name], label=f"{name} (AP={ap:.3f})")
baseline = y_test.mean()
ax.axhline(baseline, color="grey", ls="--", lw=0.8, label=f"Baseline ({baseline:.2f})")
ax.set_title("B. Precision-Recall Curves", fontweight="bold")
ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
ax.legend(fontsize=7.5)

# ── Panel C: Confusion Matrix (XGBoost) ──────────────────────────
ax = fig.add_subplot(gs[0, 2])
cm = confusion_matrix(y_test, (xgb_prob >= 0.5).astype(int))
im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
ax.set_title("C. XGBoost Confusion Matrix", fontweight="bold")
ax.set_xticks([0,1]); ax.set_yticks([0,1])
ax.set_xticklabels(["Retained","Churned"]); ax.set_yticklabels(["Retained","Churned"])
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
for i in range(2):
    for j in range(2):
        ax.text(j, i, f"{cm[i,j]:,}", ha="center", va="center",
                color="white" if cm[i,j] > cm.max()/2 else "black", fontsize=11)

# ── Panel D: XGBoost Feature Importance ──────────────────────────
ax = fig.add_subplot(gs[1, :2])
feat_imp = pd.Series(xgb.feature_importances_, index=FEATURES).sort_values(ascending=True)
top15 = feat_imp.tail(15)
colors_bar = ["#b03a2e" if v > feat_imp.quantile(0.75) else "#2471a3" for v in top15]
ax.barh(top15.index, top15.values, color=colors_bar, alpha=0.85)
ax.set_title("D. XGBoost Feature Importance (Top 15)", fontweight="bold")
ax.set_xlabel("Importance Score")
ax.tick_params(labelsize=9)

# ── Panel E: Logistic Regression Coefficients ─────────────────────
ax = fig.add_subplot(gs[1, 2])
lr_coefs = pd.Series(
    lr_pipe.named_steps["lr"].coef_[0],
    index=FEATURES
).sort_values()
colors_coef = ["#b03a2e" if v > 0 else "#1e8449" for v in lr_coefs]
ax.barh(lr_coefs.index, lr_coefs.values, color=colors_coef, alpha=0.85)
ax.axvline(0, color="black", lw=0.8)
ax.set_title("E. Logistic Regression Coefficients\n(Red=↑churn, Green=↓churn)",
             fontweight="bold", fontsize=9)
ax.tick_params(labelsize=7.5)

# ── Panel F: Churn rate by tenure cohort ─────────────────────────
ax = fig.add_subplot(gs[2, 0])
df["tenure_cohort"] = pd.cut(df["tenure_months"],
                              bins=[0,3,6,12,18,24,36],
                              labels=["0-3m","3-6m","6-12m","12-18m","18-24m","24-36m"])
cohort_churn = df.groupby("tenure_cohort", observed=True)["churned"].mean()
ax.bar(cohort_churn.index, cohort_churn.values * 100,
       color="#2471a3", alpha=0.85, edgecolor="white")
ax.set_title("F. Churn Rate by Tenure Cohort", fontweight="bold")
ax.set_xlabel("Tenure"); ax.set_ylabel("Churn Rate (%)")
ax.tick_params(labelsize=8)

# ── Panel G: Churn rate by engagement bucket ─────────────────────
ax = fig.add_subplot(gs[2, 1])
df["engagement"] = pd.cut(df["avg_hours_per_week"],
                           bins=[0,2,5,10,20,40],
                           labels=["<2h","2-5h","5-10h","10-20h","20h+"])
eng_churn = df.groupby("engagement", observed=True)["churned"].mean()
ax.bar(eng_churn.index, eng_churn.values * 100,
       color="#1e8449", alpha=0.85, edgecolor="white")
ax.set_title("G. Churn Rate by Weekly Engagement", fontweight="bold")
ax.set_xlabel("Avg Hours/Week"); ax.set_ylabel("Churn Rate (%)")
ax.tick_params(labelsize=8)

# ── Panel H: Churn rate by primary genre ─────────────────────────
ax = fig.add_subplot(gs[2, 2])
df["primary_genre"] = pd.Categorical(primary_genre)
genre_churn = df.groupby("primary_genre", observed=True)["churned"].mean().sort_values(ascending=False)
bar_colors = ["#b03a2e" if v > churned.mean() else "#2471a3" for v in genre_churn.values]
ax.bar(genre_churn.index, genre_churn.values * 100,
       color=bar_colors, alpha=0.85, edgecolor="white")
ax.axhline(churned.mean()*100, color="grey", ls="--", lw=1, label=f"Overall avg ({churned.mean():.1%})")
ax.set_title("H. Churn Rate by Primary Genre", fontweight="bold")
ax.set_xlabel("Genre"); ax.set_ylabel("Churn Rate (%)")
ax.legend(fontsize=8); ax.tick_params(labelsize=8)

plt.savefig("/home/claude/churn_main.png", dpi=150, bbox_inches="tight")
print("\nSaved: churn_main.png")


# ─────────────────────────────────────────────────────────────────
# 5. SCORE DISTRIBUTION & RETENTION STRATEGY CHART
# ─────────────────────────────────────────────────────────────────

fig2, axes2 = plt.subplots(1, 2, figsize=(14, 6))
fig2.suptitle("Churn Score Distribution & Retention Intervention Strategy",
              fontsize=13, fontweight="bold")

# Score distribution by actual outcome
ax = axes2[0]
retained_scores = xgb_prob[y_test == 0]
churned_scores  = xgb_prob[y_test == 1]
ax.hist(retained_scores, bins=40, alpha=0.6, color="#2471a3", label="Retained", density=True)
ax.hist(churned_scores,  bins=40, alpha=0.6, color="#b03a2e", label="Churned",  density=True)
ax.axvline(0.5, color="black", lw=1.2, ls="--", label="Threshold (0.5)")
ax.set_title("XGBoost Churn Score Distribution", fontweight="bold")
ax.set_xlabel("Predicted Churn Probability"); ax.set_ylabel("Density")
ax.legend(fontsize=9)

# Retention intervention matrix
ax = axes2[1]
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.set_title("Retention Intervention Framework", fontweight="bold")

zones = [
    (0.0, 0.3, "#d5e8d4", "LOW RISK\n(Monitor)\nChurn score < 30%\n→ Standard comms"),
    (0.3, 0.6, "#fff2cc", "MEDIUM RISK\n(Nurture)\nChurn score 30-60%\n→ Content recs,\n   engagement push"),
    (0.6, 1.0, "#f8cecc", "HIGH RISK\n(Intervene)\nChurn score > 60%\n→ Discount offer,\n   win-back campaign"),
]
for x0, x1, color, label in zones:
    ax.fill_between([x0, x1], [0, 0], [1, 1], alpha=0.5, color=color)
    ax.text((x0+x1)/2, 0.5, label, ha="center", va="center", fontsize=9,
            fontweight="bold", multialignment="center")

# Plot subscriber density as rug
for score, actual in zip(xgb_prob[:200], y_test.values[:200]):
    color = "#b03a2e" if actual == 1 else "#2471a3"
    ax.plot(score, np.random.uniform(0.05, 0.15), "|", color=color, alpha=0.5, ms=8)

ax.set_xlabel("Churn Score")
ax.set_yticks([])
ax.axvline(0.3, color="grey", lw=1, ls="--")
ax.axvline(0.6, color="grey", lw=1, ls="--")

plt.tight_layout()
plt.savefig("/home/claude/churn_strategy.png", dpi=150, bbox_inches="tight")
print("Saved: churn_strategy.png")

print("\n✓ All analysis complete.")
