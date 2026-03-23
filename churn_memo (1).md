# RESEARCH MEMORANDUM

**TO:** [Recruiting / Portfolio / Research Audience]  
**FROM:** Karen Colman Martinez · kcolmanmartinez@smith.edu  
**DATE:** March 2026  
**RE:** Streaming Subscriber Churn Prediction — Model Findings & Retention Recommendations

---

## Executive Summary

This memo presents a churn prediction analysis for a simulated streaming platform
(10,000 subscribers, 6.5% monthly churn rate). Using a full ML pipeline — logistic
regression, random forest, and XGBoost — I identify the behavioral and content
engagement signals that best predict subscriber cancellation and translate those
findings into a tiered retention intervention framework. The logistic regression model
achieves ROC-AUC of 0.750 and average precision of 0.234 (vs. a random baseline of
0.065), with payment failures, ad-supported plan type, and low engagement emerging as
the dominant churn drivers.

---

## Background

Monthly churn is the central metric for subscription video-on-demand (SVOD) platforms.
A 1-percentage-point reduction in monthly churn on a 50-million-subscriber base retains
500,000 subscribers per month — at an average revenue per user of ~$10/month, that
represents $60M in annualized revenue preservation before accounting for reduced
customer acquisition costs. A reliable churn model that ranks subscribers by
cancellation risk enables platforms to allocate retention spend efficiently: targeting
high-risk subscribers with win-back campaigns while avoiding costly discounts to
subscribers who would have stayed anyway.

---

## Data & Features

The analysis uses simulated subscriber data with 18 features across five categories,
calibrated to documented streaming platform behavior:

**Plan characteristics:** tenure (months), plan type (basic/standard/premium),
monthly price, ad-supported flag.

**Content engagement:** average weekly hours watched, number of distinct shows watched,
binge sessions per month, days since last watch.

**Platform behavior:** number of profiles, primary device type (TV/mobile/tablet/desktop).

**Friction signals:** customer support contacts, payment failures in the past 90 days.

**Content affinity:** primary genre (drama, comedy, sports, reality, kids, news).

The monthly churn rate of 6.5% is consistent with published SVOD industry benchmarks
(typically 5–8% monthly for mid-tier platforms).

---

## Methodology

I train and evaluate three models on an 80/20 train-test split with stratified sampling:

**Logistic Regression** (L2 regularization, StandardScaler) — serves as the
interpretable baseline. Coefficient signs directly indicate the direction of each
feature's effect on churn probability.

**Random Forest** (300 trees, max depth 8) — captures non-linear interactions between
features without assuming a parametric form. Feature importance scores reflect the
average reduction in node impurity.

**XGBoost** (300 rounds, learning rate 0.05, class weight balanced) — gradient boosting
with explicit handling of class imbalance via `scale_pos_weight`. Typically the
strongest performer in production churn systems.

**Primary evaluation metric:** Average Precision (area under the Precision-Recall curve),
which is more informative than ROC-AUC under class imbalance. A random classifier
would score 0.065 on this dataset; all three models materially exceed that baseline.

---

## Results

### Table 1: Model Performance

| Model | ROC-AUC | Avg Precision | Lift over Baseline |
|-------|---------|---------------|-------------------|
| Logistic Regression | 0.750 | 0.234 | 3.6× |
| Random Forest | 0.746 | 0.209 | 3.2× |
| XGBoost | 0.677 | 0.140 | 2.2× |

*Baseline Avg Precision = 0.065 (churn rate). Lift = model AP / baseline AP.*

Logistic regression is the top performer on both metrics — an unusual result that
reflects the relatively linear underlying data-generating process. In production
settings with richer feature sets and stronger non-linear interactions, tree-based
models typically outperform logistic regression.

### Table 2: Top Churn Drivers (Logistic Regression Coefficients)

| Feature | Direction | Interpretation |
|---------|-----------|----------------|
| Payment failures | ↑ churn | Strongest single predictor |
| Has ads (basic plan) | ↑ churn | Ad-tier subscribers churn ~2× premium rate |
| Support contacts | ↑ churn | Friction → disengagement |
| Days since last watch | ↑ churn | Recency is a leading indicator |
| Primary genre: news | ↑ churn | Lower platform loyalty |
| Avg hours/week | ↓ churn | Strongest retention signal |
| Binge sessions | ↓ churn | Series completion creates commitment |
| Primary genre: sports | ↓ churn | Live content drives appointment viewing |
| Multi-profile household | ↓ churn | Household adoption = institutional stickiness |
| Tenure | ↓ churn | Survivors of early window are loyal |

---

## Business Recommendations

**1. Automate payment failure response.**
Payment failure is the single strongest churn signal. An automated retention sequence
(grace period notification, easy payment update UI, one-click retry) triggered within
24 hours of a failed charge is the highest-ROI intervention available.

**2. Prioritize onboarding for new subscribers.**
The cohort analysis shows churn is highest in the first three months. A structured
onboarding flow — personalized series recommendations, completion nudges, multi-profile
setup prompts — targets the window of maximum risk with the lowest intervention cost.

**3. Deploy tiered retention spend using churn scores.**
Rather than applying blanket retention offers, score all subscribers monthly and
segment into three tiers: monitor (score < 30%), nurture (30–60%), and intervene
(> 60%). This concentrates discount and win-back spend on subscribers who are
genuinely at risk while preserving margin on those who would retain anyway.

**4. Investigate ad-tier churn economics.**
If ad-supported subscribers churn at materially higher rates, the incremental ad
revenue may be partially or fully offset by elevated acquisition costs to replace
churned subscribers. A targeted upgrade offer — timed around high-engagement moments
such as season premieres or live sports — could improve both retention and ARPU
simultaneously.

**5. Build engagement early.**
Weekly hours watched is the strongest retention signal. Content recommendation quality
and notification strategy (e.g., "new episodes of shows you watch" alerts) directly
influence this metric and should be treated as retention investments, not just
engagement plays.

---

## Limitations & Extensions

- **Live data** would add temporal patterns (seasonal churn, content release effects,
  price change responses) not captured in the simulated dataset.
- **Survival modeling** (Cox proportional hazards or discrete-time hazard model) would
  more rigorously handle time-to-churn and censored observations.
- **Temporal cross-validation** (train on months 1–12, test on months 13–18) would
  test model stability against concept drift.
- **Feature interactions** — particularly early tenure × low engagement — may warrant
  explicit engineering in a production model.

---

## References

- Ascarza, E. (2018). "Retention Futility: Targeting High-Risk Customers Might Be
  Ineffective." *Journal of Marketing Research*, 55(1): 80–98.
- Subscription Insider / Antenna (2022–2024). SVOD churn benchmarks, streaming industry reports.
- Chen, T. & Guestrin, C. (2016). "XGBoost: A Scalable Tree Boosting System." *KDD 2016*.
