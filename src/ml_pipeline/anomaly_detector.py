"""
src/ml_pipeline/anomaly_detector.py
════════════════════════════════════════════════════════════════════
VERSION 2.0 — UPGRADED BEHAVIORAL ANOMALY DETECTOR
════════════════════════════════════════════════════════════════════

CHANGES FROM V1 → V2:
━━━━━━━━━━━━━━━━━━━━━
[NEW] Alternative data features joined into detection:
      mobile_wallet_balance_std, airtime_recharge_freq_60d,
      essential_payments_consistency are now fed into the
      Isolation Forest alongside V1 behavioral columns.
      This allows the detector to flag distress in THIN-FILE
      borrowers who have no formal DPD history.

[NEW] Thin-file borrowers get a separate detection pass:
      For debtors with thin_file_flag=1, the Isolation Forest
      runs ONLY on alternative data columns (no FICO leakage).
      This prevents the model penalising someone for lacking
      a formal credit file rather than for actual distress signals.

[NEW] Precision / Recall / F1 metrics printed at end of run
      (V1 had no classification evaluation on the anomaly detector).
      Ground-truth proxy: age_90_plus_days > 0 used as default label.

[NEW] contamination tuned to 0.08 (8%) to match realistic SA
      portfolio distress rates. V1 used 0.05 (5%) which underestimated.

[CHANGED] Output table: model_outputs_anomaly — same as V1 but now
          includes alt_anomaly_flag (thin-file specific flag) column.

[UNCHANGED] V1 anomaly_score and anomaly_flag columns preserved.
            V1 Isolation Forest configuration preserved for the
            banked borrower cohort.
"""

import os
import sqlite3
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score, f1_score


# ── V1 behavioral feature set (unchanged) ────────────────────────────────
V1_BEHAVIORAL_FEATURES = [
    'post_payday_burn_rate',
    'transaction_velocity_30d',
    'avg_transaction_amount',
    'gambling_merchant_ratio',
    'late_night_tx_pct',
    'balance_volatility',
    'overdraft_count_90d',
    'cash_advance_pct',
    'p2p_transfer_volume',
    'digital_engagement_score',
    'password_reset_frequency',
    'category_diversity',
]

# ── [NEW V2] Alternative data features for thin-file detection ────────────
ALT_DATA_FEATURES = [
    'mobile_money_inflows_30d',
    'mobile_wallet_balance_std',
    'airtime_recharge_freq_60d',
    'essential_payments_consistency',
]

# Combined feature set for banked borrowers in V2
V2_COMBINED_FEATURES = V1_BEHAVIORAL_FEATURES + ALT_DATA_FEATURES


def detect_behavioral_anomalies():
    print("=" * 62)
    print("  V2 Behavioral Anomaly Radar — Isolation Forest")
    print("=" * 62)

    # ── 1. Connect and pull all three tables ──────────────────────────────
    db_path = os.path.join(os.path.dirname(__file__), '../../outputs/ifrs9_engine.db')
    conn    = sqlite3.connect(db_path)

    print("[1/5] Loading behavioral + alternative data from SQL...")

    # V1 behavioral data
    behavioral_df = pd.read_sql("SELECT * FROM debtors_behavioral", conn)

    # [NEW V2] Alternative data
    alt_df = pd.read_sql("SELECT * FROM debtors_alternative_data", conn)

    # [NEW V2] Financial table for thin_file_flag + ground-truth proxy
    financial_df = pd.read_sql(
        "SELECT debtor_id, thin_file_flag, age_90_plus_days FROM debtors_financial",
        conn
    )

    # ── 2. Merge all feature blocks ───────────────────────────────────────
    # [NEW V2] Merge alternative data; V1 only used behavioral
    merged = (
        behavioral_df
        .merge(alt_df[['debtor_id'] + ALT_DATA_FEATURES], on='debtor_id', how='left')
        .merge(financial_df, on='debtor_id', how='left')
    )

    # Split into banked vs thin-file cohorts
    banked_mask    = merged['thin_file_flag'] == 0
    thin_file_mask = merged['thin_file_flag'] == 1

    print(f"     Banked borrowers:     {banked_mask.sum():,}")
    print(f"     Thin-file borrowers:  {thin_file_mask.sum():,}")

    # ── 3a. [V1 COHORT] Isolation Forest on banked borrowers ─────────────
    # [CHANGED] contamination: 0.05 → 0.08 (reflects SA distress reality)
    # [CHANGED] features: V1 cols only → V2 combined cols for banked cohort
    print("[2/5] Training Isolation Forest on BANKED cohort (V2 combined features)...")

    banked_features = merged.loc[banked_mask, V2_COMBINED_FEATURES].fillna(0)

    iso_banked = IsolationForest(
        n_estimators=300,
        contamination=0.08,    # [CHANGED] was 0.05
        random_state=42,
        n_jobs=-1,
    )
    banked_preds  = iso_banked.fit_predict(banked_features)
    banked_scores = iso_banked.decision_function(banked_features) * -1

    # ── 3b. [NEW V2] Separate thin-file detection pass ────────────────────
    # Thin-file borrowers: ONLY alternative data features
    # Rationale: feeding FICO-correlated behavioral features into thin-file
    # detection would penalise them for lacking formal banking history,
    # not for actual distress signals.
    print("[3/5] Training Isolation Forest on THIN-FILE cohort (alt-data only)...")

    thin_features = merged.loc[thin_file_mask, ALT_DATA_FEATURES].fillna(0)

    iso_thin = IsolationForest(
        n_estimators=300,
        contamination=0.10,   # thin-file expected to show higher volatility
        random_state=42,
        n_jobs=-1,
    )
    thin_preds  = iso_thin.fit_predict(thin_features)
    thin_scores = iso_thin.decision_function(thin_features) * -1

    # ── 4. Assemble output ────────────────────────────────────────────────
    print("[4/5] Assembling anomaly output table...")

    result_scores = np.zeros(len(merged))
    result_flags  = np.zeros(len(merged), dtype=int)
    alt_flags     = np.zeros(len(merged), dtype=int)

    result_scores[banked_mask]  = banked_scores
    result_scores[thin_file_mask] = thin_scores

    result_flags[banked_mask]     = (banked_preds == -1).astype(int)
    result_flags[thin_file_mask]  = (thin_preds   == -1).astype(int)

    # [NEW V2] alt_anomaly_flag: set for thin-file borrowers who are flagged
    alt_flags[thin_file_mask] = (thin_preds == -1).astype(int)

    results_df = pd.DataFrame({
        'debtor_id':       merged['debtor_id'],
        'anomaly_score':   result_scores,
        'anomaly_flag':    result_flags,
        'alt_anomaly_flag': alt_flags,   # [NEW V2] thin-file specific
    })

    # ── 5. Precision / Recall / F1 evaluation ─────────────────────────────
    # [NEW V2] V1 had no evaluation metrics on the anomaly detector
    print("[5/5] Evaluating anomaly detector performance...")
    ground_truth = (merged['age_90_plus_days'] > 0).astype(int)
    gt_mask      = ~merged['age_90_plus_days'].isna()

    if gt_mask.sum() > 0:
        y_true = ground_truth[gt_mask]
        y_pred = results_df['anomaly_flag'][gt_mask]
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall    = recall_score(y_true, y_pred, zero_division=0)
        f1        = f1_score(y_true, y_pred, zero_division=0)
        print(f"\n  ── Anomaly Detector Classification Metrics ──")
        print(f"     Precision : {precision:.4f}")
        print(f"     Recall    : {recall:.4f}")
        print(f"     F1-Score  : {f1:.4f}")
        print(f"     (Ground truth proxy: age_90_plus_days > 0)")

    # ── 6. Write to database ───────────────────────────────────────────────
    results_df.to_sql('model_outputs_anomaly', con=conn, if_exists='replace', index=False)
    conn.close()

    total_flagged      = results_df['anomaly_flag'].sum()
    thin_flagged       = results_df['alt_anomaly_flag'].sum()
    print(f"\n  ✓ Total anomalies flagged   : {total_flagged:,}")
    print(f"  ✓ Thin-file anomalies       : {thin_flagged:,}")
    print(f"  ✓ Output table              : model_outputs_anomaly")
    return results_df


if __name__ == "__main__":
    detect_behavioral_anomalies()
