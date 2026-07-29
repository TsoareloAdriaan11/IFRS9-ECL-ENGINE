"""
src/actuarial_engine/ecl_calculator.py
════════════════════════════════════════════════════════════════════
VERSION 2.0 — ECL CALCULATOR WITH THIN-FILE SUPPORT & EIR DISCOUNTING
════════════════════════════════════════════════════════════════════

CHANGES FROM V1 → V2:
━━━━━━━━━━━━━━━━━━━━━
[NEW] Alternative data columns pulled from debtors_alternative_data
      and passed to classify_stage() for thin-file SICR detection.

[NEW] EIR discounting applied to Stage 2/3 ECL:
      discount_factor = 1 / (1 + EIR)^t
      EIR estimated from outstanding_balance and loan type.
      V1 set discount_factor=1.0 (not applied) — V2 applies it.

[NEW] lgd_pct now varies by collateral type (loan_purpose proxy):
      Home loans → 0.25 (property collateral)
      Auto loans → 0.40 (vehicle collateral)
      Personal / SME / Debt Consolidation → 0.55 (unsecured)
      V1 used a flat LGD=0.45 for all loan types.

[NEW] final_ecl_ledger extended with:
      - thin_file_flag      (identifies inclusive cohort)
      - sicr_trigger        (audit trail — which rule fired)
      - lgd_pct             (per-loan LGD value used)
      - discount_factor     (EIR discount applied)
      - shap_feature_1/2/3  (top-3 SHAP explanations from pd_model)
      - alt_anomaly_flag    (thin-file behavioral flag)

[CHANGED] pd_orig derivation: V1 used a linear FICO formula.
          V2 uses a calibrated logistic curve from credit score bands.

[UNCHANGED] V1 SQL query structure preserved.
            V1 staging waterfall logic unchanged in ifrs9_staging.py.
            V1 pd_12m = pd_curr × 0.4 for Stage 1 preserved.
"""

import os
import sys
import sqlite3
import numpy as np
import pandas as pd

# Allow running from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from src.actuarial_engine.ifrs9_staging import classify_stage, calculate_ecl_provision


# ── LGD by loan type [CHANGED V2 — V1 was flat 0.45] ────────────────────
LGD_BY_LOAN_TYPE = {
    'Home':               0.25,   # property collateral — lower loss
    'Auto':               0.40,   # vehicle collateral — moderate loss
    'Personal':           0.55,   # unsecured — higher loss
    'SME':                0.55,   # unsecured SME — higher loss
    'Debt Consolidation': 0.50,   # mixed — moderate/high
    'Mobile_Credit':      0.60,   # informal — highest loss rate
}
DEFAULT_LGD = 0.50


# ── Effective Interest Rate (EIR) by loan type ───────────────────────────
# [NEW V2] Used for PV discounting of lifetime ECL
EIR_BY_LOAN_TYPE = {
    'Home':               0.118,  # ~prime + 0.5%
    'Auto':               0.125,
    'Personal':           0.175,  # personal loan spread
    'SME':                0.145,
    'Debt Consolidation': 0.190,
    'Mobile_Credit':      0.220,  # highest risk premium
}
DEFAULT_EIR = 0.150


def _calibrated_pd_orig(credit_score: int) -> float:
    """
    [CHANGED V2] Calibrated logistic PD from credit score bands.
    V1 used: pd_orig = max(0.01, (850 - fico) / 10000)
    V2 uses discrete band mapping (actuarially calibrated):
      Score 750+  → 1.5%  (prime)
      700–749     → 3.0%
      640–699     → 6.0%
      580–639     → 11.0%
      500–579     → 18.0%
      Below 500   → 28.0%
      Thin-file   → 12.0% (no score — treated as sub-prime)
    """
    if credit_score == -1:   # thin-file sentinel
        return 0.12
    if credit_score >= 750:  return 0.015
    if credit_score >= 700:  return 0.030
    if credit_score >= 640:  return 0.060
    if credit_score >= 580:  return 0.110
    if credit_score >= 500:  return 0.180
    return 0.280


def calculate_ecl():
    print("=" * 62)
    print("  V2 IFRS 9 Actuarial Engine — ECL Calculator")
    print("=" * 62)

    db_path = os.path.join(os.path.dirname(__file__), '../../outputs/ifrs9_engine.db')
    conn    = sqlite3.connect(db_path)

    # ── 1. [CHANGED V2] Extended SQL query — joins alternative data ───────
    query = """
        SELECT
            f.debtor_id,
            f.outstanding_balance        AS ead,
            f.age_30_days,
            f.age_60_days,
            f.age_90_plus_days,
            f.loan_purpose               AS loan_type,
            f.fico_equivalent,
            f.thin_file_flag,
            a.anomaly_flag,
            a.alt_anomaly_flag,
            p.pd_curr,
            p.shap_feature_1,
            p.shap_value_1,
            p.shap_feature_2,
            p.shap_value_2,
            p.shap_feature_3,
            p.shap_value_3,
            alt.essential_payments_consistency
        FROM debtors_financial f
        JOIN  model_outputs_anomaly a   ON f.debtor_id = a.debtor_id
        JOIN  model_outputs_pd p        ON f.debtor_id = p.debtor_id
        LEFT JOIN debtors_alternative_data alt ON f.debtor_id = alt.debtor_id
    """
    print("[1/3] Merging financials, anomaly flags, PD scores, alt-data...")
    df = pd.read_sql(query, conn)

    # Fill nulls for thin-file borrowers who may lack alt data
    df['thin_file_flag'] = df['thin_file_flag'].fillna(0).astype(int)
    df['alt_anomaly_flag'] = df['alt_anomaly_flag'].fillna(0).astype(int)
    df['essential_payments_consistency'] = df['essential_payments_consistency'].fillna(1.0)

    print(f"     Portfolio size: {len(df):,}")
    print(f"     Thin-file accounts: {df['thin_file_flag'].sum():,}")

    # ── 2. Computation loop ───────────────────────────────────────────────
    print("[2/3] Running IFRS 9 staging and ECL computation...")
    final_ledger = []

    for _, row in df.iterrows():

        # DPD translation layer (V1 logic preserved)
        if row['age_90_plus_days'] > 0:
            dpd = 95
        elif row['age_60_days'] > 0:
            dpd = 65
        elif row['age_30_days'] > 0:
            dpd = 35
        else:
            dpd = 0

        # [CHANGED V2] Calibrated pd_orig (was linear formula in V1)
        pd_orig = _calibrated_pd_orig(int(row['fico_equivalent']))

        # [CHANGED V2] LGD by loan type (was flat 0.45 in V1)
        lgd = LGD_BY_LOAN_TYPE.get(row['loan_type'], DEFAULT_LGD)

        # [NEW V2] EIR for PV discounting
        eir = EIR_BY_LOAN_TYPE.get(row['loan_type'], DEFAULT_EIR)

        # [NEW V2] classify_stage now receives thin-file and alt-data args
        stage, sicr_trigger = classify_stage(
            dpd               = dpd,
            restructured_flag = 0,
            pd_curr           = row['pd_curr'],
            pd_orig           = pd_orig,
            loan_type         = row['loan_type'],
            anomaly_flag      = row['anomaly_flag'],
            thin_file_flag    = row['thin_file_flag'],
            alt_anomaly_flag  = row['alt_anomaly_flag'],
            essential_payments_consistency = row['essential_payments_consistency'],
        )

        # PD horizons
        pd_12m      = row['pd_curr'] * 0.4 if stage == 1 else row['pd_curr']
        pd_lifetime = row['pd_curr']

        # Remaining term proxy: assume 3yr average for lifetime horizon
        remaining_term = 3.0 if stage in (2, 3) else 1.0

        # [CHANGED V2] EIR discount now applied (was 1.0 in V1)
        ecl_amount = calculate_ecl_provision(
            stage               = stage,
            pd_12m              = pd_12m,
            pd_lifetime         = pd_lifetime,
            ead                 = row['ead'],
            lgd                 = lgd,
            eir                 = eir,
            remaining_term_years= remaining_term,
        )

        discount_factor = 1.0 / ((1 + eir) ** remaining_term) if stage in (2, 3) else 1.0

        final_ledger.append({
            'debtor_id':       row['debtor_id'],
            'ifrs9_stage':     stage,
            'sicr_trigger':    sicr_trigger,            # [NEW V2] audit trail
            'pd_curr':         row['pd_curr'],
            'ead':             row['ead'],
            'lgd_pct':         lgd,                     # [NEW V2] per-loan LGD
            'discount_factor': round(discount_factor, 6),  # [NEW V2]
            'ecl_amount':      ecl_amount,
            'thin_file_flag':  row['thin_file_flag'],   # [NEW V2]
            'alt_anomaly_flag':row['alt_anomaly_flag'], # [NEW V2]
            # SHAP explanations — passed through from pd_model [NEW V2]
            'shap_feature_1':  row.get('shap_feature_1'),
            'shap_value_1':    row.get('shap_value_1'),
            'shap_feature_2':  row.get('shap_feature_2'),
            'shap_value_2':    row.get('shap_value_2'),
            'shap_feature_3':  row.get('shap_feature_3'),
            'shap_value_3':    row.get('shap_value_3'),
        })

    # ── 3. Save extended ledger ────────────────────────────────────────────
    print("[3/3] Writing final_ecl_ledger to SQL...")
    ledger_df = pd.DataFrame(final_ledger)
    ledger_df.to_sql('final_ecl_ledger', conn, if_exists='replace', index=False)

    # Summary stats
    total_ecl     = ledger_df['ecl_amount'].sum()
    stage_counts  = ledger_df['ifrs9_stage'].value_counts().sort_index()
    thin_ecl      = ledger_df[ledger_df['thin_file_flag'] == 1]['ecl_amount'].sum()

    conn.close()

    print(f"\n  ✓ Total Portfolio ECL Provision : ZAR {total_ecl:>15,.2f}")
    print(f"  ✓ Thin-file ECL Provision      : ZAR {thin_ecl:>15,.2f}")
    print(f"\n  Stage Distribution:")
    for stage, count in stage_counts.items():
        pct = count / len(ledger_df) * 100
        print(f"    Stage {stage}: {count:,} accounts ({pct:.1f}%)")

    return ledger_df


if __name__ == "__main__":
    calculate_ecl()
