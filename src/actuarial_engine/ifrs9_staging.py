"""
src/actuarial_engine/ifrs9_staging.py
════════════════════════════════════════════════════════════════════
VERSION 2.0 — IFRS 9 STAGING ENGINE (EXPANDED)
════════════════════════════════════════════════════════════════════

CHANGES FROM V1 → V2:
━━━━━━━━━━━━━━━━━━━━━
[NEW] Step 5b — Alternative Data SICR Trigger for thin-file borrowers:
      If thin_file_flag=1 AND alt_anomaly_flag=1 AND
      essential_payments_consistency < 0.4:
          → Stage 2 (SICR via alternative data signal)
      This allows the engine to classify distress in unbanked
      borrowers who have no DPD history to trigger conventional SICR.

[NEW] loan_type now includes 'Mobile_Credit' — a product type common
      in informal lending and mobile-money backed credit facilities.

[CHANGED] calculate_ecl_provision: added EIR discount factor to the
      function signature (was always 1.0 in V1 — not applied).
      V2 applies the discount: ECL = PD × EAD × LGD / (1+r)^t
      where r = EIR and t = remaining loan term in years.

[UNCHANGED] All 5 V1 staging steps preserved exactly (Steps 1–5).
            V1 abs_thresholds dict preserved and extended.
            V1 function signatures backward-compatible.
"""


# ── Absolute PD jump thresholds by loan type ─────────────────────────────
# [CHANGED V2] Added 'Mobile_Credit' — mobile-money backed facilities
ABS_THRESHOLDS = {
    'Personal':          0.08,
    'SME':               0.10,
    'Home':              0.05,
    'Auto':              0.06,
    'Debt Consolidation':0.09,
    'Mobile_Credit':     0.07,   # [NEW V2] informal / mobile-money product
}


def classify_stage(
    dpd:             int,
    restructured_flag: int,
    pd_curr:         float,
    pd_orig:         float,
    loan_type:       str,
    anomaly_flag:    int,
    rebuttal_flag:   int   = 0,
    # ── [NEW V2] thin-file parameters ─────────────────────────────────────
    thin_file_flag:              int   = 0,
    alt_anomaly_flag:            int   = 0,
    essential_payments_consistency: float = 1.0,
) -> tuple:
    """
    Classify an account into IFRS 9 Stage 1, 2, or 3.

    Returns
    ───────
    (stage: int, trigger: str)
    stage   — 1, 2, or 3
    trigger — which rule fired (for audit trail in SHAP dashboard)

    Decision Waterfall (6 steps):
    ─────────────────────────────
    Step 1  — Hard credit-impaired rule (DPD ≥ 90) → Stage 3
    Step 2  — DPD rebuttable presumption (DPD ≥ 30) → Stage 2
    Step 3  — Restructured / forbearance flag → Stage 2
    Step 4  — Quantitative PD triggers → Stage 2
    Step 5a — Behavioral corroboration (V1) → Stage 2
    Step 5b — [NEW V2] Alt-data SICR for thin-file borrowers → Stage 2
    Step 6  — Default → Stage 1
    """

    # ── Step 1: Hard Credit-Impaired Rule ────────────────────────────────
    if dpd >= 90:
        return 3, "DPD_90_PLUS"

    # ── Step 2: DPD Rebuttable Presumption ───────────────────────────────
    if dpd >= 30 and not rebuttal_flag:
        return 2, "DPD_30_REBUTTABLE"

    # ── Step 3: Restructured / Forbearance ───────────────────────────────
    if restructured_flag == 1:
        return 2, "RESTRUCTURED_FORBEARANCE"

    # ── Step 4: Quantitative PD Triggers ────────────────────────────────
    pd_multiple = pd_curr / pd_orig if pd_orig > 0 else 999.0
    pd_jump     = pd_curr - pd_orig
    threshold   = ABS_THRESHOLDS.get(loan_type, 0.05)

    if pd_multiple >= 2.0:
        return 2, f"PD_MULTIPLE_{pd_multiple:.2f}x"
    if pd_jump >= threshold:
        return 2, f"PD_ABS_JUMP_{pd_jump:.4f}"

    # ── Step 5a: V1 Behavioral Corroboration ────────────────────────────
    if anomaly_flag == 1 and pd_multiple >= 1.5:
        return 2, "BEHAVIORAL_ANOMALY_CORROBORATION"

    # ── Step 5b: [NEW V2] Alternative Data SICR for Thin-File Borrowers ──
    # Rationale: unbanked borrowers have no DPD history to trigger conventional
    # SICR. Instead, a collapse in digital payment consistency combined with
    # a behavioral anomaly flag serves as the equivalent early-warning signal.
    if (thin_file_flag == 1
            and alt_anomaly_flag == 1
            and essential_payments_consistency < 0.40):
        return 2, "THIN_FILE_ALT_DATA_SICR"

    # ── Step 6: Performing ────────────────────────────────────────────────
    return 1, "PERFORMING"


def calculate_ecl_provision(
    stage:          int,
    pd_12m:         float,
    pd_lifetime:    float,
    ead:            float,
    lgd:            float,
    discount_factor:float = 1.0,   # [CHANGED V2] now actually applied
    eir:            float = 0.0,   # [NEW V2] Effective Interest Rate for discounting
    remaining_term_years: float = 1.0,  # [NEW V2] loan horizon in years
) -> float:
    """
    Calculate the ECL provision in ZAR.

    ECL = EAD × PD × LGD × discount_factor
    where discount_factor = 1 / (1 + EIR)^t   [NEW V2 — V1 always used 1.0]

    Stage 1: 12-month ECL (PD over next year only)
    Stage 2/3: Lifetime ECL (PD over remaining loan term, discounted to PV)
    """
    if stage == 1:
        ecl = pd_12m * ead * lgd

    elif stage in (2, 3):
        # [NEW V2] Apply EIR discount if provided
        if eir > 0 and remaining_term_years > 0:
            discount_factor = 1.0 / ((1 + eir) ** remaining_term_years)
        ecl = pd_lifetime * ead * lgd * discount_factor

    else:
        raise ValueError(f"Invalid IFRS 9 stage: {stage}. Must be 1, 2, or 3.")

    return round(max(ecl, 0.0), 2)
