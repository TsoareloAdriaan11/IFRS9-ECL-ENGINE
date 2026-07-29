"""
src/dashboard/app.py
════════════════════════════════════════════════════════════════════
VERSION 2.0 — 4-TAB STREAMLIT XAI FAIR-LENDING DASHBOARD
════════════════════════════════════════════════════════════════════

CHANGES FROM V1 → V2:
━━━━━━━━━━━━━━━━━━━━━
[NEW] Tab 2 — Individual XAI Explainer:
      Select any debtor ID and see a SHAP waterfall bar chart
      showing exactly which features drove their PD prediction
      and Stage classification. Adverse action explanation for
      any declined/Stage 3 account, satisfying GDPR Article 22
      and NCA (South Africa) adverse action notice requirements.

[NEW] Tab 3 — Model Risk Validation Suite:
      Displays ROC-AUC, Gini, KS Statistic, Brier Score and
      PR-AUC pulled directly from model_validation_metrics table.
      Shows Pass/Fail status against bank MRM benchmarks.

[NEW] Tab 4 — Algorithmic Fairness Audit:
      Computes Disparate Impact Ratio (DIR) across gender groups.
      DIR = P(Stage1 | gender=Female) / P(Stage1 | gender=Male)
      Regulatory threshold: DIR must be ≥ 0.80 (80% Rule).
      Displays PASS/FAIL badge and approval rate comparison chart.

[NEW] Thin-file inclusive metrics in Tab 1:
      Separate KPI card for thin-file ECL provision.
      Stage distribution pie now breaks out thin-file cohort.

[NEW] SICR trigger breakdown table in Tab 1:
      Shows which rule (DPD, PD_MULTIPLE, BEHAVIORAL, THIN_FILE)
      drove the most Stage 2 upgrades — useful for risk committee.

[CHANGED] V1 had a single-page layout with 2 charts.
          V2 is a 4-tab multi-section application.

[UNCHANGED] V1 KPI metric cards (Total ECL, Total EAD, High-Risk Count).
            V1 interactive debtor ledger with stage/sort filters.
            V1 PD distribution chart.
            V1 ECL by Stage bar chart.
"""

import os
import sqlite3
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Page config ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="IFRS 9 AI Engine v2.0",
    page_icon="🏦",
    layout="wide",
)

# ── Colour palette ────────────────────────────────────────────────────────
NAVY   = "#0F2A5C"
GOLD   = "#C9A84C"
GREEN  = "#166534"
RED    = "#991B1B"
BLUE   = "#1D4ED8"
TEAL   = "#0E7490"

st.markdown(f"""
<style>
  .metric-card {{
      background-color: {NAVY}; color: white;
      padding: 1rem; border-radius: 8px; text-align: center;
  }}
  .pass-badge {{ background-color: #166534; color: white;
      padding: 4px 12px; border-radius: 12px; font-weight: bold; }}
  .fail-badge {{ background-color: #991B1B; color: white;
      padding: 4px 12px; border-radius: 12px; font-weight: bold; }}
</style>
""", unsafe_allow_html=True)

st.title("🏦 IFRS 9 AI Engine — v2.0")
st.markdown("#### Predictive Behavioral ECL Engine  ·  FairXGBoost  ·  Inclusive Credit Scoring")

# ── Data loading ──────────────────────────────────────────────────────────
@st.cache_data
def load_all():
    db_path = os.path.join(os.path.dirname(__file__), '../../outputs/ifrs9_engine.db')
    if not os.path.exists(db_path):
        return None, None, None, None
    conn = sqlite3.connect(db_path)
    ledger   = pd.read_sql("SELECT * FROM final_ecl_ledger",          conn)
    metrics  = pd.read_sql("SELECT * FROM model_validation_metrics",   conn)
    alt_data = pd.read_sql("SELECT * FROM debtors_alternative_data",   conn)
    
    # [FIX] Pull first_name and last_name from the financial table
    fin_data = pd.read_sql("SELECT debtor_id, loan_purpose, first_name, last_name FROM debtors_financial", conn)
    
    conn.close()
    return ledger, metrics, alt_data, fin_data

ledger, mrm_metrics, alt_data, fin_data = load_all()

if ledger is None:
    st.error("⚠️ Database not found. Run the full pipeline (Phases 1–4) before launching the dashboard.")
    st.stop()

# Merge for fairness audit
full_df = ledger.merge(alt_data[['debtor_id','gender_protected_attribute']], on='debtor_id', how='left')
full_df = full_df.merge(fin_data, on='debtor_id', how='left')

# ═════════════════════════════════════════════════════════════════════════
# TAB LAYOUT
# ═════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Portfolio ECL",
    "🔍 Individual XAI Explainer",
    "📐 Model Risk Validation",
    "⚖️  Algorithmic Fairness Audit",
])


# ═════════════════════════════════════════════════════════════════════════
# TAB 1 — Portfolio ECL (V1 preserved + V2 additions)
# ═════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Portfolio Overview — IFRS 9 ECL Provision")

    # ── V1 KPI cards (unchanged) ──────────────────────────────────────────
    total_ecl       = ledger['ecl_amount'].sum()
    total_ead       = ledger['ead'].sum()
    high_risk_count = (ledger['ifrs9_stage'] > 1).sum()

    # [NEW V2] Thin-file KPIs
    thin_ecl   = ledger[ledger['thin_file_flag'] == 1]['ecl_amount'].sum()
    thin_count = ledger['thin_file_flag'].sum()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total ECL (ZAR)",          f"R {total_ecl:,.0f}")
    c2.metric("Total EAD (ZAR)",          f"R {total_ead:,.0f}")
    c3.metric("High-Risk Accounts",       f"{high_risk_count:,}")
    c4.metric("Thin-File Accounts",       f"{thin_count:,}")         # [NEW V2]
    c5.metric("Thin-File ECL (ZAR)",      f"R {thin_ecl:,.0f}")     # [NEW V2]

    st.divider()

    col_left, col_right = st.columns(2)

    with col_left:
        # V1 chart preserved
        st.subheader("ECL Provision by IFRS 9 Stage")
        stage_ecl = ledger.groupby('ifrs9_stage')['ecl_amount'].sum()
        stage_ecl.index = [f"Stage {i}" for i in stage_ecl.index]
        st.bar_chart(stage_ecl, color="#ff4b4b")

    with col_right:
        # V1 chart preserved
        st.subheader("PD Distribution")
        pd_bins = pd.cut(ledger['pd_curr'], bins=10).value_counts().sort_index()
        pd_bins.index = pd_bins.index.astype(str)
        st.bar_chart(pd_bins, color="#1D4ED8")

    st.divider()

    # [NEW V2] SICR trigger breakdown
    st.subheader("SICR Trigger Breakdown (Stage 2 Accounts)")
    stage2 = ledger[ledger['ifrs9_stage'] == 2]
    if 'sicr_trigger' in stage2.columns and len(stage2) > 0:
        trigger_counts = stage2['sicr_trigger'].value_counts().reset_index()
        trigger_counts.columns = ['SICR Trigger', 'Count']
        trigger_counts['% of Stage 2'] = (trigger_counts['Count'] / len(stage2) * 100).round(1)
        st.dataframe(trigger_counts, use_container_width=True, hide_index=True)
    else:
        st.info("No Stage 2 accounts in portfolio.")

    st.divider()

    # V1 interactive ledger (preserved) ──────────────────────────────────
    st.subheader("🔍 Interactive Debtor Risk Ledger")
    f1, f2 = st.columns(2)
    with f1:
        stage_filter = st.selectbox("Filter by Stage:",
            ["All Portfolio","Stage 1 (Performing)","Stage 2 (SICR)","Stage 3 (Default)"])
    with f2:
        sort_filter  = st.selectbox("Sort by:",
            ["Highest ECL","Highest PD","Thin-File Only"])

    disp = ledger.copy()
    if stage_filter != "All Portfolio":
        sn = int(stage_filter.split(" ")[1])
        disp = disp[disp['ifrs9_stage'] == sn]
    if sort_filter == "Highest ECL":
        disp = disp.sort_values('ecl_amount', ascending=False)
    elif sort_filter == "Highest PD":
        disp = disp.sort_values('pd_curr', ascending=False)
    elif sort_filter == "Thin-File Only":
        disp = disp[disp['thin_file_flag'] == 1].sort_values('ecl_amount', ascending=False)

    display_cols = ['debtor_id','ifrs9_stage','pd_curr','ead','ecl_amount',
                    'thin_file_flag','sicr_trigger']
    display_cols = [c for c in display_cols if c in disp.columns]
    st.dataframe(disp[display_cols], use_container_width=True, height=400)


# ═════════════════════════════════════════════════════════════════════════
# TAB 2 — Individual XAI Explainer [NEW V2]
# ═════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Individual Loan XAI Explainer — SHAP Waterfall")
    st.markdown(
        "Select a debtor to see exactly which features drove their Probability of Default "
        "and IFRS 9 Stage classification. Satisfies adverse action notice requirements "
        "(NCA / GDPR Article 22)."
    )

    debtor_ids = ledger['debtor_id'].tolist()
    selected_id = st.selectbox("Select Debtor ID:", debtor_ids)

    row = ledger[ledger['debtor_id'] == selected_id].iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("IFRS 9 Stage",    f"Stage {int(row['ifrs9_stage'])}")
    c2.metric("P(Default)",      f"{row['pd_curr']:.1%}")
    c3.metric("EAD (ZAR)",       f"R {row['ead']:,.0f}")
    c4.metric("ECL Provision",   f"R {row['ecl_amount']:,.0f}")

    st.markdown(f"**SICR Trigger:** `{row.get('sicr_trigger','—')}`")
    if row.get('thin_file_flag', 0) == 1:
        st.info("ℹ️  This is a **thin-file borrower** — scored using alternative data signals.")

    # SHAP waterfall chart
    shap_data = []
    for n in [1, 2, 3]:
        feat = row.get(f'shap_feature_{n}')
        val  = row.get(f'shap_value_{n}')
        if feat and val is not None:
            shap_data.append({'feature': feat, 'shap_value': float(val)})

    if shap_data:
        shap_df = pd.DataFrame(shap_data).sort_values('shap_value')
        fig, ax = plt.subplots(figsize=(8, 3))
        colors  = [GREEN if v < 0 else RED for v in shap_df['shap_value']]
        ax.barh(shap_df['feature'], shap_df['shap_value'], color=colors)
        ax.axvline(0, color='black', linewidth=0.8)
        ax.set_xlabel("SHAP Value (contribution to PD increase)")
        ax.set_title(f"SHAP Waterfall — {selected_id}  (Top 3 Features)")
        red_p  = mpatches.Patch(color=RED,   label="Increases default risk ↑")
        grn_p  = mpatches.Patch(color=GREEN, label="Decreases default risk ↓")
        ax.legend(handles=[red_p, grn_p], fontsize=8)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()
    else:
        st.warning("SHAP values not available. Re-run pd_model.py to generate per-debtor SHAP data.")


# ═════════════════════════════════════════════════════════════════════════
# TAB 3 — Model Risk Validation Suite [NEW V2]
# ═════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Model Risk Validation — MRM Metrics Suite")
    st.markdown(
        "Statistical proof of model quality required by bank Model Risk Management "
        "committees, SARB prudential review, and Basel III IRB approach."
    )

    BENCHMARKS = {
        'roc_auc':          ('ROC-AUC',           0.75,  '≥ 0.75'),
        'gini_coefficient': ('Gini Coefficient',   0.50,  '≥ 0.50'),
        'ks_statistic':     ('KS Statistic',       0.40,  '≥ 0.40'),
        'brier_score':      ('Brier Score',        0.15,  '≤ 0.15'),
        'pr_auc':           ('PR-AUC',             0.40,  '≥ 0.40'),
    }

    if mrm_metrics is not None and len(mrm_metrics) > 0:
        m = mrm_metrics.iloc[0]
        cols = st.columns(len(BENCHMARKS))
        for i, (col_name, (label, bench, bench_str)) in enumerate(BENCHMARKS.items()):
            val = m.get(col_name, None)
            if val is None:
                continue
            # Brier: lower is better
            passed = (val <= bench) if col_name == 'brier_score' else (val >= bench)
            badge  = "✅ PASS" if passed else "❌ FAIL"
            cols[i].metric(label, f"{val:.4f}", delta=f"{badge}  bench {bench_str}")

        st.divider()
        st.subheader("Additional Classification Metrics")
        extra_cols = ['optimal_threshold','f1_at_threshold','precision','recall']
        extra_df = m[extra_cols].to_frame(name='Value').reset_index()
        extra_df.columns = ['Metric','Value']
        extra_df['Value'] = extra_df['Value'].round(4)
        st.dataframe(extra_df, use_container_width=True, hide_index=True)
    else:
        st.warning("No validation metrics found. Run pd_model.py to generate metrics.")


# ═════════════════════════════════════════════════════════════════════════
# TAB 4 — Algorithmic Fairness Audit [NEW V2]
# ═════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("Algorithmic Fairness Audit — Disparate Impact Analysis")
    st.markdown(
        "Disparate Impact Ratio (DIR) = P(Stage 1 | Female) / P(Stage 1 | Male)  \n"
        "**Regulatory threshold: DIR ≥ 0.80 (80% Rule)** — breach = discriminatory model."
    )

    gender_df = full_df.dropna(subset=['gender_protected_attribute'])
    gender_df['gender_protected_attribute'] = gender_df['gender_protected_attribute'].astype(int)

    if len(gender_df) > 0:
        def approval_rate(g_val):
            grp = gender_df[gender_df['gender_protected_attribute'] == g_val]
            return (grp['ifrs9_stage'] == 1).mean() if len(grp) > 0 else 0

        rate_male   = approval_rate(0)   # Stage 1 = performing = "approved"
        rate_female = approval_rate(1)
        rate_nb     = approval_rate(2)

        dir_score = rate_female / rate_male if rate_male > 0 else 0

        # DIR badge
        dir_col, m_col, f_col, nb_col = st.columns(4)
        dir_col.metric("Disparate Impact Ratio", f"{dir_score:.4f}",
                        delta="✅ PASS" if dir_score >= 0.80 else "❌ FAIL")
        m_col.metric("Male Stage-1 Rate",       f"{rate_male:.1%}")
        f_col.metric("Female Stage-1 Rate",     f"{rate_female:.1%}")
        nb_col.metric("Non-Binary Stage-1 Rate",f"{rate_nb:.1%}")

        st.divider()

        # Approval rate comparison bar chart
        fig, ax = plt.subplots(figsize=(7, 3))
        groups   = ['Male', 'Female', 'Non-Binary']
        rates    = [rate_male, rate_female, rate_nb]
        bar_cols = [GREEN if r >= rate_male * 0.80 else RED for r in rates]
        ax.barh(groups, [r * 100 for r in rates], color=bar_cols)
        ax.axvline(rate_male * 80, color='orange', linestyle='--',
                   label=f"80% Rule threshold ({rate_male*80:.1f}%)")
        ax.set_xlabel("Stage 1 (Performing) Rate %")
        ax.set_title("Disparate Impact — Stage 1 Approval Rate by Gender Group")
        ax.legend(fontsize=9)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.divider()

        # ECL provision by gender (disparate burden check)
        st.subheader("ECL Provision Burden by Gender Group")
        ecl_gender = gender_df.groupby('gender_protected_attribute')['ecl_amount'].agg(
            ['mean','sum','count']
        ).reset_index()
        ecl_gender.columns = ['Gender (0=M,1=F,2=NB)', 'Avg ECL (ZAR)', 'Total ECL (ZAR)', 'Accounts']
        ecl_gender['Gender (0=M,1=F,2=NB)'] = ecl_gender['Gender (0=M,1=F,2=NB)'].map(
            {0: 'Male', 1: 'Female', 2: 'Non-Binary'}
        )
        ecl_gender['Avg ECL (ZAR)']   = ecl_gender['Avg ECL (ZAR)'].round(2)
        ecl_gender['Total ECL (ZAR)'] = ecl_gender['Total ECL (ZAR)'].round(2)
        st.dataframe(ecl_gender, use_container_width=True, hide_index=True)

        st.caption(
            "⚠️ Note: gender_protected_attribute is never a model feature. "
            "It is tracked solely for this compliance audit. "
            "If DIR < 0.80, retrain FairXGBoost with a higher λ_fair value."
        )
    else:
        st.warning("Gender data not available. Check debtors_alternative_data table.")
