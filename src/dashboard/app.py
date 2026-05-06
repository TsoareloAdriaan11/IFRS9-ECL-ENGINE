import streamlit as st
import pandas as pd
import sqlite3
import os

# --- PAGE SETUP ---
st.set_page_config(page_title="IFRS 9 AI Engine", page_icon="🏦", layout="wide")
st.title("🏦 IFRS 9 Expected Credit Loss (ECL) Engine")
st.markdown("### Interactive Portfolio Risk Dashboard")
st.write("This dashboard visualizes the outputs of our synthetic banking pipeline, merging XGBoost default probabilities with Isolation Forest anomaly detection to calculate IFRS 9 regulatory capital provisions.")

# --- DATA LOADING ---
@st.cache_data
def load_data():
    db_path = os.path.join(os.path.dirname(__file__), '../../outputs/ifrs9_engine.db')
    if not os.path.exists(db_path):
        return None
    
    conn = sqlite3.connect(db_path)
    df = pd.read_sql("SELECT * FROM final_ecl_ledger", conn)
    conn.close()
    return df

df = load_data()

if df is None:
    st.error("⚠️ Database not found! Please run the backend pipeline (Phases 1-4) to generate the SQLite database before launching the dashboard.")
    st.stop()

# --- TOP KPI METRICS ---
st.divider()
col1, col2, col3 = st.columns(3)

total_ecl = df['ecl_amount'].sum()
total_ead = df['ead'].sum()
high_risk_count = len(df[df['ifrs9_stage'] > 1])

col1.metric("Total Expected Credit Loss (ZAR)", f"R {total_ecl:,.2f}")
col2.metric("Total Exposure at Default (ZAR)", f"R {total_ead:,.2f}")
col3.metric("High-Risk Accounts (Stage 2 & 3)", f"{high_risk_count:,}")

# --- VISUALIZATIONS ---
st.divider()
row1_col1, row1_col2 = st.columns(2)

with row1_col1:
    st.subheader("Total ECL Provision by Stage")
    stage_ecl = df.groupby('ifrs9_stage')['ecl_amount'].sum()
    st.bar_chart(stage_ecl, color="#ff4b4b")

with row1_col2:
    st.subheader("AI Probability of Default (PD) Distribution")
    # Grouping PD into bins for visualization
    pd_bins = pd.cut(df['pd_curr'], bins=10).value_counts().sort_index()
    pd_bins.index = pd_bins.index.astype(str)
    st.bar_chart(pd_bins)

# --- INTERACTIVE LEDGER ---
st.divider()
st.subheader("🔍 Interactive Debtor Risk Ledger")

# Filters
filter_col1, filter_col2 = st.columns(2)
with filter_col1:
    stage_filter = st.selectbox("Filter by IFRS 9 Stage:", ["All Portfolio", "Stage 1 (Performing)", "Stage 2 (Underperforming - SICR)", "Stage 3 (Default)"])
with filter_col2:
    sort_filter = st.selectbox("Sort Accounts By:", ["Highest ECL Provision", "Highest Probability of Default"])

# Apply Filters
display_df = df.copy()

if stage_filter != "All Portfolio":
    stage_num = int(stage_filter.split(" ")[1]) # Extracts the 1, 2, or 3
    display_df = display_df[display_df['ifrs9_stage'] == stage_num]

if sort_filter == "Highest ECL Provision":
    display_df = display_df.sort_values(by='ecl_amount', ascending=False)
else:
    display_df = display_df.sort_values(by='pd_curr', ascending=False)

# Display Table
st.dataframe(
    display_df[['debtor_id', 'ifrs9_stage', 'pd_curr', 'ead', 'ecl_amount']], 
    use_container_width=True,
    height=400
)
