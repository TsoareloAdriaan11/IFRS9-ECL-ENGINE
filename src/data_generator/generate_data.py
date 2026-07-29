"""
src/data_generator/generate_data.py
════════════════════════════════════════════════════════════════════
VERSION 2.0 — DEBIASED ALTERNATIVE DATA GENERATOR
════════════════════════════════════════════════════════════════════
"""

import os
import numpy as np
import pandas as pd
from faker import Faker
from sqlalchemy import (
    create_engine, Column, Integer, Float, String, ForeignKey, Text
)
from sqlalchemy.orm import declarative_base

# ── Database setup ────────────────────────────────────────────────────────
db_path = os.path.join(os.path.dirname(__file__), '../../outputs/ifrs9_engine.db')
os.makedirs(os.path.dirname(db_path), exist_ok=True)
engine  = create_engine(f'sqlite:///{db_path}', echo=False)
Base    = declarative_base()
fake    = Faker('en_GB')  # Close enough locale for general names

# ═══════════════════════════════════════════════════════════════════════════
# SQL SCHEMA — TABLE DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════

class DebtorFinancial(Base):
    """V1 TABLE + V2 UPGRADES (Names & Thin-file flags)"""
    __tablename__ = 'debtors_financial'
    debtor_id             = Column(String, primary_key=True)
    first_name            = Column(String)  # [NEW] For Streamlit UI
    last_name             = Column(String)  # [NEW] For Streamlit UI
    thin_file_flag        = Column(Integer) # [NEW] 1 = no formal credit file
    fico_equivalent       = Column(Integer) # -1 for thin-file borrowers
    dti_ratio             = Column(Float)
    outstanding_balance   = Column(Float)
    current_ratio         = Column(Float)
    age_30_days           = Column(Float)
    age_60_days           = Column(Float)
    age_90_plus_days      = Column(Float)
    loan_to_income        = Column(Float)
    payment_history_score = Column(Float)
    employment_years      = Column(Integer)
    loan_purpose          = Column(String)

class DebtorBehavioral(Base):
    """V1 TABLE — Unchanged schema"""
    __tablename__ = 'debtors_behavioral'
    debtor_id                = Column(String, ForeignKey('debtors_financial.debtor_id'), primary_key=True)
    post_payday_burn_rate    = Column(Integer)
    transaction_velocity_30d = Column(Integer)
    avg_transaction_amount   = Column(Float)
    gambling_merchant_ratio  = Column(Float)
    late_night_tx_pct        = Column(Float)
    balance_volatility       = Column(Float)
    overdraft_count_90d      = Column(Integer)
    cash_advance_pct         = Column(Float)
    p2p_transfer_volume      = Column(Float)
    digital_engagement_score = Column(Integer)
    password_reset_frequency = Column(Integer)
    category_diversity       = Column(Float)

class DebtorAlternativeData(Base):
    """NEW V2 TABLE — Alternative financial transaction signals for unbanked borrowers"""
    __tablename__ = 'debtors_alternative_data'
    debtor_id                      = Column(String, ForeignKey('debtors_financial.debtor_id'), primary_key=True)
    mobile_money_inflows_30d       = Column(Float)
    mobile_wallet_balance_std      = Column(Float)
    airtime_recharge_freq_60d      = Column(Integer)
    essential_payments_consistency = Column(Float)
    device_fingerprint_id          = Column(Text)
    geographic_cluster_id          = Column(Text)
    gender_protected_attribute     = Column(Integer) # 0=Male, 1=Female (AUDIT ONLY)

# ── GEOGRAPHIC CLUSTER MAP ─────────────────────────────────────────────────
SA_GEO_CLUSTERS = [
    'GC_GP_SOWETO_01',       'GC_GP_ALEXANDRA_02',    'GC_GP_SANDTON_03',
    'GC_WC_KHAYELITSHA_04',  'GC_WC_MITCHELLSPLAIN_05','GC_WC_STELLENBOSCH_06',
    'GC_KZN_UMLAZI_07',      'GC_KZN_DURBANCENTRAL_08','GC_EC_MDANTSANE_09',
    'GC_LP_POLOKWANE_10',    'GC_MP_NELSPRUIT_11',    'GC_NW_RUSTENBURG_12',
]
HIGH_DENSITY_CLUSTERS = {
    'GC_GP_SOWETO_01', 'GC_WC_KHAYELITSHA_04',
    'GC_WC_MITCHELLSPLAIN_05', 'GC_KZN_UMLAZI_07', 'GC_EC_MDANTSANE_09'
}

# ═══════════════════════════════════════════════════════════════════════════
# DATA GENERATION
# ═══════════════════════════════════════════════════════════════════════════

def generate_correlated_data(num_records=10000, random_seed=42):
    print(f"Generating {num_records:,} synthetic SA debtor profiles with alternative data...")
    np.random.seed(random_seed)
    Faker.seed(random_seed)
    
    stress_factor = np.random.beta(a=2, b=5, size=num_records) 
    
    # ── [NEW] IDs, Names, and Thin-File Status ───────────────────────────
    debtor_ids = [f"SA-{i+1:06d}" for i in range(num_records)]
    first_names = [fake.first_name() for _ in range(num_records)]
    last_names = [fake.last_name() for _ in range(num_records)]
    thin_file_mask = np.random.random(num_records) < 0.20  # 20% Unbanked
    
    # ── Financial Data ───────────────────────────────────────────────────
    incomes = np.random.lognormal(mean=11.5, sigma=0.8, size=num_records)
    
    fico = 850 - (stress_factor * 400) - np.random.normal(0, 30, num_records)
    fico_out = np.clip(fico, 300, 850).astype(int)
    fico_out[thin_file_mask] = -1  # -1 sentinel value for XGBoost to treat as missing
    
    dti = 0.2 + (stress_factor * 0.4) + np.random.normal(0, 0.05, num_records)
    outstanding_bal = incomes * np.random.uniform(0.1, 1.5, num_records)
    
    age_30 = np.where(stress_factor > 0.4, outstanding_bal * np.random.uniform(0.1, 0.3), 0)
    age_60 = np.where(stress_factor > 0.6, outstanding_bal * np.random.uniform(0.2, 0.5), 0)
    age_90 = np.where(stress_factor > 0.8, outstanding_bal * np.random.uniform(0.4, 0.8), 0)

    financial_df = pd.DataFrame({
        'debtor_id': debtor_ids,
        'first_name': first_names,             # [NEW]
        'last_name': last_names,               # [NEW]
        'thin_file_flag': thin_file_mask.astype(int), # [NEW]
        'fico_equivalent': fico_out,
        'dti_ratio': np.clip(dti, 0.1, 0.8),
        'outstanding_balance': np.round(outstanding_bal, 2),
        'current_ratio': np.clip(2.5 - (stress_factor * 2), 0.5, 3.0),
        'age_30_days': np.round(age_30, 2),
        'age_60_days': np.round(age_60, 2),
        'age_90_plus_days': np.round(age_90, 2),
        'loan_to_income': np.clip(outstanding_bal / incomes, 0.1, 2.0),
        'payment_history_score': np.clip(1.0 - (stress_factor * 0.8), 0.1, 1.0),
        'employment_years': np.clip(np.random.normal(7, 4, num_records) - (stress_factor*2), 0, 30).astype(int),
        'loan_purpose': np.random.choice(['Auto', 'Home', 'Personal', 'SME', 'Debt Consolidation'], size=num_records, p=[0.2, 0.3, 0.25, 0.15, 0.1])
    })

    # ── Behavioral Data ──────────────────────────────────────────────────
    behavioral_df = pd.DataFrame({
        'debtor_id': debtor_ids,
        'post_payday_burn_rate': np.clip(25 - (stress_factor * 20) + np.random.normal(0, 2, num_records), 1, 28).astype(int),
        'transaction_velocity_30d': np.clip(30 + (stress_factor * 50) + np.random.normal(0, 10, num_records), 5, 150).astype(int),
        'avg_transaction_amount': np.round(incomes / np.random.uniform(20, 100, num_records), 2),
        'gambling_merchant_ratio': np.clip(stress_factor * np.random.uniform(0.1, 0.4), 0, 0.5),
        'late_night_tx_pct': np.clip(stress_factor * np.random.uniform(0.05, 0.3), 0, 0.4),
        'balance_volatility': np.clip(0.1 + (stress_factor * 0.6), 0.05, 0.9),
        'overdraft_count_90d': np.where(stress_factor > 0.5, np.random.randint(1, 6, num_records), 0),
        'cash_advance_pct': np.clip(stress_factor * np.random.uniform(0, 0.5), 0, 0.8),
        'p2p_transfer_volume': np.round(incomes * np.random.uniform(0.05, 0.3, num_records), 2),
        'digital_engagement_score': np.clip(10 - (stress_factor * 8) + np.random.normal(0, 1, num_records), 0, 14).astype(int),
        'password_reset_frequency': np.where(stress_factor > 0.7, np.random.randint(1, 4, num_records), 0),
        'category_diversity': np.clip(5.0 - (stress_factor * 3), 1.0, 6.0)
    })

    # ── [NEW] Alternative Data (V2) ──────────────────────────────────────
    geo_clusters = np.random.choice(SA_GEO_CLUSTERS, size=num_records)
    geo_clusters[thin_file_mask] = np.random.choice(list(HIGH_DENSITY_CLUSTERS), size=thin_file_mask.sum())

    mobile_inflows = np.where(thin_file_mask, np.random.lognormal(8.5, 0.6, num_records), np.random.lognormal(9.5, 0.8, num_records))
    mobile_inflows *= (1 - stress_factor * 0.4)
    
    wallet_balance_std = np.clip(0.2 + stress_factor * 0.6 + np.random.normal(0, 0.1, num_records), 0.05, 1.2)
    airtime_freq = np.clip(np.random.poisson(lam=8, size=num_records) - (stress_factor * 4).astype(int), 0, 20).astype(int)
    
    ess_consistency = np.clip(0.9 - (stress_factor * 0.6) + np.random.normal(0, 0.1, num_records), 0.0, 1.0)
    ess_consistency[thin_file_mask] = np.clip(ess_consistency[thin_file_mask] + (airtime_freq[thin_file_mask] / 40), 0.0, 1.0)

    def _make_device_id(i, geo):
        if geo in HIGH_DENSITY_CLUSTERS and np.random.random() < 0.15:
            shared_pool = int(geo.split('_')[-1]) * 1000
            return f"DEV-SHARED-{shared_pool + np.random.randint(0, 50):05d}"
        return f"DEV-{i+1:08d}"

    device_ids = [_make_device_id(i, geo_clusters[i]) for i in range(num_records)]
    gender = np.random.choice([0, 1], size=num_records, p=[0.48, 0.52])

    alternative_df = pd.DataFrame({
        'debtor_id': debtor_ids,
        'mobile_money_inflows_30d': np.round(mobile_inflows, 2),
        'mobile_wallet_balance_std': np.round(wallet_balance_std, 4),
        'airtime_recharge_freq_60d': airtime_freq,
        'essential_payments_consistency': np.round(ess_consistency, 4),
        'device_fingerprint_id': device_ids,
        'geographic_cluster_id': geo_clusters,
        'gender_protected_attribute': gender,
    })

    return financial_df, behavioral_df, alternative_df

# ═══════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    Base.metadata.create_all(engine)
    fin_df, beh_df, alt_df = generate_correlated_data(10000)
    
    print("Writing to SQL Database...")
    fin_df.to_sql('debtors_financial', con=engine, if_exists='replace', index=False)
    beh_df.to_sql('debtors_behavioral', con=engine, if_exists='replace', index=False)
    alt_df.to_sql('debtors_alternative_data', con=engine, if_exists='replace', index=False)
    
    thin_count = fin_df['thin_file_flag'].sum()
    print(f"Database generated successfully at /outputs/ifrs9_engine.db")
    print(f"-> Total Profiles: 10,000")
    print(f"-> Thin-File Borrowers: {thin_count:,} ({thin_count/10000:.0%})")
