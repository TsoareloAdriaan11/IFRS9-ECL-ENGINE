import pandas as pd
import numpy as np
from faker import Faker
from sqlalchemy import create_engine, Column, Integer, Float, String, ForeignKey
from sqlalchemy.orm import declarative_base
import os

# Ensure outputs go to the right directory
db_path = os.path.join(os.path.dirname(__file__), '../../outputs/ifrs9_engine.db')
engine = create_engine(f'sqlite:///{db_path}', echo=False)
Base = declarative_base()
fake = Faker()

class DebtorFinancial(Base):
    __tablename__ = 'debtors_financial'
    
    debtor_id = Column(String, primary_key=True)
    fico_equivalent = Column(Integer)
    dti_ratio = Column(Float)
    outstanding_balance = Column(Float)
    current_ratio = Column(Float)
    age_30_days = Column(Float)
    age_60_days = Column(Float)
    age_90_plus_days = Column(Float)
    loan_to_income = Column(Float)
    payment_history_score = Column(Float)
    employment_years = Column(Integer)
    loan_purpose = Column(String)

class DebtorBehavioral(Base):
    __tablename__ = 'debtors_behavioral'
    
    debtor_id = Column(String, ForeignKey('debtors_financial.debtor_id'), primary_key=True)
    post_payday_burn_rate = Column(Integer)
    transaction_velocity_30d = Column(Integer)
    avg_transaction_amount = Column(Float)
    gambling_merchant_ratio = Column(Float)
    late_night_tx_pct = Column(Float)
    balance_volatility = Column(Float)
    overdraft_count_90d = Column(Integer)
    cash_advance_pct = Column(Float)
    p2p_transfer_volume = Column(Float)
    digital_engagement_score = Column(Integer)
    password_reset_frequency = Column(Integer)
    category_diversity = Column(Float)

def generate_correlated_data(num_records=10000):
    print(f"Generating {num_records} synthetic SA debtor profiles...")
    stress_factor = np.random.beta(a=2, b=5, size=num_records) 
    
    # Financial Data
    debtor_ids = [f"SA-{fake.unique.random_int(min=100000, max=999999)}" for _ in range(num_records)]
    incomes = np.random.lognormal(mean=11.5, sigma=0.8, size=num_records)
    
    fico = 850 - (stress_factor * 400) - np.random.normal(0, 30, num_records)
    dti = 0.2 + (stress_factor * 0.4) + np.random.normal(0, 0.05, num_records)
    
    outstanding_bal = incomes * np.random.uniform(0.1, 1.5, num_records)
    age_30 = np.where(stress_factor > 0.4, outstanding_bal * np.random.uniform(0.1, 0.3), 0)
    age_60 = np.where(stress_factor > 0.6, outstanding_bal * np.random.uniform(0.2, 0.5), 0)
    age_90 = np.where(stress_factor > 0.8, outstanding_bal * np.random.uniform(0.4, 0.8), 0)

    financial_df = pd.DataFrame({
        'debtor_id': debtor_ids,
        'fico_equivalent': np.clip(fico, 300, 850).astype(int),
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

    # Behavioral Data
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

    return financial_df, behavioral_df

if __name__ == "__main__":
    Base.metadata.create_all(engine)
    fin_df, beh_df = generate_correlated_data(10000)
    print("Writing to SQL Database...")
    fin_df.to_sql('debtors_financial', con=engine, if_exists='replace', index=False)
    beh_df.to_sql('debtors_behavioral', con=engine, if_exists='replace', index=False)
    print("Database generated at /outputs/ifrs9_engine.db")
