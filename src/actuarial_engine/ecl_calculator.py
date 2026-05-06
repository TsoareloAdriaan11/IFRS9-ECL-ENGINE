import sqlite3
import pandas as pd
import os
from src.actuarial_engine.ifrs9_staging import classify_stage, calculate_ecl_provision

def calculate_ecl():
    print("Initializing Phase 4: IFRS 9 Actuarial Engine...")
    db_path = os.path.join(os.path.dirname(__file__), '../../outputs/ifrs9_engine.db')
    conn = sqlite3.connect(db_path)
    
    # 1. THE PERFECTED SQL QUERY
    # Pulling exactly what exists in your database
    query = """
        SELECT 
            f.debtor_id, 
            f.outstanding_balance as ead,
            f.age_30_days,
            f.age_60_days,
            f.age_90_plus_days,
            f.loan_purpose as loan_type,
            f.fico_equivalent,
            a.anomaly_flag, 
            p.pd_curr
        FROM debtors_financial f
        JOIN model_outputs_anomaly a ON f.debtor_id = a.debtor_id
        JOIN model_outputs_pd p ON f.debtor_id = p.debtor_id
    """
    
    print("Merging Financials, AI Anomalies, and AI PDs...")
    df = pd.read_sql(query, conn)
    
    # Setup Actuarial Parameters
    LGD = 0.45 
    final_ledger = []
    
    print("Executing Strict IFRS 9 Staging and ECL Computation...")
    
    # 2. THE COMPUTATION LOOP WITH TRANSLATION LAYER
    for index, row in df.iterrows():
        
        # --- Translation Layer ---
        # 1. Derive a single DPD integer from the arrears buckets
        if row['age_90_plus_days'] > 0:
            dpd = 95
        elif row['age_60_days'] > 0:
            dpd = 65
        elif row['age_30_days'] > 0:
            dpd = 35
        else:
            dpd = 0
            
        # 2. Derive a mock Original PD based on their FICO score (Lower FICO = Higher starting PD)
        # 3. Assume no loans are restructured for this synthetic dataset
        pd_orig = max(0.01, (850 - row['fico_equivalent']) / 10000) 
        restructured_flag = 0 
        
        # --- The Staging Engine ---
        stage = classify_stage(
            dpd=dpd, 
            restructured_flag=restructured_flag, 
            pd_curr=row['pd_curr'], 
            pd_orig=pd_orig, 
            loan_type=row['loan_type'], 
            anomaly_flag=row['anomaly_flag']
        )
        
        # Calculate provision
        pd_12m = row['pd_curr'] * 0.4 if stage == 1 else row['pd_curr']
        pd_lifetime = row['pd_curr']
        
        ecl_amount = calculate_ecl_provision(
            stage=stage, 
            pd_12m=pd_12m, 
            pd_lifetime=pd_lifetime, 
            ead=row['ead'], 
            lgd=LGD
        )
        
        final_ledger.append({
            'debtor_id': row['debtor_id'],
            'ifrs9_stage': stage,
            'pd_curr': row['pd_curr'],
            'ead': row['ead'],
            'ecl_amount': ecl_amount
        })
        
    # 3. SAVE TO DATABASE
    ledger_df = pd.DataFrame(final_ledger)
    ledger_df.to_sql('final_ecl_ledger', conn, if_exists='replace', index=False)
    
    total_ecl = ledger_df['ecl_amount'].sum()
    conn.close()
    
    print(f"SUCCESS: Engine Execution Complete.")
    print(f"Total Portfolio ECL Provision Required: ZAR {total_ecl:,.2f}")

if __name__ == "__main__":
    calculate_ecl()
