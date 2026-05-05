import pandas as pd
import sqlite3
import os

def calculate_ecl():
    print("Initializing Phase 4: IFRS 9 Actuarial Engine...")
    
    # 1. Connect to Database
    db_path = os.path.join(os.path.dirname(__file__), '../../outputs/ifrs9_engine.db')
    conn = sqlite3.connect(db_path)
    
    # 2. Pull all the pieces together
    print("Merging Financials, AI Anomalies, and AI PDs...")
    query = """
        SELECT 
            f.debtor_id, 
            f.current_balance, 
            f.age_90_plus_days,
            a.anomaly_flag,
            p.pd_curr
        FROM debtors_financial f
        JOIN model_outputs_anomaly a ON f.debtor_id = a.debtor_id
        JOIN model_outputs_pd p ON f.debtor_id = p.debtor_id
    """
    df = pd.read_sql(query, conn)
    
    # 3. IFRS 9 Staging Logic (The Rules)
    print("Applying IFRS 9 Staging Logic...")
    def determine_stage(row):
        # Stage 3: Actual Default (90+ days late)
        if row['age_90_plus_days'] > 0:
            return 3
        # Stage 2: Significant Increase in Credit Risk (AI caught bad behavior or high PD)
        elif row['anomaly_flag'] == 1 or row['pd_curr'] > 0.15:
            return 2
        # Stage 1: Performing normally
        else:
            return 1
            
    df['ifrs9_stage'] = df.apply(determine_stage, axis=1)
    
    # 4. Calculate ECL Variables
    # LGD (Loss Given Default): Assuming a standard 45% loss severity if they default
    LGD = 0.45 
    
    # EAD (Exposure At Default): We will use the current balance for simplicity
    df['ead'] = df['current_balance']
    
    # 5. The Master ECL Formula
    print("Calculating Expected Credit Loss (ECL) in ZAR...")
    df['ecl_amount'] = df['pd_curr'] * LGD * df['ead']
    
    # Rounding for financial reporting
    df['ecl_amount'] = df['ecl_amount'].round(2)
    
    # 6. Save final ledger to Database
    print("Writing Final ECL Ledger to database...")
    results_df = df[['debtor_id', 'ifrs9_stage', 'pd_curr', 'ead', 'ecl_amount']]
    results_df.to_sql('final_ecl_ledger', con=conn, if_exists='replace', index=False)
    
    conn.close()
    
    total_ecl = df['ecl_amount'].sum()
    print(f"Complete! Total Portfolio Expected Credit Loss: R {total_ecl:,.2f}")

if __name__ == "__main__":
    calculate_ecl()
