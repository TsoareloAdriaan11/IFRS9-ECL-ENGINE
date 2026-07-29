"""
src/dashboard/generate_dashboard.py
════════════════════════════════════════════════════════════════════
VERSION 2.0 — EXECUTIVE STATIC DASHBOARD
════════════════════════════════════════════════════════════════════
"""
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def generate_executive_dashboard():
    print("Initializing Phase 5: Executive Visual Reporting...")
    
    # 1. Connect to Database and Load Final Ledger + Names
    db_path = os.path.join(os.path.dirname(__file__), '../../outputs/ifrs9_engine.db')
    conn = sqlite3.connect(db_path)
    
    # [CHANGED V2] JOIN added to pull first_name and last_name for the Watchlist
    query = """
        SELECT 
            l.*, 
            f.first_name, 
            f.last_name 
        FROM final_ecl_ledger l
        JOIN debtors_financial f ON l.debtor_id = f.debtor_id
    """
    df = pd.read_sql(query, conn)
    conn.close()
    
    # 2. Setup the visual canvas
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('IFRS 9 Expected Credit Loss (ECL) Executive Dashboard v2.0', fontsize=22, fontweight='bold')
    
    # -- Chart 1: Total ECL Provision by Stage (Top Left) --
    stage_ecl = df.groupby('ifrs9_stage')['ecl_amount'].sum().reset_index()
    sns.barplot(data=stage_ecl, x='ifrs9_stage', y='ecl_amount', ax=axes[0, 0], palette='viridis')
    axes[0, 0].set_title('Total ECL Provision by IFRS 9 Stage', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('IFRS 9 Stage')
    axes[0, 0].set_ylabel('ECL Amount (ZAR)')
    
    # -- Chart 2: Probability of Default Distribution (Top Right) --
    # [CHANGED V2] Added hue='thin_file_flag' to show Banked vs Unbanked risk overlap
    df['Borrower Type'] = df['thin_file_flag'].map({0: 'Banked', 1: 'Thin-File (Alternative)'})
    sns.histplot(data=df, x='pd_curr', hue='Borrower Type', multiple="stack", bins=50, kde=True, 
                 ax=axes[0, 1], palette={'Banked': '#1D4ED8', 'Thin-File (Alternative)': '#C9A84C'})
    axes[0, 1].set_title('AI-Predicted Probability of Default (PD) Distribution', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Predicted PD (%)')
    axes[0, 1].set_ylabel('Number of Debtors')
    
    # -- Chart 3: Portfolio Exposure Pie Chart (Bottom Left) --
    stage_ead = df.groupby('ifrs9_stage')['ead'].sum()
    axes[1, 0].pie(stage_ead, labels=[f'Stage {i}' for i in stage_ead.index], autopct='%1.1f%%', 
                   startangle=90, colors=sns.color_palette('pastel'), 
                   wedgeprops={'edgecolor': 'black', 'linewidth': 1})
    axes[1, 0].set_title('Total Exposure at Default (EAD) by Stage', fontsize=14, fontweight='bold')
    
    # -- Chart 4: High-Risk Watchlist (Bottom Right) --
    top_risk = df.nlargest(10, 'ecl_amount').copy()
    # [CHANGED V2] Display human names instead of truncated SA- IDs
    top_risk['full_name'] = top_risk['first_name'] + ' ' + top_risk['last_name']
    
    sns.barplot(data=top_risk, x='ecl_amount', y='full_name', ax=axes[1, 1], palette='Reds_r')
    axes[1, 1].set_title('Top 10 Highest Risk Accounts (ECL Target Watchlist)', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('ECL Provision (ZAR)')
    axes[1, 1].set_ylabel('Borrower Name')
    
    # 3. Finalize and Save
    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(__file__), '../../outputs/executive_dashboard.png')
    # Create dir if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Complete! V2 Executive Dashboard rendered and saved to outputs/executive_dashboard.png")

if __name__ == "__main__":
    generate_executive_dashboard()
