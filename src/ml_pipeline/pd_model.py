import pandas as pd
import sqlite3
import os
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def train_pd_model():
    print("Initializing Probability of Default (PD) Engine...")
    
    # 1. Connect to Database and Join Tables
    db_path = os.path.join(os.path.dirname(__file__), '../../outputs/ifrs9_engine.db')
    conn = sqlite3.connect(db_path)
    
    query = """
        SELECT f.*, b.* 
        FROM debtors_financial f
        JOIN debtors_behavioral b ON f.debtor_id = b.debtor_id
    """
    df = pd.read_sql(query, conn)
    
    # Drop duplicate debtor_id column from the SQL JOIN
    df = df.loc[:,~df.columns.duplicated()].copy()

    # 2. Define the Target Variable (IFRS 9 Definition of Default)
    # 1 = Default (90+ Days Late), 0 = Performing
    df['default_flag'] = (df['age_90_plus_days'] > 0).astype(int)
    
    # 3. PREVENT DATA LEAKAGE! 
    # We must drop the arrears columns, or the model will cheat.
    features_to_drop = ['debtor_id', 'age_30_days', 'age_60_days', 'age_90_plus_days', 'loan_purpose', 'default_flag']
    X = df.drop(columns=features_to_drop)
    y = df['default_flag']
    
    print(f"Training XGBoost on {len(X.columns)} financial and behavioral features...")
    
    # 4. Train/Test Split & Model Training
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize XGBoost Classifier
    model = xgb.XGBClassifier(
        n_estimators=200, 
        learning_rate=0.05, 
        max_depth=4, 
        eval_metric='logloss'
    )
    
    model.fit(X_train, y_train)
    
    # 5. Generate PD Predictions for the entire portfolio
    print("Calculating PD percentages for all 10,000 debtors...")
    # predict_proba returns [Probability of 0, Probability of 1]. We want index 1 (Prob of Default)
    df['pd_curr'] = model.predict_proba(X)[:, 1]
    
    # 6. SHAP Auditor Interpretability
    print("Generating SHAP Explainability Matrix...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)
    
    # Save the SHAP plot as an image so you can view it
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_train, show=False)
    plot_path = os.path.join(os.path.dirname(__file__), '../../outputs/shap_auditor_report.png')
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
    
    # 7. Format Output and Save to Database
    results_df = df[['debtor_id', 'pd_curr']].copy()
    # Round to 4 decimal places for clean percentage reading (e.g., 0.0845 -> 8.45%)
    results_df['pd_curr'] = results_df['pd_curr'].round(4) 
    
    print("Writing PD outputs to Database...")
    results_df.to_sql('model_outputs_pd', con=conn, if_exists='replace', index=False)
    
    conn.close()
    
    print(f"Complete! Auditor SHAP report saved to outputs/shap_auditor_report.png")

if __name__ == "__main__":
    train_pd_model()
