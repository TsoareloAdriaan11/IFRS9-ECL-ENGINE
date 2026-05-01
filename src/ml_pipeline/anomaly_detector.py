import pandas as pd
import sqlite3
import os
from sklearn.ensemble import IsolationForest

def detect_behavioral_anomalies():
    print("Initializing Behavioral Anomaly Radar (Isolation Forest)...")
    
    # 1. Connect to the SQLite Database
    db_path = os.path.join(os.path.dirname(__file__), '../../outputs/ifrs9_engine.db')
    conn = sqlite3.connect(db_path)
    
    # 2. Pull ONLY the behavioral data
    print("Extracting behavioral telemetry...")
    query = "SELECT * FROM debtors_behavioral"
    behavioral_df = pd.read_sql(query, conn)
    
    # Isolate features (drop the ID column so the model only looks at behavior)
    features = behavioral_df.drop(columns=['debtor_id'])
    
    # 3. Train the Isolation Forest
    # contamination=0.05 means we expect 5% of the portfolio to exhibit extreme distress signals
    print("Training Isolation Forest (Contamination = 5%)...")
    iso_forest = IsolationForest(n_estimators=300, contamination=0.05, random_state=42)
    
    # Fit the model and predict
    # Predict returns 1 for normal, -1 for anomaly. We will convert this to 0 (normal) and 1 (anomaly).
    predictions = iso_forest.fit_predict(features)
    
    # decision_function returns an anomaly score (lower/negative = more anomalous)
    # We invert it so higher score = higher risk
    scores = iso_forest.decision_function(features) * -1 
    
    # 4. Format the Output
    results_df = pd.DataFrame({
        'debtor_id': behavioral_df['debtor_id'],
        'anomaly_score': scores,
        'anomaly_flag': (predictions == -1).astype(int) # Converts -1 to 1 (True), and 1 to 0 (False)
    })
    
    # 5. Push results back to the database
    print("Writing anomaly scores back to SQL database...")
    results_df.to_sql('model_outputs_anomaly', con=conn, if_exists='replace', index=False)
    
    conn.close()
    
    # Quick sanity check printout
    anomalies_found = results_df['anomaly_flag'].sum()
    print(f"Complete! Isolated {anomalies_found} severe behavioral anomalies.")
    
    return results_df

if __name__ == "__main__":
    detect_behavioral_anomalies()
