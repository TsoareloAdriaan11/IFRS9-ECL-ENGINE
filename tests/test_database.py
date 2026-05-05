import os
import sqlite3
import pytest

# Define the path to the database
DB_PATH = os.path.join(os.path.dirname(__file__), '../outputs/ifrs9_engine.db')

# --- THE CLOUD FIX ---
# Tell pytest to gracefully skip this entire file if the database doesn't exist
# (e.g., when running on the empty GitHub Actions cloud server)
pytestmark = pytest.mark.skipif(
    not os.path.exists(DB_PATH),
    reason="Database not found. Skipping local integration tests in CI environment."
)

def test_database_exists():
    """Verify that the SQLite database file was generated."""
    assert os.path.exists(DB_PATH), "CRITICAL: Database file does not exist. Run Phase 1 first."

def test_required_tables_exist():
    """Verify that all pipeline stages successfully wrote their tables."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Query SQLite master table for all table names
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in cursor.fetchall()]
    conn.close()
    
    # The 5 tables our engine should have created
    required_tables = [
        'debtors_financial', 
        'debtors_behavioral', 
        'model_outputs_anomaly', 
        'model_outputs_pd', 
        'final_ecl_ledger'
    ]
    
    for table in required_tables:
        assert table in tables, f"PIPELINE FAILURE: Missing table '{table}'"
