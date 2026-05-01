import pytest
from src.data_generator.generate_data import generate_correlated_data

def test_data_generation_shape():
    """
    Tests that the synthetic data generator creates the correct number of records
    and that both financial and behavioral dataframes align.
    """
    num_records = 100
    # We test on 100 records to keep the CI pipeline fast
    fin_df, beh_df = generate_correlated_data(num_records)
    
    # Assertions to prove the data is structured correctly
    assert len(fin_df) == num_records, "Financial dataframe row count mismatch"
    assert len(beh_df) == num_records, "Behavioral dataframe row count mismatch"
    
    # Assert primary keys align
    assert list(fin_df['debtor_id']) == list(beh_df['debtor_id']), "Debtor IDs do not match between tables"
    
    # Assert specific columns exist based on the actuarial spec
    assert 'fico_equivalent' in fin_df.columns
    assert 'gambling_merchant_ratio' in beh_df.columns
