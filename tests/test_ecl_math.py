import pytest

# Extracting the core logic from our ecl_calculator.py for isolated testing
def determine_stage(age_90_plus_days, anomaly_flag, pd_curr):
    if age_90_plus_days > 0:
        return 3
    elif anomaly_flag == 1 or pd_curr > 0.15:
        return 2
    else:
        return 1

def calculate_ecl(pd_curr, lgd, ead):
    return round(pd_curr * lgd * ead, 2)

def test_ifrs9_staging_logic():
    """Test every possible regulatory staging condition."""
    # Condition 1: 90 Days Late MUST be Stage 3, regardless of AI
    assert determine_stage(age_90_plus_days=1, anomaly_flag=0, pd_curr=0.05) == 3
    
    # Condition 2: Behavioral Anomaly MUST trigger Stage 2 (SICR)
    assert determine_stage(age_90_plus_days=0, anomaly_flag=1, pd_curr=0.05) == 2
    
    # Condition 3: High PD (>15%) MUST trigger Stage 2 (SICR)
    assert determine_stage(age_90_plus_days=0, anomaly_flag=0, pd_curr=0.20) == 2
    
    # Condition 4: Clean profile MUST be Stage 1
    assert determine_stage(age_90_plus_days=0, anomaly_flag=0, pd_curr=0.05) == 1

def test_ecl_formula():
    """Test the monetary ECL calculation: PD * LGD * EAD"""
    # Scenario: 10% Probability of Default, 45% Loss Severity, R100,000 Exposure
    # Expected Loss = 0.10 * 0.45 * 100000 = 4500.00
    assert calculate_ecl(pd_curr=0.10, lgd=0.45, ead=100000) == 4500.00
    
    # Scenario: 0% Probability of Default
    assert calculate_ecl(pd_curr=0.0, lgd=0.45, ead=50000) == 0.00
