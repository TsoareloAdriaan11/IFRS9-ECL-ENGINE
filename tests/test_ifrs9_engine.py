import pytest
from src.actuarial_engine.ifrs9_staging import classify_stage, calculate_ecl_provision

def test_stage_3_hard_rule():
    """Test Step 1: DPD >= 90 overrides everything to Stage 3."""
    stage = classify_stage(dpd=95, restructured_flag=0, pd_curr=0.05, pd_orig=0.04, loan_type='Personal', anomaly_flag=0)
    assert stage == 3, "90+ DPD failed to trigger Stage 3"

def test_stage_2_pd_multiple():
    """Test Step 4: PD Multiple >= 2.0x triggers Stage 2."""
    # PD Orig = 2%, PD Curr = 5% (2.5x multiple)
    stage = classify_stage(dpd=5, restructured_flag=0, pd_curr=0.05, pd_orig=0.02, loan_type='SME', anomaly_flag=0)
    assert stage == 2, "PD Multiple >= 2.0 failed to trigger Stage 2"

def test_stage_1_clean_loan():
    """Test Step 6: Clean loan remains Stage 1."""
    # PD Orig = 2%, PD Curr = 2.2% (1.1x multiple, below threshold)
    stage = classify_stage(dpd=0, restructured_flag=0, pd_curr=0.022, pd_orig=0.02, loan_type='Home', anomaly_flag=0)
    assert stage == 1, "Clean loan failed to remain in Stage 1"

def test_ecl_calculation():
    """Test the ECL formula: PD * EAD * LGD"""
    # Stage 1 uses 12-month PD (e.g., 5%), EAD = R100k, LGD = 40%
    ecl = calculate_ecl_provision(stage=1, pd_12m=0.05, pd_lifetime=0.15, ead=100000, lgd=0.40)
    
    # Expected: 0.05 * 100000 * 0.40 = 2000
    assert ecl == 2000.00, "Stage 1 ECL calculation is mathematically incorrect"
