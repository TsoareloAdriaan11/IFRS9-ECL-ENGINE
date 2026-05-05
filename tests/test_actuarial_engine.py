import sys
import os

# --- THE PATHING FIX ---
# This forces the GitHub runner to look at the root directory to find 'src'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from src.actuarial_engine.ifrs9_staging import classify_stage, calculate_ecl_provision

# --- STAGING LOGIC TESTS ---

def test_stage_3_hard_rule():
    """Test Step 1: DPD >= 90 is automatic Stage 3."""
    stage = classify_stage(dpd=95, restructured_flag=0, pd_curr=0.05, pd_orig=0.04, loan_type='Personal', anomaly_flag=0)
    assert stage == 3, "Failed: 90+ DPD should be Stage 3"

def test_stage_2_dpd_30_no_rebuttal():
    """Test Step 2: DPD >= 30 without rebuttal triggers Stage 2."""
    stage = classify_stage(dpd=45, restructured_flag=0, pd_curr=0.05, pd_orig=0.04, loan_type='Personal', anomaly_flag=0, rebuttal_flag=0)
    assert stage == 2, "Failed: 30+ DPD without rebuttal should be Stage 2"

def test_stage_1_dpd_30_with_rebuttal():
    """Test Step 2 Bypass: DPD >= 30 WITH rebuttal falls through to Stage 1."""
    stage = classify_stage(dpd=45, restructured_flag=0, pd_curr=0.05, pd_orig=0.04, loan_type='Personal', anomaly_flag=0, rebuttal_flag=1)
    assert stage == 1, "Failed: 30+ DPD with successful rebuttal should remain Stage 1"

def test_stage_2_restructured():
    """Test Step 3: Restructured loans trigger Stage 2."""
    stage = classify_stage(dpd=0, restructured_flag=1, pd_curr=0.05, pd_orig=0.04, loan_type='Personal', anomaly_flag=0)
    assert stage == 2, "Failed: Restructured flag should trigger Stage 2"

def test_stage_2_pd_multiple():
    """Test Step 4: PD Multiple >= 2.0x triggers Stage 2."""
    stage = classify_stage(dpd=0, restructured_flag=0, pd_curr=0.05, pd_orig=0.02, loan_type='SME', anomaly_flag=0)
    assert stage == 2, "Failed: PD Multiple >= 2.0 should trigger Stage 2"

def test_stage_2_pd_jump():
    """Test Step 4: Absolute PD jump > threshold triggers Stage 2."""
    stage = classify_stage(dpd=0, restructured_flag=0, pd_curr=0.13, pd_orig=0.02, loan_type='SME', anomaly_flag=0)
    assert stage == 2, "Failed: Absolute PD jump above threshold should trigger Stage 2"

def test_stage_2_behavioral_corroboration():
    """Test Step 5: Anomaly + 1.5x Multiple triggers Stage 2."""
    stage = classify_stage(dpd=0, restructured_flag=0, pd_curr=0.08, pd_orig=0.05, loan_type='Personal', anomaly_flag=1)
    assert stage == 2, "Failed: Anomaly + >1.5x Multiple should trigger Stage 2"

def test_stage_1_clean_loan():
    """Test Step 6: Clean loan remains Stage 1."""
    stage = classify_stage(dpd=0, restructured_flag=0, pd_curr=0.03, pd_orig=0.025, loan_type='Home', anomaly_flag=0)
    assert stage == 1, "Failed: Clean loan should remain Stage 1"

# --- ECL CALCULATION TESTS ---

def test_ecl_stage_1():
    """Test ECL formula for Stage 1 (12-month PD)."""
    ecl = calculate_ecl_provision(stage=1, pd_12m=0.05, pd_lifetime=0.15, ead=100000, lgd=0.40)
    assert ecl == 2000.00, "Failed: Stage 1 ECL math is incorrect"

def test_ecl_stage_2_with_discount():
    """Test ECL formula for Stage 2 (Lifetime PD) with Discount Factor."""
    ecl = calculate_ecl_provision(stage=2, pd_12m=0.05, pd_lifetime=0.15, ead=100000, lgd=0.40, discount_factor=0.90)
    assert ecl == 5400.00, "Failed: Stage 2 ECL math with discount factor is incorrect"
