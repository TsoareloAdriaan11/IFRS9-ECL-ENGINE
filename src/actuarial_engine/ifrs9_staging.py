def classify_stage(dpd, restructured_flag, pd_curr, pd_orig, loan_type, anomaly_flag, rebuttal_flag=0):
    """
    Determines the IFRS 9 Stage (1, 2, or 3) based on a strict 5-step decision waterfall.
    """
    abs_thresholds = {
        'Personal': 0.08,
        'SME': 0.10,
        'Home': 0.05,
        'Auto': 0.06,
        'Debt Consolidation': 0.09
    }

    # Step 1: Hard Credit-Impaired Rule
    if dpd >= 90:
        return 3

    # Step 2: DPD Rebuttable Presumption
    if dpd >= 30 and not rebuttal_flag:
        return 2

    # Step 3: Restructured / Forbearance Flag
    if restructured_flag == 1:
        return 2

    # Step 4: Quantitative PD Triggers
    pd_multiple = pd_curr / pd_orig if pd_orig > 0 else 999.0
    pd_jump = pd_curr - pd_orig
    threshold = abs_thresholds.get(loan_type, 0.05)
    
    if pd_multiple >= 2.0 or pd_jump >= threshold:
        return 2

    # Step 5: Behavioral Corroboration Trigger
    if anomaly_flag == 1 and pd_multiple >= 1.5:
        return 2

    # Step 6: Default
    return 1

def calculate_ecl_provision(stage, pd_12m, pd_lifetime, ead, lgd, discount_factor=1.0):
    """
    Calculates the final ECL Rand provision.
    """
    if stage == 1:
        ecl = pd_12m * ead * lgd
    elif stage in [2, 3]:
        ecl = pd_lifetime * ead * lgd * discount_factor
    else:
        raise ValueError("Invalid IFRS 9 Stage.")
        
    return round(ecl, 2)
