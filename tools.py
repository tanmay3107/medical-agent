import os

# Tool 1: The Calculator
def check_vitals(systolic, diastolic, heart_rate):
    """
    Analyzes vital signs to determine medical risk level.
    Returns: 'Normal', 'Elevated', or 'Critical'.
    """
    risk = "Normal"
    
    # Simple logic for the demo
    if systolic > 140 or diastolic > 90:
        risk = "Elevated"
    if systolic > 180 or diastolic > 120 or heart_rate > 100:
        risk = "Critical"
        
    return f"Vitals Analysis: {risk} Risk. (BP: {systolic}/{diastolic}, HR: {heart_rate})"

# Tool 2: The Writer (File System Access)
def write_referral_letter(patient_name, diagnosis, recommendation):
    """
    Writes a formal medical referral letter to a text file.
    Useful when a patient is High Risk or Critical.
    """
    filename = f"Referral_{patient_name.replace(' ', '_')}.txt"
    content = f"""
    URGENT MEDICAL REFERRAL
    -----------------------
    Patient: {patient_name}
    Date: 2026-02-01
    
    Diagnosis: {diagnosis}
    
    Recommendation:
    {recommendation}
    
    Signed,
    AI Triage Agent
    """
    
    with open(filename, "w") as f:
        f.write(content)
        
    return f"✅ Referral letter successfully saved to {filename}"