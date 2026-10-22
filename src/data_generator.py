"""
Data Generator for Motor Insurance Claims and Policy History
Generates realistic insurance claims dataset with both legitimate claims and anomalous/fraudulent patterns.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def generate_insurance_dataset(num_samples: int = 5000, fraud_ratio: float = 0.12, random_state: int = 42) -> pd.DataFrame:
    np.random.seed(random_state)
    
    num_fraud = int(num_samples * fraud_ratio)
    num_normal = num_samples - num_fraud
    
    # -----------------------------
    # 1. Normal / Legitimate Claims
    # -----------------------------
    normal_tenure = np.random.exponential(scale=60, size=num_normal) + 6  # 6 to 250+ months
    normal_tenure = np.clip(normal_tenure, 6, 240).astype(int)
    
    normal_age = np.clip(np.random.normal(42, 12, num_normal), 18, 75).astype(int)
    
    normal_premium = np.clip(np.random.normal(1200, 250, num_normal), 600, 2400).round(2)
    normal_deductible = np.random.choice([500, 1000, 1500, 2000], size=num_normal, p=[0.45, 0.35, 0.15, 0.05])
    normal_coverage = np.random.choice([100000, 300000, 500000], size=num_normal, p=[0.5, 0.35, 0.15])
    
    normal_incident_types = np.random.choice(
        ['Multi-vehicle Collision', 'Single Vehicle Collision', 'Parked Car', 'Vehicle Theft'],
        size=num_normal, p=[0.42, 0.38, 0.15, 0.05]
    )
    
    normal_severities = []
    for it in normal_incident_types:
        if it == 'Vehicle Theft':
            normal_severities.append('Total Loss')
        elif it == 'Parked Car':
            normal_severities.append(np.random.choice(['Trivial Damage', 'Minor Damage'], p=[0.6, 0.4]))
        else:
            normal_severities.append(np.random.choice(['Minor Damage', 'Major Damage', 'Total Loss'], p=[0.5, 0.35, 0.15]))
            
    normal_hours_probs = np.array([
        0.01, 0.01, 0.01, 0.01, 0.01, 0.02, # 00:00 - 05:00
        0.05, 0.09, 0.10, 0.07, 0.06, 0.06, # 06:00 - 11:00
        0.07, 0.07, 0.07, 0.08, 0.09, 0.06, # 12:00 - 17:00
        0.04, 0.03, 0.02, 0.01, 0.01, 0.01  # 18:00 - 23:00
    ])
    normal_hours_probs = normal_hours_probs / normal_hours_probs.sum()
    normal_hours = np.random.choice(range(24), size=num_normal, p=normal_hours_probs)
    
    normal_witnesses = np.random.poisson(lam=1.5, size=num_normal).clip(0, 4)
    normal_police_report = np.random.choice(['YES', 'NO'], size=num_normal, p=[0.75, 0.25])
    normal_vehicles = np.random.choice([1, 2, 3], size=num_normal, p=[0.55, 0.38, 0.07])
    normal_injuries = np.random.choice([0, 1, 2], size=num_normal, p=[0.65, 0.25, 0.10])
    normal_past_claims = np.random.choice([0, 1, 2, 3], size=num_normal, p=[0.70, 0.22, 0.06, 0.02])
    normal_vehicle_age = np.random.randint(1, 15, size=num_normal)
    
    normal_vehicle_val = np.clip(np.random.normal(25000 - normal_vehicle_age * 1000, 5000), 4000, 65000)
    
    # Claim amounts proportional to severity
    normal_claim_amounts = []
    normal_injury_claims = []
    normal_property_claims = []
    normal_vehicle_claims = []
    
    for i in range(num_normal):
        sev = normal_severities[i]
        v_val = normal_vehicle_val[i]
        inj = normal_injuries[i]
        
        if sev == 'Trivial Damage':
            v_claim = np.random.uniform(500, 3000)
            p_claim = np.random.uniform(0, 500)
        elif sev == 'Minor Damage':
            v_claim = np.random.uniform(2500, 8000)
            p_claim = np.random.uniform(500, 2000)
        elif sev == 'Major Damage':
            v_claim = np.random.uniform(8000, min(24000, v_val * 0.75))
            p_claim = np.random.uniform(2000, 8000)
        else: # Total Loss
            v_claim = np.random.uniform(v_val * 0.75, v_val * 0.95)
            p_claim = np.random.uniform(3000, 12000)
            
        inj_claim = (np.random.uniform(2000, 9000) * inj) if inj > 0 else 0
        tot = round(v_claim + p_claim + inj_claim, 2)
        normal_claim_amounts.append(tot)
        normal_injury_claims.append(round(inj_claim, 2))
        normal_property_claims.append(round(p_claim, 2))
        normal_vehicle_claims.append(round(v_claim, 2))
        
    normal_labels = np.zeros(num_normal, dtype=int)

    # -----------------------------
    # 2. Anomalous / Fraudulent Claims
    # -----------------------------
    # Typical fraud typologies:
    # A. Staged crash / inflated claim shortly after policy inception (jump-in claim)
    # B. Phantom / exaggerated bodily injury claim with trivial vehicle damage
    # C. High claims filed during 1 AM - 4 AM without police report or witnesses
    # D. Serial fraudster with frequent claims claiming near or above 100% vehicle value
    
    fraud_tenure = np.random.choice(
        [np.random.randint(1, 4), np.random.randint(4, 18), np.random.randint(20, 80)],
        size=num_fraud, p=[0.45, 0.40, 0.15]
    )
    
    fraud_age = np.clip(np.random.normal(36, 11, num_fraud), 19, 70).astype(int)
    fraud_premium = np.clip(np.random.normal(1150, 300, num_fraud), 550, 2500).round(2)
    fraud_deductible = np.random.choice([500, 1000, 1500, 2000], size=num_fraud, p=[0.60, 0.25, 0.10, 0.05])
    fraud_coverage = np.random.choice([100000, 300000, 500000], size=num_fraud, p=[0.30, 0.40, 0.30])
    
    fraud_incident_types = np.random.choice(
        ['Single Vehicle Collision', 'Multi-vehicle Collision', 'Vehicle Theft', 'Parked Car'],
        size=num_fraud, p=[0.45, 0.25, 0.20, 0.10]
    )
    fraud_severities = np.random.choice(['Major Damage', 'Total Loss', 'Minor Damage', 'Trivial Damage'],
                                        size=num_fraud, p=[0.48, 0.38, 0.10, 0.04])
    
    # Suspicious hours (late night / early dawn)
    fraud_hours_probs = np.array([
        0.09, 0.11, 0.14, 0.15, 0.10, 0.04, # 00:00 - 05:00 (heavy concentration)
        0.02, 0.03, 0.03, 0.02, 0.02, 0.02, 
        0.02, 0.02, 0.02, 0.02, 0.03, 0.03, 
        0.03, 0.02, 0.02, 0.02, 0.03, 0.03
    ])
    fraud_hours_probs = fraud_hours_probs / fraud_hours_probs.sum()
    fraud_hours = np.random.choice(range(24), size=num_fraud, p=fraud_hours_probs)
    
    fraud_witnesses = np.random.choice([0, 1, 2], size=num_fraud, p=[0.78, 0.18, 0.04])
    fraud_police_report = np.random.choice(['NO', 'YES'], size=num_fraud, p=[0.72, 0.28])
    fraud_vehicles = np.random.choice([1, 2, 3], size=num_fraud, p=[0.68, 0.26, 0.06])
    fraud_injuries = np.random.choice([0, 1, 2], size=num_fraud, p=[0.30, 0.45, 0.25])
    fraud_past_claims = np.random.choice([0, 1, 2, 3, 4], size=num_fraud, p=[0.25, 0.30, 0.25, 0.15, 0.05])
    fraud_vehicle_age = np.random.randint(3, 18, size=num_fraud)
    fraud_vehicle_val = np.clip(np.random.normal(20000 - fraud_vehicle_age * 900, 4500), 3000, 45000)
    
    fraud_claim_amounts = []
    fraud_injury_claims = []
    fraud_property_claims = []
    fraud_vehicle_claims = []
    
    for i in range(num_fraud):
        v_val = fraud_vehicle_val[i]
        sev = fraud_severities[i]
        inj = fraud_injuries[i]
        
        # Inflated claims relative to vehicle value
        v_claim = np.random.uniform(v_val * 0.85, v_val * 1.35)
        p_claim = np.random.uniform(5000, 16000)
        # Often phantom high bodily injuries claimed even on trivial/minor accidents
        inj_claim = np.random.uniform(8000, 22000) if (inj > 0 or sev in ['Minor Damage', 'Trivial Damage']) else np.random.uniform(3000, 9000)
        
        tot = round(v_claim + p_claim + inj_claim, 2)
        fraud_claim_amounts.append(tot)
        fraud_injury_claims.append(round(inj_claim, 2))
        fraud_property_claims.append(round(p_claim, 2))
        fraud_vehicle_claims.append(round(v_claim, 2))
        
    fraud_labels = np.ones(num_fraud, dtype=int)
    
    # -----------------------------
    # 3. Combine and Build DataFrame
    # -----------------------------
    policy_tenure = np.concatenate([normal_tenure, fraud_tenure])
    insured_age = np.concatenate([normal_age, fraud_age])
    annual_premium = np.concatenate([normal_premium, fraud_premium])
    policy_deductible = np.concatenate([normal_deductible, fraud_deductible])
    policy_coverage = np.concatenate([normal_coverage, fraud_coverage])
    incident_types = np.concatenate([normal_incident_types, fraud_incident_types])
    incident_severities = np.concatenate([normal_severities, fraud_severities])
    incident_hours = np.concatenate([normal_hours, fraud_hours])
    witnesses = np.concatenate([normal_witnesses, fraud_witnesses])
    police_reports = np.concatenate([normal_police_report, fraud_police_report])
    vehicles_involved = np.concatenate([normal_vehicles, fraud_vehicles])
    injuries = np.concatenate([normal_injuries, fraud_injuries])
    past_claims = np.concatenate([normal_past_claims, fraud_past_claims])
    vehicle_age = np.concatenate([normal_vehicle_age, fraud_vehicle_age])
    vehicle_value = np.concatenate([normal_vehicle_val, fraud_vehicle_val]).round(2)
    
    total_claims = np.concatenate([normal_claim_amounts, fraud_claim_amounts])
    injury_claims = np.concatenate([normal_injury_claims, fraud_injury_claims])
    property_claims = np.concatenate([normal_property_claims, fraud_property_claims])
    vehicle_claims = np.concatenate([normal_vehicle_claims, fraud_vehicle_claims])
    fraud_ground_truth = np.concatenate([normal_labels, fraud_labels])
    
    # Generate dates and identifiers
    base_date = datetime(2025, 1, 1)
    dates = [base_date + timedelta(days=int(d)) for d in np.random.randint(0, 650, size=num_samples)]
    dates_str = [d.strftime("%Y-%m-%d") for d in dates]
    
    claim_ids = [f"CLM-2026-{10000 + i}" for i in range(num_samples)]
    policy_ids = [f"POL-{np.random.randint(100000, 999999)}" for _ in range(num_samples)]
    
    genders = np.random.choice(['MALE', 'FEMALE'], size=num_samples, p=[0.52, 0.48])
    educations = np.random.choice(['High School', 'Associate', 'Bachelor', 'Master', 'PhD'],
                                  size=num_samples, p=[0.25, 0.20, 0.35, 0.15, 0.05])
    vehicle_categories = np.random.choice(['Sedan', 'SUV', 'Compact', 'Luxury', 'Truck'],
                                          size=num_samples, p=[0.38, 0.30, 0.15, 0.10, 0.07])
    
    df = pd.DataFrame({
        'claim_id': claim_ids,
        'policy_id': policy_ids,
        'policy_tenure_months': policy_tenure,
        'insured_age': insured_age,
        'insured_sex': genders,
        'insured_education': educations,
        'annual_premium': annual_premium,
        'policy_deductible': policy_deductible,
        'policy_coverage_limit': policy_coverage,
        'incident_date': dates_str,
        'incident_type': incident_types,
        'incident_severity': incident_severities,
        'incident_hour': incident_hours,
        'number_of_vehicles_involved': vehicles_involved,
        'bodily_injuries': injuries,
        'witnesses': witnesses,
        'police_report_available': police_reports,
        'vehicle_age': vehicle_age,
        'vehicle_category': vehicle_categories,
        'vehicle_market_value': vehicle_value,
        'injury_claim': injury_claims,
        'property_claim': property_claims,
        'vehicle_claim': vehicle_claims,
        'total_claim_amount': total_claims,
        'past_claims_count': past_claims,
        'is_fraud': fraud_ground_truth
    })
    
    # Shuffle dataset
    df = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    return df

if __name__ == "__main__":
    df = generate_insurance_dataset(num_samples=5000, fraud_ratio=0.12)
    print(f"Generated {len(df)} insurance claim records.")
    print(f"Fraud distribution:\n{df['is_fraud'].value_counts(normalize=True)}")
    df.to_csv("C:/Users/srila/.gemini/antigravity/scratch/insurance_claim_fraud_detection/data/claims_dataset.csv", index=False)
    print("Saved claims_dataset.csv successfully.")
