"""
Data Preprocessing and Feature Engineering Pipeline
Handles data cleaning, domain-specific feature engineering, encoding, and scaling.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List, Dict, Any

CATEGORICAL_COLS = [
    'incident_type',
    'incident_severity',
    'police_report_available',
    'vehicle_category',
    'insured_sex'
]

NUMERICAL_BASE_COLS = [
    'policy_tenure_months',
    'insured_age',
    'annual_premium',
    'policy_deductible',
    'policy_coverage_limit',
    'incident_hour',
    'number_of_vehicles_involved',
    'bodily_injuries',
    'witnesses',
    'vehicle_age',
    'vehicle_market_value',
    'injury_claim',
    'property_claim',
    'vehicle_claim',
    'total_claim_amount',
    'past_claims_count'
]

ENGINEERED_COLS = [
    'claim_to_vehicle_ratio',
    'claim_to_premium_ratio',
    'injury_share_of_claim',
    'is_early_policy_claim',
    'is_odd_hour_incident',
    'major_damage_no_police_report',
    'zero_witness_night_incident',
    'claim_intensity_by_past_record'
]

class InsuranceDataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_columns: List[str] = []
        self.fitted = False
        self.dummy_columns: List[str] = []

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        data = df.copy()
        
        # 1. Claim amount vs Vehicle Market Value ratio
        data['claim_to_vehicle_ratio'] = data['total_claim_amount'] / (data['vehicle_market_value'] + 1e-5)
        
        # 2. Claim amount vs Annual Premium ratio
        data['claim_to_premium_ratio'] = data['total_claim_amount'] / (data['annual_premium'] + 1e-5)
        
        # 3. Share of bodily injury within total claim
        data['injury_share_of_claim'] = data['injury_claim'] / (data['total_claim_amount'] + 1e-5)
        
        # 4. Jump-in claim (policy active for 3 months or less)
        data['is_early_policy_claim'] = (data['policy_tenure_months'] <= 3).astype(float)
        
        # 5. Odd hour crash (between 23:00 and 04:00)
        data['is_odd_hour_incident'] = ((data['incident_hour'] >= 23) | (data['incident_hour'] <= 4)).astype(float)
        
        # 6. Major / Total loss with NO police report
        data['major_damage_no_police_report'] = (
            (data['incident_severity'].isin(['Major Damage', 'Total Loss'])) & 
            (data['police_report_available'] == 'NO')
        ).astype(float)
        
        # 7. Unwitnessed nighttime collision
        data['zero_witness_night_incident'] = (
            (data['witnesses'] == 0) & 
            ((data['incident_hour'] >= 21) | (data['incident_hour'] <= 5))
        ).astype(float)
        
        # 8. Past claims frequency pressure
        data['claim_intensity_by_past_record'] = (data['past_claims_count'] + 1) * (data['total_claim_amount'] / 10000.0)
        
        return data

    def fit(self, df: pd.DataFrame):
        data = self.engineer_features(df)
        
        # One-hot encode categoricals
        cat_df = pd.get_dummies(data[CATEGORICAL_COLS], drop_first=False, dtype=float)
        self.dummy_columns = list(cat_df.columns)
        
        num_features = data[NUMERICAL_BASE_COLS + ENGINEERED_COLS]
        full_features = pd.concat([num_features, cat_df], axis=1)
        
        self.feature_columns = list(full_features.columns)
        self.scaler.fit(full_features)
        self.fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted:
            raise ValueError("Preprocessor has not been fitted yet!")
            
        data = self.engineer_features(df)
        cat_df = pd.get_dummies(data[CATEGORICAL_COLS], drop_first=False, dtype=float)
        
        # Align one-hot dummy columns with training structure
        for col in self.dummy_columns:
            if col not in cat_df.columns:
                cat_df[col] = 0.0
        cat_df = cat_df[self.dummy_columns]
        
        num_features = data[NUMERICAL_BASE_COLS + ENGINEERED_COLS]
        full_features = pd.concat([num_features, cat_df], axis=1)
        full_features = full_features[self.feature_columns]
        
        scaled_array = self.scaler.transform(full_features)
        scaled_df = pd.DataFrame(scaled_array, columns=self.feature_columns, index=df.index)
        return scaled_df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)
