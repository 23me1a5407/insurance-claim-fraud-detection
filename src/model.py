"""
Isolation Forest Anomaly Detection Model
Encapsulates scikit-learn's Isolation Forest with scoring calibration,
triage thresholding, and risk factor explainability.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
import joblib
from typing import Dict, Any, List, Tuple

class InsuranceAnomalyDetector:
    def __init__(self, n_estimators: int = 200, contamination: float = 0.12, random_state: int = 42):
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.random_state = random_state
        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            max_samples='auto',
            random_state=self.random_state,
            n_jobs=-1
        )
        self.fitted = False
        self.baseline_mean: pd.Series = None
        self.baseline_std: pd.Series = None

    def fit(self, X: pd.DataFrame):
        self.model.fit(X)
        self.fitted = True
        self.baseline_mean = X.mean()
        self.baseline_std = X.std().replace(0, 1.0)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Returns 1 for normal, -1 for anomaly."""
        return self.model.predict(X)

    def decision_function(self, X: pd.DataFrame) -> np.ndarray:
        """Returns scikit-learn decision function (lower = more anomalous)."""
        return self.model.decision_function(X)

    def get_calibrated_risk_score(self, X: pd.DataFrame) -> np.ndarray:
        """
        Converts Isolation Forest decision function into a 0.0 - 100.0% Risk Score.
        Higher score = higher likelihood of being anomalous/fraudulent.
        """
        raw_scores = self.decision_function(X)
        # Shift and scale sigmoid or min-max transformation
        # Typically raw scores range roughly from -0.3 to +0.25
        # We use a logistic mapping centered around the contamination boundary ~ 0.0
        z = -12.0 * raw_scores
        prob = 1.0 / (1.0 + np.exp(-z))
        risk_pct = np.clip(prob * 100.0, 0.0, 100.0)
        return np.round(risk_pct, 2)

    def assign_triage_priority(self, risk_score: float) -> str:
        """Assigns investigator priority queue tier."""
        if risk_score >= 70.0:
            return "HIGH - Priority Investigation"
        elif risk_score >= 45.0:
            return "MEDIUM - Review Required"
        else:
            return "LOW - Standard Fast-Track"

    def explain_instance(self, single_features: pd.Series, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Explains which features deviated the most towards anomaly
        using standardized z-score deviation against legitimate baseline.
        """
        if self.baseline_mean is None or self.baseline_std is None:
            return []
            
        z_scores = (single_features - self.baseline_mean) / self.baseline_std
        # Absolute divergence
        abs_divergence = z_scores.abs().sort_values(ascending=False)
        
        explanations = []
        for feature, dev in abs_divergence.head(top_k).items():
            val = single_features[feature]
            mean_val = self.baseline_mean[feature]
            direction = "significantly higher than normal" if val > mean_val else "significantly lower than normal"
            
            # Clean human-readable label
            label = feature.replace('_', ' ').title()
            explanations.append({
                "feature": label,
                "raw_feature": feature,
                "z_deviation": round(float(dev), 2),
                "observed_value": round(float(val), 2),
                "baseline_mean": round(float(mean_val), 2),
                "impact_direction": direction
            })
        return explanations

    def save(self, filepath: str):
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath: str) -> 'InsuranceAnomalyDetector':
        return joblib.load(filepath)
