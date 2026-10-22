"""
Training and Evaluation Pipeline
Orchestrates data generation, train/test split, model fitting, evaluation, and artifact saving.
"""

import os
import json
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split

from src.data_generator import generate_insurance_dataset
from src.preprocessing import InsuranceDataPreprocessor
from src.model import InsuranceAnomalyDetector
from src.evaluate import evaluate_model_performance, plot_and_save_evaluation_charts

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
DOCS_DIR = os.path.join(BASE_DIR, "docs")

def run_training_pipeline():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(DOCS_DIR, exist_ok=True)
    
    print("=" * 60)
    print("INSURANCE CLAIM FRAUD ANOMALY DETECTION PIPELINE")
    print("Author: Bobburi Praveen Kumar | Guide: Mrs P. Neeraja")
    print("=" * 60)
    
    # 1. Dataset Generation
    dataset_path = os.path.join(DATA_DIR, "claims_dataset.csv")
    print("\n[Step 1/5] Synthesizing motor insurance policy & claims dataset...")
    df = generate_insurance_dataset(num_samples=5000, fraud_ratio=0.12, random_state=42)
    df.to_csv(dataset_path, index=False)
    print(f"Generated {len(df)} total claims (Legitimate: {(df['is_fraud']==0).sum()}, Fraudulent: {(df['is_fraud']==1).sum()})")
    
    # 2. Train-Test Split (80% Train, 20% Held-Out Test)
    print("\n[Step 2/5] Performing stratified train/held-out test split (80/20)...")
    train_df, test_df = train_test_split(df, test_size=0.20, random_state=42, stratify=df['is_fraud'])
    
    train_path = os.path.join(DATA_DIR, "train_claims.csv")
    test_path = os.path.join(DATA_DIR, "test_claims.csv")
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    print(f"Training set: {len(train_df)} records | Held-out Test set: {len(test_df)} records")
    
    # 3. Preprocessing and Feature Engineering
    print("\n[Step 3/5] Engineering features and transforming data...")
    preprocessor = InsuranceDataPreprocessor()
    X_train_scaled = preprocessor.fit_transform(train_df)
    X_test_scaled = preprocessor.transform(test_df)
    
    # Save preprocessor
    preprocessor_path = os.path.join(MODELS_DIR, "preprocessor.joblib")
    joblib.dump(preprocessor, preprocessor_path)
    print(f"Saved fitted preprocessor with {X_train_scaled.shape[1]} features.")
    
    # 4. Training Isolation Forest Anomaly Detector
    print("\n[Step 4/5] Training Isolation Forest detector...")
    # Note: In unsupervised anomaly detection, we train primarily on normal claims or general claims population
    detector = InsuranceAnomalyDetector(n_estimators=200, contamination=0.12, random_state=42)
    detector.fit(X_train_scaled)
    
    model_path = os.path.join(MODELS_DIR, "isolation_forest_model.joblib")
    detector.save(model_path)
    print(f"Saved trained Isolation Forest model to {model_path}")
    
    # 5. Model Evaluation on Held-Out Test Data
    print("\n[Step 5/5] Evaluating performance on held-out test data...")
    y_test = test_df['is_fraud'].values
    test_raw_preds = detector.predict(X_test_scaled)
    # IsolationForest returns -1 for anomaly, 1 for normal.
    # Map to binary 1 = anomaly/fraud, 0 = normal
    test_binary_anomalies = (test_raw_preds == -1).astype(int)
    test_risk_scores = detector.get_calibrated_risk_score(X_test_scaled)
    
    metrics = evaluate_model_performance(y_test, test_risk_scores, test_binary_anomalies)
    
    metrics_path = os.path.join(MODELS_DIR, "evaluation_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics summary to {metrics_path}")
    
    chart_path = os.path.join(DOCS_DIR, "model_performance_curves.png")
    plot_and_save_evaluation_charts(y_test, test_risk_scores, chart_path)
    
    # Add predicted risk scores and priority to test dataset for quick inspector triage demo
    test_results_df = test_df.copy()
    test_results_df['anomaly_risk_score'] = test_risk_scores
    test_results_df['triage_priority'] = [detector.assign_triage_priority(s) for s in test_risk_scores]
    test_results_df['predicted_as_anomaly'] = test_binary_anomalies
    
    scored_test_path = os.path.join(DATA_DIR, "scored_test_claims.csv")
    test_results_df.to_csv(scored_test_path, index=False)
    
    print("\n" + "=" * 60)
    print("TRAINING & EVALUATION SUMMARY")
    print("=" * 60)
    print(f"ROC-AUC Score:           {metrics['roc_auc']:.4f}")
    print(f"PR-AUC (Avg Precision):  {metrics['pr_auc']:.4f}")
    print(f"Overall Accuracy:        {metrics['accuracy']*100:.2f}%")
    print(f"Fraud Detection Recall:  {metrics['fraud_recall']*100:.2f}%")
    print(f"Fraud Triage Precision:  {metrics['fraud_precision']*100:.2f}%")
    print(f"Top 10% Triage Recall:   {metrics['triage_queue_efficiency']['top_10_percent']['recall']*100:.2f}%")
    print(f"Top 20% Triage Recall:   {metrics['triage_queue_efficiency']['top_20_percent']['recall']*100:.2f}%")
    print("=" * 60)
    print("Pipeline execution completed successfully!\n")

if __name__ == "__main__":
    run_training_pipeline()
