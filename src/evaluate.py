"""
Evaluation Pipeline for Isolation Forest Motor Insurance Fraud Detection
Computes ROC-AUC, PR-AUC, Confusion Matrix, Classification Report,
and Investigator Triage Queue Efficiency (Precision@K & Recall@K).
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve
)
import matplotlib.pyplot as plt
import json
from typing import Dict, Any

def evaluate_model_performance(y_true: np.ndarray, risk_scores: np.ndarray, binary_predictions: np.ndarray) -> Dict[str, Any]:
    """
    Evaluates anomaly detector predictions against ground truth fraud labels.
    Note: binary_predictions is 1 for anomaly (fraud), 0 for normal.
    """
    roc_auc = roc_auc_score(y_true, risk_scores)
    pr_auc = average_precision_score(y_true, risk_scores)
    
    cm = confusion_matrix(y_true, binary_predictions)
    tn, fp, fn, tp = cm.ravel()
    
    report = classification_report(y_true, binary_predictions, output_dict=True, target_names=['Legitimate', 'Fraudulent'])
    
    # Triage Queue Metrics: Precision and Recall at Top K percent
    df_eval = pd.DataFrame({'y_true': y_true, 'risk': risk_scores}).sort_values('risk', ascending=False).reset_index(drop=True)
    total_fraud = int(y_true.sum())
    total_cases = len(y_true)
    
    triage_metrics = {}
    for pct in [5, 10, 15, 20, 25]:
        k = int(total_cases * (pct / 100.0))
        top_k = df_eval.iloc[:k]
        fraud_in_top_k = int(top_k['y_true'].sum())
        prec_at_k = fraud_in_top_k / k if k > 0 else 0
        rec_at_k = fraud_in_top_k / total_fraud if total_fraud > 0 else 0
        triage_metrics[f"top_{pct}_percent"] = {
            "queue_depth_cases": k,
            "fraud_detected": fraud_in_top_k,
            "precision": round(prec_at_k, 4),
            "recall": round(rec_at_k, 4)
        }
        
    results = {
        "roc_auc": round(float(roc_auc), 4),
        "pr_auc": round(float(pr_auc), 4),
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp)
        },
        "accuracy": round(float((tp + tn) / total_cases), 4),
        "fraud_precision": round(float(report['Fraudulent']['precision']), 4),
        "fraud_recall": round(float(report['Fraudulent']['recall']), 4),
        "fraud_f1": round(float(report['Fraudulent']['f1-score']), 4),
        "triage_queue_efficiency": triage_metrics
    }
    return results

def plot_and_save_evaluation_charts(y_true: np.ndarray, risk_scores: np.ndarray, save_path: str):
    fpr, tpr, _ = roc_curve(y_true, risk_scores)
    precision, recall, _ = precision_recall_curve(y_true, risk_scores)
    roc_auc = roc_auc_score(y_true, risk_scores)
    pr_auc = average_precision_score(y_true, risk_scores)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 1. ROC Curve
    axes[0].plot(fpr, tpr, color='#1E88E5', lw=2.5, label=f'ROC Curve (AUC = {roc_auc:.3f})')
    axes[0].plot([0, 1], [0, 1], color='#9E9E9E', linestyle='--', lw=1.5)
    axes[0].set_xlim([0.0, 1.0])
    axes[0].set_ylim([0.0, 1.05])
    axes[0].set_xlabel('False Positive Rate (FPR)', fontsize=11)
    axes[0].set_ylabel('True Positive Rate (TPR / Recall)', fontsize=11)
    axes[0].set_title('Receiver Operating Characteristic (ROC)', fontsize=13, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc='lower right', fontsize=11)
    
    # 2. Precision-Recall Curve
    axes[1].plot(recall, precision, color='#E53935', lw=2.5, label=f'PR Curve (AP = {pr_auc:.3f})')
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel('Recall (Fraud Detection Rate)', fontsize=11)
    axes[1].set_ylabel('Precision (Investigator Accuracy)', fontsize=11)
    axes[1].set_title('Precision-Recall Curve', fontsize=13, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc='lower left', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved evaluation charts to {save_path}")
