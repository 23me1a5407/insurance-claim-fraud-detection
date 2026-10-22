# PROJECT REPORT
# An Isolation Forest Anomaly Detector-Based Approach to Insurance Claim Fraud

**Author:** BOBBURI PRAVEEN KUMAR  
**Supervisor / Guide:** Dr. P. Neeraja  
**Department:** Computer Science & Engineering / Information Technology  
**Domain:** Machine Learning & Predictive Analytics  
**Academic Year:** 2025–2026  

---

## ABSTRACT
This project proposes an Isolation Forest anomaly detector to surface suspicious motor-insurance claims, addressing the problem of insurance claim fraud through a data-driven approach. The pipeline draws on policy and claims history data, applying suitable pre-processing before training and testing the model on held-out data. The project culminates in an investigator triage queue, bridging the gap between the underlying model and everyday use. Evaluated on a held-out test split, the developed Isolation Forest framework attains a **ROC-AUC of 0.9827**, a **Precision-Recall AUC of 0.9295**, and captures **96.67% of all fraudulent claims within the top 20% of prioritized queue reviews**, dramatically accelerating forensic throughput and reducing financial exposure.

---

## CHAPTER 1: INTRODUCTION

### 1.1 Background
The insurance industry is foundational to modern socioeconomic stability, insulating individuals and commercial enterprises against catastrophic asset losses. However, the Coalition Against Insurance Fraud estimates that insurance fraud costs consumers and carriers tens of billions of dollars annually. In motor vehicle insurance, fraudulent claims inflate overall premiums by an estimated 10% to 15% for honest policyholders.

Motor insurance fraud spans various deceptive behaviors:
- **Opportunistic / Inflated Claims:** Padding legitimate claims with pre-existing damage or exaggerating bodily injury treatment expenses.
- **Pre-meditated / Staged Accidents:** Intentionally orchestrating multi-vehicle or single-vehicle collisions ("crash-for-cash").
- **Jump-in / Fabricated Policies:** Procuring an insurance policy immediately after an accident occurs and backdating or misrepresenting the incident timeline.
- **Phantom Claims:** Reporting non-existent passengers or staged vehicle thefts.

### 1.2 Problem Statement
Traditional fraud mitigation in motor insurance suffers from three systemic weaknesses:
1. **Manual Ad-hoc Auditing:** Claims adjusters handle hundreds of dossiers weekly; thorough forensic audits are time-consuming, meaning only a fraction of claims can be manually inspected.
2. **Brittle Rule-Based Systems:** Hard-coded heuristics (e.g., "claim > \$10,000 and tenure < 6 months") fail to generalize and are easily bypassed once syndicate rings learn the thresholds.
3. **Imbalance in Supervised ML:** Legitimate claims outnumber fraudulent claims by at least 10:1 or 20:1. Supervised classifiers trained on skewed labels tend to overfit the majority legitimate class, resulting in high false-negative rates and poor detection of zero-day fraud mechanisms.

### 1.3 Project Objectives
- Synthesize an end-to-end motor insurance claims pipeline incorporating policy tenure, demographic attributes, accident circumstances, witness verification, and claim financial ratios.
- Engineer domain-specific behavioral features, including claim-to-vehicle-value ratio, jump-in policy tenure flags, and late-night unwitnessed incident markers.
- Implement and optimize an **Isolation Forest anomaly detection model** leveraging recursive tree isolation without reliance on labeled fraud samples during isolation structure construction.
- Formulate a continuous calibration engine converting raw decision tree path lengths into a normalized **0.0% to 100.0% Anomaly Risk Score**.
- Develop a production-ready **Investigator Triage Queue** web interface equipped with priority tiering, feature attribution explanations, and a what-if simulator.

---

## CHAPTER 2: LITERATURE SURVEY

### 2.1 Overview of Fraud Detection Techniques
| Paradigm | Core Mechanics | Advantages | Disadvantages |
| :--- | :--- | :--- | :--- |
| **Expert Heuristics / Rules** | Static boolean checks & threshold rules | Highly transparent; easy to deploy | Inflexible; high maintenance; quickly circumvented |
| **Supervised Learning** | Logistic Regression, XGBoost, Random Forest | High accuracy on known, labeled historical patterns | Severely degraded by label sparsity; incapable of detecting new fraud typologies |
| **Distance-Based Outlier Detection** | $k$-Nearest Neighbors ($k$-NN), Local Outlier Factor (LOF) | Does not require labeled fraud data | Computationally prohibitive $\mathcal{O}(n^2)$; sensitive to curse of dimensionality |
| **Isolation-Based Outlier Detection** | **Isolation Forest (Liu, Ting & Zhou)** | Linear time complexity $\mathcal{O}(n \log n)$; robust in high dimensions; immune to class imbalance | Requires calibrated scoring and feature attribution for interpretability |

### 2.2 Why Isolation Forest?
Introduced by Fei Tony Liu, Kai Ming Ting, and Zhi-Hua Zhou (2008), Isolation Forest revolutionized anomaly detection by inverting conventional density-based thinking. Rather than profiling dense clusters of normal data and declaring points in low-density regions as outliers, Isolation Forest explicitly isolates anomalies:
1. Anomalies are **few in number** (minority instances).
2. Anomalies possess **divergent attribute values** compared to inliers.

Consequently, in a randomized recursive partition tree (an Isolation Tree), anomalous claims are isolated close to the root node with significantly fewer split steps.

---

## CHAPTER 3: MATHEMATICAL FORMULATION

### 3.1 Binary Search Tree (BST) Analogy and Average Path Length
An Isolation Tree (iTree) is structurally equivalent to a proper Binary Search Tree. For a dataset of $n$ instances, the average path length of an unsuccessful search in a BST corresponds to the expected isolation depth of random instances:
$$c(n) = 2 \left( \ln(n - 1) + 0.5772156649 \right) - \frac{2(n - 1)}{n}$$
where $0.5772156649$ represents Euler's constant ($\gamma$). Here, $c(n)$ serves as the normalization factor representing the average path length across all possible tree configurations of size $n$.

### 3.2 Anomaly Score Formulation
For a specific claim vector $x$ evaluated across an ensemble of $t$ independent isolation trees $\{T_1, T_2, \dots, T_t\}$, let $h_i(x)$ represent the path length (number of edges traversed from root to termination) in tree $T_i$.

The expected path length is:
$$E(h(x)) = \frac{1}{t} \sum_{i=1}^t h_i(x)$$

The anomaly score $s(x, n)$ is defined monotonically as:
$$s(x, n) = 2^{-\frac{E(h(x))}{c(n)}}$$

#### Mathematical Analysis of Score Boundaries:
1. **Case A ($E(h(x)) \to 0$):**  
   If the instance is isolated in very few splits near the root, $s(x, n) \to 2^0 = 1$. The claim is an **extreme anomaly (high fraud likelihood)**.
2. **Case B ($E(h(x)) \to c(n)$):**  
   If the instance's path length equals the theoretical average, $s(x, n) \to 2^{-1} = 0.5$. The instance lacks any distinguishing outlier characteristics.
3. **Case C ($E(h(x)) \to n - 1$):**  
   If the instance requires maximal recursive splitting down to the deepest leaves, $s(x, n) \to 2^{-\infty} \approx 0$. The claim is a **prototypical normal inlier**.

### 3.3 Score Calibration for Operational Triage
Raw scikit-learn decision functions satisfy $f(x) = - (s(x, n) - \text{threshold})$. To provide clear operational utility to claims adjusters, we calibrate the raw decision function via a scaled logistic mapping:
$$\text{RiskScore}(x) = \frac{100}{1 + e^{12 \cdot f(x)}} \quad [\%]$$

Triage priority bands are established as:
- **HIGH Priority ($\ge 70.0\%$):** Immediate freeze; route to Special Investigation Unit (SIU).
- **MEDIUM Priority ($45.0\% \le \text{Risk} < 70.0\%$):** Additional documentation / mechanic audit requested.
- **LOW Priority ($< 45.0\%$):** Automated fast-track clearance for prompt settlement.

---

## CHAPTER 4: SYSTEM ARCHITECTURE & METHODOLOGY

### 4.1 End-to-End Pipeline
```
[Claims & Policy Ingestion]
           │
           ▼
[Feature Engineering Engine]
   ├─ Claim-to-Vehicle Ratio
   ├─ Jump-In Policy Tenure Flag
   ├─ Unwitnessed Night Accident
   └─ Sub-claim Disparity Metrics
           │
           ▼
[Standard Scaling & Matrix Construction]
           │
           ▼
[Isolation Forest Ensemble (200 Trees)]
           │
           ▼
[Score Calibration & Triage Classification]
           │
           ▼
[Streamlit Investigator Triage Dashboard]
   ├─ Active Priority Queue
   ├─ Root Cause Explainability (Z-score Deviations)
   └─ What-If Simulation Station
```

### 4.2 Feature Engineering
The preprocessor computes eight domain-specific engineered features:
1. `claim_to_vehicle_ratio`: Ratio of total claim amount to current vehicle market value.
2. `claim_to_premium_ratio`: Ratio of claimed payout to annual policy premium.
3. `injury_share_of_claim`: Percentage of total payout allocated to bodily injury claims.
4. `is_early_policy_claim`: Binary marker flagging claims filed within 90 days ($\le 3$ months) of policy bind date.
5. `is_odd_hour_incident`: Binary marker flagging collisions between 23:00 and 04:00.
6. `major_damage_no_police_report`: Binary flag for severe/total losses lacking police documentation.
7. `zero_witness_night_incident`: Binary flag for nighttime accidents without independent witnesses.
8. `claim_intensity_by_past_record`: Interaction term between past claims frequency and claim volume.

---

## CHAPTER 5: EXPERIMENTAL RESULTS & EVALUATION

### 5.1 Dataset Partitioning
The pipeline was trained and evaluated on 5,000 multi-attribute motor insurance records with an 80/20 stratified split:
- **Training Set:** 4,000 claims (used for isolation tree construction and preprocessor fitting)
- **Held-Out Test Set:** 1,000 claims (strictly held out for unbiased benchmarking)

### 5.2 Quantitative Performance Metrics
| Metric | Benchmark Result |
| :--- | :--- |
| **ROC-AUC Score** | **0.9827** |
| **Precision-Recall AUC (PR-AUC)** | **0.9295** |
| **Overall Classification Accuracy** | **96.70%** |
| **Fraud Detection Recall** | **87.50%** |
| **Fraud Triage Precision** | **85.37%** |
| **Top 10% Queue Recall (Recall@10%)** | **77.50%** |
| **Top 20% Queue Recall (Recall@20%)** | **96.67%** |

### 5.3 Investigator Triage Queue Efficiency Analysis
In insurance operations, investigators cannot audit 100% of claims. The true test of an anomaly detector is **cumulative gains at constrained inspection capacity**:

- Auditing just the **Top 5%** highest-risk claims captures **41.67%** of all fraudulent claims in the portfolio (Precision: 100%).
- Auditing the **Top 10%** highest-risk claims captures **77.50%** of all fraudulent claims (Precision: 93.0%).
- Auditing the **Top 20%** highest-risk claims captures **96.67%** of all fraudulent claims (Precision: 58.0%).

**Operational Implication:** The claims department can intercept **over 96% of fraudulent claims** while examining only 1 out of every 5 incoming dossiers, freeing up 80% of investigator capacity to fast-track honest claims.

---

## CHAPTER 6: USER INTERFACE & INVESTIGATOR DASHBOARD

The system includes a Streamlit web application with four interactive operational modules:
1. **Investigator Triage Queue:** Real-time priority list of claims with one-click actions ("Fast-Track Approve", "Escalate to SIU", "Flag & Reject") and status persistence.
2. **Feature Attribution Inspector:** Decomposes any flagged claim into top contributing risk drivers using baseline standard deviation divergence.
3. **What-If Claim Risk Analyzer:** Interactive simulation form with preloaded real-world case studies (Jump-in claim, Exaggerated injury, Normal fender bender) enabling live fraud scoring.
4. **Batch Processing Station:** Allows adjusters to upload raw claim batches in CSV format and immediately download an exportable, risk-scored queue.

---

## CHAPTER 7: CONCLUSION & FUTURE WORK

### 7.1 Conclusion
The developed system demonstrates that an Isolation Forest anomaly detector, paired with domain-aware feature engineering, provides a highly accurate, scalable, and operationally viable solution for motor insurance fraud detection. By decoupling anomaly detection from labeled fraud datasets, the system overcomes severe class imbalance and captures emergent fraud strategies. The integrated investigator triage queue bridges machine learning predictions and everyday insurance claims administration.

### 7.2 Future Scope
- **Integration with Vehicle Telematics (IoT):** Ingesting high-frequency accelerometer and GPS data from onboard telematics devices to verify collision severity and timestamp accuracy.
- **Deep Semi-Supervised Ensembles:** Combining Isolation Forest with deep autoencoders or Graph Neural Networks (GNNs) to detect organized fraud rings and multi-party collusion networks.
- **Explainable AI (SHAP / LIME):** Generating tree-SHAP explanations natively inside real-time adjuster workflows.

---

## REFERENCES
1. Liu, F. T., Ting, K. M., & Zhou, Z. H. (2008). "Isolation Forest." *Eighth IEEE International Conference on Data Mining*, pp. 413-422.
2. Coalition Against Insurance Fraud (2024). *The Impact of Insurance Fraud on the Global Economy*.
3. Phua, C., Lee, V., Smith, K., & Gayler, R. (2010). "A Comprehensive Survey of Data Mining-based Fraud Detection Research." *Artificial Intelligence Review*, 34(1), pp. 1-14.
4. Pedregosa, F., et al. (2011). "Scikit-learn: Machine Learning in Python." *Journal of Machine Learning Research*, 12, pp. 2825-2830.
