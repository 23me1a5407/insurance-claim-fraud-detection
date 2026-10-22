# VIVA VOCE & TECHNICAL INTERVIEW PREPARATION GUIDE
## An Isolation Forest Anomaly Detector-Based Approach to Insurance Claim Fraud

**Student Name:** BOBBURI PRAVEEN KUMAR  
**Supervisor / Guide:** Mrs P. Neeraja  
**Domain:** Machine Learning & Predictive Analytics  

---

### Q1: What is the core motivation of using Isolation Forest instead of Supervised ML (like Logistic Regression or Random Forest)?
**Answer:**  
In real-world insurance environments:
1. **Severe Class Imbalance:** Fraudulent claims constitute a tiny minority (< 5%–10% of total claims). Supervised algorithms tend to maximize overall accuracy by predicting everything as the legitimate majority class, resulting in high false-negative rates.
2. **Lack of High-Quality Fraud Labels:** Many frauds go undetected or are discovered months later after settlements. Building reliable ground-truth labels for supervised training is expensive and lagging.
3. **Emergence of Novel Fraud Schemes (Zero-Day Fraud):** Supervised classifiers can only detect fraud patterns present in historical training data. Isolation Forest makes **no assumption about prior fraud patterns**; it detects anything statistically deviant from standard claims behavior.

---

### Q2: How does an Isolation Tree (iTree) isolate anomalies? What is the mathematical principle?
**Answer:**  
Isolation Forest works on the axiom that **anomalies are 'few and different'**.  
- To build an Isolation Tree, a feature is selected at random, and a random split value is chosen between the feature's minimum and maximum values.
- This recursive partitioning continues until points are isolated into individual leaf nodes or a maximum depth is reached.
- Normal, dense points require many random partitions to isolate (deep in the tree, large path length $h(x)$).
- Outliers/anomalous claims reside in sparse regions of the feature space and are partitioned off in very few splits (near the root, short path length $h(x)$).

---

### Q3: What is the significance of $c(n)$ in the Isolation Forest anomaly score formula?
**Answer:**  
$c(n)$ is the **average path length of an unsuccessful search in a Binary Search Tree (BST)** constructed over $n$ instances:
$$c(n) = 2 \left( \ln(n - 1) + 0.5772156649 \right) - \frac{2(n - 1)}{n}$$
where $0.5772156649$ is Euler's constant.  
Because an Isolation Tree is structurally equivalent to a BST, $c(n)$ provides a mathematically rigorous normalization factor. By dividing the empirical average path length $E(h(x))$ by $c(n)$ in the exponent:
$$s(x, n) = 2^{-\frac{E(h(x))}{c(n)}}$$
the anomaly score is bounded between $0$ and $1$, independent of the dataset sample size.

---

### Q4: How do you interpret an anomaly score $s(x, n)$?
**Answer:**
- **$s \to 1.0$ (Path length $E(h(x)) \to 0$):** Strong anomaly. The claim requires very few splits to isolate, indicating high suspicion of fraud.
- **$s \approx 0.5$ (Path length $E(h(x)) \approx c(n)$):** Instance has typical average depth; no distinct outlier characteristics.
- **$s \to 0.0$ (Path length $E(h(x)) \to n - 1$):** Standard legitimate claim residing deep in dense feature space.

---

### Q5: What domain-specific features did you engineer, and why are they critical for fraud detection?
**Answer:**  
Raw policy data alone is insufficient. We engineered 8 insurance-specific behavioral indicators:
1. **Claim-to-Vehicle-Value Ratio:** A claim exceeding 80%–100% of the actual vehicle market value strongly indicates inflated total loss.
2. **Jump-in Policy Tenure Flag:** Claims occurring within the first 90 days ($\le 3$ months) of policy purchase often indicate pre-existing damage insured right before claim filing.
3. **Unwitnessed Night Collision Marker:** Collisions occurring between 23:00 and 04:00 with zero independent witnesses have a significantly higher fraud probability.
4. **Severe Damage Without Police Report:** Legitimate major crashes and total vehicle losses almost always involve police involvement. Absence of a police report for major damage is a strong red flag.
5. **Injury Share Disparity:** High bodily injury claims on minor or parked car damage flag fabricated medical billing.

---

### Q6: What was the contamination parameter set to, and how is it chosen?
**Answer:**  
The `contamination` parameter specifies the expected proportion of outliers in the dataset. We set `contamination = 0.12` (12%), aligning with empirical industry estimates by the Coalition Against Insurance Fraud (typically 10%–15% for motor insurance). In production, this threshold can also be dynamically adjusted based on the investigative capacity of the Special Investigation Unit (SIU).

---

### Q7: What are the key evaluation metrics achieved by your model?
**Answer:**  
On the held-out test dataset (1,000 unseen claims):
- **ROC-AUC Score:** **0.9827** (demonstrating excellent discriminative ability between legitimate and fraudulent distributions across all classification thresholds)
- **Precision-Recall AUC (PR-AUC):** **0.9295** (crucial for imbalanced domains)
- **Fraud Recall:** **87.50%**
- **Fraud Precision:** **85.37%**
- **Recall@Top-20% Queue:** **96.67%** (Reviewing just the top 20% prioritized claims captures nearly 97% of all fraud cases)

---

### Q8: What is an Investigator Triage Queue, and why is it important in this project?
**Answer:**  
In everyday insurance operations, machine learning models often fail to be adopted if they only provide a raw binary prediction (0 or 1).  
An **Investigator Triage Queue** bridges this gap by:
1. Ranking incoming claims by their **calibrated risk score (0% to 100%)**.
2. Categorizing claims into actionable operational tiers: **High Priority (SIU Audit)**, **Medium Priority (Verification)**, and **Low Priority (Fast-Track Automated Approval)**.
3. Providing **Explainability / Root-Cause Attribution** showing the claims adjuster exactly which features (e.g. claim-to-vehicle ratio, odd hours, no police report) triggered the anomaly flag.
4. Enabling one-click workflow actions (Approve, Escalate to SIU, Reject) to track claims in real time.

---

### Q9: What are the limitations of Isolation Forest, and how can they be mitigated?
**Answer:**
1. **Axis-Aligned Splits:** Standard Isolation Forest chooses random splits parallel to coordinate axes. In cases with complex diagonal correlations, extended Isolation Forest (EIF) with random hyperplanes can be used.
2. **Interpretability:** Unlike decision trees, an ensemble of random trees does not provide a single interpretable decision path. We mitigated this by building a z-score feature attribution engine that quantifies how many standard deviations each feature diverged from the legitimate baseline.
3. **Collusion Rings:** Isolation Forest evaluates claims individually. Multi-party organized fraud rings (e.g. staged accidents involving multiple collusion policies) are better addressed by combining Isolation Forest with Graph Neural Networks (GNNs).
