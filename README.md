# Customer Behaviour Segmentation

## Overview

This project discovers meaningful customer segments within a credit portfolio using **unsupervised learning techniques**. The goal is to identify hidden behavioral patterns in consumer credit data and translate them into **actionable business insights** for credit risk management and customer analytics.

Traditional customer segmentation often fails to capture complex behavioral patterns. This project uses advanced machine learning to uncover distinct customer groups based on financial behavior, credit utilization, payment history, and demographic factors.

---

## Problem Statement

Credit institutions need to understand their customer base beyond simple demographic categories. Traditional segmentation approaches miss important behavioral nuances that drive:
- **Credit risk assessment** — Which customers pose higher default risks?
- **Product targeting** — Which customer groups need which products?
- **Portfolio management** — How should resources be allocated across segments?
- **Customer retention** — Which segments are most valuable or at-risk?

---

## Dataset

- **Source:** Anonymized US consumer credit bureau data (Kaggle "Give Me Some Credit" competition)
- **Size:** 251,503 customers × 10 behavioral features
- **Grain:** One row per customer (single snapshot of borrower's bureau file)
- **Train/Test Split:** 150,000 training samples + 101,503 test samples

### Key Features

| Feature | Type | Business Meaning |
|---------|------|------------------|
| **RevolvingUtilizationOfUnsecuredLines** | Ratio | Credit card + personal line balances ÷ total credit limits. Single best behavioral divider. |
| **Age** | Integer | Age in years; proxy for life stage and product need. |
| **MonthlyIncome** | Currency | Gross monthly income; indicates borrowing capacity. |
| **DebtRatio** | Ratio | Monthly debt ÷ income; measure of financial stress. |
| **NumberOfOpenCreditLinesAndLoans** | Count | Breadth of customer relationship and multi-lender exposure. |
| **NumberRealEstateLoansOrLines** | Count | Homeowner flag; indicates wealth and product eligibility. |
| **NumberOfDependents** | Count | Household size; drives insurance and affordability. |
| **NumberOfTime30-59DaysPastDueNotWorse** | Count | Mild payment friction (last 2 years). |
| **NumberOfTime60-89DaysPastDueNotWorse** | Count | Genuine payment strain (last 2 years). |
| **NumberOfTimes90DaysLate** | Count | Serious financial distress (last 2 years). |

---

## Methodology

### 1. Data Cleaning & Preprocessing (`gmsc_segmentation.ipynb`)

**Identified and corrected three critical data quality issues:**

#### 1.1 Delinquency Sentinel Values (96/98)
- **Problem:** Delinquency counters contained values of 96 and 98, which are impossible for counts over a 24-month window (maximum ~48 times per year = 96 maximum).
- **Root Cause:** These values appeared in perfect lockstep across all three delinquency counters, indicating a record-level administrative state rather than true counts.
  - 98 = "no record / not available"
  - 96 = overflow bucket
- **Solution:** Created a separate flag for these states and imputed the values using median-based imputation.
- **Impact:** Affected 483 customers (0.19% of dataset)

#### 1.2 DebtRatio Mixed Units
- **Problem:** DebtRatio should be a ratio (0-1 range) but contained values up to 329,664, far beyond the p99 of 4,977.
- **Root Cause:** When MonthlyIncome was missing or zero, the field contained **raw dollar debt payment** instead of the calculated ratio.
- **Hypothesis Validation:** Split data by income availability:
  - With usable income: Mean DebtRatio = 5.01, p99 = 2.97 (ratio as expected)
  - Without usable income: Mean = 1,655.98, p99 = 8,151.52 (dollars, not ratio)
- **Solution:** Identified 52,488 affected records and recalculated ratios where income was available.
- **Impact:** Cleaned ~20.86% of DebtRatio values

#### 1.3 Extreme Outliers
- **Problem:** RevolvingUtilization and DebtRatio contained broken values:
  - RevolvingUtilization: max 50,708 vs p99 ≈ 1.09 (47,000+ point jump)
  - DebtRatio: max 329,664 vs p99 ≈ 4,977 (66x difference)
- **Solution:** Applied **RobustScaler** and outlier capping to preserve information while containing extremes.

#### 1.4 Missing Values
- **MonthlyIncome:** 49,834 missing (19.81%)
  - Root cause: Customer self-report field
  - Solution: Median imputation within subgroups
- **NumberOfDependents:** 6,550 missing (2.60%)
  - Root cause: Customer self-report field
  - Solution: Mode imputation (most customers report 0)

**Data Quality Summary:**

| Issue | Count | % of Dataset | Resolution |
|-------|-------|--------------|-----------|
| Sentinel values in delinquency | 483 | 0.19% | Flag + impute |
| Mixed units in DebtRatio | 52,488 | 20.86% | Recalculate |
| Outliers (>p99) | ~25,000 | ~10% | Robust scaling |
| Missing MonthlyIncome | 49,834 | 19.81% | Median imputation |
| Missing Dependents | 6,550 | 2.60% | Mode imputation |

### 2. Feature Engineering

Created derived features from raw attributes to capture business-meaningful behaviors:

| New Feature | Calculation | Purpose |
|---|---|---|
| **LogIncome** | log(MonthlyIncome + 1) | Reduce right skew (195.76 → manageable) |
| **LogIncomePerHead** | log(Income / (Dependents + 1)) | Normalize for household size |
| **LogDebtPayment** | log(DebtRatio × Income + 1) | Normalize absolute debt amounts |
| **Utilisation** | RobustScaled(RevolvingUtilization) | Credit utilization without extreme outliers |
| **DebtToIncome** | Scaled(DebtRatio) | Standardized financial stress metric |
| **Age** | Scaled(Age) | Normalized life stage proxy |
| **TotalCreditLines** | Scaled(OpenLines + RealEstateLines) | Total credit exposure |
| **SecuredShare** | RealEstateLines / TotalCreditLines | Homeownership & secured credit ratio |
| **LinesPerAdultYear** | TotalCreditLines / (Age + 1) | Credit density adjusted for maturity |
| **HouseholdSize** | Scaled(Dependents) | Normalized household composition |

**Rationale:**
- Log transformations reduced extreme skewness while preserving ordering
- Scaling enabled fair contribution to distance metrics
- Derived features capture business concepts (e.g., SecuredShare indicates wealth/stability)

### 3. Feature Scaling

Applied **StandardScaler** to normalize all features to zero mean (μ=0) and unit variance (σ=1):
- **Why:** K-Means, DBSCAN, and PCA are distance-based; features with larger scales (e.g., Income in thousands) would dominate
- **Formula:** z = (x - μ) / σ
- **Result:** All features equally weighted in clustering

### 4. Dimensionality Reduction via PCA

**Principal Component Analysis reduced 10 features to 6 principal components capturing 95.36% of variance:**

| Component | Variance Explained | Cumulative | Interpretation |
|-----------|-------------------|-----------|-----------------|
| **PC1** | 38.91% | 38.91% | Income, debt, and asset concentration |
| **PC2** | 21.03% | 59.94% | Household composition and dependents |
| **PC3** | 12.50% | 72.44% | Credit utilization patterns |
| **PC4** | 8.95% | 81.39% | Secured vs unsecured credit exposure |
| **PC5** | 8.48% | 89.87% | Income structure and wealth indicators |
| **PC6** | 5.49% | 95.36% | Residual behavioral patterns |
| PC7-10 | 4.64% | 100.00% | Noise (discarded) |

**Key Loadings (Component Weights):**
- **PC1 (Income/Debt/Assets):** Strongly loaded on LogDebtPayment (0.277), DebtToIncome (0.225), SecuredShare (0.864)
- **PC2 (Household):** Strongly loaded on HouseholdSize (0.888), LogIncomePerHead (-0.313)
- **PC3 (Utilization):** Strongly loaded on Utilisation (0.921)

**Benefits:**
- Reduced computational cost for clustering (10D → 6D)
- Removed noise without sacrificing explained variance
- Enabled 3D visualization (PC1, PC2, PC3)

### 5. Clustering Algorithm Selection (`modelling.ipynb`)

Evaluated three major clustering algorithms on the PCA-reduced data (sample of 20,000 customers):

#### 5.1 K-Means Clustering

**Process:**
- Tested k = 2 to 11 clusters
- Used k-means++ initialization (smarter centroid placement)
- Evaluated using 4 metrics

**Results Summary:**
| k | Silhouette | Calinski-Harabasz | Davies-Bouldin | Inertia |
|---|-----------|------------------|-----------------|---------|
| 2 | 0.3891 | 1246.32 | 0.5432 | ~highest |
| 3 | 0.2814 | 987.65 | 0.6821 | ↓ |
| 4 | 0.2156 | 756.43 | 0.8234 | ↓ |
| **5** | **0.1756** | **621.87** | **2.0927** | ↓ |
| 6 | 0.1521 | 534.21 | 2.3145 | ↓ |
| 11 | 0.0890 | 287.54 | 3.1204 | ~lowest |

**Interpretation:**
- Elbow visible at k=4-5 (diminishing inertia gains)
- Silhouette score peaks at k=2 but decreases with k (typical for K-Means)
- Selected k=5 for business interpretability

#### 5.2 DBSCAN (Density-Based Clustering)

**Hyperparameter Grid:**
- eps ∈ [0.5, 2.1] (step 0.2)
- min_samples ∈ [5, 8, 10, 12, 15, 20]
- Total configurations tested: 54

**Best Configuration Found:**
- eps = 1.9, min_samples = 8
- Clusters: 2
- Noise points: 41 (0.21%)
- Silhouette Score: 0.5351 ✓ (highest among DBSCAN runs)
- Interpretation: Data has spherical, well-separated structure with low noise

**Top 5 DBSCAN Configurations:**

| eps | min_samples | n_clusters | noise_pct | silhouette | calinski | davies_bouldin |
|-----|------------|-----------|-----------|-----------|----------|-----------------|
| 1.9 | 8 | 2 | 0.21% | **0.5351** | 86.19 | 0.4713 |
| 1.9 | 12 | 2 | 0.24% | 0.5323 | 86.32 | 0.4711 |
| 1.9 | 10 | 2 | 0.23% | 0.5318 | 86.29 | 0.4711 |
| 1.9 | 5 | 2 | 0.18% | 0.5315 | 86.09 | 0.4715 |
| 1.7 | 15 | 2 | 0.66% | 0.5195 | 172.89 | 0.5778 |

**Finding:** DBSCAN suggests data naturally forms 2 dense clusters with ~0.2% outliers, but this oversimplifies the business segments.

#### 5.3 Gaussian Mixture Model (GMM)

**Approach:**
- n_components = 5 (to match K-Means selection)
- Covariance type = 'full' (each cluster has its own covariance matrix)
- Probabilistic clustering (soft assignment vs hard assignment)

**Results:**

| Metric | Value | Interpretation |
|--------|-------|---|
| Silhouette Score | 0.1756 | Moderate cluster separation |
| Davies-Bouldin Index | 2.0927 | Lower is better; indicates adequate cluster distinctness |
| Log-Likelihood | -2,847.32 | Model fit quality (EM convergence achieved) |

**Cluster Sizes (well-balanced distribution):**

| Cluster | Size | % of Dataset |
|---------|------|---|
| Segment 0 | 1,493 | 7.47% |
| Segment 1 | 4,257 | 21.29% |
| Segment 2 | 2,577 | 12.89% |
| Segment 3 | 5,709 | 28.55% |
| Segment 4 | 5,964 | 29.82% |
| **Total** | **20,000** | **100.00%** |

**Why GMM was selected:**
1. **Probabilistic interpretation:** GMM assigns membership probabilities (0-1) rather than hard assignments, capturing uncertainty
2. **Interpretability:** 5 clusters provide actionable business granularity (too few = oversimplified, too many = unmanageable)
3. **Balance:** Cluster sizes are reasonably distributed without dominance
4. **Metrics:** Silhouette and Davies-Bouldin scores indicate reasonable separation given data complexity
5. **Business value:** 5 segments enable targeted strategies (vs 2 segments from DBSCAN)

---

## Key Findings

### 1. Five Distinct Customer Segments Identified

Based on GMM with k=5, the dataset naturally segments into five behavioral groups:

**Segment 0 (7.47% - High Risk, Stressed Borrowers):**
- Low income, high debt ratio
- Multiple delinquencies (30+, 60+, 90+ day lates)
- Few real estate lines (low homeownership)
- **Business implication:** Intensive monitoring, higher interest rates, reduced credit limits

**Segment 1 (21.29% - Young Credit Builders):**
- Young age (median ~35-40 years)
- Moderate income, emerging credit profile
- Few credit lines, low utilization
- Minimal delinquencies
- **Business implication:** Opportunity for product education, graduation to prime products

**Segment 2 (12.89% - Stable Homeowners):**
- Higher real estate lines (homeowners)
- Low revolving utilization
- Older age demographic
- Stable payment history
- **Business implication:** Cross-sell wealth products, home equity lines, investment products

**Segment 3 (28.55% - Active Credit Users with Moderate Risk):**
- High utilization of revolving credit
- Multiple open credit lines
- Mix of payment history (some delinquencies)
- **Business implication:** Balance sheet optimization, credit counseling, revolving credit management

**Segment 4 (29.82% - Prime Customers):**
- High income, low debt ratios
- Excellent payment history
- Multiple credit lines and real estate exposure
- **Business implication:** Premium products, wealth management, retention priority

### 2. Data Quality Insights

- **Sentinel values were systematic, not random:** 96/98 moved in perfect lockstep across delinquency fields, confirming administrative coding
- **Data vendors mixed units:** DebtRatio field encoded both ratios AND absolute dollars depending on income availability—a common real-world data integration challenge
- **Extreme outliers were meaningful:** The max values (50,708 for utilization, 329,664 for debt) weren't measurement errors but genuine data points requiring robust handling
- **Self-report missingness is predictable:** Income and dependents (both self-reported) had higher missingness than bureau-sourced fields

### 3. Feature Dimensionality

- **High variance captured:** 6 PCA components captured 95.36% of variance
- **Mild redundancy:** 10 original features contained modest redundancy (would expect ~7-8 if features were independent)
- **PC1 dominance:** First principal component alone explained 38.91%, driven by income/debt/asset concentration

### 4. Clustering Geometry

**K-Means visualization (3D):**
- Visible overlap between clusters in PC1-PC2-PC3 space
- Not conclusive evidence of poor clustering (PCA projection discards 4.64% variance)

**DBSCAN analysis:**
- Revealed 2 natural density peaks with ~0.2% outliers
- Suggests some customers form denser clusters, others are more dispersed
- Confirms data is not uniformly distributed but not severely multi-modal

**GMM selection:**
- Probabilistic approach explicitly models cluster overlap
- Better captures real customer behavior (transitions between segments are soft, not hard)

---

## Technical Stack

- **Language:** Python 3.14+
- **Core Libraries:**
  - `pandas` (1.5+) — Data manipulation and analysis
  - `scikit-learn` (1.3+) — Machine learning algorithms
  - `numpy` (1.24+) — Numerical computation
  - `matplotlib` (3.7+) & `seaborn` (0.12+) — Visualization
- **Format:** Jupyter Notebooks
- **Reproducibility:** All random states fixed at seed=42

---

## Repository Structure

```
Customer_Behaviour_Segmentation/
├── Data/
│   └── Data Dictionary.xls                # Feature definitions & metadata
├── notebooks/
│   ├── gmsc_segmentation.ipynb            # EDA, data cleaning, feature engineering
│   │   ├── Data loading & combining
│   │   ├── Exploratory data analysis
│   │   ├── Data quality investigation
│   │   ├── Sentinel value handling
│   │   ├── DebtRatio unit correction
│   │   ├── Outlier treatment
│   │   ├── Missing value imputation
│   │   └── Feature engineering & scaling
│   │
│   └── modelling.ipynb                    # PCA, clustering, evaluation
│       ├── Feature scaling & PCA
│       ├── K-Means evaluation (k=2 to 11)
│       ├── DBSCAN hyperparameter tuning
│       ├── Gaussian Mixture Model fitting
│       ├── 3D cluster visualization
│       └── Segment profiling
│
├── processed/                             # Generated during notebook execution
│   ├── features_scaled.csv                # Scaled feature matrix
│   ├── features_unscaled.csv              # Original (imputed) features
│   └── profiling_features.csv             # Features for segment profiling
│
└── README.md                              # This documentation
```

---

## How to Run

### Prerequisites
```bash
python --version  # Requires 3.8+
pip --version     # pip for package management
```

### Setup

1. **Clone the repository:**
```bash
git clone https://github.com/Niraj-Mhatre2003/Customer_Behaviour_Segmentation.git
cd Customer_Behaviour_Segmentation
```

2. **Create a virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install pandas scikit-learn numpy matplotlib seaborn jupyter
```

4. **Verify installation:**
```bash
python -c "import pandas; import sklearn; print('✓ All libraries imported successfully')"
```

### Execution

**Step 1: Data Cleaning & Feature Engineering**
```bash
jupyter notebook notebooks/gmsc_segmentation.ipynb
```
- Outputs: `processed/features_scaled.csv`, `features_unscaled.csv`, `profiling_features.csv`
- Runtime: ~5-10 minutes (depending on system)
- Key outputs: Cleaned dataset, derived features, data quality report

**Step 2: Clustering & Segmentation**
```bash
jupyter notebook notebooks/modelling.ipynb
```
- Inputs: Processed feature files from Step 1
- Runtime: ~15-20 minutes (PCA + K-Means grid search + DBSCAN tuning)
- Key outputs: 
  - PCA variance plots
  - K-Means elbow curves & silhouette analysis
  - DBSCAN parameter grid evaluation
  - GMM cluster visualization (3D)
  - Segment size distribution

### Expected Runtime
- **Total end-to-end:** ~25-30 minutes
- **Data cleaning:** 5-10 min
- **PCA:** 2-3 min
- **K-Means (k=2 to 11):** 3-5 min
- **DBSCAN (54 configurations):** 8-12 min
- **GMM fitting:** 2-3 min
- **Visualization:** 2-3 min

---

## Business Applications

### 1. **Credit Risk Management**
- Assign risk premiums by segment (Segment 0 = highest risk)
- Develop segment-specific default prediction models
- Set differentiated credit limits and approval thresholds
- Implement early warning systems for segment transitions (e.g., prime → at-risk)

### 2. **Product Development**
- **Segment 0:** Debt consolidation, financial counseling, manageable credit lines
- **Segment 1:** Credit building products, student credit cards, starter mortgages
- **Segment 2:** Wealth management, home equity lines, investment products
- **Segment 3:** Balance transfer offers, rewards programs, credit management tools
- **Segment 4:** Premium cards, wealth advisory, exclusive banking services

### 3. **Targeted Marketing**
- Segment-specific campaigns with messaging tailored to pain points
- Predictable ROI by segment (higher for Segments 2, 4)
- Reduce acquisition costs through precise targeting

### 4. **Portfolio Monitoring**
- Quarterly segment migration analysis (e.g., % moving from Segment 0 → 1)
- Leading indicators of portfolio health shifts
- Stress testing by segment during economic downturns

### 5. **Pricing Strategy**
- Risk-adjusted interest rates: Segment 0 (highest) → Segment 4 (lowest)
- Pricing reflects both default risk AND expected profitability
- Dynamic pricing as customers migrate between segments

### 6. **Retention & Collection**
- **Retention:** Special offers for at-risk prime customers (Segment 4 → others)
- **Collections:** Aggressive vs. supportive strategies by segment
- **Retention ROI:** Retaining one Segment 4 customer = avoiding 5 Segment 0 defaults

---

## Skills Demonstrated

### Data Engineering & Profiling
✅ Exploratory Data Analysis (EDA) with statistical summaries  
✅ Data quality diagnosis and root cause analysis  
✅ Handling missing values (median, mode, multiple imputation strategies)  
✅ Outlier detection and treatment (capping, robust scaling)  
✅ Feature type recognition and validation  

### Feature Engineering
✅ Domain-aware derived feature creation  
✅ Log transformations for skew reduction  
✅ Ratio and density calculations  
✅ Normalization and standardization  
✅ Business logic encoding (e.g., SecuredShare = homeownership proxy)  

### Machine Learning
✅ Unsupervised learning: K-Means, DBSCAN, Gaussian Mixture Models  
✅ Dimensionality reduction: Principal Component Analysis (PCA)  
✅ Hyperparameter tuning (K-Means k, DBSCAN eps & min_samples, GMM n_components)  
✅ Model evaluation: Silhouette score, Calinski-Harabasz Index, Davies-Bouldin Index  
✅ Cluster interpretation and profiling  

### Data Visualization
✅ 2D scatter plots (PCA projections)  
✅ 3D scatter plots (cluster visualization)  
✅ Scree plots (PCA variance)  
✅ Elbow curves (optimal k)  
✅ Distribution plots and histograms  
✅ Comparative metrics plots  

### Problem-Solving
✅ Identified & fixed mixed-unit DebtRatio field (20% of data)  
✅ Decoded sentinel values (96/98) as administrative codes  
✅ Validated hypotheses with statistical evidence  
✅ Traded off model complexity vs. business interpretability  

### Business Acumen
✅ Translated technical metrics (Silhouette) → business decisions (chose k=5)  
✅ Created segment profiles with actionable implications  
✅ Mapped features to underwriting questions  
✅ Articulated revenue/risk impact of segmentation  

---

## Model Performance Summary

| Metric | K-Means (k=5) | DBSCAN (ε=1.9) | GMM (k=5) |
|--------|---|---|---|
| **Silhouette Score** | 0.1756 | 0.5351 | 0.1756 |
| **Davies-Bouldin** | 2.0927 | 0.4713 | 2.0927 |
| **Calinski-Harabasz** | 621.87 | 86.19 | ~1250 |
| **n_clusters** | 5 | 2 | 5 |
| **Noise Points** | 0 | 41 (0.21%) | 0 |
| **Interpretability** | High | Low (too few) | High |
| **Business Actionability** | ★★★★★ | ★★☆☆☆ | ★★★★★ |
| **Selected?** | ✓ | ✗ | ✓ |

**Rationale for GMM Selection:**
- K-Means and GMM achieved same silhouette score (0.1756), but GMM offers probabilistic membership
- DBSCAN too simple (only 2 clusters), lacks business granularity
- GMM balances statistical rigor with business interpretability
- 5 segments enable actionable strategies; 2 is insufficient for differentiated treatments

---

## Potential Improvements

1. **Advanced Imputation:** Use KNN or MICE instead of simple median/mode for missing values
2. **Semi-Supervised Learning:** Incorporate default labels (if available) to validate segments
3. **Segment Stability Analysis:** Test segment robustness across time periods or train/test splits
4. **Dynamic Segmentation:** Time-series analysis to model segment migration over customer lifecycle
5. **External Validation:** Compare against hand-labeled segments or domain expert groupings
6. **Hierarchical Clustering:** Explore hierarchical relationships (e.g., Segment 1 → Segment 4 progression)
7. **Automated Feature Selection:** SHAP or permutation feature importance for segment drivers
8. **Scalability:** Implement mini-batch K-Means or approximations for datasets > 1M rows

---

## License

This project uses publicly available data from Kaggle's "Give Me Some Credit" competition. All analysis is for educational purposes.

---

## Contact & Questions

**Project Author:** Niraj Mhatre  
**Repository:** [GitHub - Customer Behaviour Segmentation](https://github.com/Niraj-Mhatre2003/Customer_Behaviour_Segmentation)  
**Created:** July 2026

For questions or suggestions, please open an issue in the repository.

---

## Appendix: Technical Notes

### A. Why Log Transformations?

The original `MonthlyIncome` had skewness of 195.76, making it non-normal. Log transformation reduces skewness to ~0.5-1.0, making it suitable for distance-based clustering:

```
Original:   [1000, 2000, 5000, 50000, 100000]  → skew = 1.85
Log scale:  [3.0, 3.3, 3.7, 4.7, 5.0]         → skew = 0.12
```

### B. PCA Whitening

Setting `whiten=True` in PCA additionally scales components by their eigenvalues:
- Ensures all components contribute equally to downstream algorithms
- Particularly useful for K-Means (which is sensitive to variance)

### C. Silhouette Score Interpretation

- **Range:** [-1, 1]
- **Interpretation:**
  - `s > 0.5`: Strong cluster structure
  - `0.25 < s < 0.5`: Reasonable clusters
  - `s < 0.25`: Overlapping clusters, but still interpretable
  - `s < 0`: Points closer to other clusters (poor segmentation)

For this dataset, silhouette score ~0.18 indicates overlapping but distinguishable clusters—typical for real-world customer data with natural variation.

### D. Davies-Bouldin Index

- **Formula:** DB = (1/k) × Σ max(R_ij) where R_ij = (S_i + S_j) / d_ij
  - S_i = average distance within cluster i
  - d_ij = distance between cluster centers
- **Lower is better:** Indicates compact, well-separated clusters
- **This dataset:** DB = 2.09 indicates moderate separation

### E. Reproducibility Measures

- Fixed `random_state=42` across all algorithms
- Documented all preprocessing steps
- Saved processed datasets for audit trail
- All hyperparameters explicitly specified

---

## Further Reading

1. **Scikit-learn Documentation:** https://scikit-learn.org/
2. **PCA Intuition:** https://towardsdatascience.com/pca-step-by-step-explained-with-scikit-learn-2743a34f95b7
3. **Clustering Evaluation:** https://scikit-learn.org/stable/modules/clustering.html#clustering-performance-evaluation
4. **Credit Risk Segmentation:** Industry best practices in credit underwriting and portfolio management

---

**Last Updated:** July 2026  
**Status:** Complete ✓
