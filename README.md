# Democracy Clustering Analysis

> 📊 **Two-part unsupervised ML study revealing governance archetypes across 167 countries — extended with SHAP explainability and digital freedom dimensions**

A research-grade political data science project combining K-means, hierarchical clustering, PCA, t-SNE, and XGBoost surrogate models to uncover democracy patterns in global governance data. Part 1 applies unsupervised clustering to the EIU Democracy Index. Part 2 extends the analysis with Freedom House internet freedom data and SHAP explainability to explain why countries like Hungary and India defy simple classification.

**[🌐 View Live Interactive Dashboard →](https://rosalinatorres888.github.io/democracy-clustering-analysis/index.html)**

[![Status](https://img.shields.io/badge/Status-Complete-success?style=flat-square)](https://github.com/rosalinatorres888/democracy-clustering-analysis)
[![Python](https://img.shields.io/badge/Python-3.9+-blue?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Clustering](https://img.shields.io/badge/K--means-Unsupervised-blueviolet?style=flat-square)](https://github.com/rosalinatorres888/democracy-clustering-analysis)
[![SHAP](https://img.shields.io/badge/SHAP-Explainability-orange?style=flat-square)](https://github.com/rosalinatorres888/democracy-clustering-analysis)
[![Countries](https://img.shields.io/badge/Countries-167-blue?style=flat-square)](https://github.com/rosalinatorres888/democracy-clustering-analysis)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)

---

## Two-Part Structure

| | Part 1 | Part 2 |
|---|---|---|
| **File** | `my_democracy-clustering-notebook-torres.ipynb` | `shap_explainer_v2.ipynb` |
| **Data** | EIU Democracy Index | EIU 2024 + Freedom House FOTN 2025 |
| **Features** | 5 democracy dimensions | 6 dimensions (+ internet freedom) |
| **Method** | K-means + Hierarchical + PCA + t-SNE | AgglomerativeClustering + XGBoost surrogate |
| **Explainability** | Silhouette + Bootstrap + ARI | SHAP TreeExplainer |
| **Clusters** | 4 named regime archetypes | 4 named digital governance profiles |

---

## Part 1 — EIU Democracy Index Clustering

### Data Source
**Economist Intelligence Unit (EIU) Democracy Index** — 167 countries scored across 5 dimensions:
- Electoral process and pluralism
- Functioning of government
- Political participation
- Political culture
- Civil liberties

### Methodology

**1. Preprocessing**
- StandardScaler normalization across all 5 dimensions
- Cross-sectional dataset (single-year snapshot — no time series imputation required)
- Missing value handling via SimpleImputer

**2. Optimal Cluster Selection**
- KElbow visualizer (k=2 to k=10) for distortion score analysis
- Silhouette scoring across k range:

| k | Silhouette Score |
|---|---|
| 2 | 0.459 |
| 3 | 0.385 |
| 4 | 0.297 |
| 5 | 0.253 |
| 6 | 0.248 |
| 7 | 0.261 |

- Political science theory (regime type literature) informed final cluster selection
- Elbow method confirmed diminishing returns beyond k=4

**3. Clustering**
- K-means (k-means++ initialization, random_state=42, n_init=10)
- Hierarchical clustering (Ward's linkage) for structural validation
- Dendrogram analysis confirming cluster boundaries

**4. Dimensionality Reduction**
- PCA (2 components) for cluster visualization and biplot interpretation
- t-SNE (perplexity calibrated to dataset size) for non-linear pattern detection

**5. Validation**
- Bootstrap stability testing (30 iterations, Adjusted Rand Index)
- Confusion matrix comparison against EIU expert regime classifications
- Transition country identification via distance-to-centroid scoring


### Part 1 Results — 4 Named Regime Archetypes

| Cluster | Name | Countries | Profile |
|---|---|---|---|
| 0 | **Liberal Democracy** | 39 | High electoral integrity, strong civil liberties, high political participation |
| 1 | **Participatory Democracy** | 41 | Democratic with strong participation, moderate institutional performance |
| 2 | **Hybrid Regime** | 66 | Mixed democratic and authoritarian features, weak institutions |
| 3 | **Electoral Autocracy** | 21 | Limited political freedoms, controlled elections, weak civil society |

**Key transition countries identified** (high distance-to-centroid scores): Ecuador and others at cluster boundaries representing transitional regimes.

---

## Part 2 — SHAP Explainability + Digital Freedom Extension

### Why Part 2?
Part 1 revealed that countries like Hungary, India, and Turkey resist clean classification by traditional democracy metrics alone. Part 2 adds **internet freedom** as a sixth dimension to capture digital authoritarianism — a governance pattern not captured by EIU's original five dimensions.

### Data Sources
- **EIU Democracy Index 2024** — 5 core dimensions
- **Freedom House Freedom on the Net (FOTN) 2025** — internet freedom scores (0–100)
- Merged dataset: `data/democracy_v2_dataset.csv` — 167 countries, 6 features + `surveillance_gap`

### 4 Digital Governance Profiles

| Cluster | Name | Defining Characteristics |
|---|---|---|
| 0 | **Digital Democracies** | High EIU score + high internet freedom — Nordic/Western Europe |
| 1 | **Constrained Democracies** | Democratic institutions but restricted digital space |
| 2 | **Digital Hybrids** | Mixed governance + selective internet control (Hungary, India, Turkey) |
| 3 | **Hard Authoritarians** | Low EIU score + restricted internet — China, Russia |

### SHAP Explainability

**Why XGBoost surrogate?** Clustering is unsupervised — it produces labels but no feature importance scores. The surrogate approach trains XGBoost to predict cluster membership, then uses SHAP TreeExplainer to explain which features drove each country's assignment.

**Surrogate architecture:**
```python
surrogate = xgb.XGBClassifier(
    n_estimators=300, max_depth=4,
    learning_rate=0.05, subsample=0.8,
    colsample_bytree=0.8, random_state=42
)
```

**SHAP outputs generated:**
- Global feature importance bar chart (all clusters combined)
- Per-cluster beeswarm plots (4 separate figures)
- Country-level waterfall plots — Hungary and India
- Decision plot — all Digital Hybrids countries
- Mean absolute SHAP pivot table by feature and cluster

**Key finding:** For the Digital Hybrids cluster, `internet_freedom` and `political_culture` are the dominant features — not electoral process. This explains why Hungary scores relatively well on EIU metrics but lands in the hybrid cluster when digital freedom is included.

---

## Tech Stack

| Component | Technology |
|---|---|
| **Language** | Python 3.9+ |
| **Data manipulation** | pandas, numpy |
| **Clustering** | scikit-learn (KMeans, AgglomerativeClustering) |
| **Dimensionality reduction** | PCA, t-SNE (sklearn) |
| **Visualization** | matplotlib, seaborn, Plotly Express, plotly.graph_objects |
| **Clustering validation** | factoextra (via yellowbrick KElbow), silhouette_score, ARI |
| **Explainability** | SHAP TreeExplainer, XGBoost surrogate |
| **Statistical methods** | Bootstrap resampling (30 iterations), Ward's linkage |


---

## Repository Structure

```
democracy-clustering-analysis/
├── my_democracy-clustering-notebook-torres.ipynb  # Part 1 — EIU clustering analysis
├── shap_explainer_v2.ipynb                        # Part 2 — SHAP + digital freedom extension
├── democracy-clustering-notebook-torres.html      # Rendered HTML notebook (Part 1)
├── index.html                                     # GitHub Pages entry point
├── dashboard/                                     # Dashboard components
├── docs/                                          # Documentation
├── react-democracy-viz/src/                       # React visualization components
├── README.md
└── .gitignore
```

**Data files** (not committed — sourced externally):
- `democracy_index.csv` — EIU Democracy Index (download from EIU)
- `data/democracy_v2_dataset.csv` — EIU 2024 + Freedom House FOTN 2025 merged dataset

---

## Installation

```bash
git clone https://github.com/rosalinatorres888/democracy-clustering-analysis.git
cd democracy-clustering-analysis

pip install pandas numpy scikit-learn matplotlib seaborn plotly yellowbrick scipy xgboost shap
```

**Run Part 1:**
Open `my_democracy-clustering-notebook-torres.ipynb` in Jupyter. Update the data path in Cell 5 to point to your local EIU Democracy Index CSV.

**Run Part 2:**
Open `shap_explainer_v2.ipynb`. Requires `data/democracy_v2_dataset.csv` (EIU 2024 + Freedom House FOTN 2025 merged).

---

## Academic Context

**Course:** IE6400 — Data Analytics Engineering
**Institution:** Northeastern University — MS Data Analytics Engineering (EDGE Program)
**Term:** Spring 2025

**Research applications demonstrated:**
- Unsupervised ML for political science classification
- Surrogate model explainability for clustering outputs
- Multi-source data integration (EIU + Freedom House)
- Digital authoritarianism detection via internet freedom dimensions
- Transition regime identification at cluster boundaries

---

## Author

**Rosalina Torres** — ML/AI Engineer
MS Data Analytics Engineering @ Northeastern University (EDGE Program)
Expected Graduation: August 2026 · 4.0 GPA

- **Portfolio:** [rosalina.sites.northeastern.edu](https://rosalina.sites.northeastern.edu)
- **LinkedIn:** [linkedin.com/in/rosalina-torres](https://linkedin.com/in/rosalina-torres)
- **GitHub:** [@rosalinatorres888](https://github.com/rosalinatorres888)
- **Email:** torres.ros@northeastern.edu

---

## License

MIT License — See LICENSE file for details

---

*Part of an ML/AI engineering portfolio demonstrating unsupervised learning, dimensionality reduction, model explainability, and political data science.*

---

## Interactive Dashboard

**`democracy_clustering_dashboard.html`** — a standalone 5-section interactive dashboard built with Plotly.js:

| Section | Content |
|---|---|
| **01 Country Explorer** | Radar chart for any country across all 5 EIU dimensions + borderline flags |
| **02 US Decline** | Bar chart of dimensional scores + historical trend (2015–2024) |
| **03 Democracy Clustering** | Scatter plot with toggleable cluster boundaries + 3-method comparison |
| **04 Trends** | Line chart 2017–2024 — all countries, US vs Norway, borderline cases |
| **05 Key Insights** | Research summary with correlation findings |

**Clustering method comparison (Adjusted Rand Index vs expert labels):**

| Method | ARI Score | Notes |
|---|---|---|
| Hierarchical Clustering | **0.78** | Best alignment with EIU expert classifications |
| K-Means | 0.57 | Good overall, less precise at regime boundaries |
| Gaussian Mixture Model | 0.52 | Probabilistic — useful for borderline cases |

**Feature correlations with regime type:**
- Civil liberties: **0.90**
- Electoral process: **0.86**

Open `democracy_clustering_dashboard.html` directly in any browser — no server required.

**Live:** [rosalinatorres888.github.io/democracy-clustering-analysis/index.html](https://rosalinatorres888.github.io/democracy-clustering-analysis/index.html)

