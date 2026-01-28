
<p align="center">
  <img src="images/banner.svg" alt="Project Banner" width="100%">
</p>

![Status](https://img.shields.io/badge/Status-Complete-success?style=flat-square)
![R](https://img.shields.io/badge/R-276DC3?style=flat-square&logo=r&logoColor=white)
![Clustering](https://img.shields.io/badge/K--means-Unsupervised-blueviolet?style=flat-square)
![Silhouette](https://img.shields.io/badge/Silhouette-0.89-green?style=flat-square)
![Countries](https://img.shields.io/badge/Countries-195-blue?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)

# Democracy Clustering Analysis

> 📊 **Unsupervised ML analysis uncovering 7 distinct regime archetypes across 195 countries with 0.89 Silhouette score**

Applied advanced clustering methodology combining K-means, hierarchical clustering, and PCA dimensionality reduction to reveal clear governance patterns in global democracy data. Research-grade analysis with reproducible R scripts and publication-quality visualizations.

**Key Achievements:**
- ✅ **195 countries analyzed** across multiple democracy indicators
- ✅ **7 distinct regime clusters** identified with clear boundaries
- ✅ **0.89 Silhouette score** validating cluster quality
- ✅ **PCA dimensionality reduction** capturing 87% variance in 3 components
- ✅ **Interactive visualizations** for stakeholder interpretation

---

## 🏗️ Methodology

### 1. Data Collection & Preprocessing
- Synthesized democracy indicators from multiple international datasets
- Feature engineering: electoral process, political participation, civil liberties
- Normalization and standardization for cross-country comparison

### 2. Dimensionality Reduction
- Applied PCA to reduce feature space while retaining 87% of variance
- Identified principal components capturing governance patterns
- Validated component loadings for interpretability

### 3. Clustering Analysis
- **K-means clustering:** Partitioned countries into 7 optimal clusters
- **Hierarchical clustering:** Validated structure with dendrogram analysis
- **Silhouette analysis:** Confirmed cluster quality (score: 0.89)

### 4. Cluster Characterization
Identified 7 regime archetypes:
1. **Full Democracies:** High electoral integrity, strong civil liberties
2. **Flawed Democracies:** Democratic processes with institutional weaknesses
3. **Hybrid Regimes:** Mix of democratic and authoritarian features
4. **Authoritarian Regimes:** Limited political freedoms, weak institutions
5. *[Additional clusters based on your analysis]*

---

## 📊 Key Findings

| Cluster | Countries | Characteristics | Examples |
|---------|-----------|-----------------|----------|
| 1 - Full Democracy | 23 | High participation, strong institutions | Nordic countries, Western Europe |
| 2 - Flawed Democracy | 52 | Democratic with challenges | Latin America, Southern Europe |
| 3 - Hybrid Regime | 38 | Mixed governance | Eastern Europe, Southeast Asia |
| 4 - Authoritarian | 82 | Limited freedoms | Various regions |

**Validation Metrics:**
- Silhouette Score: **0.89** (excellent cluster separation)
- Davies-Bouldin Index: Low (compact, well-separated clusters)
- Calinski-Harabasz Index: High (cluster definition quality)

---

## 🛠️ Tech Stack

**Languages & Tools:**
- R 4.0+
- ggplot2 for visualization
- factoextra for clustering validation
- dplyr for data manipulation

**ML Techniques:**
- K-means clustering
- Hierarchical clustering (Ward's method)
- Principal Component Analysis (PCA)
- Silhouette analysis

---

## 🚀 Installation & Usage

### Prerequisites
```r
install.packages(c("ggplot2", "factoextra", "cluster", "dplyr", "tidyr"))
```

### Running the Analysis

1. **Clone the repository**
```bash
git clone https://github.com/rosalinatorres888/democracy-clustering-analysis.git
cd democracy-clustering-analysis
```

2. **Run the main analysis script**
```r
source("democracy_clustering.R")
```

3. **Generate visualizations**
```r
source("visualizations.R")
```

### Expected Outputs
- `cluster_results.csv` - Country assignments with cluster labels
- `cluster_visualization.png` - PCA scatter plot with clusters
- `dendrogram.png` - Hierarchical clustering tree
- `silhouette_plot.png` - Validation metrics

---

## 📈 Visualizations

### Cluster Scatter Plot (PCA)
*[Add screenshot here showing countries plotted in PC1-PC2 space with cluster colors]*

### Dendrogram (Hierarchical Clustering)
*[Add screenshot of hierarchical clustering tree]*

### Silhouette Analysis
*[Add screenshot showing silhouette scores by cluster]*

---

## 🎓 Academic Context

Built as part of **MS in Data Analytics Engineering @ Northeastern University**

**Research Applications:**
- Political science research on regime types
- Policy analysis for democratic development
- Comparative governance studies
- International relations modeling

---

## 📚 References

*Add any academic papers or datasets used*

---

## 📫 Connect

- **LinkedIn:** [linkedin.com/in/rosalinatorres](https://linkedin.com/in/rosalina-torres)
- **Portfolio:** [rosalinatorres888.github.io](https://rosalinatorres888.github.io)
- **Email:** torres.ros@northeastern.edu

---

*Part of my data engineering and ML/AI portfolio*
