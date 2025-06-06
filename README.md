# Democracy Clustering Analysis

Using K-means and hierarchical clustering to analyze democracy data and reveal insights about regime types and democratic transitions beyond traditional classifications.

## [View Interactive Dashboard](https://rosalinatorres888.github.io/democracy-clustering-analysis/)

## Geographic Visualization of Democracy Index

This visualization shows the democracy index by country, revealing patterns of democratic development across the globe:

![Democracy Index by Country](docs/democracy_map.png)

## About This Project

This project uses machine learning techniques to cluster countries based on various democracy indicators, identifying patterns that go beyond traditional classifications of democratic, hybrid, and authoritarian regimes.

Key findings include:
- Identification of transition countries at cluster boundaries
- Patterns of regional democratic development
- Insights into the relationship between various democracy indicators

## 🧪 Methodology & Workflow

### ▶ Data Ingestion & Cleaning

- Cleaned and standardized Democracy Index
- Removed missing values and normalized features

### ▶ Dimensionality Reduction

- Applied PCA: retained 4 components (82% variance explained)

### ▶ Clustering

- Used **K-Means** and **Agglomerative Clustering**
- Chose optimal k using Elbow + Silhouette analysis

### ▶ Evaluation

- Compared clusters with regime types (full, flawed, hybrid, authoritarian)
- Created **Democratic Stability Index** to flag borderline cases

## Technologies Used
- Python
- scikit-learn
- Plotly
- Pandas
- NumPy

## How to Run
1. Clone this repository
2. Install the required packages: `pip install -r requirements.txt`
3. Run the Jupyter notebook: `jupyter notebook my_democracy-clustering-notebook-torres.ipynb`
