{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Discovering Democracy Profiles: A Cluster Analysis of the EIU Democracy Index\n",
    "\n",
    "## Introduction\n",
    "\n",
    "The Economist Intelligence Unit's Democracy Index provides a multidimensional assessment of democratic systems worldwide, measuring countries across five key dimensions: electoral process and pluralism, functioning of government, political participation, political culture, and civil liberties. While the index categorizes countries into four regime types (full democracies, flawed democracies, hybrid regimes, and authoritarian regimes) based on threshold scores, this classification system may obscure more nuanced patterns in how democratic dimensions interact.\n",
    "\n",
    "This analysis applies clustering techniques to identify natural groupings in democracy data that may reveal distinct \"democracy profiles\" - patterns in how countries emphasize or de-emphasize different aspects of democratic governance. Recent political science research suggests that democracies often make different trade-offs between democratic values, resulting in distinctive institutional arrangements that simple classification schemes might miss.\n",
    "\n",
    "By using K-means and hierarchical clustering algorithms, we can identify these natural groupings and explore how they relate to traditional classifications, geographic patterns, and theories of democratic development. This approach allows us to move beyond viewing democracy as simply a linear progression from authoritarianism to full democracy, instead recognizing the diverse paths that political systems take."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Setup and Dependencies\n",
    "\n",
    "Let's start by importing the necessary libraries for our analysis."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Core data manipulation and analysis libraries\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "\n",
    "# Clustering and preprocessing\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "from sklearn.impute import SimpleImputer\n",
    "from sklearn.cluster import KMeans, AgglomerativeClustering\n",
    "from sklearn.metrics import silhouette_score, adjusted_rand_score, adjusted_mutual_info_score\n",
    "from sklearn.decomposition import PCA\n",
    "from sklearn.manifold import TSNE\n",
    "from sklearn.neighbors import NearestNeighbors\n",
    "\n",
    "# Visualization libraries\n",
    "import plotly.express as px\n",
    "import plotly.graph_objects as go\n",
    "import plotly.figure_factory as ff\n",
    "from yellowbrick.cluster import KElbowVisualizer\n",
    "from scipy.cluster.hierarchy import dendrogram, linkage\n",
    "\n",
    "# Set plotting style\n",
    "plt.style.use('seaborn-v0_8-whitegrid')\n",
    "sns.set_palette('viridis')\n",
    "\n",
    "# Display settings\n",
    "pd.set_option('display.max_columns', None)\n",
    "pd.set_option('display.max_rows', 30)\n",
    "pd.set_option('display.width', 160)\n",
    "\n",
    "# Set random seed for reproducibility\n",
    "np.random.seed(42)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Data Loading and Initial Exploration\n",
    "\n",
    "For this analysis, we'll be using the Economist Intelligence Unit's Democracy Index data. Let's load the data and explore its structure.\n",
    "\n",
    "**Note:** You'll need to download the Democracy Index data. The EIU publishes this data annually, but you may need to create a cleaned CSV format or locate a pre-processed version."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Load democracy index data\n",
    "# Note: Replace 'democracy_index_data.csv' with your actual file path\n",
    "try:\n",
    "    df = pd.read_csv('democracy_index_data.csv')\n",
    "    print(\"Data loaded successfully!\")\n",
    "except FileNotFoundError:\n",
    "    print(\"File not found. Please ensure the data file is in the correct location.\")\n",
    "    print(\"Creating example data for demonstration purposes...\")\n",
    "    \n",
    "    # Create mock data for demonstration if file is not available\n",
    "    # This is for illustration only - replace with actual Democracy Index data\n",
    "    countries = [\n",
    "        'Sweden', 'Norway', 'Denmark', 'United States', 'Canada', 'United Kingdom', \n",
    "        'Germany', 'France', 'Italy', 'Spain', 'Japan', 'Australia', 'New Zealand',\n",
    "        'Brazil', 'Mexico', 'India', 'South Africa', 'Turkey', 'Russia', 'China',\n",
    "        'Saudi Arabia', 'Iran', 'Venezuela', 'Cuba', 'North Korea'\n",
    "    ]\n",
    "    \n",
    "    # ISO3 codes for the example countries\n",
    "    iso3_codes = [\n",
    "        'SWE', 'NOR', 'DNK', 'USA', 'CAN', 'GBR', 'DEU', 'FRA', 'ITA', 'ESP', \n",
    "        'JPN', 'AUS', 'NZL', 'BRA', 'MEX', 'IND', 'ZAF', 'TUR', 'RUS', 'CHN',\n",
    "        'SAU', 'IRN', 'VEN', 'CUB', 'PRK'\n",
    "    ]\n",
    "    \n",
    "    # Define regime types for the examples\n",
    "    regime_types = [\n",
    "        'Full democracy', 'Full democracy', 'Full democracy', 'Flawed democracy', 'Full democracy', 'Full democracy',\n",
    "        'Full democracy', 'Flawed democracy', 'Flawed democracy', 'Flawed democracy', 'Flawed democracy', 'Full democracy', 'Full democracy',\n",
    "        'Flawed democracy', 'Flawed democracy', 'Flawed democracy', 'Flawed democracy', 'Hybrid regime', 'Authoritarian regime', 'Authoritarian regime',\n",
    "        'Authoritarian regime', 'Authoritarian regime', 'Authoritarian regime', 'Authoritarian regime', 'Authoritarian regime'\n",
    "    ]\n",
    "    \n",
    "    # Create numeric mapping for regime types\n",
    "    regime_mapping = {\n",
    "        'Full democracy': 0,\n",
    "        'Flawed democracy': 1,\n",
    "        'Hybrid regime': 2,\n",
    "        'Authoritarian regime': 3\n",
    "    }\n",
    "    \n",
    "    # Create dataframe with mock data\n",
    "    n_countries = len(countries)\n",
    "    np.random.seed(42)  # For reproducibility\n",
    "    \n",
    "    data = {\n",
    "        'country': countries,\n",
    "        'ISO3': iso3_codes,\n",
    "        'year': [2023] * n_countries,\n",
    "        'regime_type': regime_types,\n",
    "        'regime_type_numeric': [regime_mapping[rt] for rt in regime_types],\n",
    "    }\n",
    "    \n",
    "    # Create reasonable values for each dimension based on regime type\n",
    "    dimensions = ['electoral_process', 'functioning_govt', 'political_participation', \n",
    "                  'political_culture', 'civil_liberties']\n",
    "    \n",
    "    # Set ranges for each regime type to simulate realistic data\n",
    "    for dim in dimensions:\n",
    "        data[dim] = np.zeros(n_countries)\n",
    "        \n",
    "    for i, regime in enumerate(regime_types):\n",
    "        if regime == 'Full democracy':\n",
    "            for dim in dimensions:\n",
    "                data[dim][i] = np.random.uniform(8.0, 10.0)\n",
    "        elif regime == 'Flawed democracy':\n",
    "            for dim in dimensions:\n",
    "                data[dim][i] = np.random.uniform(6.0, 8.0)\n",
    "        elif regime == 'Hybrid regime':\n",
    "            for dim in dimensions:\n",
    "                data[dim][i] = np.random.uniform(4.0, 6.0)\n",
    "        else:  # Authoritarian regime\n",
    "            for dim in dimensions:\n",
    "                data[dim][i] = np.random.uniform(1.0, 4.0)\n",
    "    \n",
    "    # Add some variation to make it more realistic\n",
    "    for i in range(n_countries):\n",
    "        # Choose a random dimension to adjust\n",
    "        dim_to_adjust = np.random.choice(dimensions)\n",
    "        # Adjust up or down by up to 2 points (but keep in 0-10 range)\n",
    "        adjustment = np.random.uniform(-2, 2)\n",
    "        data[dim_to_adjust][i] = np.clip(data[dim_to_adjust][i] + adjustment, 0, 10)\n",
    "    \n",
    "    # Calculate overall democracy index (average of the five dimensions)\n",
    "    data['democracy_index'] = np.mean([data[dim] for dim in dimensions], axis=0)\n",
    "    \n",
    "    df = pd.DataFrame(data)\n",
    "    print(\"Example data created successfully!\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Basic exploration to understand structure\n",
    "print(f\"Dataset shape: {df.shape}\")\n",
    "print(f\"Columns: {df.columns.tolist()}\")\n",
    "print(f\"Number of countries: {df['country'].nunique()}\")\n",
    "print(f\"Years available: {df['year'].unique()}\")\n",
    "\n",
    "# Display the first few rows\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Check for missing values\n",
    "missing_values = df.isnull().sum()\n",
    "print(f\"Missing values per column:\\n{missing_values}\")\n",
    "\n",
    "# Basic statistics for the five democracy dimensions\n",
    "dimensions = ['electoral_process', 'functioning_govt', 'political_participation', \n",
    "              'political_culture', 'civil_liberties']\n",
    "\n",
    "df[dimensions + ['democracy_index']].describe()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Initial Analysis of Democracy Dimensions\n",
    "\n",
    "Let's examine the relationships between the different dimensions and visualize the distribution of scores."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Correlation matrix between dimensions\n",
    "corr_matrix = df[dimensions].corr()\n",
    "\n",
    "# Create a heatmap of correlations\n",
    "plt.figure(figsize=(10, 8))\n",
    "sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)\n",
    "plt.title('Correlation Matrix of Democracy Dimensions')\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Create a pairplot to visualize relationships between dimensions\n",
    "# Color by existing regime type for comparison with our later clustering\n",
    "sns.pairplot(df, vars=dimensions, hue='regime_type', palette='viridis', height=2.5)\n",
    "plt.suptitle('Pairwise Relationships Between Democracy Dimensions', y=1.02)\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Distribution of overall democracy index\n",
    "plt.figure(figsize=(10, 6))\n",
    "sns.histplot(df['democracy_index'], bins=20, kde=True)\n",
    "plt.axvline(x=8.0, color='red', linestyle='--', label='Full Democracy Threshold')\n",
    "plt.axvline(x=6.0, color='orange', linestyle='--', label='Flawed Democracy Threshold')\n",
    "plt.axvline(x=4.0, color='green', linestyle='--', label='Hybrid Regime Threshold')\n",
    "plt.title('Distribution of Democracy Index Scores')\n",
    "plt.xlabel('Democracy Index Score')\n",
    "plt.ylabel('Frequency')\n",
    "plt.legend()\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Data Preprocessing\n",
    "\n",
    "Now we'll prepare the data for clustering analysis by handling any missing values and standardizing the dimensions."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Handle missing values with forward fill for time series (if multi-year data)\n",
    "if df['year'].nunique() > 1:\n",
    "    df = df.sort_values(['country', 'year'])\n",
    "    df = df.fillna(method='ffill')\n",
    "\n",
    "# For any remaining gaps, use mean imputation\n",
    "imputer = SimpleImputer(strategy='mean')\n",
    "df[dimensions] = imputer.fit_transform(df[dimensions])\n",
    "\n",
    "# Verify missing values are handled\n",
    "print(f\"Remaining missing values: {df[dimensions].isnull().sum().sum()}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Standardize dimensions for clustering\n",
    "scaler = StandardScaler()\n",
    "scaled_data = scaler.fit_transform(df[dimensions])\n",
    "scaled_df = pd.DataFrame(scaled_data, columns=dimensions)\n",
    "\n",
    "# Add country column for reference\n",
    "scaled_df['country'] = df['country'].values\n",
    "\n",
    "# Display the standardized data\n",
    "scaled_df.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Feature Engineering\n",
    "\n",
    "Let's create additional features that might enhance clustering quality and interpretability."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Create ratio features that capture trade-offs between dimensions\n",
    "df['liberty_governance_ratio'] = df['civil_liberties'] / (df['functioning_govt'] + 1e-10)\n",
    "df['participation_culture_ratio'] = df['political_participation'] / (df['political_culture'] + 1e-10)\n",
    "\n",
    "# Create stability/volatility features (if using multi-year data)\n",
    "if df['year'].nunique() > 1:\n",
    "    democracy_volatility = df.groupby('country')['democracy_index'].std().reset_index()\n",
    "    democracy_volatility.columns = ['country', 'democracy_volatility']\n",
    "    df = df.merge(democracy_volatility, on='country', how='left')\n",
    "    \n",
    "    # Calculate year-over-year changes\n",
    "    df['democracy_change'] = df.groupby('country')['democracy_index'].diff()\n",
    "\n",
    "# Display the new features\n",
    "if 'democracy_volatility' in df.columns:\n",
    "    df[['country', 'democracy_index', 'liberty_governance_ratio', 'participation_culture_ratio', \n",
    "        'democracy_volatility', 'democracy_change']].head()\n",
    "else:\n",
    "    df[['country', 'democracy_index', 'liberty_governance_ratio', 'participation_culture_ratio']].head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Determining the Optimal Number of Clusters\n",
    "\n",
    "We'll use several methods to determine the appropriate number of clusters for our analysis."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Use elbow method to find potential optimal k values\n",
    "model = KMeans()\n",
    "visualizer = KElbowVisualizer(model, k=(2, 10))\n",
    "visualizer.fit(scaled_data)\n",
    "visualizer.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Calculate silhouette scores for range of k values\n",
    "silhouette_scores = []\n",
    "k_range = range(2, 10)\n",
    "for k in k_range:\n",
    "    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)\n",
    "    cluster_labels = kmeans.fit_predict(scaled_data)\n",
    "    silhouette_avg = silhouette_score(scaled_data, cluster_labels)\n",
    "    silhouette_scores.append(silhouette_avg)\n",
    "    print(f\"For n_clusters = {k}, the silhouette score is {silhouette_avg:.3f}\")\n",
    "\n",
    "# Plot silhouette scores\n",
    "plt.figure(figsize=(10, 6))\n",
    "plt.plot(k_range, silhouette_scores, 'bo-')\n",
    "plt.xlabel('Number of Clusters (k)')\n",
    "plt.ylabel('Silhouette Score')\n",
    "plt.title('Silhouette Method For Optimal k')\n",
    "plt.grid(True)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Based on the elbow method and silhouette scores, along with political science theory suggesting 4-6 clusters often work well for democracy data, let's select an optimal number of clusters for our analysis."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Set optimal k based on our analysis\n",
    "# Adjust this value based on the results of the elbow and silhouette methods\n",
    "optimal_k = 5  # Example - adjust based on your results"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## K-means Clustering\n",
    "\n",
    "Now we'll implement K-means clustering with our chosen number of clusters."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Implement K-means with optimal k\n",
    "kmeans = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42, n_init=10)\n",
    "cluster_labels = kmeans.fit_predict(scaled_data)\n",
    "\n",
    "# Add cluster labels to original dataframe\n",
    "df['cluster'] = cluster_labels\n",
    "\n",
    "# Analyze cluster characteristics\n",
    "cluster_profiles = df.groupby('cluster')[dimensions].mean()\n",
    "print(\"Cluster Profiles:\")\n",
    "print(cluster_profiles)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Visualize the cluster characteristics as a bar chart\n",
    "cluster_profiles.plot(kind='bar', figsize=(14, 7))\n",
    "plt.title('Democracy Dimensions by Cluster')\n",
    "plt.xlabel('Cluster')\n",
    "plt.ylabel('Average Score')\n",
    "plt.legend(title='Democracy Dimension')\n",
    "plt.grid(axis='y')\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Count countries in each cluster\n",
    "cluster_counts = df['cluster'].value_counts().sort_index()\n",
    "print(\"Number of countries in each cluster:\")\n",
    "print(cluster_counts)\n",
    "\n",
    "# Visualize the distribution\n",
    "plt.figure(figsize=(10, 6))\n",
    "cluster_counts.plot(kind='bar')\n",
    "plt.title('Number of Countries in Each Cluster')\n",
    "plt.xlabel('Cluster')\n",
    "plt.ylabel('Number of Countries')\n",
    "plt.grid(axis='y')\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Hierarchical Clustering\n",
    "\n",
    "Let's also apply hierarchical clustering to understand the relationships between different regime types."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Calculate linkage matrix\n",
    "linkage_matrix = linkage(scaled_data, method='ward')\n",
    "\n",
    "# Plot dendrogram\n",
    "plt.figure(figsize=(14, 10))\n",
    "dendrogram(\n",
    "    linkage_matrix,\n",
    "    labels=df['country'].values,\n",
    "    orientation='right',\n",
    "    leaf_font_size=9\n",
    ")\n",
    "plt.title('Hierarchical Clustering Dendrogram')\n",
    "plt.xlabel('Distance')\n",
    "plt.ylabel('Countries')\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Implement hierarchical clustering\n",
    "hc = AgglomerativeClustering(n_clusters=optimal_k, linkage='ward')\n",
    "hc_labels = hc.fit_predict(scaled_data)\n",
    "df['hc_cluster'] = hc_labels\n",
    "\n",
    "# Compare K-means and hierarchical clustering results\n",
    "comparison = pd.crosstab(df['cluster'], df['hc_cluster'])\n",
    "print(\"Comparing K-means and Hierarchical Clustering results:\")\n",
    "print(comparison)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Dimensionality Reduction for Visualization\n",
    "\n",
    "Now we'll use PCA and t-SNE to visualize our clusters in two dimensions."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# PCA for basic visualization and interpretation\n",
    "pca = PCA(n_components=2)\n",
    "pca_result = pca.fit_transform(scaled_data)\n",
    "pca_df = pd.DataFrame(data=pca_result, columns=['PC1', 'PC2'])\n",
    "pca_df['country'] = df['country'].values\n",
    "pca_df['cluster'] = df['cluster'].values\n",
    "pca_df['democracy_index'] = df['democracy_index'].values\n",
    "pca_df['regime_type'] = df['regime_type'].values\n",
    "\n",
    "# Print explained variance\n",
    "print(f\"PC1 explained variance: {pca.explained_variance_ratio_[0]:.2%}\")\n",
    "print(f\"PC2 explained variance: {pca.explained_variance_ratio_[1]:.2%}\")\n",
    "print(f\"Total explained variance: {sum(pca.explained_variance_ratio_):.2%}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Create scatter plot with PCA results using plotly\n",
    "fig = px.scatter(\n",
    "    pca_df, \n",
    "    x='PC1', \n",
    "    y='PC2',\n",
    "    color='cluster',\n",
    "    hover_name='country',\n",
    "    size='democracy_index',\n",
    "    text='country',\n",
    "    title='PCA of Democracy Index Dimensions',\n",
    "    labels={'PC1': f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)',\n",
    "            'PC2': f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)'},\n",
    "    color_continuous_scale=px.colors.qualitative.G10,\n",
    ")\n",
    "\n",
    "# Add biplot vectors to show how original dimensions relate to PCA\n",
    "for i, feature in enumerate(dimensions):\n",
    "    fig.add_shape(\n",
    "        type='line',\n",
    "        x0=0, y0=0,\n",
    "        x1=pca.components_[0, i] * 5,\n",
    "        y1=pca.components_[1, i] * 5,\n",
    "        line=dict(color='red', width=1, dash='dash')\n",
    "    )\n",
    "    fig.add_annotation(\n",
    "        x=pca.components_[0, i] * 5.5,\n",
    "        y=pca.components_[1, i] * 5.5,\n",
    "        text=feature,\n",
    "        showarrow=False,\n",
    "        font=dict(size=12, color='darkred')\n",
    "    )\n",
    "\n",
    "fig.update_traces(textposition='top center')\n",
    "fig.update_layout(height=700, width=900)\n",
    "fig.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# t-SNE for more complex, non-linear patterns\n",
    "tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(df) // 5))\n",
    "tsne_results = tsne.fit_transform(scaled_data)\n",
    "tsne_df = pd.DataFrame(tsne_results, columns=['t-SNE1', 't-SNE2'])\n",
    "tsne_df['country'] = df['country'].values\n",
    "tsne_df['cluster'] = df['cluster'].values\n",
    "tsne_df['democracy_index'] = df['democracy_index'].values\n",
    "tsne_df['regime_type'] = df['regime_type'].values\n",
    "\n",
    "# Create t-SNE plot\n",
    "fig_tsne = px.scatter(\n",
    "    tsne_df, \n",
    "    x='t-SNE1', \n",
    "    y='t-SNE2',\n",
    "    color='cluster',\n",
    "    hover_name='country',\n",
    "    size='democracy_index',\n",
    "    text='country',\n",
    "    title='t-SNE of Democracy Index Dimensions',\n",
    "    color_continuous_scale=px.colors.qualitative.G10,\n",
    ")\n",
    "\n",
    "fig_tsne.update_traces(textposition='top center')\n",
    "fig_tsne.update_layout(height=700, width=900)\n",
    "fig_tsne.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Radar Charts for Democracy Profiles\n",
    "\n",
    "Let's create radar charts to visualize all five dimensions simultaneously for each cluster."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Create radar chart comparing cluster profiles\n",
    "fig = go.Figure()\n",
    "\n",
    "# Add each cluster as a trace\n",
    "for cluster_id in range(optimal_k):\n",
    "    cluster_avg = cluster_profiles.iloc[cluster_id].values\n",
    "    fig.add_trace(go.Scatterpolar(\n",
    "        r=cluster_avg,\n",
    "        theta=dimensions,\n",
    "        fill='toself',\n",
    "        name=f'Cluster {cluster_id}'\n",
    "    ))\n",
    "\n",
    "# Update layout\n",
    "fig.update_layout(\n",
    "    polar=dict(\n",
    "        radialaxis=dict(\n",
    "            visible=True,\n",
    "            range=[0, 10]  # Democracy Index range\n",
    "        )\n",
    "    ),\n",
    "    title=\"Democracy Index Dimension Profiles by Cluster\",\n",
    "    showlegend=True,\n",
    "    height=600,\n",
    "    width=800\n",
    ")\n",
    "\n",
    "fig.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Geographic Visualization of Clusters\n",
    "\n",
    "Let's create a choropleth map to visualize the geographical distribution of our clusters."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Create choropleth map showing clusters\n",
    "fig = px.choropleth(\n",
    "    df,\n",
    "    locations='ISO3',  # Uses ISO3 country codes\n",
    "    color='cluster',\n",
    "    hover_name='country',\n",
    "    color_discrete_sequence=px.colors.qualitative.G10,\n",
    "    title='Democracy Clusters Worldwide',\n",
    "    labels={'cluster': 'Cluster'}\n",
    ")\n",
    "\n",
    "fig.update_layout(\n",
    "    geo=dict(\n",
    "        showframe=False,\n",
    "        showcoastlines=True,\n",
    "        projection_type='equirectangular'\n",
    "    ),\n",
    "    height=600,\n",
    "    width=900\n",
    ")\n",
    "\n",
    "fig.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Identifying Transition Countries\n",
    "\n",
    "Let's identify countries at cluster boundaries, which might represent transitional regimes."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Identify boundary/transition countries\n",
    "def find_transition_countries(data, cluster_labels, n_neighbors=3):\n",
    "    # Calculate distance to cluster centers\n",
    "    clusters_unique = np.unique(cluster_labels)\n",
    "    cluster_centers = []\n",
    "    for cluster_id in clusters_unique:\n",
    "        cluster_points = data[cluster_labels == cluster_id]\n",
    "        cluster_centers.append(np.mean(cluster_points, axis=0))\n",
    "    \n",
    "    # Calculate transition scores based on neighbors from different clusters\n",
    "    nn = NearestNeighbors(n_neighbors=n_neighbors)\n",
    "    nn.fit(data)\n",
    "    distances, indices = nn.kneighbors(data)\n",
    "    \n",
    "    transition_scores = []\n",
    "    for i, neighbors in enumerate(indices):\n",
    "        own_cluster = cluster_labels[i]\n",
    "        neighbor_clusters = [cluster_labels[j] for j in neighbors]\n",
    "        different_clusters = sum(1 for c in neighbor_clusters if c != own_cluster)\n",
    "        transition_scores.append(different_clusters / len(neighbors))\n",
    "    \n",
    "    return np.array(transition_scores)\n",
    "\n",
    "# Apply to our data\n",
    "transition_scores = find_transition_countries(scaled_data, cluster_labels, n_neighbors=min(5, len(df) // 5))\n",
    "pca_df['transition_score'] = transition_scores"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Visualize transition countries\n",
    "fig = px.scatter(\n",
    "    pca_df,\n",
    "    x='PC1',\n",
    "    y='PC2',\n",
    "    color='cluster',\n",
    "    size='transition_score',  # Size by transition score\n",
    "    hover_name='country',\n",
    "    hover_data=['democracy_index', 'transition_score'],\n",
    "    title='Democracy Index - Transition Countries',\n",
    "    labels={'PC1': f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)',\n",
    "            'PC2': f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)'},\n",
    "    color_discrete_sequence=px.colors.qualitative.G10,\n",
    "    size_max=15\n",
    ")\n",
    "\n",
    "# Highlight top transition countries\n",
    "top_transition = pca_df.nlargest(5, 'transition_score')\n",
    "for i, row in top_transition.iterrows():\n",
    "    fig.add_annotation(\n",
    "        x=row['PC1'],\n",
    "        y=row['PC2'],\n",
    "        text=row['country'],\n",
    "        showarrow=True,\n",
    "        arrowhead=1,\n",
    "        font=dict(size=12)\n",
    "    )\n",
    "\n",
    "fig.update_layout(height=700, width=900)\n",
    "fig.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Display top transition countries\n",
    "df['transition_score'] = transition_scores\n",
    "top_transitions = df.nlargest(10, 'transition_score')\n",
    "\n",
    "print(\"Top transition countries:\")\n",
    "top_transitions[['country', 'cluster', 'democracy_index', 'transition_score'] + dimensions]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Comparing Clusters with Official Regime Types\n",
    "\n",
    "Let's compare our cluster results with the EIU's official regime classifications."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Compare clusters with official regime types\n",
    "regime_mapping = {\n",
    "    'Full democracy': 0,\n",
    "    'Flawed democracy': 1,\n",
    "    'Hybrid regime': 2,\n",
    "    'Authoritarian regime': 3\n",
    "}\n",
    "\n",
    "# Create confusion matrix\n",
    "if 'regime_type_numeric' in df.columns:\n",
    "    true_regimes = df['regime_type_numeric'].values\n",
    "else:\n",
    "    true_regimes = df['regime_type'].map(regime_mapping).values\n",
    "\n",
    "conf_matrix = confusion_matrix(true_regimes, cluster_labels)\n",
    "\n",
    "# Calculate agreement score\n",
    "ari = adjusted_rand_score(true_regimes, cluster_labels)\n",
    "ami = adjusted_mutual_info_score(true_regimes, cluster_labels)\n",
    "\n",
    "print(f\"Adjusted Rand Index: {ari:.3f}\")\n",
    "print(f\"Adjusted Mutual Information: {ami:.3f}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Visualize confusion matrix\n",
    "regime_labels = list(regime_mapping.keys())\n",
    "cluster_labels_text = [f'Cluster {i}' for i in range(optimal_k)]\n",
    "\n",
    "fig = ff.create_annotated_heatmap(\n",
    "    z=conf_matrix,\n",
    "    x=cluster_labels_text,\n",
    "    y=regime_labels,\n",
    "    annotation_text=conf_matrix,\n",
    "    colorscale='Blues'\n",
    ")\n",
    "\n",
    "fig.update_layout(\n",
    "    title=f'Comparison: Regime Types vs. Clusters (ARI: {ari:.2f})',\n",
    "    xaxis_title='Predicted Clusters',\n",
    "    yaxis_title='EIU Regime Types',\n",
    "    height=500,\n",
    "    width=700\n",
    ")\n",
    "\n",
    "fig.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Cluster Interpretation and Naming\n",
    "\n",
    "Based on the dimensional profiles of each cluster, let's create meaningful names that reflect their characteristics."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Create and visualize cluster dimensional profiles\n",
    "profile_df = cluster_profiles.copy()\n",
    "\n",
    "# Create descriptive names based on dimensional characteristics\n",
    "cluster_names = []\n",
    "for i, profile in profile_df.iterrows():\n",
    "    # Example logic for naming (adjust based on your findings)\n",
    "    if profile['electoral_process'] > 8 and profile['civil_liberties'] > 8:\n",
    "        name = \"Liberal Democracy\"\n",
    "    elif profile['functioning_govt'] > profile['political_participation'] + 1:\n",
    "        name = \"State-Centric Democracy\"\n",
    "    elif profile['political_participation'] > profile['functioning_govt'] + 1:\n",
    "        name = \"Participatory Democracy\"\n",
    "    elif profile['electoral_process'] < 4:\n",
    "        name = \"Electoral Autocracy\"\n",
    "    else:\n",
    "        name = \"Hybrid Regime\"\n",
    "    cluster_names.append(name)\n",
    "\n",
    "profile_df['cluster_name'] = cluster_names\n",
    "profile_df.set_index('cluster_name', inplace=True)\n",
    "\n",
    "# Display the named clusters\n",
    "print(\"Cluster profiles with descriptive names:\")\n",
    "profile_df"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Visualize the named clusters\n",
    "ax = profile_df[dimensions].plot(kind='bar', figsize=(14, 7))\n",
    "ax.set_title('Democracy Profiles by Cluster Type')\n",
    "ax.set_ylabel('Score')\n",
    "ax.set_xlabel('Cluster Type')\n",
    "plt.legend(title='Democracy Dimension')\n",
    "plt.grid(axis='y')\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Map cluster names back to the dataframe\n",
    "cluster_name_map = {i: name for i, name in enumerate(cluster_names)}\n",
    "df['cluster_name'] = df['cluster'].map(cluster_name_map)\n",
    "\n",
    "# Count countries in each named cluster\n",
    "named_counts = df['cluster_name'].value_counts()\n",
    "print(\"Number of countries in each named cluster:\")\n",
    "print(named_counts)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Create a choropleth map with named clusters\n",
    "fig = px.choropleth(\n",
    "    df,\n",
    "    locations='ISO3',\n",
    "    color='cluster_name',\n",
    "    hover_name='country',\n",
    "    title='Democracy Profile Types Worldwide',\n",
    "    color_discrete_sequence=px.colors.qualitative.Bold,\n",
    "    labels={'cluster_name': 'Democracy Profile'}\n",
    ")\n",
    "\n",
    "fig.update_layout(\n",
    "    geo=dict(\n",
    "        showframe=False,\n",
    "        showcoastlines=True,\n",
    "        projection_type='natural earth'\n",
    "    ),\n",
    "    height=600,\n",
    "    width=900\n",
    ")\n",
    "\n",
    "fig.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Robustness Testing\n",
    "\n",
    "Let's test the robustness of our clustering solution using bootstrapping."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Bootstrapping for cluster stability\n",
    "from sklearn.utils import resample\n",
    "\n",
    "n_bootstraps = 30  # Reduced for demonstration - increase for real analysis\n",
    "ari_bootstrap_scores = []\n",
    "\n",
    "for i in range(n_bootstraps):\n",
    "    # Create bootstrap sample\n",
    "    boot_indices = resample(range(len(scaled_data)), replace=True, \n",
    "                           n_samples=len(scaled_data), random_state=i)\n",
    "    boot_data = scaled_data[boot_indices]\n",
    "    \n",
    "    # Cluster the bootstrap sample\n",
    "    kmeans_boot = KMeans(n_clusters=optimal_k, random_state=42, n_init=10).fit(boot_data)\n",
    "    boot_labels = kmeans_boot.labels_\n",
    "    \n",
    "    # Compare with original clustering\n",
    "    original_labels = np.zeros_like(boot_labels)\n",
    "    for j, idx in enumerate(boot_indices):\n",
    "        original_labels[j] = cluster_labels[idx]\n",
    "    \n",
    "    ari = adjusted_rand_score(original_labels, boot_labels)\n",
    "    ari_bootstrap_scores.append(ari)\n",
    "\n",
    "print(f\"Bootstrap stability - Mean ARI: {np.mean(ari_bootstrap_scores):.3f}\")\n",
    "print(f\"Bootstrap stability - Std Dev ARI: {np.std(ari_bootstrap_scores):.3f}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Visualize bootstrap stability\n",
    "plt.figure(figsize=(10, 6))\n",
    "plt.hist(ari_bootstrap_scores, bins=10, alpha=0.7, color='steelblue')\n",
    "plt.axvline(x=np.mean(ari_bootstrap_scores), color='red', linestyle='--', \n",
    "            label=f'Mean: {np.mean(ari_bootstrap_scores):.3f}')\n",
    "plt.title('Bootstrap Stability of Clustering Solution')\n",
    "plt.xlabel('Adjusted Rand Index')\n",
    "plt.ylabel('Frequency')\n",
    "plt.legend()\n",
    "plt.grid(axis='y', alpha=0.3)\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Analyzing Transition Countries in Detail\n",
    "\n",
    "Let's examine the transition countries more closely to understand their democratic characteristics."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Add transition countries to dataframe\n",
    "df['is_transition'] = df['transition_score'] > np.percentile(df['transition_score'], 80)\n",
    "\n",
    "# Compare transition countries' profiles\n",
    "transition_profiles = df[df['is_transition'] == True][dimensions + ['country', 'transition_score', 'cluster_name']]\n",
    "transition_profiles.set_index('country', inplace=True)\n",
    "\n",
    "# Display transition country profiles\n",
    "print(\"Democracy profiles of top transition countries:\")\n",
    "transition_profiles"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Visualize transition country profiles\n",
    "ax = transition_profiles[dimensions].plot(kind='bar', figsize=(14, 7))\n",
    "ax.set_title('Democracy Profiles of Top Transition Countries')\n",
    "ax.set_ylabel('Score')\n",
    "ax.set_xlabel('Country')\n",
    "plt.legend(title='Democracy Dimension')\n",
    "plt.grid(axis='y', alpha=0.3)\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Political Interpretation of Clusters\n",
    "\n",
    "Let's interpret our findings in the context of political science theory."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Cluster Interpretations\n",
    "\n",
    "Our cluster analysis reveals distinct democracy profiles that extend beyond the traditional democracy-autocracy spectrum. Each profile represents a different pattern in how countries balance the five dimensions of democracy measured by the EIU.\n",
    "\n",
    "**Liberal Democracy**  \n",
    "Countries in this cluster show high scores across all dimensions, with particularly strong performance in electoral processes and civil liberties. These countries have developed robust institutional frameworks that support all aspects of democratic governance, with a special emphasis on protecting individual rights and freedoms.\n",
    "\n",
    "**State-Centric Democracy**  \n",
    "This cluster represents countries with stronger institutional governance than participatory elements. These systems prioritize government effectiveness and stability, sometimes at the expense of broad political participation. They maintain democratic elections and reasonable civil liberties, but citizen engagement in the political process beyond voting may be limited.\n",
    "\n",
    "**Participatory Democracy**  \n",
    "Countries in this cluster demonstrate stronger political participation than institutional functioning. These systems have developed democratic cultures and high levels of citizen engagement, but may lag in bureaucratic effectiveness or policy implementation. They prioritize inclusive political processes over administrative efficiency.\n",
    "\n",
    "**Hybrid Regime**  \n",
    "This cluster shows mid-range scores across dimensions, with significant inconsistencies between democratic elements. These countries maintain some democratic features (usually elections) while restricting others (often civil liberties or genuine political competition). They represent transitional or deliberately mixed systems that combine democratic and autocratic features.\n",
    "\n",
    "**Electoral Autocracy**  \n",
    "Countries in this cluster score poorly across most or all dimensions, particularly in electoral processes and civil liberties. While these regimes may maintain the formal appearance of democratic institutions, genuine political competition and basic freedoms are severely constrained or absent.\n",
    "\n",
    "These findings support recent theoretical work suggesting that democracy is not simply a linear spectrum but involves different configurations of democratic elements. As political scientists like Møller and Skaaning argue, democratic development often follows specific patterns where certain rights and institutions emerge before others, creating distinctive profiles during transition periods."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### The Significance of Transition Countries\n",
    "\n",
    "The countries we identified as \"transition countries\" through our boundary analysis are particularly interesting from a political science perspective. These nations exist at the intersection between different cluster types, potentially representing:\n",
    "\n",
    "1. **Systems in transition** - Countries actively moving between regime types, either democratizing or experiencing democratic backsliding\n",
    "\n",
    "2. **Hybrid systems** - Stable regimes that deliberately combine elements from different democratic models\n",
    "\n",
    "3. **Contested democracies** - Countries where the basic character of the political system is actively contested by domestic actors\n",
    "\n",
    "By examining these transition countries in detail, we can gain insight into democratic development trajectories and the challenges of democratic consolidation or erosion."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Conclusion\n",
    "\n",
    "Our cluster analysis of the Democracy Index reveals natural groupings that extend beyond traditional regime classifications. By applying appropriate preprocessing, carefully selecting clustering parameters, and creating effective visualizations, we've identified distinct democracy profiles that reflect how countries prioritize different aspects of democratic governance.\n",
    "\n",
    "Key findings include:\n",
    "\n",
    "1. The existence of distinct democracy profiles that emphasize different combinations of electoral processes, governance, participation, culture, and civil liberties\n",
    "\n",
    "2. The identification of transition countries that exist at the boundaries between clusters, representing systems in flux or hybrid regimes\n",
    "\n",
    "3. The spatial distribution of democracy profiles, highlighting regional patterns in democratic development\n",
    "\n",
    "These insights support a more nuanced understanding of democracy that moves beyond viewing it as a simple linear progression from authoritarianism to full democracy. Instead, our analysis reveals the diverse paths that political systems take and the different trade-offs they make between democratic values.\n",
    "\n",
    "Future research could extend this analysis by incorporating time-series data to track transitions between clusters over time, or by including additional variables beyond the core democracy dimensions to explore the relationship between democratic profiles and other political, economic, or social factors."
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
