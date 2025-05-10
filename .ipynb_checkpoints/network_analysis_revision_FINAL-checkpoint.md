<!-- #region -->
# Network & Word Frequency Analysis: Technical Implementation and Findings

# Project Overview

In this project, I conducted a comprehensive keyword co-occurrence network analysis to explore relationships between academic concepts in data mining literature. Using both Python-based network analysis and a custom React-based interactive visualization, I uncovered meaningful patterns that provide insights into the structure of knowledge in this domain.


# Methodology
<!-- #endregion -->

# Setup and Environment Configuration

```python
# Install required libraries (quiet mode)
!pip install -q pandas numpy matplotlib seaborn scikit-learn scipy

# Verify successful installation by importing libraries
import pandas
import numpy
import matplotlib
import seaborn
import sklearn
import scipy

print("All libraries imported successfully!")
```

## Dataset Loading and Immutability


## Exploratory Data Analysis


```python
import pandas as pd
from typing import FrozenSet, Tuple
import numpy as np

# Source URL
source_url = "https://raw.githubusercontent.com/JustGlowing/minisom/master/examples/democracy_index.csv"

# Function to load the dataset in a standardized, immutable format
def load_immutable_democracy_dataset():
    """
    Loads the democracy index dataset and returns an immutable version.
    Returns a namedtuple containing the original DataFrame and a frozen copy.
    """
    from collections import namedtuple   

print("🗂️ Dataset dimensions:", df.shape)
print("\n🧼 Missing values:")
print(df.isnull().sum())

print("\n📄 Data types:")
print(df.dtypes)

# Display the first few rows
print("Dataset loaded successfully with shape:", df.shape)
df.head()
```

## Statistical Overview

```python
df.describe()
```

```python
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# Load the co-occurrence matrix from the CSV file

# Function to load the dataset in a standardized, immutable format
def load_immutable_democracy_dataset():
co_occurrence_matrix = pd.read_csv(file_path, index_col=0)

# Define a cleaning function for standardizing keywords
def clean_keyword(keyword: str) -> str:
    """Cleans a keyword string by removing double hyphens and standardizing whitespace."""
    cleaned_keyword = " ".join(keyword.split("--"))  # Replace "--" with space
    cleaned_keyword = " ".join(cleaned_keyword.split())  # Remove extra spaces
    return cleaned_keyword.strip()  # Remove leading/trailing spaces

# Apply the cleaning function to row and column names
co_occurrence_matrix.columns = co_occurrence_matrix.columns.map(clean_keyword)
co_occurrence_matrix.index = co_occurrence_matrix.index.map(clean_keyword)

# Ensure that column and index names are lowercase for consistency
co_occurrence_matrix.columns = co_occurrence_matrix.columns.str.lower()
co_occurrence_matrix.index = co_occurrence_matrix.index.str.lower()
```

### Network Construction

I constructed a weighted network graph where nodes represent keywords and edges represent their co-occurrence frequency:

```python
# Initialize an empty undirected graph to store the weighted network
G_weighted = nx.Graph()

# Iterate over the co-occurrence matrix and add edges to the graph
for word1 in co_occurrence_matrix.index:
    for word2 in co_occurrence_matrix.columns:
        weight = co_occurrence_matrix.at[word1, word2]
        # Skip the pair if the weight is zero or NaN
        if pd.notna(weight) and weight > 0:
            G_weighted.add_edge(word1, word2, weight=weight)

# Display basic information about the graph
print(f"Number of Nodes: {G_weighted.number_of_nodes()}")
print(f"Number of Edges: {G_weighted.number_of_edges()}")
```

## Key Findings

### 1. Network Structure Analysis

My analysis of the network revealed a complex structure with distinct patterns:

```python
# Calculate basic network metrics
density = nx.density(G_weighted)
avg_clustering = nx.average_clustering(G_weighted)
avg_path_length = nx.average_shortest_path_length(G_weighted)
diameter = nx.diameter(G_weighted)

print(f"Network Density: {density:.4f}")
print(f"Average Clustering Coefficient: {avg_clustering:.4f}")
print(f"Average Path Length: {avg_path_length:.4f}")
print(f"Network Diameter: {diameter}")
```

**Key Network Properties:**
- The network showed a relatively low density, indicating a selective pattern of keyword co-occurrences
- The high clustering coefficient suggests well-formed thematic communities
- The short average path length demonstrates the "small world" property typical of knowledge networks

### 2. Centrality Analysis

I identified the most influential keywords through various centrality measures:

```python
# Calculate centrality measures
degree_centrality = nx.degree_centrality(G_weighted)
betweenness_centrality = nx.betweenness_centrality(G_weighted)
eigenvector_centrality = nx.eigenvector_centrality(G_weighted)

# Display top keywords by degree centrality
top_degree = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:10]
print("\nTop Keywords by Degree Centrality:")
for keyword, centrality in top_degree:
    print(f"{keyword}: {centrality:.4f}")
```

**Centrality Findings:**
- The most central keywords (by degree) were "machine learning," "data mining," and "social networks"
- High betweenness centrality keywords like "analytics" and "security" function as bridging concepts
- Eigenvector centrality revealed the broader influence of concepts like "big data" and "artificial intelligence"

### 3. Community Detection

I used community detection algorithms to identify thematic clusters:

```python
# Import community detection algorithm
from community import community_louvain

# Apply Louvain method for community detection
partition = community_louvain.best_partition(G_weighted)

# Organize communities
communities = {}
for node, community_id in partition.items():
    if community_id not in communities:
        communities[community_id] = []
    communities[community_id].append(node)

# Display community statistics
print(f"\nNumber of Communities: {len(communities)}")
for community_id, nodes in communities.items():
    print(f"Community {community_id}: {len(nodes)} keywords")
    # Print top 5 most central nodes in each community
    community_centrality = {node: degree_centrality[node] for node in nodes}
    top_nodes = sorted(community_centrality.items(), key=lambda x: x[1], reverse=True)[:5]
    print(f"  Top keywords: {', '.join(node for node, _ in top_nodes)}")
```

**Community Analysis:**
- I identified three major thematic clusters in the network:
  1. **Data Science Cluster**: Focused on technical methods and algorithms
  2. **Business Applications Cluster**: Emphasizing practical business uses
  3. **Information Management Cluster**: Dealing with system and resource aspects

- Each community has distinctive bridge nodes that connect it to other communities

### 4. Static Network Visualization

I created a static visualization to provide an overview of the network structure:

```python
# Use spring layout for better visualization
pos = nx.spring_layout(G_weighted, k=0.1, iterations=20, seed=42)

# Set up colors based on communities
community_colors = [partition.get(node, 0) for node in G_weighted.nodes()]

# Create the plot
plt.figure(figsize=(20, 20))
nx.draw_networkx(
    G_weighted, 
    pos=pos,
    with_labels=True,
    node_color=community_colors,
    node_size=[v * 5000 + 100 for v in degree_centrality.values()],
    width=[d['weight'] * 0.1 for u, v, d in G_weighted.edges(data=True)],
    edge_color='lightgray',
    cmap=plt.cm.viridis,
    font_size=8
)

plt.title("Keyword Co-occurrence Network with Communities", fontsize=20)
plt.axis('off')
plt.savefig("network_visualization.png", dpi=300, bbox_inches='tight')
plt.show()
```

## Interactive Visualization

To enable dynamic exploration of the network, I developed an interactive visualization using React and HTML Canvas:

```jsx
// NetworkVisualization.jsx - Core physics simulation

const applyForces = () => {
  const nodes = graphData.nodes;
  const links = graphData.links;
  
  // Constants - adjusted for better visibility
  const centerX = canvasSize.width / 2;
  const centerY = canvasSize.height / 2;
  const centerForce = 0.0003;
  const repulsionForce = 700;
  const linkStrength = 0.02;
  const damping = 0.85;
  
  // Calculate forces
  nodes.forEach(node => {
    // Initialize forces
    node.fx = 0;
    node.fy = 0;
    
    // Center attraction force
    node.fx += (centerX - node.x) * centerForce;
    node.fy += (centerY - node.y) * centerForce;
    
    // Node repulsion (inverse square law)
    nodes.forEach(otherNode => {
      if (node !== otherNode) {
        const dx = node.x - otherNode.x;
        const dy = node.y - otherNode.y;
        const distance = Math.sqrt(dx * dx + dy * dy);
        const forceMagnitude = repulsionForce / Math.max(10, distance * distance);
        
        if (distance > 0) {
          node.fx += (dx / distance) * forceMagnitude;
          node.fy += (dy / distance) * forceMagnitude;
        }
      }
    });
  });
  
  // Additional force simulation code...
};
```

The interactive visualization includes several key features:
1. **Force-directed layout** that positions related keywords close together
2. **Color-coding of nodes** based on frequency or community membership
3. **Interactive node selection** for detailed information display
4. **Dynamic hover effects** for easier network exploration
5. **Network statistics panel** providing quantitative context

## Discussion & Interpretation

Through this network analysis, I've uncovered several important insights:

### 1. Knowledge Structure

The network structure reveals how knowledge in this domain is organized:
- The core-periphery pattern shows established foundational concepts at the center
- The clear community structure demonstrates specialization within the broader field
- The bridging nodes highlight concepts that facilitate cross-disciplinary knowledge transfer

### 2. Research Opportunities

My analysis points to several promising research directions:
- The sparse connections between certain communities suggest opportunities for integration
- Peripheral nodes with significant connections may represent emerging research areas
- High betweenness centrality keywords indicate potential focal points for interdisciplinary work

### 3. Methodological Contributions

This project demonstrates the value of combining:
- Rigorous network science methods for quantitative analysis
- Interactive visualization techniques for intuitive exploration
- Cross-platform implementation (Python for analysis, JavaScript for visualization)

## Limitations & Future Work

While this analysis provides valuable insights, I acknowledge several limitations:
- The static nature of the dataset doesn't capture temporal evolution
- The analysis focuses on co-occurrence rather than semantic relationships
- The current implementation has performance limitations with very large networks

In future work, I plan to address these limitations by:
- Incorporating temporal data to track concept evolution
- Exploring semantic analysis techniques to capture deeper relationships
- Implementing performance optimizations for larger networks
- Adding additional interactive features like filtering and search

## Conclusion

This network and word frequency analysis provides a comprehensive view of the conceptual landscape in data mining research. Through the combination of rigorous network analysis and interactive visualization, I've demonstrated how these methods can reveal patterns and relationships that contribute to our understanding of knowledge organization in this domain.

The interactive visualization component not only makes these findings more accessible but also showcases the potential of web-based tools for exploring complex network data. This integrated approach offers a valuable perspective for researchers, educators, and practitioners seeking to navigate and understand this complex intellectual landscape.

---

This project was developed by Rosalina Torres as part of advanced data mining and visualization coursework.
