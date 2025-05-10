
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import stats

print("=== Advanced Activity Analysis ===")

# Load the processed data
df = pd.read_csv("processed_permutation_entropy_complexity.csv")
print(f"Loaded data with {len(df)} rows")

# Create a directory for visualizations
os.makedirs("analysis_results", exist_ok=True)

# 1. Analyze by activity pairs
activity_pairs = [
    ('walking', 'running'),
    ('climbingup', 'climbingdown'),
    ('walking', 'climbingup'),
    ('running', 'climbingdown')
]

print("\n=== Activity Pair Comparisons ===")
for act1, act2 in activity_pairs:
    print(f"\nComparing {act1} vs {act2}:")
    
    # Get data for both activities
    act1_data = df[df['Activity'] == act1]
    act2_data = df[df['Activity'] == act2]
    
    # Perform t-test on PE values
    ttest_result = stats.ttest_ind(
        act1_data['PE'].values,
        act2_data['PE'].values,
        equal_var=False
    )
    print(f"  T-test on PE: t={ttest_result.statistic:.4f}, p={ttest_result.pvalue:.4f}")
    
    # Perform t-test on Complexity values
    ttest_result = stats.ttest_ind(
        act1_data['Complexity'].values,
        act2_data['Complexity'].values,
        equal_var=False
    )
    print(f"  T-test on Complexity: t={ttest_result.statistic:.4f}, p={ttest_result.pvalue:.4f}")
    
    # Create visualization
    plt.figure(figsize=(10, 8))
    
    plt.scatter(
        act1_data['PE'], 
        act1_data['Complexity'],
        label=act1,
        alpha=0.7,
        s=50
    )
    
    plt.scatter(
        act2_data['PE'], 
        act2_data['Complexity'],
        label=act2,
        alpha=0.7,
        s=50
    )
    
    # Calculate and plot centroids
    act1_centroid = (act1_data['PE'].mean(), act1_data['Complexity'].mean())
    act2_centroid = (act2_data['PE'].mean(), act2_data['Complexity'].mean())
    
    plt.scatter(
        act1_centroid[0],
        act1_centroid[1],
        marker='*',
        color='red',
        s=200,
        label=f"{act1} centroid"
    )
    
    plt.scatter(
        act2_centroid[0],
        act2_centroid[1],
        marker='*',
        color='blue',
        s=200,
        label=f"{act2} centroid"
    )
    
    plt.xlabel('Permutation Entropy')
    plt.ylabel('Statistical Complexity')
    plt.title(f'PE vs Complexity: {act1.capitalize()} vs {act2.capitalize()}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    filename = f"analysis_results/PE_vs_Complexity_{act1}_vs_{act2}.png"
    plt.savefig(filename)
    plt.close()
    
    print(f"  Saved visualization to {filename}")

# 2. Analyze by axis
print("\n=== Analysis by Axis ===")
axes = df['Axis'].unique()

for axis in axes:
    axis_data = df[df['Axis'] == axis]
    
    print(f"\nAnalysis for {axis}:")
    
    # Calculate mean PE and Complexity for each activity on this axis
    activity_stats = axis_data.groupby('Activity').agg({
        'PE': ['mean', 'std'],
        'Complexity': ['mean', 'std']
    }).reset_index()
    
    print(activity_stats)
    
    # Create visualization
    plt.figure(figsize=(10, 8))
    
    for activity in df['Activity'].unique():
        activity_axis_data = axis_data[axis_data['Activity'] == activity]
        
        plt.scatter(
            activity_axis_data['PE'],
            activity_axis_data['Complexity'],
            label=activity,
            alpha=0.7,
            s=50
        )
    
    plt.xlabel('Permutation Entropy')
    plt.ylabel('Statistical Complexity')
    plt.title(f'PE vs Complexity by Activity for {axis}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    filename = f"analysis_results/PE_vs_Complexity_{axis}.png"
    plt.savefig(filename)
    plt.close()
    
    print(f"  Saved visualization to {filename}")

# 3. Create summary table
print("\n=== Activity Summary Statistics ===")
activity_summary = df.groupby(['Activity', 'Axis']).agg({
    'PE': ['mean', 'std', 'min', 'max'],
    'Complexity': ['mean', 'std', 'min', 'max']
}).reset_index()

# Save summary to CSV
summary_file = "analysis_results/activity_summary_statistics.csv"
activity_summary.to_csv(summary_file)
print(f"Saved summary statistics to {summary_file}")

# 4. Create overall visualization with ellipses
plt.figure(figsize=(12, 10))

from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

def confidence_ellipse(x, y, ax, n_std=2.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    
    # Using a special case to obtain the eigenvalues of this
    # two-dimensional dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the standard deviation of x from the square root of the variance
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)
    
    # Calculating the standard deviation of y from the square root of the variance
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

# Colors for different activities
colors = {
    'walking': 'blue',
    'running': 'red',
    'climbingup': 'green',
    'climbingdown': 'purple'
}

# Create the main plot
fig, ax = plt.subplots(figsize=(12, 10))

# Plot each activity with confidence ellipses
for activity in df['Activity'].unique():
    activity_data = df[df['Activity'] == activity]
    
    # Plot the data points
    ax.scatter(
        activity_data['PE'],
        activity_data['Complexity'],
        label=activity,
        color=colors[activity],
        alpha=0.6,
        s=50
    )
    
    # Add confidence ellipse
    confidence_ellipse(
        activity_data['PE'].values,
        activity_data['Complexity'].values,
        ax,
        n_std=1.0,
        edgecolor=colors[activity],
        linestyle='--',
        linewidth=2
    )

ax.set_xlabel('Permutation Entropy')
ax.set_ylabel('Statistical Complexity')
ax.set_title('PE vs Complexity with Confidence Ellipses')
ax.legend()
ax.grid(True, alpha=0.3)

# Save the figure
plt.savefig("analysis_results/PE_vs_Complexity_with_ellipses.png")
plt.close()
print("Saved visualization with confidence ellipses")

print("\nAnalysis complete!")
