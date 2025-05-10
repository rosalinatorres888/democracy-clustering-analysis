
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
from scipy import stats

print("=== Analyzing Full Permutation Entropy Dataset ===")

# Define the path to the full dataset
full_dataset_path = "/Users/rosalinatorres/corrected_permutation_entropy_complexity.csv"

# Check if the file exists
if not os.path.exists(full_dataset_path):
    print(f"Error: File not found at {full_dataset_path}")
    print("Please specify the correct path to your full dataset.")
    exit()

# Load the full dataset
print(f"Loading full dataset from {full_dataset_path}...")
df = pd.read_csv(full_dataset_path)
print(f"Loaded data with {len(df)} rows")
print(f"Columns: {df.columns.tolist()}")

# Create directory for results
os.makedirs("full_analysis_results", exist_ok=True)

# Display dataset summary
print("\nDataset Summary:")
print(f"Number of subjects: {df['Subject'].nunique()}")
print(f"Activities: {sorted(df['Activity'].unique())}")
print(f"Axes: {sorted(df['Axis'].unique())}")
print(f"Dimensions: {sorted(df['Dimension'].unique())}")
print(f"Delays: {sorted(df['Delay'].unique())}")
print(f"Signal lengths: {sorted(df['Signal length'].unique())}")

# 1. Create heatmap of PE values by dimension and delay
print("\nCreating heatmaps for PE and Complexity by dimension and delay...")

# Group by dimension and delay and calculate mean PE and Complexity
heatmap_data = df.groupby(['Dimension', 'Delay'])[['Permutation entropy', 'Complexity']].mean().reset_index()
heatmap_pe = heatmap_data.pivot(index='Dimension', columns='Delay', values='Permutation entropy')
heatmap_complexity = heatmap_data.pivot(index='Dimension', columns='Delay', values='Complexity')

# Create PE heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(heatmap_pe, annot=True, cmap='viridis', fmt='.4f')
plt.title('Mean Permutation Entropy by Dimension and Delay')
plt.savefig("full_analysis_results/PE_heatmap_by_dim_delay.png")
plt.close()

# Create Complexity heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(heatmap_complexity, annot=True, cmap='viridis', fmt='.4f')
plt.title('Mean Complexity by Dimension and Delay')
plt.savefig("full_analysis_results/Complexity_heatmap_by_dim_delay.png")
plt.close()

# 2. Analysis by activity for dimension=3, delay=1 (most common parameters)
print("\nAnalyzing activities with dimension=3, delay=1...")

# Filter for dimension=3, delay=1
filtered_df = df[(df['Dimension'] == 3) & (df['Delay'] == 1)]

# Group by Activity and calculate statistics
activity_stats = filtered_df.groupby('Activity').agg({
    'Permutation entropy': ['mean', 'std'],
    'Complexity': ['mean', 'std']
}).reset_index()

print(activity_stats)

# Create activity comparison plot
plt.figure(figsize=(12, 6))
activities = filtered_df['Activity'].unique()
x = np.arange(len(activities))
width = 0.35

pe_means = [filtered_df[filtered_df['Activity'] == act]['Permutation entropy'].mean() for act in activities]
pe_std = [filtered_df[filtered_df['Activity'] == act]['Permutation entropy'].std() for act in activities]

complexity_means = [filtered_df[filtered_df['Activity'] == act]['Complexity'].mean() for act in activities]
complexity_std = [filtered_df[filtered_df['Activity'] == act]['Complexity'].std() for act in activities]

plt.bar(x - width/2, pe_means, width, label='Permutation Entropy', yerr=pe_std, capsize=5)
plt.bar(x + width/2, complexity_means, width, label='Complexity', yerr=complexity_std, capsize=5)

plt.xlabel('Activity')
plt.ylabel('Value')
plt.title('Permutation Entropy and Complexity by Activity (Dimension=3, Delay=1)')
plt.xticks(x, activities)
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("full_analysis_results/activity_comparison_dim3_delay1.png")
plt.close()

# 3. Analysis by axis for dimension=3, delay=1
print("\nAnalyzing axes with dimension=3, delay=1...")

# Group by Axis and calculate statistics
axis_stats = filtered_df.groupby(['Activity', 'Axis']).agg({
    'Permutation entropy': ['mean', 'std'],
    'Complexity': ['mean', 'std']
}).reset_index()

print(axis_stats)

# Create axis comparison plot
plt.figure(figsize=(14, 8))

# Get unique activities and axes
activities = filtered_df['Activity'].unique()
axes = filtered_df['Axis'].unique()

# Create subplot for each activity
fig, axs = plt.subplots(1, len(activities), figsize=(20, 6), sharey=True)
fig.suptitle('Permutation Entropy by Axis for Each Activity (Dimension=3, Delay=1)')

for i, activity in enumerate(activities):
    activity_data = filtered_df[filtered_df['Activity'] == activity]
    
    # Group by axis
    axis_means = [activity_data[activity_data['Axis'] == axis]['Permutation entropy'].mean() for axis in axes]
    axis_std = [activity_data[activity_data['Axis'] == axis]['Permutation entropy'].std() for axis in axes]
    
    axs[i].bar(axes, axis_means, yerr=axis_std, capsize=5)
    axs[i].set_title(activity)
    axs[i].set_xlabel('Axis')
    if i == 0:
        axs[i].set_ylabel('Permutation Entropy')
    axs[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("full_analysis_results/axis_comparison_dim3_delay1.png")
plt.close()

# 4. ANOVA test for activities
print("\nPerforming ANOVA test between activities...")
activities = filtered_df['Activity'].unique()

# ANOVA for PE
f_stat_pe, p_value_pe = stats.f_oneway(
    *[filtered_df[filtered_df['Activity'] == act]['Permutation entropy'].values for act in activities]
)
print(f"ANOVA for Permutation Entropy: F={f_stat_pe:.4f}, p={p_value_pe:.4f}")

# ANOVA for Complexity
f_stat_complexity, p_value_complexity = stats.f_oneway(
    *[filtered_df[filtered_df['Activity'] == act]['Complexity'].values for act in activities]
)
print(f"ANOVA for Complexity: F={f_stat_complexity:.4f}, p={p_value_complexity:.4f}")

# 5. Create PE vs Complexity scatter plots for each parameter combination
print("\nCreating PE vs Complexity scatter plots for different parameter combinations...")

# Get unique dimensions and delays
dimensions = sorted(df['Dimension'].unique())
delays = sorted(df['Delay'].unique())

# Create scatter plots for each combination
for dim in dimensions:
    for delay in delays:
        # Filter data for this combination
        param_df = df[(df['Dimension'] == dim) & (df['Delay'] == delay)]
        
        if len(param_df) > 0:
            plt.figure(figsize=(12, 10))
            
            for activity in df['Activity'].unique():
                activity_data = param_df[param_df['Activity'] == activity]
                
                plt.scatter(
                    activity_data['Permutation entropy'],
                    activity_data['Complexity'],
                    label=activity,
                    alpha=0.7,
                    s=50
                )
            
            plt.xlabel('Permutation Entropy')
            plt.ylabel('Complexity')
            plt.title(f'PE vs Complexity (Dimension={dim}, Delay={delay})')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            filename = f"full_analysis_results/PE_vs_Complexity_dim{dim}_delay{delay}.png"
            plt.savefig(filename)
            plt.close()
            print(f"  Saved {filename}")

# 6. Create 3D scatter plot with Dimension, Delay, and PE
print("\nCreating 3D visualization of parameter space...")

fig = plt.figure(figsize=(14, 12))
ax = fig.add_subplot(111, projection='3d')

# Get mean values for each combination
mean_data = df.groupby(['Dimension', 'Delay', 'Activity']).agg({
    'Permutation entropy': 'mean',
    'Complexity': 'mean'
}).reset_index()

# Create color map for activities
activities = mean_data['Activity'].unique()
colors = ['b', 'r', 'g', 'purple']
activity_colors = dict(zip(activities, colors))

# Plot each activity
for activity in activities:
    activity_data = mean_data[mean_data['Activity'] == activity]
    
    ax.scatter(
        activity_data['Dimension'],
        activity_data['Delay'],
        activity_data['Permutation entropy'],
        label=activity,
        alpha=0.7,
        s=100,
        c=activity_colors[activity]
    )

ax.set_xlabel('Dimension')
ax.set_ylabel('Delay')
ax.set_zlabel('Permutation Entropy')
ax.set_title('Permutation Entropy in Parameter Space by Activity')
ax.legend()

plt.savefig("full_analysis_results/3D_parameter_space.png")
plt.close()
print("  Saved 3D parameter space visualization")

# 7. Summary analysis
print("\nCreating overall summary report...")

# Create summary file
with open("full_analysis_results/analysis_summary.txt", "w") as summary_file:
    summary_file.write("=== Human Activity Analysis using Permutation Entropy ===\n\n")
    
    summary_file.write("Dataset Summary:\n")
    summary_file.write(f"Total data points: {len(df)}\n")
    summary_file.write(f"Subjects: {df['Subject'].nunique()}\n")
    summary_file.write(f"Activities: {', '.join(sorted(df['Activity'].unique()))}\n")
    summary_file.write(f"Axes: {', '.join(sorted(df['Axis'].unique()))}\n")
    summary_file.write(f"Dimensions: {', '.join(map(str, sorted(df['Dimension'].unique())))}\n")
    summary_file.write(f"Delays: {', '.join(map(str, sorted(df['Delay'].unique())))}\n")
    summary_file.write(f"Signal lengths: {', '.join(map(str, sorted(df['Signal length'].unique())))}\n\n")
    
    summary_file.write("Key Findings:\n")
    
    # Optimal parameters
    optimal_params = df.groupby(['Dimension', 'Delay']).apply(
        lambda x: stats.f_oneway(
            *[x[x['Activity'] == act]['Permutation entropy'].values for act in x['Activity'].unique()]
        )[0]  # F-statistic
    ).reset_index(name='F_statistic')
    
    best_params = optimal_params.loc[optimal_params['F_statistic'].idxmax()]
    
    summary_file.write(f"1. Optimal parameters for distinguishing activities:\n")
    summary_file.write(f"   Dimension: {best_params['Dimension']}, Delay: {best_params['Delay']}\n")
    summary_file.write(f"   (F-statistic: {best_params['F_statistic']:.4f})\n\n")
    
    # Activity differences
    summary_file.write(f"2. Activity discrimination:\n")
    summary_file.write(f"   ANOVA for PE: F={f_stat_pe:.4f}, p={p_value_pe:.4f}\n")
    summary_file.write(f"   ANOVA for Complexity: F={f_stat_complexity:.4f}, p={p_value_complexity:.4f}\n\n")
    
    # Best axis
    axis_f_stats = {}
    for axis in df['Axis'].unique():
        axis_data = filtered_df[filtered_df['Axis'] == axis]
        f_stat, _ = stats.f_oneway(
            *[axis_data[axis_data['Activity'] == act]['Permutation entropy'].values for act in activities]
        )
        axis_f_stats[axis] = f_stat
    
    best_axis = max(axis_f_stats, key=axis_f_stats.get)
    
    summary_file.write(f"3. Best axis for activity discrimination: {best_axis}\n")
    summary_file.write(f"   (F-statistic: {axis_f_stats[best_axis]:.4f})\n\n")
    
    # Activity characteristics
    summary_file.write("4. Activity characteristics (Dimension=3, Delay=1):\n")
    for activity in activities:
        pe_mean = filtered_df[filtered_df['Activity'] == activity]['Permutation entropy'].mean()
        complexity_mean = filtered_df[filtered_df['Activity'] == activity]['Complexity'].mean()
        summary_file.write(f"   {activity}: PE={pe_mean:.4f}, Complexity={complexity_mean:.4f}\n")

print("\nAnalysis complete! Results saved to full_analysis_results/ directory")
