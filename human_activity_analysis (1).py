# # **Human Activity Monitoring: Time Series Feature Extraction**
#
# **Author:** Rosalina Torres  
# **Course:** IE6400 – Data Analytics Engineering, Northeastern University  
# **Term:** Spring 2025  
# **Contact:** 📧 torres.ros@northeastern.edu
# 🔗 [LinkedIn](#) | [GitHub](#)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
from itertools import product
import os
import ordpy  # For computing permutation entropy
from tqdm import tqdm  # For progress tracking

# Set plot style for better visualization
try:
    plt.style.use('seaborn-v0_8-whitegrid')  # For newer matplotlib versions
except:
    try:
        plt.style.use('seaborn-whitegrid')  # For older matplotlib versions
    except:
        print("Default style will be used for plots")

# Set seaborn context for better readability
sns.set_context("notebook", font_scale=1.2)

# Define parameters
subjects = range(1, 16)  # Subjects 1 to 15
axes = ['attr_x', 'attr_y', 'attr_z']  # Accelerometer axes
signal_lengths = [1024, 2048, 4096]  # Signal lengths to analyze
dimensions = [3, 4, 5, 6]  # Embedding dimensions
delays = [1, 2, 3]  # Embedding delays
activities = ['walking', 'running', 'climbingup', 'climbingdown']  # Activities

# ============================================================================
# Task 1: Load the required data into 'df' dataframe
# ============================================================================

def validate_file(file_path):
    """Validate CSV file for proper structure and data types."""
    expected_columns = ['id', 'attr_time', 'attr_x', 'attr_y', 'attr_z']
    expected_dtypes = {
        'id': 'int64',
        'attr_time': 'int64',
        'attr_x': 'float64',
        'attr_y': 'float64',
        'attr_z': 'float64'
    }
    
    try:
        df = pd.read_csv(file_path)
        
        # Check if columns match expected
        if list(df.columns) != expected_columns:
            print(f"❌ Columns mismatch in {file_path}")
            return False, None
            
        # Check for missing values
        if df.isnull().sum().sum() > 0:
            print(f"❌ Missing values found in {file_path}")
            return False, None
            
        # Check data types
        for col, dtype in expected_dtypes.items():
            if df[col].dtype != dtype:
                print(f"❌ Data type mismatch in {file_path}. {col}: {dtype} expected.")
                return False, None
        
        print(f"✅ {file_path} is valid.")
        return True, df
    except Exception as e:
        print(f"❌ Error with file {file_path}: {e}")
        return False, None

def load_all_files():
    """Load all valid accelerometer data files."""
    # Get all CSV files that match the pattern
    file_patterns = [
        "s*_walking_chest.csv", 
        "s*_running_chest.csv",
        "s*_climbingUp_chest.csv", 
        "s*_climbingDown_chest.csv"
    ]
    
    all_files = []
    for pattern in file_patterns:
        all_files.extend(glob.glob(pattern))
    
    print(f"Found {len(all_files)} files matching the patterns.")
    
    # Validate and load all files
    valid_dfs = []
    for file in all_files:
        is_valid, df = validate_file(file)
        if is_valid:
            # Extract subject number and activity from filename
            subject = int(file.split('_')[0][1:])  # Extract number from 's1', 's2', etc.
            if 'walking' in file:
                activity = 'walking'
            elif 'running' in file:
                activity = 'running'
            elif 'climbingUp' in file or 'climbingup' in file:
                activity = 'climbingup'
            elif 'climbingDown' in file or 'climbingdown' in file:
                activity = 'climbingdown'
            else:
                activity = 'unknown'
            
            # Add subject and activity columns to the dataframe
            df['subject'] = subject
            df['activity'] = activity
            valid_dfs.append(df)
    
    if not valid_dfs:
        print("No valid files found!")
        return None
    
    # Combine all dataframes
    combined_df = pd.concat(valid_dfs, ignore_index=True)
    print(f"Successfully loaded data with {combined_df.shape[0]} rows and {combined_df.shape[1]} columns.")
    return combined_df

# Load all files into a single dataframe
df = load_all_files()

# If the data is already processed and available as a CSV, load it instead
def load_processed_data():
    """Load preprocessed data if available."""
    processed_path = 'processed_permutation_entropy_complexity.csv'
    if os.path.exists(processed_path):
        print(f"Loading preprocessed data from {processed_path}")
        return pd.read_csv(processed_path)
    return None

processed_df = load_processed_data()
if processed_df is not None:
    print("Using preprocessed data.")
    df = processed_df

# ============================================================================
# Task 2: Compute Permutation Entropy and Complexity
# ============================================================================

# Helper function for Shannon entropy calculation
def s_entropy(freq_list):
    """Compute Shannon entropy for a frequency distribution.
    
    Parameters:
    -----------
    freq_list : list or array
        List of frequencies (probabilities)
        
    Returns:
    --------
    float
        Shannon entropy value
    """
    # Filter out zero frequencies to avoid log(0)
    freq_list = [f for f in freq_list if f != 0]
    # Calculate Shannon entropy: -sum(p*log(p))
    sh_entropy = -sum(f * np.log(f) for f in freq_list)
    return sh_entropy

# Function for computing permutation entropy
def p_entropy(op):
    """Compute normalized permutation entropy.
    
    Parameters:
    -----------
    op : list or array
        Ordinal pattern frequencies
        
    Returns:
    --------
    float
        Permutation entropy value (normalized between 0 and 1)
    """
    ordinal_pat = op
    # Maximum entropy (for normalization)
    max_entropy = np.log(len(ordinal_pat))
    # Convert frequencies to probabilities
    p = np.divide(np.array(ordinal_pat), float(sum(ordinal_pat)))
    # Calculate normalized permutation entropy
    return s_entropy(p) / max_entropy

# Function for computing complexity (based on Jensen-Shannon Divergence)
def complexity(op):
    """Compute complexity measure based on Jensen-Shannon divergence.
    
    Parameters:
    -----------
    op : list or array
        Ordinal pattern frequencies
        
    Returns:
    --------
    float
        Complexity value
    
    Notes:
    ------
    The complexity is defined as:
    Comp_JS = Q_o * JSdivergence * PE
    where:
    - Q_o is a normalizing constant
    - JSdivergence is the Jensen-Shannon divergence
    - PE is the permutation entropy
    """
    # Calculate permutation entropy
    pe = p_entropy(op)
    
    # Calculate normalizing constant Q_o
    constant1 = (0.5 + ((1 - 0.5) / len(op))) * np.log(0.5 + ((1 - 0.5) / len(op)))
    constant2 = ((1 - 0.5) / len(op)) * np.log((1 - 0.5) / len(op)) * (len(op) - 1)
    constant3 = 0.5 * np.log(len(op))
    Q_o = -1 / (constant1 + constant2 + constant3)

    # Calculate probabilities
    temp_op_prob = np.divide(op, sum(op))
    # Calculate mixture distribution
    temp_op_prob2 = (0.5 * temp_op_prob) + (0.5 * (1 / len(op)))
    
    # Calculate Jensen-Shannon Divergence
    JSdivergence = (s_entropy(temp_op_prob2) - 0.5 * s_entropy(temp_op_prob) - 0.5 * np.log(len(op)))
    
    # Calculate complexity
    Comp_JS = Q_o * JSdivergence * pe
    return Comp_JS

def compute_ordpy_pattern_dist(signal, dimension, delay):
    """Compute ordinal pattern distribution using ordpy.
    
    Parameters:
    -----------
    signal : array-like
        Time series data
    dimension : int
        Embedding dimension
    delay : int
        Embedding delay
        
    Returns:
    --------
    array
        Ordinal pattern distribution
    """
    # Use ordpy to get the ordinal pattern distribution
    pattern_dist = ordpy.ordinal_distribution(signal, dimension, delay)
    return pattern_dist

def compute_pe_complexity(df_data, results_file='pe_complexity_results.csv'):
    """Compute Permutation Entropy and Complexity for all parameter combinations.
    
    Parameters:
    -----------
    df_data : DataFrame
        DataFrame containing the time series data
    results_file : str, optional
        File path to save results, by default 'pe_complexity_results.csv'
        
    Returns:
    --------
    DataFrame
        Results containing PE and Complexity for all combinations
    """
    results = []
    
    # Generate all combinations of parameters
    all_combinations = list(product(
        subjects, activities, axes, signal_lengths, dimensions, delays
    ))
    
    print(f"Computing PE and Complexity for {len(all_combinations)} combinations...")
    
    # Process each combination
    for subject, activity, axis, signal_length, dimension, delay in tqdm(all_combinations):
        try:
            # Filter data for the current combination
            filtered_df = df_data[
                (df_data['subject'] == subject) & 
                (df_data['activity'] == activity)
            ]
            
            if filtered_df.empty:
                print(f"No data for Subject={subject}, Activity={activity}")
                continue
                
            # Get the signal data for the specified axis
            signal = filtered_df[axis].values[:signal_length]
            
            # If signal is shorter than required length, skip
            if len(signal) < signal_length:
                print(f"Signal too short: {len(signal)} < {signal_length}")
                continue
                
            # Compute ordinal pattern distribution
            pattern_dist = compute_ordpy_pattern_dist(signal, dimension, delay)
            
            # Compute PE and Complexity
            pe_value = p_entropy(pattern_dist)
            complexity_value = complexity(pattern_dist)
            
            # Store the results
            results.append({
                'Subject': subject,
                'Activity': activity,
                'Accelerometer axis': axis,
                'Signal length': signal_length,
                'Dimension': dimension,
                'Delay': delay,
                'Permutation entropy': pe_value,
                'Complexity': complexity_value
            })
            
            print(f"Processed: PE={pe_value:.5f}, Complexity={complexity_value:.5f}")
            
        except Exception as e:
            print(f"Error processing combination: {subject}, {activity}, {axis}, {signal_length}, {dimension}, {delay}")
            print(f"Error details: {e}")
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results to CSV
    if results_file:
        results_df.to_csv(results_file, index=False)
        print(f"Results saved to {results_file}")
    
    return results_df

# Check if we need to compute PE and Complexity or if we have it already
if 'Permutation entropy' in df.columns and 'Complexity' in df.columns:
    print("PE and Complexity already computed.")
    df_results = df
else:
    print("Computing PE and Complexity for all combinations...")
    df_results = compute_pe_complexity(df)

# ============================================================================
# Task 3: Filter Data for a Specific Subject, Axis, and Signal Length
# ============================================================================

# Select a specific subject, axis, and signal length
selected_subject = 3
selected_axis = 'attr_x'
selected_signal_length = 2048

# Filter the data
filtered_data = df_results[
    (df_results['Subject'] == selected_subject) &
    (df_results['Accelerometer axis'] == selected_axis) &
    (df_results['Signal length'] == selected_signal_length)
]

print(f"\nFiltered data for Subject {selected_subject}, Axis {selected_axis}, Signal Length {selected_signal_length}:")
print(filtered_data.head())

# ============================================================================
# Task 4: Identify Optimum Dimension and Delay for Walking vs Running
# ============================================================================

def create_pe_complexity_scatter(data, activities, title):
    """Create scatter plot of PE vs Complexity for the given activities.
    
    Parameters:
    -----------
    data : DataFrame
        Filtered data
    activities : list
        List of activities to compare
    title : str
        Plot title
    """
    plt.figure(figsize=(14, 10))
    
    # Filter for the specified activities
    activity_data = data[data['Activity'].isin(activities)]
    
    # Create subplots for each dimension-delay combination
    dim_delay_combinations = activity_data[['Dimension', 'Delay']].drop_duplicates()
    
    # Calculate the grid size
    n_combinations = len(dim_delay_combinations)
    n_cols = min(3, n_combinations)
    n_rows = (n_combinations + n_cols - 1) // n_cols
    
    # Create the scatter plots
    for i, (_, row) in enumerate(dim_delay_combinations.iterrows()):
        dim, delay = row['Dimension'], row['Delay']
        
        # Create subplot
        plt.subplot(n_rows, n_cols, i + 1)
        
        # Filter data for current dimension and delay
        current_data = activity_data[
            (activity_data['Dimension'] == dim) &
            (activity_data['Delay'] == delay)
        ]
        
        # Create scatter plot
        for activity in activities:
            act_data = current_data[current_data['Activity'] == activity]
            plt.scatter(
                act_data['Permutation entropy'],
                act_data['Complexity'],
                label=activity,
                alpha=0.7,
                s=100
            )
        
        # Add labels and legend
        plt.xlabel('Permutation Entropy')
        plt.ylabel('Complexity')
        plt.title(f'Dimension={dim}, Delay={delay}')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    return plt

# Create scatter plot for walking vs running
walking_vs_running_plot = create_pe_complexity_scatter(
    filtered_data,
    ['walking', 'running'],
    f'Permutation Entropy vs Complexity: Walking vs Running\nSubject {selected_subject}, Axis {selected_axis}, Signal Length {selected_signal_length}'
)

# Determine the optimum dimension and delay for walking vs running
def find_optimal_parameters(data, activities):
    """Find the optimal dimension and delay for separating activities.
    
    Parameters:
    -----------
    data : DataFrame
        Filtered data
    activities : list
        List of activities to compare
        
    Returns:
    --------
    tuple
        (optimal_dimension, optimal_delay, separation_score)
    """
    activity_data = data[data['Activity'].isin(activities)]
    
    best_separation = 0
    optimal_dim = None
    optimal_delay = None
    
    # Check each dimension-delay combination
    for dim, delay in product(dimensions, delays):
        current_data = activity_data[
            (activity_data['Dimension'] == dim) &
            (activity_data['Delay'] == delay)
        ]
        
        if len(current_data) < 2:
            continue
            
        # Group by activity
        grouped = current_data.groupby('Activity')
        
        # If not all activities are present, skip
        if len(grouped) < len(activities):
            continue
            
        # Calculate centroids for each activity
        centroids = grouped[['Permutation entropy', 'Complexity']].mean()
        
        # Calculate pairwise distances between centroids
        distances = []
        for i, act1 in enumerate(activities):
            for j, act2 in enumerate(activities[i+1:], i+1):
                if act1 in centroids.index and act2 in centroids.index:
                    # Euclidean distance between centroids
                    dist = np.sqrt(
                        (centroids.loc[act1, 'Permutation entropy'] - centroids.loc[act2, 'Permutation entropy'])**2 +
                        (centroids.loc[act1, 'Complexity'] - centroids.loc[act2, 'Complexity'])**2
                    )
                    distances.append(dist)
        
        # Average distance as separation score
        if distances:
            avg_distance = np.mean(distances)
            if avg_distance > best_separation:
                best_separation = avg_distance
                optimal_dim = dim
                optimal_delay = delay
    
    return optimal_dim, optimal_delay, best_separation

# Find optimal parameters for walking vs running
optimal_dim_wr, optimal_delay_wr, separation_wr = find_optimal_parameters(
    filtered_data, ['walking', 'running']
)

print(f"\nOptimal parameters for separating walking vs running:")
print(f"Dimension: {optimal_dim_wr}, Delay: {optimal_delay_wr}, Separation Score: {separation_wr:.5f}")

# ============================================================================
# Task 5: Identify Optimum Dimension and Delay for Climbing Up vs Climbing Down
# ============================================================================

# Create scatter plot for climbing up vs climbing down
climbing_plot = create_pe_complexity_scatter(
    filtered_data,
    ['climbingup', 'climbingdown'],
    f'Permutation Entropy vs Complexity: Climbing Up vs Climbing Down\nSubject {selected_subject}, Axis {selected_axis}, Signal Length {selected_signal_length}'
)

# Find optimal parameters for climbing up vs climbing down
optimal_dim_cd, optimal_delay_cd, separation_cd = find_optimal_parameters(
    filtered_data, ['climbingup', 'climbingdown']
)

print(f"\nOptimal parameters for separating climbing up vs climbing down:")
print(f"Dimension: {optimal_dim_cd}, Delay: {optimal_delay_cd}, Separation Score: {separation_cd:.5f}")

# ============================================================================
# Additional Analysis: Compare All Activities
# ============================================================================

# Create scatter plot for all activities
all_activities_plot = create_pe_complexity_scatter(
    filtered_data,
    activities,
    f'Permutation Entropy vs Complexity: All Activities\nSubject {selected_subject}, Axis {selected_axis}, Signal Length {selected_signal_length}'
)

# ============================================================================
# Summary of Results
# ============================================================================

print("\n===================== SUMMARY OF RESULTS =====================")
print(f"Total data points analyzed: {len(df_results)}")
print(f"Parameters varied:")
print(f"  - Subjects: 1 to 15")
print(f"  - Activities: {activities}")
print(f"  - Accelerometer axes: {axes}")
print(f"  - Signal lengths: {signal_lengths}")
print(f"  - Embedding dimensions: {dimensions}")
print(f"  - Embedding delays: {delays}")
print("\nOptimal parameters for activity separation:")
print(f"  - Walking vs Running: Dimension={optimal_dim_wr}, Delay={optimal_delay_wr}")
print(f"  - Climbing Up vs Climbing Down: Dimension={optimal_dim_cd}, Delay={optimal_delay_cd}")
print("==============================================================")

# Save results to file
summary_file = 'activity_analysis_summary.txt'
with open(summary_file, 'w') as f:
    f.write("===================== SUMMARY OF RESULTS =====================\n")
    f.write(f"Total data points analyzed: {len(df_results)}\n")
    f.write(f"Parameters varied:\n")
    f.write(f"  - Subjects: 1 to 15\n")
    f.write(f"  - Activities: {activities}\n")
    f.write(f"  - Accelerometer axes: {axes}\n")
    f.write(f"  - Signal lengths: {signal_lengths}\n")
    f.write(f"  - Embedding dimensions: {dimensions}\n")
    f.write(f"  - Embedding delays: {delays}\n")
    f.write("\nOptimal parameters for activity separation:\n")
    f.write(f"  - Walking vs Running: Dimension={optimal_dim_wr}, Delay={optimal_delay_wr}\n")
    f.write(f"  - Climbing Up vs Climbing Down: Dimension={optimal_dim_cd}, Delay={optimal_delay_cd}\n")
    f.write("==============================================================\n")

print(f"Summary saved to {summary_file}")
