
import pandas as pd
import matplotlib.pyplot as plt
import os

print("=== Examining Processed Data ===")

# Load the processed data file
processed_file = "processed_permutation_entropy_complexity.csv"
if os.path.exists(processed_file):
    df = pd.read_csv(processed_file)
    
    print(f"File loaded: {processed_file}")
    print(f"Number of rows: {len(df)}")
    print(f"Columns: {df.columns.tolist()}")
    
    print("\nSample of the data:")
    print(df.head())
    
    # Check what activities are present
    if 'Activity' in df.columns:
        activities = df['Activity'].unique()
        print(f"\nActivities found: {activities}")
        
        # Check how many samples for each activity
        print("\nSamples per activity:")
        print(df['Activity'].value_counts())
    
    # Check what subjects are present
    if 'Subject' in df.columns:
        subjects = df['Subject'].unique()
        print(f"\nSubjects found: {subjects}")
        print(f"Number of subjects: {len(subjects)}")
    
    # Create a visualization of PE vs Complexity
    if all(col in df.columns for col in ['PE', 'Complexity', 'Activity']):
        print("\nCreating PE vs Complexity visualization...")
        
        plt.figure(figsize=(10, 8))
        for activity in df['Activity'].unique():
            activity_data = df[df['Activity'] == activity]
            plt.scatter(
                activity_data['PE'],
                activity_data['Complexity'],
                label=activity,
                alpha=0.7,
                s=50
            )
        
        plt.xlabel('Permutation Entropy')
        plt.ylabel('Statistical Complexity')
        plt.title('PE vs Complexity for Different Activities')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save the plot
        plt.savefig("PE_vs_Complexity_from_processed_data.png")
        print("Plot saved as: PE_vs_Complexity_from_processed_data.png")
else:
    print(f"File not found: {processed_file}")
