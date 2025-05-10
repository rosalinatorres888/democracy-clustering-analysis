
import os
import pandas as pd
import glob

print("=== Looking for Larger Datasets ===")

# Find all CSV files
csv_files = glob.glob("**/*.csv", recursive=True)

print(f"Found {len(csv_files)} CSV files:")
for file in csv_files:
    try:
        # Get file size
        file_size = os.path.getsize(file) / 1024  # Size in KB
        
        # Try to read the file and count rows
        try:
            df = pd.read_csv(file)
            row_count = len(df)
            col_count = len(df.columns)
        except Exception as e:
            row_count = "Error reading"
            col_count = "Error reading"
        
        print(f"  - {file}: {file_size:.1f} KB, {row_count} rows, {col_count} columns")
    except Exception as e:
        print(f"  - {file}: Error getting size - {e}")

# Look for any files with close to 6480 rows
large_files = []
for file in csv_files:
    try:
        df = pd.read_csv(file)
        if len(df) > 1000:  # Looking for files with over 1000 rows
            large_files.append((file, len(df)))
    except:
        pass

if large_files:
    print("\nLarger files that might contain your full dataset:")
    for file, rows in large_files:
        print(f"  - {file}: {rows} rows")
else:
    print("\nNo files found with over 1000 rows")

# Check directories for potential data sources
print("\nChecking directories for potential data sources:")
for root, dirs, files in os.walk('.', topdown=True):
    for dirname in dirs:
        dir_path = os.path.join(root, dirname)
        try:
            dir_files = os.listdir(dir_path)
            file_count = len(dir_files)
            print(f"  - {dir_path}: {file_count} files")
        except:
            print(f"  - {dir_path}: Error reading directory")
