
import os
import glob

# Find all CSV files in current directory and subdirectories
csv_files = glob.glob("**/*.csv", recursive=True)

print("=== All CSV Files ===")
print(f"Found {len(csv_files)} CSV files:")
for file in csv_files:
    print(f"  - {file}")
    
    # Try to check the first few lines of each CSV to understand its content
    try:
        with open(file, 'r') as f:
            first_line = f.readline().strip()
            print(f"    First line: {first_line}")
    except Exception as e:
        print(f"    Error reading file: {e}")
