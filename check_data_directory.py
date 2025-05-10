
import os

print("=== Data Directory Check ===")
print(f"Current directory: {os.getcwd()}")

# Check the data directory
data_dir = "data"
if os.path.exists(data_dir) and os.path.isdir(data_dir):
    print(f"\nContents of {data_dir} directory:")
    data_files = os.listdir(data_dir)
    
    # Show first 10 files in data directory
    for i, item in enumerate(data_files[:10]):
        print(f"  - {item}")
    
    if len(data_files) > 10:
        print(f"  ... and {len(data_files) - 10} more files")
    
    # Look for chest accelerometer files
    chest_files = [f for f in data_files if f.endswith('_chest.csv')]
    print(f"\nFound {len(chest_files)} chest accelerometer files in data directory")
    if chest_files:
        print("First 5 chest accelerometer files:")
        for f in chest_files[:5]:
            print(f"  - {f}")
else:
    print(f"The {data_dir} directory doesn't exist or is not accessible")
