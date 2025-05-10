
import os

print("=== Directory Check ===")
print(f"Current directory: {os.getcwd()}")
print("\nFiles and folders in this directory:")
for item in os.listdir():
    print(f"  - {item}")

print("\nAccelerometer CSV files:")
acc_files = [f for f in os.listdir() if f.endswith('_chest.csv')]
print(f"Found {len(acc_files)} accelerometer files")
if acc_files:
    print("First 5 accelerometer files:")
    for f in acc_files[:5]:
        print(f"  - {f}")
