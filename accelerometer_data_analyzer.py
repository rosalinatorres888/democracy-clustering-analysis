#!/usr/bin/env python3
"""
accelerometer_data_analyzer.py - A script for analyzing accelerometer data

This script reads accelerometer data from CSV files, performs analysis,
and visualizes the results.

Usage:
    python accelerometer_data_analyzer.py input_file.csv [options]

Options:
    --output FILE       Output file for analysis results (default: output.csv)
    --visualize         Generate visualization plots
    --threshold VAL     Threshold value for peak detection (default: 1.5)
    --filter            Apply smoothing filter to the data
    --help              Show this help message
"""

import sys
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Analyze accelerometer data.')
    parser.add_argument('input_file', help='Input CSV file with accelerometer data')
    parser.add_argument('--output', default='output.csv', help='Output file for analysis results')
    parser.add_argument('--visualize', action='store_true', help='Generate visualization plots')
    parser.add_argument('--threshold', type=float, default=1.5, help='Threshold for peak detection')
    parser.add_argument('--filter', action='store_true', help='Apply smoothing filter to the data')
    return parser.parse_args()


def load_data(file_path):
    """Load accelerometer data from CSV file."""
    try:
        print(f"Loading data from {file_path}...")
        df = pd.read_csv(file_path)
        
        # Check for required columns
        required_columns = ['timestamp', 'x', 'y', 'z']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"Error: Missing required columns: {', '.join(missing_columns)}")
            print("Expected columns: timestamp, x, y, z")
            print(f"Found columns: {', '.join(df.columns)}")
            sys.exit(1)
            
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)


def preprocess_data(df, apply_filter=False):
    """Preprocess the accelerometer data."""
    # Calculate magnitude
    df['magnitude'] = np.sqrt(df['x']**2 + df['y']**2 + df['z']**2)
    
    # Apply smoothing filter if requested
    if apply_filter:
        print("Applying smoothing filter...")
        # Apply a Savitzky-Golay filter
        df['x_filtered'] = signal.savgol_filter(df['x'], window_length=15, polyorder=2)
        df['y_filtered'] = signal.savgol_filter(df['y'], window_length=15, polyorder=2)
        df['z_filtered'] = signal.savgol_filter(df['z'], window_length=15, polyorder=2)
        df['magnitude_filtered'] = signal.savgol_filter(df['magnitude'], window_length=15, polyorder=2)
    
    return df


def analyze_data(df, threshold):
    """Analyze the accelerometer data."""
    results = {}
    
    # Basic statistics
    results['mean_x'] = df['x'].mean()
    results['mean_y'] = df['y'].mean()
    results['mean_z'] = df['z'].mean()
    results['std_x'] = df['x'].std()
    results['std_y'] = df['y'].std()
    results['std_z'] = df['z'].std()
    
    # Calculate peaks
    column = 'magnitude_filtered' if 'magnitude_filtered' in df.columns else 'magnitude'
    peaks, _ = signal.find_peaks(df[column], height=threshold)
    results['num_peaks'] = len(peaks)
    results['peak_indices'] = peaks.tolist()
    
    # Calculate frequency domain features using FFT
    if 'timestamp' in df.columns and len(df) > 1:
        time_diff = np.diff(df['timestamp']).mean()
        if time_diff > 0:
            sampling_rate = 1 / time_diff
            
            # Apply FFT to magnitude
            fft_result = np.abs(np.fft.rfft(df[column]))
            fft_freq = np.fft.rfftfreq(len(df[column]), 1/sampling_rate)
            
            # Find dominant frequency
            dominant_idx = np.argmax(fft_result[1:]) + 1  # Skip DC component
            results['dominant_frequency'] = fft_freq[dominant_idx]
            results['fft_results'] = list(zip(fft_freq, fft_result))
    
    return results


def visualize_data(df, results, output_prefix):
    """Generate visualization plots."""
    print("Generating visualizations...")
    
    # Time domain plot
    plt.figure(figsize=(12, 8))
    
    # Plot raw data
    plt.subplot(2, 1, 1)
    plt.plot(df['timestamp'], df['x'], 'r-', label='X', alpha=0.5)
    plt.plot(df['timestamp'], df['y'], 'g-', label='Y', alpha=0.5)
    plt.plot(df['timestamp'], df['z'], 'b-', label='Z', alpha=0.5)
    plt.plot(df['timestamp'], df['magnitude'], 'k-', label='Magnitude', linewidth=1.5)
    
    # If filtered data exists, plot it
    if 'x_filtered' in df.columns:
        plt.plot(df['timestamp'], df['magnitude_filtered'], 'k--', label='Filtered Magnitude', linewidth=1.5)
    
    # Mark peaks if available
    if 'peak_indices' in results:
        plt.plot(df['timestamp'].iloc[results['peak_indices']], 
                 df['magnitude'].iloc[results['peak_indices']], 
                 'ro', label='Peaks')
    
    plt.xlabel('Time')
    plt.ylabel('Acceleration')
    plt.title('Accelerometer Data')
    plt.legend()
    plt.grid(True)
    
    # FFT plot
    if 'fft_results' in results:
        plt.subplot(2, 1, 2)
        fft_freq, fft_result = zip(*results['fft_results'])
        plt.plot(fft_freq, fft_result)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Amplitude')
        plt.title('Frequency Spectrum')
        plt.grid(True)
        
        # Mark dominant frequency
        if 'dominant_frequency' in results:
            dom_freq = results['dominant_frequency']
            idx = np.abs(np.array(fft_freq) - dom_freq).argmin()
            plt.plot(fft_freq[idx], fft_result[idx], 'ro', 
                     label=f'Dominant: {dom_freq:.2f} Hz')
            plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_plot.png")
    print(f"Saved visualization to {output_prefix}_plot.png")
    plt.close()


def save_results(results, output_file):
    """Save analysis results to a CSV file."""
    print(f"Saving results to {output_file}...")
    
    # Convert results to a format suitable for saving
    save_dict = {k: v for k, v in results.items() if not isinstance(v, list)}
    
    # Save to CSV
    pd.DataFrame([save_dict]).to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")


def main():
    """Main function."""
    args = parse_arguments()
    
    print("\nAccelerometer Data Analyzer")
    print("==========================")
    
    # Load data
    df = load_data(args.input_file)
    
    # Preprocess data
    df = preprocess_data(df, apply_filter=args.filter)
    
    # Analyze data
    results = analyze_data(df, args.threshold)
    
    # Save results
    output_prefix = os.path.splitext(args.output)[0]
    save_results(results, args.output)
    
    # Visualize data if requested
    if args.visualize:
        visualize_data(df, results, output_prefix)
    
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()


