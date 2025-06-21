""" A simple script that profiles a baseline, uses nsys to profile run_baseline.py
"""

import subprocess
import pandas as pd
import os
import tempfile
import argparse
from io import StringIO  # Add this import

KERNEL_NAME_MAP = {
    'sdpa': 'pytorch_flash::flash_fwd_kernel',
}

def run_nsys_profile(provider, mode, b, t, h, d, dtype):
    """Run nsys profile on run_baseline.py with given parameters
    
    Args:
        provider: One of 'sdpa', 'fla', 'power'
        mode: One of 'fwd', 'bwd', 'fwd+bwd'
        b: batch size
        t: sequence length
        h: number of heads
        d: head dimension
        dtype: data type ('float16' or 'bfloat16')
    
    Returns:
        pandas DataFrame with kernel statistics
    """
    # Create a temporary file for the nsys report
    with tempfile.NamedTemporaryFile(suffix='.nsys-rep', delete=False) as report_file:
        # Build the nsys command
        nsys_cmd = [
            'nsys', 'profile',
            '--force-overwrite=true',
            f'--output={report_file.name}',
            'python',
            'run_baseline.py',
            f'--provider={provider}',
            f'--mode={mode}',
            f'--b={b}',
            f'--t={t}',
            f'--h={h}',
            f'--d={d}',
            f'--dtype={dtype}'
        ]
        
        # Run the nsys profile command
        try:
            subprocess.run(nsys_cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            print(f"Error running nsys profile: {e}")
            print(f"stdout: {e.stdout.decode()}")
            print(f"stderr: {e.stderr.decode()}")
            return None

        # Run nsys stats to get kernel summary
        stats_cmd = [
            'nsys', 'stats',
            '--report', 'gpukernsum',
            '--format', 'csv',
            report_file.name
        ]
        
        try:
            result = subprocess.run(stats_cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            print(f"Error running nsys stats: {e}")
            print(f"stdout: {e.stdout}")
            print(f"stderr: {e.stderr}")
            return None

        # Parse the CSV output
        # Skip the first few lines of output that aren't CSV data
        csv_lines = []
        capture = False
        for line in result.stdout.split('\n'):
            if line.startswith('Time (%)'):
                capture = True
            if capture and line.strip():
                csv_lines.append(line)
        
        # Convert to DataFrame
        if len(csv_lines) > 1:  # Need at least header + one data row
            df = pd.read_csv(StringIO('\n'.join(csv_lines)))
            
            # Convert nanoseconds to milliseconds for time columns
            time_columns = ['Total Time (ns)', 'Avg (ns)', 'Med (ns)', 'Min (ns)', 'Max (ns)', 'StdDev (ns)']
            for col in time_columns:
                if col in df.columns:
                    new_col = col.replace('(ns)', '(ms)')
                    df[new_col] = df[col] / 1e6  # Convert ns to ms
                    df = df.drop(columns=[col])
            
            # Filter rows based on kernel name if provider exists in KERNEL_NAME_MAP
            if provider in KERNEL_NAME_MAP:
                kernel_name = KERNEL_NAME_MAP[provider]
                df = df[df['Name'].str.contains(kernel_name, regex=True, na=False)]
            
            # Add profiling parameters as metadata
            df['provider'] = provider
            return df
        else:
            print("No kernel data found in nsys output")
            return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--provider', type=str, default='sdpa', choices=['sdpa', 'fla', 'power'])
    parser.add_argument('--mode', type=str, default='fwd', choices=['fwd', 'bwd', 'fwd+bwd'])
    parser.add_argument('--b', type=int, default=2)
    parser.add_argument('--t', type=int, default=1024)
    parser.add_argument('--h', type=int, default=16)
    parser.add_argument('--d', type=int, default=64)
    parser.add_argument('--dtype', type=str, default='bfloat16', choices=['float16', 'bfloat16'])
    parser.add_argument('--output', type=str, help='Output CSV file to save results')
    
    args = parser.parse_args()
    
    # Run profiling
    df = run_nsys_profile(
        args.provider,
        args.mode,
        args.b,
        args.t,
        args.h,
        args.d,
        args.dtype
    )
    
    if df is not None:
        print("\nKernel Statistics:")
        print(df)
        
        if args.output:
            df.to_csv(args.output, index=False)
            print(f"\nResults saved to {args.output}")

if __name__ == '__main__':
    main()
