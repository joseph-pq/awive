#!/usr/bin/env python3
"""Extract validation data columns for analysis pipeline.

This script transforms flow validation results into a standardized format
compatible with the optimization pipeline. It extracts only the essential
columns needed for flow formula testing:

Extracted columns:
- timestamp: Unix timestamp in milliseconds
- senamhi_flow: Reference flow measurement (m³/s)
- senamhi_level: Water level measurement (meters)
- velocimetry: Array of velocity measurements across the channel cross-section

Usage:
    python extract_validation_columns.py

Default files:
    Input:  flow_validation_results_3005_1006.csv
    Output: _senamhi_data_3005_1006.csv
"""

import pandas as pd


def extract_matching_columns(input_file, output_file):
    """
    Extract matching columns from flow validation results.

    Args:
        input_file: Path to the flow validation results CSV
        output_file: Path to save the extracted data
    """
    # Read the input file
    print(f"Reading {input_file}...")
    df = pd.read_csv(input_file)

    # Define the columns we need to extract
    required_columns = [
        "timestamp",
        "senamhi_flow",
        "senamhi_level",
        "velocimetry",
    ]

    # Check if all required columns exist
    missing_columns = [
        col for col in required_columns if col not in df.columns
    ]
    if missing_columns:
        print(f"Error: Missing columns in input file: {missing_columns}")
        print(f"Available columns: {df.columns.tolist()}")

    # Extract only the required columns
    df_extracted = df[required_columns].copy()

    # Save to output file
    print(f"Saving extracted data to {output_file}...")
    df_extracted.to_csv(output_file, index=False)

    print(f"Done! Extracted {len(df_extracted)} rows.")
    print(f"Columns: {list(df_extracted.columns)}")

    # Show sample of the data
    print("\nFirst few rows:")
    print(df_extracted.head())


if __name__ == "__main__":
    # Default file paths
    input_file = "flow_validation_results_3005_1006.csv"
    output_file = "_senamhi_data_3005_1006.csv"

    extract_matching_columns(input_file, output_file)
