#!/usr/bin/env python3
"""
Main script to run complete flow formula validation.

This script:
1. Downloads videos from ThingsBoard/Google Drive
2. Processes each video to calculate flow
3. Compares with SENAMHI reference data
4. Generates comprehensive statistical analysis and visualizations
"""

import argparse
import logging
import sys
from pathlib import Path

# Add the awive package to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from awive.algorithms.flow_analysis import FlowValidationAnalyzer
from awive.algorithms.validate_formula import validate_flow_formula


def main():
    """Main function to run the validation pipeline."""
    parser = argparse.ArgumentParser(
        description="Complete flow formula validation pipeline"
    )
    parser.add_argument(
        "--start-date",
        "-s",
        default="2025-04-29 00:00:00",
        help="Start date for validation (YYYY-MM-DD HH:MM:SS)",
    )
    parser.add_argument(
        "--end-date",
        "-e",
        default="2025-05-13 00:00:00",
        help="End date for validation (YYYY-MM-DD HH:MM:SS)",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default="validation_output",
        help="Output directory for all results",
    )
    parser.add_argument(
        "--skip-processing",
        "-k",
        action="store_true",
        help="Skip video processing and use existing CSV file",
    )
    parser.add_argument(
        "--csv-file",
        "-c",
        help="Existing CSV file to analyze (when skipping processing)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    csv_file = None

    if not args.skip_processing:
        print("=" * 60)
        print("STEP 1: PROCESSING VIDEOS AND CALCULATING FLOWS")
        print("=" * 60)

        # Run the validation process
        csv_file = output_dir / "flow_validation_results.csv"

        try:
            results = validate_flow_formula(
                start_date=args.start_date,
                end_date=args.end_date,
                output_file=str(csv_file),
            )

            if not results:
                print(
                    "❌ No validation results obtained. Check your ThingsBoard connection and date range."
                )
                return 1

            print(f"✅ Successfully processed {len(results)} videos")

        except Exception as e:
            print(f"❌ Error during video processing: {e}")
            logging.exception("Video processing failed")
            return 1

    else:
        if args.csv_file:
            csv_file = Path(args.csv_file)
        else:
            csv_file = output_dir / "flow_validation_results.csv"

        if not csv_file.exists():
            print(f"❌ CSV file not found: {csv_file}")
            return 1

    print("\n" + "=" * 60)
    print("STEP 2: STATISTICAL ANALYSIS AND VISUALIZATION")
    print("=" * 60)

    try:
        # Create analyzer
        analyzer = FlowValidationAnalyzer(csv_file)

        # Generate plots
        plots_dir = output_dir / "plots"
        analyzer.create_visualizations(plots_dir)
        print(f"✅ Plots generated in {plots_dir}")

        # Generate report
        report_file = output_dir / "validation_report.txt"
        analyzer.generate_report(report_file)
        print(f"✅ Report generated: {report_file}")

        # Print key statistics
        stats = analyzer.calculate_statistics()
        print("\n" + "=" * 40)
        print("KEY VALIDATION METRICS")
        print("=" * 40)
        print(f"Sample Size:       {stats['sample_size']} videos")
        print(f"R-squared:         {stats['r2']:.3f}")
        print(f"RMSE:              {stats['rmse']:.3f} m³/s")
        print(f"MAPE:              {stats['mape']:.2f}%")
        print(f"Bias:              {stats['bias']:.3f} m³/s")
        print(f"Nash-Sutcliffe:    {stats['nash_sutcliffe_efficiency']:.3f}")
        print(f"Pearson r:         {stats['pearson_correlation']:.3f}")
        print(f"Spearman r:        {stats['spearman_correlation']:.3f}")

        # Interpretation
        print("\n" + "=" * 40)
        print("INTERPRETATION")
        print("=" * 40)

        if stats["r2"] > 0.8:
            print(
                "🟢 Strong correlation between calculated and reference flows"
            )
        elif stats["r2"] > 0.6:
            print(
                "🟡 Moderate correlation between calculated and reference flows"
            )
        else:
            print("🔴 Weak correlation between calculated and reference flows")

        if stats["nash_sutcliffe_efficiency"] > 0.75:
            print("🟢 Excellent model performance (NSE > 0.75)")
        elif stats["nash_sutcliffe_efficiency"] > 0.5:
            print("🟡 Good model performance (NSE > 0.5)")
        else:
            print("🔴 Poor model performance (NSE < 0.5)")

        if stats["mape"] < 10:
            print("🟢 Low prediction error (MAPE < 10%)")
        elif stats["mape"] < 25:
            print("🟡 Moderate prediction error (MAPE < 25%)")
        else:
            print("🔴 High prediction error (MAPE > 25%)")

        print(f"\n✅ Complete validation analysis saved to: {output_dir}")

        return 0

    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        logging.exception("Analysis failed")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
