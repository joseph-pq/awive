"""Statistical analysis and visualization for flow formula validation."""

import csv
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


class FlowValidationAnalyzer:
    """Analyzer for flow formula validation results."""

    def __init__(self, csv_file: str | Path):
        """Initialize the analyzer with validation results.

        Args:
            csv_file: Path to the CSV file with validation results
        """
        self.csv_file = Path(csv_file)
        self.df = None
        self.load_data()

    def load_data(self) -> None:
        """Load and preprocess the validation data."""
        if not self.csv_file.exists():
            raise FileNotFoundError(f"CSV file not found: {self.csv_file}")

        self.df = pd.read_csv(self.csv_file)

        # Convert timestamp to datetime if needed
        if "timestamp_dt" in self.df.columns:
            self.df["timestamp_dt"] = pd.to_datetime(self.df["timestamp_dt"])

        # Remove any rows with missing critical data
        self.df = self.df.dropna(subset=["calculated_flow", "senamhi_flow"])

        print(f"Loaded {len(self.df)} validation records")

    def calculate_statistics(self) -> dict:
        """Calculate comprehensive statistical metrics.

        Returns:
            Dictionary with statistical results
        """
        if self.df is None or len(self.df) == 0:
            raise ValueError("No data available for analysis")

        calculated = self.df["calculated_flow"].values
        senamhi = self.df["senamhi_flow"].values

        # Basic error metrics
        absolute_errors = np.abs(calculated - senamhi)
        relative_errors = np.abs(calculated - senamhi) / senamhi * 100

        # Statistical metrics
        mae = np.mean(absolute_errors)
        rmse = np.sqrt(np.mean((calculated - senamhi) ** 2))
        mape = np.mean(relative_errors)
        bias = np.mean(calculated - senamhi)
        r2 = stats.pearsonr(calculated, senamhi)[0] ** 2

        # Correlation coefficients
        pearson_r, pearson_p = stats.pearsonr(calculated, senamhi)
        spearman_r, spearman_p = stats.spearmanr(calculated, senamhi)

        # Nash-Sutcliffe Efficiency
        nse = 1 - (
            np.sum((senamhi - calculated) ** 2)
            / np.sum((senamhi - np.mean(senamhi)) ** 2)
        )

        # Index of Agreement (Willmott)
        ioa = 1 - (
            np.sum((calculated - senamhi) ** 2)
            / np.sum(
                (
                    np.abs(calculated - np.mean(senamhi))
                    + np.abs(senamhi - np.mean(senamhi))
                )
                ** 2
            )
        )

        # Normality tests for residuals
        residuals = calculated - senamhi
        shapiro_stat, shapiro_p = stats.shapiro(residuals)

        results = {
            "sample_size": len(calculated),
            "mae": mae,
            "rmse": rmse,
            "mape": mape,
            "bias": bias,
            "r2": r2,
            "pearson_correlation": pearson_r,
            "pearson_p_value": pearson_p,
            "spearman_correlation": spearman_r,
            "spearman_p_value": spearman_p,
            "nash_sutcliffe_efficiency": nse,
            "index_of_agreement": ioa,
            "shapiro_wilk_stat": shapiro_stat,
            "shapiro_wilk_p": shapiro_p,
            "mean_calculated": np.mean(calculated),
            "mean_senamhi": np.mean(senamhi),
            "std_calculated": np.std(calculated),
            "std_senamhi": np.std(senamhi),
        }

        return results

    def create_visualizations(self, output_dir: str | Path = "plots") -> None:
        """Create comprehensive visualizations.

        Args:
            output_dir: Directory to save plots
        """
        if self.df is None:
            raise ValueError("No data available for plotting")

        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

        # Set style
        plt.style.use("seaborn-v0_8")
        sns.set_palette("husl")

        # 1. Scatter plot with 1:1 line
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.scatter(
            self.df["senamhi_flow"],
            self.df["calculated_flow"],
            alpha=0.6,
            s=50,
        )

        # Add 1:1 line
        min_val = min(
            self.df["senamhi_flow"].min(), self.df["calculated_flow"].min()
        )
        max_val = max(
            self.df["senamhi_flow"].max(), self.df["calculated_flow"].max()
        )
        ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2)

        # Add regression line
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            self.df["senamhi_flow"], self.df["calculated_flow"]
        )
        line = slope * self.df["senamhi_flow"] + intercept
        ax.plot(
            self.df["senamhi_flow"],
            line,
            "g-",
            linewidth=2,
            label=f"Regression (R²={r_value**2:.3f})",
        )

        ax.set_xlabel("SENAMHI Flow (m³/s)", fontsize=12)
        ax.set_ylabel("Calculated Flow (m³/s)", fontsize=12)
        ax.set_title(
            "Flow Formula Validation: Calculated vs SENAMHI", fontsize=14
        )
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(
            output_dir / "scatter_plot.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

        # 2. Time series comparison
        if "timestamp_dt" in self.df.columns:
            fig, ax = plt.subplots(figsize=(14, 6))
            ax.plot(
                self.df["timestamp_dt"],
                self.df["senamhi_flow"],
                "o-",
                label="SENAMHI",
                linewidth=2,
                markersize=4,
            )
            ax.plot(
                self.df["timestamp_dt"],
                self.df["calculated_flow"],
                "s-",
                label="Calculated",
                linewidth=2,
                markersize=4,
            )
            ax.set_xlabel("Date", fontsize=12)
            ax.set_ylabel("Flow (m³/s)", fontsize=12)
            ax.set_title("Time Series Comparison", fontsize=14)
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(
                output_dir / "time_series.png", dpi=300, bbox_inches="tight"
            )
            plt.close()

        # 3. Residuals analysis
        residuals = self.df["calculated_flow"] - self.df["senamhi_flow"]

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

        # Residuals vs predicted
        ax1.scatter(self.df["senamhi_flow"], residuals, alpha=0.6)
        ax1.axhline(y=0, color="r", linestyle="--")
        ax1.set_xlabel("SENAMHI Flow (m³/s)")
        ax1.set_ylabel("Residuals")
        ax1.set_title("Residuals vs SENAMHI Flow")
        ax1.grid(True, alpha=0.3)

        # Histogram of residuals
        ax2.hist(residuals, bins=20, alpha=0.7, edgecolor="black")
        ax2.set_xlabel("Residuals")
        ax2.set_ylabel("Frequency")
        ax2.set_title("Distribution of Residuals")
        ax2.grid(True, alpha=0.3)

        # Q-Q plot for normality
        stats.probplot(residuals, dist="norm", plot=ax3)
        ax3.set_title("Q-Q Plot (Normality Check)")

        # Box plot of relative errors
        ax4.boxplot(self.df["relative_error"])
        ax4.set_ylabel("Relative Error (%)")
        ax4.set_title("Box Plot of Relative Errors")
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            output_dir / "residuals_analysis.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

        # 4. Error metrics visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Absolute errors
        ax1.plot(
            range(len(self.df)), self.df["absolute_error"], "o-", markersize=4
        )
        ax1.set_xlabel("Sample Index")
        ax1.set_ylabel("Absolute Error (m³/s)")
        ax1.set_title("Absolute Errors Over Time")
        ax1.grid(True, alpha=0.3)

        # Relative errors
        ax2.plot(
            range(len(self.df)), self.df["relative_error"], "o-", markersize=4
        )
        ax2.set_xlabel("Sample Index")
        ax2.set_ylabel("Relative Error (%)")
        ax2.set_title("Relative Errors Over Time")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            output_dir / "error_metrics.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

        print(f"Plots saved to {output_dir}")

    def generate_report(
        self, output_file: str | Path = "validation_report.txt"
    ) -> None:
        """Generate a comprehensive validation report.

        Args:
            output_file: Path to save the report
        """
        stats_results = self.calculate_statistics()

        report = f"""
FLOW FORMULA VALIDATION REPORT
===============================

Dataset Information:
-------------------
• Sample Size: {stats_results["sample_size"]} videos
• Analysis Period: {self.df["timestamp_dt"].min()} to {self.df["timestamp_dt"].max()}

Flow Statistics:
---------------
• SENAMHI Flow: {stats_results["mean_senamhi"]:.3f} ± {stats_results["std_senamhi"]:.3f} m³/s
• Calculated Flow: {stats_results["mean_calculated"]:.3f} ± {stats_results["std_calculated"]:.3f} m³/s
• Bias: {stats_results["bias"]:.3f} m³/s

Error Metrics:
-------------
• Mean Absolute Error (MAE): {stats_results["mae"]:.3f} m³/s
• Root Mean Square Error (RMSE): {stats_results["rmse"]:.3f} m³/s
• Mean Absolute Percentage Error (MAPE): {stats_results["mape"]:.2f}%

Correlation Analysis:
--------------------
• Pearson Correlation: {stats_results["pearson_correlation"]:.3f} (p-value: {stats_results["pearson_p_value"]:.3e})
• Spearman Correlation: {stats_results["spearman_correlation"]:.3f} (p-value: {stats_results["spearman_p_value"]:.3e})
• R-squared: {stats_results["r2"]:.3f}

Model Performance Indices:
-------------------------
• Nash-Sutcliffe Efficiency: {stats_results["nash_sutcliffe_efficiency"]:.3f}
• Index of Agreement: {stats_results["index_of_agreement"]:.3f}

Residuals Analysis:
------------------
• Shapiro-Wilk Test for Normality: W = {stats_results["shapiro_wilk_stat"]:.3f}, p = {stats_results["shapiro_wilk_p"]:.3f}
• Residuals are {"normally" if stats_results["shapiro_wilk_p"] > 0.05 else "not normally"} distributed (α = 0.05)

Interpretation:
--------------
"""

        # Add interpretation based on metrics
        if stats_results["r2"] > 0.8:
            report += (
                "• Strong correlation between calculated and reference flows\n"
            )
        elif stats_results["r2"] > 0.6:
            report += "• Moderate correlation between calculated and reference flows\n"
        else:
            report += (
                "• Weak correlation between calculated and reference flows\n"
            )

        if stats_results["nash_sutcliffe_efficiency"] > 0.75:
            report += "• Excellent model performance (NSE > 0.75)\n"
        elif stats_results["nash_sutcliffe_efficiency"] > 0.5:
            report += "• Good model performance (NSE > 0.5)\n"
        else:
            report += "• Poor model performance (NSE < 0.5)\n"

        if abs(stats_results["bias"]) < stats_results["std_senamhi"] * 0.1:
            report += "• Low systematic bias in predictions\n"
        else:
            report += "• Significant systematic bias detected\n"

        report += f"""
Recommendations:
---------------
• Consider {"accepting" if stats_results["r2"] > 0.7 else "improving"} the current formula
• {"No significant" if abs(stats_results["bias"]) < stats_results["std_senamhi"] * 0.1 else "Significant"} calibration adjustment needed
• Focus on {"maintaining" if stats_results["mape"] < 20 else "reducing"} prediction uncertainty

Data Quality Notes:
------------------
• {len(self.df)} out of total videos successfully processed
• Check for any systematic errors in video quality or processing parameters
• Consider environmental factors affecting flow measurements
"""

        with open(output_file, "w") as f:
            f.write(report)

        print(f"Report saved to {output_file}")


def main():
    """Main function for standalone execution."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze flow formula validation results"
    )
    parser.add_argument(
        "csv_file", help="Path to CSV file with validation results"
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default="analysis_output",
        help="Output directory for plots and report",
    )

    args = parser.parse_args()

    # Create analyzer
    analyzer = FlowValidationAnalyzer(args.csv_file)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Generate visualizations
    analyzer.create_visualizations(output_dir)

    # Generate report
    analyzer.generate_report(output_dir / "validation_report.txt")

    # Print statistics
    stats = analyzer.calculate_statistics()
    print("\nKey Statistics:")
    print(f"R² = {stats['r2']:.3f}")
    print(f"RMSE = {stats['rmse']:.3f} m³/s")
    print(f"MAPE = {stats['mape']:.2f}%")
    print(f"Bias = {stats['bias']:.3f} m³/s")


if __name__ == "__main__":
    main()
