import ast
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import time
from awive.algorithms.water_flow import (
    get_water_flow,
    get_water_flow_experimental,
    get_water_flow_experimental_v2,
    get_water_flow_trapz,
)
from awive.config import Config

AWIVE_FP = Path("/root/.config/nflow/awive.yaml")

flow_formulas = {
    "standard": get_water_flow,
    "spline": get_water_flow_experimental,
    "linear": get_water_flow_experimental_v2,
    "trapz": get_water_flow_trapz,
}


def compute_water_flow(
    flow_formulas: dict,
    depths: np.ndarray,
    vels: np.ndarray,
    old_depth: float,
    roughness: float,
    current_depth: float,
) -> float:
    """Compute water flow using the specified formula.

    Args:
        flow_formulas: Dictionary of flow formula functions.
        depths: Array of depths (N,2) (meters, meters). First column is depth,
        second is distance between depths.
        vels: Array of velocities (N,) (m/s).
        old_depth: Depth when depths were measured (meters).
        roughness: Roughness coefficient (Manning's n).
        current_depth: Current depth (meters).

    Returns:
        float: Water flow (m^3/s).
    """
    water_flow_result = {}
    time_used = {}
    for name, func in flow_formulas.items():
        time_start = time.time()
        water_flow_result[name] = func(
            depths=depths,
            vels=vels,
            old_depth=old_depth,
            roughness=roughness,
            current_depth=current_depth,
        )
        time_end = time.time()
        time_used[name] = (time_end - time_start) * 1000  # milliseconds

    return (water_flow_result, time_used)


def run_compute_water_flow(
    awive_path: Path = AWIVE_FP,
    data_path: Path = Path("/root/awive/senamhi_data.csv"),
    flow_formulas: dict = flow_formulas,
    output_path: Path | None = None,
) -> None:
    """Run water flow computation using the awive configuration and data file.

    Args:
        awive_path: Path to the awive configuration file.
        data_path: Path to the CSV data file.
        flow_formulas: Dictionary of flow formula functions.
        output_path: Path to output CSV file.
    """
    awive_cfg = Config.from_fp(awive_path)

    # Use pandas to read the CSV file
    df = pd.read_csv(data_path)

    depths = awive_cfg.water_flow.profile.depths_meters(
        awive_cfg.preprocessing.ppm
    )
    old_depth = awive_cfg.water_flow.profile.height
    roughness = awive_cfg.water_flow.roughness

    if output_path is not None:
        with output_path.open("w") as out:
            out.write(
                "timestamp,current_depth,senamhi,"
                + ",".join(flow_formulas.keys())
                + ","
                + ",".join(f"time_{name}" for name in flow_formulas.keys())
                + "\n"
            )

    for _, row in df.iterrows():
        timestamp = row["timestamp"]
        current_depth = row[
            "senamhi_level"
        ]  # Using senamhi_level as current_depth
        reference_flow = row["senamhi_flow"]
        # Parse the velocimetry array from string to numpy array
        vels_str = row["velocimetry"]
        try:
            # Parse the string representation of the list
            vels_list = ast.literal_eval(vels_str)
            vels = np.array(vels_list, dtype=float)
            # print(f"Parsed velocities: {vels}")
        except (ValueError, SyntaxError) as e:
            print(
                f"Error parsing velocimetry data for timestamp {timestamp}: {e}"
            )
            continue

        water_flow, time_used = compute_water_flow(
            flow_formulas=flow_formulas,
            depths=depths,
            vels=vels,
            old_depth=old_depth,
            roughness=roughness,
            current_depth=current_depth,
        )

        if output_path is not None:
            with open(output_path, "a") as out:
                out.write(
                    f"{timestamp},{current_depth},{reference_flow},"
                    + ",".join(f"{wf:.3f}" for wf in water_flow.values())
                    + ","
                    + ",".join(f"{tu:.4f}" for tu in time_used.values())
                    + "\n"
                )


if __name__ == "__main__":
    argparse = argparse.ArgumentParser()

    argparse.add_argument(
        "--output", type=Path, help="Path to output CSV file"
    )
    argparse.add_argument(
        "--data",
        type=Path,
        help="Path to data CSV file (timestamp, senamhi_flow, senamhi_level, velocimetry)",
    )
    argparse.add_argument(
        "--plot",
        action="store_true",
        help="If provided, only plot the results from the output CSV file",
    )

    args = argparse.parse_args()

    if args.output is None:
        args.output = Path("/root/awive/_water_flow_results.csv")

    if args.data is None:
        args.data = Path("/root/awive/_senamhi_data.csv")

    if not args.plot:
        run_compute_water_flow(
            awive_path=AWIVE_FP,
            data_path=args.data,
            flow_formulas=flow_formulas,
            output_path=args.output,
        )
    else:
        import matplotlib
        import matplotlib.pyplot as plt

        matplotlib.use("TkAgg")

        df = pd.read_csv(args.output)
        flow_label = ["senamhi"]
        flow_label.extend(list(flow_formulas.keys()))
        for name in flow_label:
            plt.plot(
                df["timestamp"],
                df[name],
                label=f"Water Flow - {name}",
            )

        plt.xlabel("Timestamp")
        plt.ylabel("Water Flow (m^3/s)")
        plt.title("Water Flow Computation Comparison")
        plt.legend()
        plt.grid()
        plt.savefig("/root/awive/_water_flow_comparison.png")
        plt.show()
