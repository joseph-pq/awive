import argparse
import datetime as dt
import logging
from pathlib import Path

import numpy as np
import sympy as sp
import yaml

import awive.config
from awive.algorithms.otv import run_otv
from awive.algorithms.water_flow import get_simplest_water_flow, get_water_flow

AWIVE_FP = Path("/root/.config/nflow/awive.yaml")
LOG = logging.getLogger(__name__)


def process_video(
    awive_fp: Path,
    area: float,
    ts: dt.datetime | None = None,
    wlevel: float | None = None,
) -> None:
    """Process video.

    Args:
        awive_fp: Path to the awive config file.
        area: Area of the water flow.
        ts: Timestamp of the data. If None, use current time.
        wlevel: Current water level. If None, use simplest water flow calculation.
    """
    ts = ts if ts is not None else dt.datetime.now(dt.UTC)
    raw: dict
    raw, _ = run_otv(awive_fp)
    velocimetry = [
        d.get("velocity", "-") for _, d in raw.items() if isinstance(d, dict)
    ]

    current_water_depth = wlevel
    if current_water_depth is None:
        water_flow: float = get_simplest_water_flow(area=area, velocities=raw)
    else:
        awive_cfg = awive.config.Config.from_fp(awive_fp)
        depths = awive_cfg.water_flow.profile.depths_meters(
            awive_cfg.preprocessing.ppm,
            awive_cfg.preprocessing.resolution,
        )
        water_flow: float = get_water_flow(
            depths=depths,
            vels=np.array(velocimetry, dtype=float),
            old_depth=awive_cfg.water_flow.profile.height,
            roughness=awive_cfg.water_flow.roughness,
            current_depth=current_water_depth,
        )

    velocimetry_array = [float(v) for v in velocimetry]
    river_width = float(depths[-1, 1] - depths[0, 1])
    print(depths[-1, 0], depths[0, 0])
    print(f"{velocimetry_array=}")
    for i in range(len(velocimetry_array)):
        print(f"  Region {i + 1}: {velocimetry_array[i]} m/s")
    print(f"Water flow: {water_flow:.3f} m³/s")
    print(f"River width: {river_width:.3f} m")
    water_flow_array = float(water_flow)
    data_save = {
        "timestamp": ts,
        "velocimetry": velocimetry_array,
        "water_flow": water_flow_array,
        "river_width": river_width,
    }

    with open("/root/awive/data.yaml", "w") as f:
        yaml.dump(data_save, f)


def velocimetry(
    awive_fp: Path,
    video_fp: Path,
    ts: dt.datetime | None = None,
    wlevel: float | None = None,
    resolution: float = 1.0,
) -> None:
    """Process video."""
    # Replace the video path in the awive config file
    awive_cfg = awive.config.Config.from_fp(awive_fp)
    awive_cfg.dataset.video_fp = video_fp
    awive_cfg.preprocessing.resolution = resolution
    awive_fp.write_text(yaml.dump(awive_cfg.model_dump(mode="json"), indent=4))

    process_video(
        awive_fp=awive_fp,
        area=awive_cfg.water_flow.area,
        ts=ts,
        wlevel=wlevel,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_fp", type=Path, help="Path to the video file")
    parser.add_argument(
        "--wlevel",
        type=float,
    )
    args = parser.parse_args()
    velocimetry(
        awive_fp=AWIVE_FP,
        video_fp=args.video_fp,
        wlevel=args.wlevel,
    )
