#!/usr/bin/env python3
"""Fetch and process ThingsBoard sensor data with multi-resolution support.

This script downloads real-world data from ThingsBoard IoT platform and processes
it using AWIVE velocimetry algorithms:

1. Authenticate with ThingsBoard API and fetch telemetry data
2. Download associated videos from Google Drive
3. Process videos using AWIVE velocimetry with configurable scaling factors
4. Calculate water flow for each resolution/scaling factor
5. Save results to CSV with timestamps, flows, levels, and velocimetry data

Usage:
    python fetch_thingsboard_data.py -s "2025-04-29 00:00:00" -e "2025-05-13 00:00:00"
    python fetch_thingsboard_data.py -s "2025-05-30 00:00:00" -e "2025-06-10 00:00:00" -r 0.9 1.0 1.1
"""

import argparse
import contextlib
import csv
import datetime as dt
import io
import logging
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Optional

import requests
import yaml
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from tqdm import tqdm

# Add the awive package to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import AWIVE modules
from awive.algorithms import velocimetry
from thingsboard import (
    CREDENTIALS_FILE,
    DEVICE_ID,
    ENTITY_TYPE,
    PASSWORD,
    SCOPES,
    TB_URL,
    USERNAME,
)


# Configure beautiful logging with colors and timestamps
class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors and emojis"""

    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"

    EMOJIS = {
        "DEBUG": "🔍",
        "INFO": "✅",
        "WARNING": "⚠️",
        "ERROR": "❌",
        "CRITICAL": "💥",
    }

    def format(self, record):
        color = self.COLORS.get(record.levelname, "")
        emoji = self.EMOJIS.get(record.levelname, "")

        # Format timestamp
        timestamp = dt.datetime.fromtimestamp(record.created).strftime(
            "%H:%M:%S"
        )

        # Format message with color and emoji
        message = (
            f"{color}{emoji} [{timestamp}] {record.getMessage()}{self.RESET}"
        )
        return message


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

# Apply colored formatter
for handler in logging.getLogger().handlers:
    handler.setFormatter(ColoredFormatter())

# Suppress annoying googleapiclient warnings
logging.getLogger("googleapiclient.discovery_cache").setLevel(logging.ERROR)
logging.getLogger("googleapiclient.discovery").setLevel(logging.WARNING)

LOG = logging.getLogger(__name__)


def tb_login() -> str:
    """Login to ThingsBoard and get JWT token."""
    url = f"{TB_URL}/api/auth/login"
    payload = {"username": USERNAME, "password": PASSWORD}
    r = requests.post(url, json=payload)
    r.raise_for_status()
    token = r.json()["token"]
    LOG.info("✅ Logged in to ThingsBoard")
    return token


def get_timeseries(
    jwt: str,
    entity_type: str,
    entity_id: str,
    keys: str,
    start_ts: int,
    end_ts: int,
    limit: int = 1000,
) -> dict:
    """Get timeseries data from ThingsBoard."""
    headers = {
        "Content-Type": "application/json",
        "X-Authorization": f"Bearer {jwt}",
    }
    params = {
        "keys": keys,
        "startTs": start_ts,
        "endTs": end_ts,
        "limit": limit,
        "orderBy": "ASC",
    }
    url = (
        f"{TB_URL}/api/plugins/telemetry/{entity_type}/{entity_id}/"
        "values/timeseries"
    )
    resp = requests.get(url, headers=headers, params=params)
    resp.raise_for_status()
    return resp.json()


def get_drive_service() -> Any:
    """Get authenticated Google Drive service."""
    creds = service_account.Credentials.from_service_account_file(
        CREDENTIALS_FILE, scopes=SCOPES
    )
    return build("drive", "v3", credentials=creds)


def get_video_filename(video_id: str) -> str:
    """Get filename for a video from Google Drive."""
    try:
        service = get_drive_service()
        file_metadata = service.files().get(fileId=video_id).execute()
        return file_metadata.get("name", f"unknown_{video_id}")
    except Exception as e:
        LOG.warning(f"Could not get filename for {video_id}: {e}")
        return f"unknown_{video_id}"


def find_original_video(duplicate_filename: str) -> Optional[str]:
    """Find original video file ID given a duplicate filename."""
    if not duplicate_filename.endswith("_b.mp4"):
        return None

    original_filename = duplicate_filename.replace("_b.mp4", ".mp4")
    try:
        service = get_drive_service()
        folder_id = "1qlgEpPPZscPmqwz96xLcgYTtyCBePdrs"
        query = f"name='{original_filename}' and '{folder_id}' in parents"
        results = (
            service.files().list(q=query, fields="files(id, name)").execute()
        )
        files = results.get("files", [])

        if files:
            LOG.info(
                f"Found original: {original_filename} -> {files[0]['id']}"
            )
            return files[0]["id"]

        LOG.warning(f"Original video not found: {original_filename}")
        return None
    except Exception as e:
        LOG.error(
            f"Error searching for original video {original_filename}: {e}"
        )
        return None


def download_video_from_gdrive(
    video_id: str, output_path: Path
) -> tuple[Optional[Path], Optional[str]]:
    """Download video from Google Drive using official API with optimizations.

    Optimizations applied:
    - 10MB chunk size (vs 1MB default) for fewer API calls
    - Better progress tracking with file size and download speed
    - Efficient memory handling

    Args:
        video_id: Google Drive file ID
        output_path: Local path where to save the video

    Returns:
        Tuple of (Path to downloaded file, filename) if successful,
        (None, None) otherwise
    """
    try:
        # Get Drive service
        service = get_drive_service()

        # Get file metadata first to check if it exists and get name + size
        file_metadata = (
            service.files().get(fileId=video_id, fields="name,size").execute()
        )
        file_name = file_metadata.get("name", f"video_{video_id}")
        file_size = file_metadata.get("size")

        if file_size:
            file_size = int(file_size)
            LOG.info(
                f"🔽 Downloading: {file_name} ({file_size / (1024 * 1024):.1f} MB)"
            )
        else:
            LOG.info(f"🔽 Downloading: {file_name}")

        # Download file content
        request = service.files().get_media(fileId=video_id)

        # Create directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Download in chunks with progress bar - OPTIMIZED
        fh = io.BytesIO()
        # Use larger chunk size for faster downloads (10MB vs 1MB default)
        downloader = MediaIoBaseDownload(
            fh, request, chunksize=10 * 1024 * 1024
        )

        # Create optimized progress bar with speed tracking
        if file_size:
            # Use file size for accurate progress with speed
            with tqdm(
                total=file_size,
                desc="📥 Download",
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{rate_fmt}]",
            ) as pbar:
                done = False
                while not done:
                    status, done = downloader.next_chunk()
                    if status:
                        downloaded_bytes = int(status.resumable_progress)
                        pbar.update(downloaded_bytes - pbar.n)
                pbar.clear()
                pbar.close()
        else:
            # Fallback to percentage-based progress
            with tqdm(
                total=100,
                desc="📥 Download",
                unit="%",
                bar_format="{desc}: {percentage:3.0f}%|{bar}| {elapsed}",
            ) as pbar:
                done = False
                last_progress = 0
                while not done:
                    status, done = downloader.next_chunk()
                    if status:
                        current_progress = int(status.progress() * 100)
                        pbar.update(current_progress - last_progress)
                        last_progress = current_progress

        # Write to file efficiently
        with open(output_path, "wb") as f:
            f.write(fh.getvalue())

        LOG.info(f"✅ Download complete: {file_name}")
        return output_path, file_name

    except Exception as e:
        LOG.error(f"Failed to download video {video_id}: {e}")
        return None, None


def process_single_video(
    video_id: str,
    timestamp: int,
    water_level: Optional[float] = None,
    temp_dir: Path = Path("/tmp/data_extraction/awive_fetching"),
    resolution: Optional[list[float]] = None,
) -> list[dict]:
    """Process a single video and return calculated flow.

    Args:
        video_id: Google Drive video ID
        timestamp: Timestamp of the video in milliseconds
        water_level: Water level for the calculation
        temp_dir: Temporary directory for video storage
        resolution: List of resolution values to test

    Returns:
        List of dictionaries with processing results for each resolution
    """
    if resolution is None:
        resolution = [1.0]

    # Download video
    video_path = temp_dir / f"video_{video_id}.mp4"
    downloaded_path, filename = download_video_from_gdrive(
        video_id, video_path
    )

    if downloaded_path is None:
        LOG.error(f"Failed to download video {video_id}")
        return []

    LOG.info(f"🎬 Processing: {filename}")

    try:
        # Process video with velocimetry
        awive_fp = Path("/root/.config/nflow/awive.yaml")
        # Convert timestamp to datetime
        video_ts = dt.datetime.fromtimestamp(
            timestamp / 1000, tz=dt.timezone.utc
        )

        # Save results
        all_results = []

        # Show processing progress
        with tqdm(
            total=len(resolution),
            desc="🤖 Process",
            bar_format="{desc}: {n}/{total}|{bar}| {elapsed}",
        ) as pbar:
            pbar.set_description("🤖 Processing velocimetry")
            pbar.refresh()

            for res_value in resolution:
                LOG.info(f"\nUsing resolution: {res_value}x")

                # Track processing time
                start_time = time.time()
                try:
                    # Call velocimetry function
                    velocimetry.velocimetry(
                        awive_fp=awive_fp,
                        video_fp=downloaded_path,
                        ts=video_ts,
                        wlevel=water_level,
                        resolution=res_value,
                    )
                finally:
                    # Restore stdout
                    processing_time = time.time() - start_time

                pbar.update(1)
                pbar.refresh()

                # Read results from the generated data.yaml
                with open("/root/awive/data.yaml") as f:
                    yaml_data = yaml.safe_load(f)
                LOG.info(
                    f"✅ Processing complete: "
                    f"Flow = {yaml_data['water_flow']:.3f} m³/s "
                    f"(took {processing_time:.1f}s)"
                )

                # Store results before cleanup
                result_data = {
                    "video_id": video_id,
                    "timestamp": timestamp,
                    "calculated_flow": yaml_data["water_flow"],
                    "velocimetry": yaml_data["velocimetry"],
                    "resolution": res_value,
                    "processing_time_seconds": round(processing_time, 2),
                }
                all_results.append(result_data)

        return all_results

    except Exception as e:
        LOG.error(f"Failed to process video {video_id}: {e}")
        return []

    finally:
        # ALWAYS clean up downloaded video, regardless of success or failure
        try:
            if downloaded_path and downloaded_path.exists():
                downloaded_path.unlink()
                LOG.info(f"✅ Cleaned up video file: {downloaded_path.name}")
        except Exception as cleanup_error:
            LOG.warning(f"Failed to clean up video file: {cleanup_error}")


def find_closest_reference_data(
    video_timestamp: int,
    flow_data: dict[int, float],
    level_data: dict[int, float],
    max_diff_minutes: int = 30,
) -> tuple[Optional[float], Optional[float], Optional[int], Optional[int]]:
    """Find closest SENAMHI flow and level data to a video timestamp."""
    max_diff_ms = max_diff_minutes * 60 * 1000

    # Find closest flow
    closest_flow = None
    closest_flow_ts = None
    min_flow_diff = float("inf")
    for ts, flow in flow_data.items():
        diff = abs(video_timestamp - ts)
        if diff < min_flow_diff and diff <= max_diff_ms:
            min_flow_diff = diff
            closest_flow = flow
            closest_flow_ts = ts

    # Find closest level
    closest_level = None
    closest_level_ts = None
    min_level_diff = float("inf")
    for ts, level in level_data.items():
        diff = abs(video_timestamp - ts)
        if diff < min_level_diff and diff <= max_diff_ms:
            min_level_diff = diff
            closest_level = level
            closest_level_ts = ts

    return closest_flow, closest_level, closest_flow_ts, closest_level_ts


def save_result_to_csv(
    result: dict, output_file: str, is_first: bool = False
) -> None:
    """Save a single result to CSV file."""
    fieldnames = [
        "timestamp",
        "timestamp_dt",
        "senamhi_flow",
        "senamhi_level",
        "velocimetry",
        "resolution",
        "calculated_flow",
        "processing_time_seconds",
        "absolute_error",
        "relative_error",
        "video_id",
        "flow_timestamp",
        "level_timestamp",
        "flow_time_diff_minutes",
        "level_time_diff_minutes",
    ]

    mode = "w" if is_first else "a"
    with open(output_file, mode, newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if is_first:
            writer.writeheader()
        writer.writerow(result)


def fetch_and_process(
    start_date: str,
    end_date: str,
    output_file: str,
    resolution: Optional[list[float]] = None,
) -> list[dict]:
    """Fetch data from ThingsBoard and process videos.

    Args:
        start_date: Start date (YYYY-MM-DD HH:MM:SS)
        end_date: End date (YYYY-MM-DD HH:MM:SS)
        output_file: Output CSV file path
        resolution: List of scaling factors (default: [1.0])

    Returns:
        List of processing results
    """
    if resolution is None:
        resolution = [1.0]

    LOG.info("🚀 Starting data fetch and processing")

    # Login and get data
    jwt = tb_login()

    start_dt = dt.datetime.strptime(start_date, "%Y-%m-%d %H:%M:%S")
    end_dt = dt.datetime.strptime(end_date, "%Y-%m-%d %H:%M:%S")
    start_ts = int(start_dt.timestamp() * 1000)
    end_ts = int(end_dt.timestamp() * 1000)

    LOG.info(f"📅 Fetching data from {start_date} to {end_date}")

    tb_data = get_timeseries(
        jwt,
        ENTITY_TYPE,
        DEVICE_ID,
        keys="video_id,water_flow_senamhi,water_level_senamhi",
        start_ts=start_ts,
        end_ts=end_ts,
        limit=2000,
    )

    if "video_id" not in tb_data:
        LOG.error("❌ No video_id data found in ThingsBoard response")
        return []

    # Process videos and filter duplicates
    all_video_data = {
        item["ts"]: item["value"] for item in tb_data["video_id"]
    }
    video_data = {}
    filtered_count = 0
    processed_video_ids = set()

    LOG.info(
        f"Processing {len(all_video_data)} videos and filtering duplicates..."
    )

    for timestamp, video_id in all_video_data.items():
        filename = get_video_filename(video_id)
        LOG.info(f"Retrieved filename: {filename} for video ID: {video_id}")

        # Handle duplicates
        if filename.endswith("_b.mp4"):
            original_video_id = find_original_video(filename)
            if original_video_id:
                if original_video_id in processed_video_ids:
                    filtered_count += 1
                    LOG.info(
                        f"Original video {original_video_id} already processed, "
                        f"skipping duplicate {filename}"
                    )
                    continue
                video_data[timestamp] = original_video_id
                processed_video_ids.add(original_video_id)
                original_filename = filename.replace("_b.mp4", ".mp4")
                LOG.info(
                    f"Replaced duplicate {filename} with original {original_filename}"
                )
            else:
                filtered_count += 1
                LOG.warning(
                    f"Skipping duplicate {filename} - original not found"
                )
                continue
        else:
            if video_id in processed_video_ids:
                filtered_count += 1
                LOG.info(
                    f"Video {video_id} ({filename}) already processed, "
                    f"skipping duplicate timestamp"
                )
                continue
            video_data[timestamp] = video_id
            processed_video_ids.add(video_id)
            LOG.info(f"Will process original: {filename}")

    # Get reference data
    flow_data = {
        item["ts"]: float(item["value"])
        for item in tb_data.get("water_flow_senamhi", [])
    }
    level_data = {
        item["ts"]: float(item["value"])
        for item in tb_data.get("water_level_senamhi", [])
    }

    LOG.info(
        f"📊 Found {len(all_video_data)} videos, "
        f"filtered {filtered_count} duplicates"
    )
    LOG.info(
        f"🎯 Processing {len(video_data)} unique videos with SENAMHI data"
    )

    # Create temp directory
    temp_dir = Path("/tmp/data_extraction/awive_fetching")
    temp_dir.mkdir(parents=True, exist_ok=True)

    # Process videos
    all_results = []

    # Process videos with overall progress bar
    with tqdm(
        video_data.items(),
        desc="🎬 Fetching",
        bar_format="{desc}: {n}/{total}|{bar}| {elapsed}",
    ) as video_pbar:
        for video_index, (timestamp, video_id) in enumerate(video_pbar):
            # Find closest SENAMHI flow and level data
            (
                senamhi_flow,
                water_level,
                flow_ts,
                level_ts,
            ) = find_closest_reference_data(timestamp, flow_data, level_data)

            if senamhi_flow is None:
                LOG.warning(
                    f"⚠️  No SENAMHI flow data within 30 minutes "
                    f"for video {video_id}"
                )
                continue

            # Log time differences for debugging
            video_dt = dt.datetime.fromtimestamp(
                timestamp / 1000, tz=dt.timezone.utc
            )
            if flow_ts:
                flow_dt = dt.datetime.fromtimestamp(
                    flow_ts / 1000, tz=dt.timezone.utc
                )
                flow_diff = abs((timestamp - flow_ts) / (60 * 1000))
                video_pbar.clear()
                LOG.info(
                    f"⏰ Video: {video_dt.strftime('%H:%M:%S')}, "
                    f"SENAMHI: {flow_dt.strftime('%H:%M:%S')} "
                    f"(Δ{flow_diff:.1f}min)"
                )
                video_pbar.refresh()

            # Update progress bar description
            video_pbar.set_description(
                f"🎬 Processing video {video_index + 1}/{len(video_data)}"
            )

            # Process video with all resolutions
            results = process_single_video(
                video_id=video_id,
                timestamp=timestamp,
                water_level=water_level,
                temp_dir=temp_dir,
                resolution=resolution,
            )

            if results:
                for result in results:
                    # Add SENAMHI reference data
                    result["senamhi_flow"] = senamhi_flow
                    result["senamhi_level"] = water_level
                    result["timestamp_dt"] = dt.datetime.fromtimestamp(
                        timestamp / 1000, tz=dt.timezone.utc
                    )

                    # Add timestamp information for reference data
                    result["flow_timestamp"] = flow_ts
                    result["level_timestamp"] = level_ts

                    # Calculate time differences in minutes
                    result["flow_time_diff_minutes"] = (
                        abs(timestamp - flow_ts) / (60 * 1000)
                        if flow_ts
                        else None
                    )
                    result["level_time_diff_minutes"] = (
                        abs(timestamp - level_ts) / (60 * 1000)
                        if level_ts
                        else None
                    )

                    # Calculate error metrics
                    calculated = result["calculated_flow"]
                    reference = senamhi_flow
                    result["absolute_error"] = abs(calculated - reference)
                    result["relative_error"] = (
                        abs(calculated - reference) / reference * 100
                        if reference != 0
                        else 0
                    )

                    # Save to CSV immediately (incremental save)
                    is_first_result = len(all_results) == 0
                    save_result_to_csv(result, output_file, is_first_result)

                    all_results.append(result)
                    LOG.info(
                        f"💾 Saved result {video_index + 1}: "
                        f"Calculated={calculated:.3f}, "
                        f"SENAMHI={reference:.3f}, "
                        f"Error={result['relative_error']:.2f}% "
                        f"Resolution={result['resolution']}x"
                    )
            else:
                LOG.error(f"❌ Failed to process video {video_id}")

    # Clean up temporary directory and any remaining files
    try:
        # Remove any remaining files in temp directory
        for file in temp_dir.iterdir():
            if file.is_file():
                file.unlink()
                LOG.debug(f"Cleaned up remaining file: {file}")

        # Remove directory if empty
        with contextlib.suppress(OSError):
            temp_dir.rmdir()
            LOG.info("Cleaned up temporary directory")
    except OSError as e:
        LOG.warning(
            f"Could not fully clean up temporary directory {temp_dir}: {e}"
        )

    LOG.info(
        f"📊 Fetching completed with {len(all_results)} successful results"
    )
    if all_results:
        LOG.info(f"💾 All results saved to {output_file}")

    LOG.info(
        f"✅ Fetching completed. Processed {len(all_results)} videos. "
        f"Filtered out {filtered_count} duplicate videos."
    )
    return all_results


def main() -> int:
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Fetch and process ThingsBoard data with "
        "multi-resolution support"
    )
    parser.add_argument(
        "--start-date",
        "-s",
        default="2025-04-29 00:00:00",
        help="Start date (YYYY-MM-DD HH:MM:SS)",
    )
    parser.add_argument(
        "--end-date",
        "-e",
        default="2025-05-13 00:00:00",
        help="End date (YYYY-MM-DD HH:MM:SS)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="thingsboard_data.csv",
        help="Output CSV file",
    )
    parser.add_argument(
        "--resolution",
        "-r",
        type=float,
        nargs="+",
        default=[1.0],
        help="Scaling factors (e.g., -r 0.9 1.0 1.1). Default: 1.0",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    if args.verbose:
        LOG.setLevel(logging.DEBUG)

    print("=" * 60)
    print("THINGSBOARD DATA FETCH & PROCESSING")
    print("=" * 60)
    print(f"Date range: {args.start_date} to {args.end_date}")
    print(f"Scaling factors: {args.resolution}")
    print(f"Output file: {args.output}")
    print("=" * 60)
    print()

    try:
        results = fetch_and_process(
            start_date=args.start_date,
            end_date=args.end_date,
            output_file=args.output,
            resolution=args.resolution,
        )

        if not results:
            print(
                "\n❌ No results obtained. "
                "Check your ThingsBoard connection and date range."
            )
            return 1

        print(f"\n✅ Success! Processed {len(results)} results")
        print(f"📁 Data saved to: {args.output}")

        # Show summary
        unique_videos = len({r["video_id"] for r in results})
        resolutions = sorted({r["resolution"] for r in results})

        print("\n📊 Summary:")
        print(f"   Videos processed: {unique_videos}")
        print(f"   Scaling factors: {resolutions}")
        print(f"   Total results: {len(results)}")

        return 0

    except Exception as e:
        print(f"\n❌ Error: {e}")
        logging.exception("Processing failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
