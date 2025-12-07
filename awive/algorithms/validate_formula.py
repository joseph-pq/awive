import contextlib
import csv
import datetime as dt
import io
import logging
import re
import sys
import time
from pathlib import Path
from typing import Optional

import requests
import yaml
from google.auth.transport.requests import Request
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from tqdm import tqdm

from awive.algorithms import velocimetry
from thingsboard import *


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
    """Login to ThingsBoard and get JWT token.

    Returns:
        JWT token string for authentication.
    """
    url = f"{TB_URL}/api/auth/login"
    payload = {"username": USERNAME, "password": PASSWORD}

    r = requests.post(url, json=payload)
    r.raise_for_status()

    token = r.json()["token"]
    print("JWT obtenido:", token[:20], "...")
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
    """Get timeseries data from ThingsBoard.

    Args:
        jwt: Authentication token
        entity_type: Type of entity (e.g., 'DEVICE')
        entity_id: ID of the entity
        keys: Comma-separated list of telemetry keys
        start_ts: Start timestamp in milliseconds
        end_ts: End timestamp in milliseconds
        limit: Maximum number of records to return

    Returns:
        Dictionary with timeseries data
    """
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
        f"{TB_URL}/api/plugins/telemetry/{entity_type}/"
        f"{entity_id}/values/timeseries"
    )

    resp = requests.get(url, headers=headers, params=params)
    resp.raise_for_status()

    return resp.json()


def get_validation_data(
    start_date: str = "2025-04-29 00:00:00",
    end_date: str = "2025-05-13 00:00:00",
) -> dict:
    """Get validation data from ThingsBoard for the specified date range.

    Args:
        start_date: Start date in format 'YYYY-MM-DD HH:MM:SS'
        end_date: End date in format 'YYYY-MM-DD HH:MM:SS'

    Returns:
        Dictionary with timeseries data containing video_id,
        water_flow_senamhi, and water_level_senamhi
    """
    jwt = tb_login()

    # Convert dates to timestamps in milliseconds
    start_dt = dt.datetime.strptime(start_date, "%Y-%m-%d %H:%M:%S")
    end_dt = dt.datetime.strptime(end_date, "%Y-%m-%d %H:%M:%S")

    start_ts = int(start_dt.timestamp() * 1000)
    end_ts = int(end_dt.timestamp() * 1000)

    LOG.info(f"Fetching data from {start_date} to {end_date}")
    LOG.info(f"Timestamps: {start_ts} to {end_ts}")

    return get_timeseries(
        jwt,
        ENTITY_TYPE,
        DEVICE_ID,
        keys="video_id,water_flow_senamhi,water_level_senamhi",
        start_ts=start_ts,
        end_ts=end_ts,
        limit=2000,  # Increase limit for larger date range
    )


def find_closest_reference_data(
    video_timestamp: int,
    flow_data: dict[int, float],
    level_data: dict[int, float],
    max_diff_minutes: int = 30,
) -> tuple[Optional[float], Optional[float], Optional[int], Optional[int]]:
    """Find the closest SENAMHI flow and level data to a video timestamp.

    Args:
        video_timestamp: Video timestamp in milliseconds
        flow_data: Dictionary of {timestamp: flow_value}
        level_data: Dictionary of {timestamp: level_value}
        max_diff_minutes: Maximum allowed time difference in minutes

    Returns:
        Tuple of (closest_flow, closest_level, flow_timestamp, level_timestamp)
        Returns None values if no data within time threshold
    """
    max_diff_ms = max_diff_minutes * 60 * 1000

    # Find closest flow data
    closest_flow = None
    closest_flow_ts = None
    min_flow_diff = float("inf")

    for ts, flow in flow_data.items():
        diff = abs(video_timestamp - ts)
        if diff < min_flow_diff and diff <= max_diff_ms:
            min_flow_diff = diff
            closest_flow = flow
            closest_flow_ts = ts

    # Find closest level data
    closest_level = None
    closest_level_ts = None
    min_level_diff = float("inf")

    for ts, level in level_data.items():
        diff = abs(video_timestamp - ts)
        if diff < min_level_diff and diff <= max_diff_ms:
            min_level_diff = diff
            closest_level = level
            closest_level_ts = ts

    # Log the time differences for debugging
    if closest_flow is not None:
        flow_diff_min = min_flow_diff / (60 * 1000)
        LOG.debug(
            f"Found closest flow data: {flow_diff_min:.2f} minutes difference"
        )

    if closest_level is not None:
        level_diff_min = min_level_diff / (60 * 1000)
        LOG.debug(
            f"Found closest level data: "
            f"{level_diff_min:.2f} minutes difference"
        )

    return closest_flow, closest_level, closest_flow_ts, closest_level_ts


def find_original_video(duplicate_filename: str) -> Optional[str]:
    """Find the original video file ID in Google Drive given a duplicate filename.

    Args:
        duplicate_filename: Filename ending with '_b.mp4'

    Returns:
        Google Drive file ID of the original video (without _b) or None if not found
    """
    if not duplicate_filename.endswith("_b.mp4"):
        return None

    # Get the original filename by removing '_b' suffix
    original_filename = duplicate_filename.replace("_b.mp4", ".mp4")

    try:
        service = get_drive_service()

        # Search for the original file by name in the specific folder
        folder_id = "1qlgEpPPZscPmqwz96xLcgYTtyCBePdrs"  # From user's link
        query = f"name='{original_filename}' and '{folder_id}' in parents"

        results = (
            service.files().list(q=query, fields="files(id, name)").execute()
        )
        files = results.get("files", [])

        if files:
            original_file = files[0]  # Take the first match
            LOG.info(
                f"Found original: {original_filename} -> {original_file['id']}"
            )
            return original_file["id"]

        # If not found in specific folder, try global search as fallback
        LOG.debug(f"Not found in target folder, trying global search...")
        global_query = f"name='{original_filename}'"
        global_results = (
            service.files()
            .list(q=global_query, fields="files(id, name)")
            .execute()
        )
        global_files = global_results.get("files", [])

        if global_files:
            original_file = global_files[0]
            LOG.info(
                f"Found original (global): {original_filename} -> {original_file['id']}"
            )
            return original_file["id"]

        LOG.warning(f"Original video not found: {original_filename}")
        return None

    except Exception as e:
        LOG.error(
            f"Error searching for original video {original_filename}: {e}"
        )
        return None


def get_video_filename(video_id: str) -> str:
    """Get filename for a video from Google Drive.

    Args:
        video_id: Google Drive file ID

    Returns:
        Filename of the video
    """
    try:
        service = get_drive_service()
        file_metadata = service.files().get(fileId=video_id).execute()
        return file_metadata.get("name", f"unknown_{video_id}")
    except Exception as e:
        LOG.warning(f"Could not get filename for {video_id}: {e}")
        return f"unknown_{video_id}"


def get_drive_service():
    """Get authenticated Google Drive service.

    Returns:
        Google Drive API service object
    """
    creds = service_account.Credentials.from_service_account_file(
        CREDENTIALS_FILE, scopes=SCOPES
    )
    return build("drive", "v3", credentials=creds)


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
    temp_dir: Path = Path("/tmp/awive_validation"),
    resolution: list[float] = (1.0,),
) -> list[dict]:
    """Process a single video and return calculated flow.

    Args:
        video_id: Google Drive video ID
        timestamp: Timestamp of the video in milliseconds
        water_level: Water level for the calculation
        temp_dir: Temporary directory for video storage
        resolution: List of resolution values to test (px/m)

    Returns:
        List of dictionaries with processing results for each resolution
    """
    # Download video
    video_path = temp_dir / f"video_{video_id}.mp4"
    downloaded_path, filename = download_video_from_gdrive(
        video_id, video_path
    )

    if downloaded_path is None:
        LOG.error(f"Failed to download video {video_id}")
        return None

    LOG.info(f"🎬 Processing: {filename}")

    try:
        # Process video with velocimetry
        awive_fp = Path("/root/.config/nflow/awive.yaml")
        # Convert timestamp to datetime
        video_ts = dt.datetime.fromtimestamp(
            timestamp / 1000, tz=dt.timezone.utc
        )

        # Save results
        results = []

        # Show processing progress
        with tqdm(
            total=len(resolution),
            desc="⚙️  Process",
            bar_format="{desc}: {n}/{total}|{bar}| {elapsed}",
        ) as pbar:
            pbar.set_description("⚙️  Velocimetry")
            pbar.update(1)

            for res_value in resolution:
                LOG.info(f"   - Using resolution: {res_value} px/m")

                # Call velocimetry function
                velocimetry.velocimetry(
                    awive_fp=awive_fp,
                    video_fp=downloaded_path,
                    ts=video_ts,
                    wlevel=water_level,
                    resolution=res_value,
                )
                pbar.update(1)

                pbar.set_description("📄 Reading results")
                # Read results from the generated data.yaml
                with open("/root/awive/data.yaml") as f:
                    results = yaml.safe_load(f)
                pbar.update(1)

                LOG.info(
                    f"✅ Processing complete: Flow = {results['water_flow']:.3f} m³/s"
                )

                # Store results before cleanup
                result_data = {
                    "video_id": video_id,
                    "timestamp": timestamp,
                    "calculated_flow": results["water_flow"],
                    "velocimetry": results["velocimetry"],
                    "processing_timestamp": results["timestamp"],
                    "water_level_used": water_level,
                    "resolution": res_value,
                }
                results.append(result_data)

        return results

    except Exception as e:
        LOG.error(f"Failed to process video {video_id}: {e}")
        return None

    finally:
        # ALWAYS clean up downloaded video, regardless of success or failure
        try:
            if downloaded_path and downloaded_path.exists():
                downloaded_path.unlink()
                LOG.info(f"✅ Cleaned up video file: {downloaded_path.name}")
        except Exception as cleanup_error:
            LOG.warning(
                f"Failed to clean up video file {downloaded_path}: "
                f"{cleanup_error}"
            )


def save_result_to_csv(
    result: dict, output_file: str, is_first: bool = False
) -> None:
    """Save a single validation result to CSV file.

    Args:
        result: Dictionary with validation results
        output_file: Path to CSV file
        is_first: Whether this is the first result (write headers)
    """
    csv_path = Path(output_file)
    fieldnames = [
        "video_id",
        "timestamp",
        "timestamp_dt",
        "calculated_flow",
        "senamhi_flow",
        "senamhi_level",
        "water_level_used",
        "absolute_error",
        "relative_error",
        "velocimetry",
        "processing_timestamp",
        "flow_timestamp",
        "level_timestamp",
        "flow_time_diff_minutes",
        "level_time_diff_minutes",
        "resolution",
    ]

    # Write headers only for the first result
    mode = "w" if is_first else "a"
    with open(csv_path, mode, newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if is_first:
            writer.writeheader()
        writer.writerow(result)


def validate_flow_formula(
    start_date: str = "2025-04-29 00:00:00",
    end_date: str = "2025-05-13 00:00:00",
    output_file: str = "validation_results.csv",
) -> list[dict]:
    """Main function to validate the flow formula.

    Args:
        start_date: Start date for validation data
        end_date: End date for validation data
        output_file: Output CSV file name

    Returns:
        List of validation results
    """
    LOG.info("Starting flow formula validation")

    # Get data from ThingsBoard
    tb_data = get_validation_data(start_date, end_date)

    # Parse the data and match video_id with water_flow and water_level
    validation_results = []

    # Create temporary directory for videos
    temp_dir = Path("/tmp/awive_validation")
    temp_dir.mkdir(parents=True, exist_ok=True)

    if "video_id" not in tb_data:
        LOG.error("No video_id data found in ThingsBoard response")
        return validation_results

    # Process all videos and filter duplicates based on filename
    all_video_data = {
        item["ts"]: item["value"] for item in tb_data["video_id"]
    }

    video_data = {}
    filtered_count = 0
    processed_video_ids = set()  # Track video IDs to avoid duplicates

    LOG.info(
        f"Processing {len(all_video_data)} videos and filtering duplicates..."
    )

    for timestamp, video_id in all_video_data.items():
        # Get filename to check if it's a duplicate
        filename = get_video_filename(video_id)
        LOG.info(f"Retrieved filename: {filename} for video ID: {video_id}")

        # If it's a duplicate (_b.mp4), try to find the original
        if filename.endswith("_b.mp4"):
            original_video_id = find_original_video(filename)
            if original_video_id:
                # Check if we already have this original video ID
                if original_video_id in processed_video_ids:
                    filtered_count += 1
                    LOG.info(
                        f"Original video {original_video_id} already processed, skipping duplicate {filename}"
                    )
                    continue

                # Replace duplicate with original
                video_data[timestamp] = original_video_id
                processed_video_ids.add(original_video_id)
                original_filename = filename.replace("_b.mp4", ".mp4")
                LOG.info(
                    f"Replaced duplicate {filename} with original {original_filename}"
                )
            else:
                # Skip if we can't find the original
                filtered_count += 1
                LOG.warning(
                    f"Skipping duplicate {filename} - original not found"
                )
                continue
        else:
            # It's already an original video
            # Check if we already have this video ID processed
            if video_id in processed_video_ids:
                filtered_count += 1
                LOG.info(
                    f"Video {video_id} ({filename}) already processed, skipping duplicate timestamp"
                )
                continue

            video_data[timestamp] = video_id
            processed_video_ids.add(video_id)
            LOG.info(f"Will process original: {filename}")

    flow_data = {
        item["ts"]: float(item["value"])
        for item in tb_data.get("water_flow_senamhi", [])
    }
    level_data = {
        item["ts"]: float(item["value"])
        for item in tb_data.get("water_level_senamhi", [])
    }

    LOG.info(
        f"📊 Found {len(all_video_data)} videos, filtered {filtered_count} duplicates"
    )
    LOG.info(
        f"🎯 Processing {len(video_data)} unique videos with SENAMHI data"
    )

    # Process videos with overall progress bar
    with tqdm(
        video_data.items(),
        desc="🎬 Validating",
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
                    f"⚠️  No SENAMHI flow data within 30 minutes for video {video_id}"
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
                LOG.info(
                    f"⏰ Video: {video_dt.strftime('%H:%M:%S')}, "
                    f"SENAMHI: {flow_dt.strftime('%H:%M:%S')} "
                    f"(Δ{flow_diff:.1f}min)"
                )

            # Update progress bar description
            video_pbar.set_description(
                f"🎬 Processing video {video_index + 1}/{len(video_data)}"
            )

            # Process the video
            results = process_single_video(
                video_id=video_id,
                timestamp=timestamp,
                water_level=water_level,
                temp_dir=temp_dir,
            )

            if results is not None:
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
                    is_first_result = len(validation_results) == 0
                    save_result_to_csv(result, output_file, is_first_result)

                    validation_results.append(result)
                    LOG.info(
                        f"💾 Saved result {video_index + 1}: "
                        f"Calculated={calculated:.3f}, "
                        f"SENAMHI={reference:.3f}, "
                        f"Error={result['relative_error']:.2f}% "
                        f"Resolution={result['resolution']} ppm"
                    )
            else:
                LOG.error(f"❌ Failed to process video {video_id}")

    # Log final summary
    LOG.info(
        f"📊 Validation completed with {len(validation_results)} successful results"
    )
    if validation_results:
        LOG.info(f"💾 All results saved to {output_file}")

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
        f"✅ Validation completed. Processed {len(validation_results)} videos. "
        f"Filtered out {filtered_count} duplicate videos."
    )
    return validation_results


# EJEMPLO DE USO
if __name__ == "__main__":
    # Test connection first
    jwt = tb_login()

    # Example to get recent data
    now = int(time.time() * 1000)
    start = now - 24 * 60 * 60 * 1000  # Last 24 hours

    test_data = get_timeseries(
        jwt,
        ENTITY_TYPE,
        DEVICE_ID,
        keys="video_id, water_flow_senamhi, water_level_senamhi",
        start_ts=start,
        end_ts=now,
    )

    print("=== TELEMETRÍA DE PRUEBA ===")
    print(test_data)

    # Uncomment to run full validation
    results = validate_flow_formula(
        start_date="2025-05-30 00:00:00",
        end_date="2025-06-10 00:00:00",
        output_file="flow_validation_results.csv",
    )
