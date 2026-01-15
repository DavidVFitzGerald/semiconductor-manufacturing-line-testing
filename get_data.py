"""Handles the download and extraction of the SECOM dataset."""

import logging
import zipfile
from pathlib import Path

import requests

# Configure logger
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Define data directory constant
DATA_DIR = Path("data")


def _download_data(url: str) -> Path:
    """Downloads the SECOM data from the specified URL.

    Returns:
        zip_path: Path to the downloaded zip file.
    """
    # Check if any file starting with "secom" exists in data folder
    if any(DATA_DIR.glob("secom*")):
        logger.info("SECOM data found.")
        return None

    try:
        logger.info("SECOM data not found. Downloading...")

        # Create data folder if it doesn't exist
        DATA_DIR.mkdir(parents=True, exist_ok=True)

        # Get filename from URL
        zip_filename = url.split("/")[-1]
        zip_path = DATA_DIR.joinpath(zip_filename)

        logger.info("Downloading from %s...", url)
        response = requests.get(url, timeout=30)
        response.raise_for_status()  # Raise an error for bad status codes

        with zip_path.open("wb") as f:
            f.write(response.content)
        logger.info("Download complete.")

    except requests.exceptions.RequestException as e:
        logger.exception("Error downloading file.")

    return zip_path


def _extract_data(zip_path: Path) -> None:
    """Extracts the SECOM data from the specified zip file.

    Args:
        zip_path: Path to the zip file to extract.
    """
    try:
        # Unzip the file
        logger.info("Extracting %s...", zip_path)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(DATA_DIR)
        logger.info("Extraction complete.")

        # Remove the zip file after extraction
        zip_path.unlink()
        logger.info("Cleaned up zip file.")

    except zipfile.BadZipFile as e:
        logger.exception("Error: Downloaded file is not a valid zip file.")


def download_and_extract_data(
    url: str = "https://archive.ics.uci.edu/static/public/179/secom.zip",
) -> None:
    """Downloads and extracts the SECOM data from the specified URL.

    Args:
        url: URL to download the SECOM data from.
    """
    zip_path = _download_data(url)
    if zip_path is not None:
        _extract_data(zip_path)

    logger.info("Data ready to be used.")


if __name__ == "__main__":
    url = "https://archive.ics.uci.edu/static/public/179/secom.zip"
    download_and_extract_data(url)
