"""
Blender installation for ComfyUI-UniRig mesh preprocessing.
"""

import urllib.request
import tarfile
import zipfile
import subprocess
import shutil
import sys
from pathlib import Path
from typing import Optional, Tuple

from .utils import InstallLogger, get_platform_info
from .config import BLENDER_VERSION, BLENDER_BASE_URL


def get_blender_download_url(platform_name: str, architecture: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Get Blender download URL for the platform.

    Args:
        platform_name: "linux", "macos", or "windows"
        architecture: "x64" or "arm64"

    Returns:
        tuple: (download_url, version, filename) or (None, None, None) if not found
    """
    version = BLENDER_VERSION

    urls = {
        ("linux", "x64"): (
            f"{BLENDER_BASE_URL}/blender-{version}-linux-x64.tar.xz",
            version,
            f"blender-{version}-linux-x64.tar.xz"
        ),
        ("macos", "x64"): (
            f"{BLENDER_BASE_URL}/blender-{version}-macos-x64.dmg",
            version,
            f"blender-{version}-macos-x64.dmg"
        ),
        ("macos", "arm64"): (
            f"{BLENDER_BASE_URL}/blender-{version}-macos-arm64.dmg",
            version,
            f"blender-{version}-macos-arm64.dmg"
        ),
        ("windows", "x64"): (
            f"{BLENDER_BASE_URL}/blender-{version}-windows-x64.zip",
            version,
            f"blender-{version}-windows-x64.zip"
        ),
    }

    key = (platform_name, architecture)
    if key in urls:
        url, ver, filename = urls[key]
        InstallLogger.info(f"Using Blender {ver} for {platform_name}-{architecture}")
        return url, ver, filename

    return None, None, None


def download_file(url: str, dest_path: str) -> bool:
    """Download file with progress."""
    InstallLogger.info(f"Downloading: {url}")
    InstallLogger.info(f"Destination: {dest_path}")

    last_printed_percent = [-1]

    def progress_hook(count, block_size, total_size):
        percent = int(count * block_size * 100 / total_size)
        if percent >= last_printed_percent[0] + 10 or percent >= 100:
            sys.stdout.write(f"\r[UniRig Install] Progress: {percent}%")
            sys.stdout.flush()
            last_printed_percent[0] = percent

    try:
        urllib.request.urlretrieve(url, dest_path, progress_hook)
        sys.stdout.write("\n")
        sys.stdout.flush()
        InstallLogger.info("Download complete!")
        return True
    except Exception as e:
        print(f"\n[UniRig Install] Error downloading: {e}")
        return False


def extract_archive(archive_path: str, extract_to: str) -> bool:
    """Extract tar.gz, tar.xz, zip, or handle DMG (macOS)."""
    InstallLogger.info(f"Extracting: {archive_path}")

    try:
        if archive_path.endswith(('.tar.gz', '.tar.xz', '.tar.bz2')):
            with tarfile.open(archive_path, 'r:*') as tar:
                tar.extractall(extract_to)

        elif archive_path.endswith('.zip'):
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)

        elif archive_path.endswith('.dmg'):
            InstallLogger.info("DMG detected - mounting disk image...")

            mount_result = subprocess.run(
                ['hdiutil', 'attach', '-nobrowse', archive_path],
                capture_output=True,
                text=True
            )

            if mount_result.returncode != 0:
                InstallLogger.error(f"Error mounting DMG: {mount_result.stderr}")
                return False

            mount_point = None
            for line in mount_result.stdout.split('\n'):
                if '/Volumes/' in line:
                    mount_point = line.split('\t')[-1].strip()
                    break

            if not mount_point:
                InstallLogger.error("Could not find mount point")
                return False

            try:
                blender_app = Path(mount_point) / "Blender.app"
                if blender_app.exists():
                    dest_app = Path(extract_to) / "Blender.app"
                    shutil.copytree(blender_app, dest_app)
                    InstallLogger.info(f"Copied Blender.app to: {dest_app}")
                else:
                    InstallLogger.error(f"Blender.app not found in {mount_point}")
                    return False
            finally:
                subprocess.run(['hdiutil', 'detach', mount_point], check=False)

        else:
            InstallLogger.error(f"Unknown archive format: {archive_path}")
            return False

        InstallLogger.info("Extraction complete!")
        return True

    except Exception as e:
        InstallLogger.error(f"Error extracting: {e}")
        return False


def find_blender_executable(blender_dir) -> Optional[Path]:
    """Find the blender executable in the extracted directory."""
    platform_info = get_platform_info()
    plat = platform_info["platform"]

    if plat == "windows":
        exe_pattern = "**/blender.exe"
    elif plat == "macos":
        exe_pattern = "**/MacOS/blender"
    else:  # linux
        exe_pattern = "**/blender"

    executables = list(Path(blender_dir).glob(exe_pattern))

    if executables:
        return executables[0]
    return None


def install_blender(target_dir: Path = None) -> Optional[str]:
    """
    Install Blender for mesh preprocessing.

    Args:
        target_dir: Optional target directory. If None, uses lib/blender under script directory.

    Returns:
        str: Path to Blender executable, or None if installation failed.

    Environment Variables:
        UNIRIG_SKIP_BLENDER_INSTALL: Set to '1' to skip Blender download.
            Assumes Blender is already installed and available in system PATH.
    """
    import os

    InstallLogger.header("Blender Installation")

    # Check if Blender installation is disabled
    if os.environ.get('UNIRIG_SKIP_BLENDER_INSTALL', '').lower() in ('1', 'true', 'yes'):
        InstallLogger.info("UNIRIG_SKIP_BLENDER_INSTALL is set - skipping Blender download")
        InstallLogger.info("Assuming Blender is available in system PATH")
        # Try to find blender in PATH
        blender_in_path = shutil.which('blender')
        if blender_in_path:
            InstallLogger.info(f"Found Blender in PATH: {blender_in_path}")
            return blender_in_path
        InstallLogger.warning("Blender not found in PATH - some features may not work")
        return None

    if target_dir is None:
        script_dir = Path(__file__).parent.parent.absolute()
        target_dir = script_dir / "lib" / "blender"
    else:
        target_dir = Path(target_dir)

    # Check if Blender already installed
    blender_exe = find_blender_executable(target_dir)
    if blender_exe and blender_exe.exists():
        InstallLogger.info("Blender already installed at:")
        InstallLogger.info(f"  {blender_exe}")
        InstallLogger.info("Skipping download.")
        return str(blender_exe)

    # Detect platform
    platform_info = get_platform_info()
    plat = platform_info["platform"]
    arch = platform_info["arch"]

    if not plat or not arch:
        InstallLogger.error("Could not detect platform")
        InstallLogger.info("Please install Blender manually from: https://www.blender.org/download/")
        return None

    InstallLogger.info(f"Detected platform: {plat}-{arch}")

    # Get download URL
    url, version, filename = get_blender_download_url(plat, arch)
    if not url:
        InstallLogger.error("Could not find Blender download for your platform")
        InstallLogger.info("Please install Blender manually from: https://www.blender.org/download/")
        return None

    # Create temporary download directory
    temp_dir = target_dir.parent / "_temp_blender_download"
    temp_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Download
        download_path = temp_dir / filename
        if not download_file(url, str(download_path)):
            return None

        # Extract
        target_dir.mkdir(parents=True, exist_ok=True)
        if not extract_archive(str(download_path), str(target_dir)):
            return None

        InstallLogger.success("Blender installation complete!")
        InstallLogger.info(f"Location: {target_dir}")

        # Find blender executable
        blender_exe = find_blender_executable(target_dir)

        if blender_exe:
            InstallLogger.info(f"Blender executable: {blender_exe}")
            return str(blender_exe)
        else:
            InstallLogger.warning("Could not find blender executable")
            return None

    except Exception as e:
        InstallLogger.error(f"Error during installation: {e}")
        return None

    finally:
        # Cleanup temp files
        if temp_dir.exists():
            InstallLogger.info("Cleaning up temporary files...")
            shutil.rmtree(temp_dir, ignore_errors=True)
