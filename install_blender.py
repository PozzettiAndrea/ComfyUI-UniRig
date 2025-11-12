#!/usr/bin/env python3
"""
ComfyUI-UniRig Blender Installer
Automatically downloads and installs Blender for mesh preprocessing.
"""

import os
import sys
import platform
import urllib.request
import tarfile
import zipfile
import shutil
from pathlib import Path


def get_platform_info():
    """Detect current platform and architecture."""
    system = platform.system().lower()
    machine = platform.machine().lower()

    # Map platform names
    if system == "darwin":
        plat = "macos"
        if machine == "arm64":
            arch = "arm64"
        else:
            arch = "x64"
    elif system == "linux":
        plat = "linux"
        arch = "x64"  # Most common
    elif system == "windows":
        plat = "windows"
        arch = "x64"
    else:
        plat = None
        arch = None

    return plat, arch


def get_blender_download_url(platform_name, architecture):
    """
    Get Blender 4.2 LTS download URL for the platform.

    Args:
        platform_name: "linux", "macos", or "windows"
        architecture: "x64" or "arm64"

    Returns:
        tuple: (download_url, version, filename) or (None, None, None) if not found
    """
    version = "4.2.3"
    base_url = "https://download.blender.org/release/Blender4.2"

    # Platform-specific URLs for Blender 4.2.3 LTS
    urls = {
        ("linux", "x64"): (
            f"{base_url}/blender-{version}-linux-x64.tar.xz",
            version,
            f"blender-{version}-linux-x64.tar.xz"
        ),
        ("macos", "x64"): (
            f"{base_url}/blender-{version}-macos-x64.dmg",
            version,
            f"blender-{version}-macos-x64.dmg"
        ),
        ("macos", "arm64"): (
            f"{base_url}/blender-{version}-macos-arm64.dmg",
            version,
            f"blender-{version}-macos-arm64.dmg"
        ),
        ("windows", "x64"): (
            f"{base_url}/blender-{version}-windows-x64.zip",
            version,
            f"blender-{version}-windows-x64.zip"
        ),
    }

    key = (platform_name, architecture)
    if key in urls:
        url, ver, filename = urls[key]
        print(f"[UniRig Installer] Using Blender {ver} for {platform_name}-{architecture}")
        return url, ver, filename

    return None, None, None


def download_file(url, dest_path):
    """Download file with progress."""
    print(f"[UniRig Installer] Downloading: {url}")
    print(f"[UniRig Installer] Destination: {dest_path}")

    def progress_hook(count, block_size, total_size):
        percent = int(count * block_size * 100 / total_size)
        sys.stdout.write(f"\r[UniRig Installer] Progress: {percent}%")
        sys.stdout.flush()

    try:
        urllib.request.urlretrieve(url, dest_path, progress_hook)
        sys.stdout.write("\n")
        print("[UniRig Installer] Download complete!")
        return True
    except Exception as e:
        print(f"\n[UniRig Installer] Error downloading: {e}")
        return False


def extract_archive(archive_path, extract_to):
    """Extract tar.gz, tar.xz, zip, or handle DMG (macOS)."""
    print(f"[UniRig Installer] Extracting: {archive_path}")

    try:
        if archive_path.endswith(('.tar.gz', '.tar.xz', '.tar.bz2')):
            with tarfile.open(archive_path, 'r:*') as tar:
                tar.extractall(extract_to)
        elif archive_path.endswith('.zip'):
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
        elif archive_path.endswith('.dmg'):
            # macOS DMG - mount and copy Blender.app
            print("[UniRig Installer] DMG detected - mounting disk image...")
            import subprocess

            # Mount the DMG
            mount_result = subprocess.run(
                ['hdiutil', 'attach', '-nobrowse', archive_path],
                capture_output=True,
                text=True
            )

            if mount_result.returncode != 0:
                print(f"[UniRig Installer] Error mounting DMG: {mount_result.stderr}")
                return False

            # Find the mount point from the output
            mount_point = None
            for line in mount_result.stdout.split('\n'):
                if '/Volumes/' in line:
                    mount_point = line.split('\t')[-1].strip()
                    break

            if not mount_point:
                print("[UniRig Installer] Error: Could not find mount point")
                return False

            try:
                # Copy Blender.app to destination
                blender_app = Path(mount_point) / "Blender.app"
                if blender_app.exists():
                    dest_app = Path(extract_to) / "Blender.app"
                    shutil.copytree(blender_app, dest_app)
                    print(f"[UniRig Installer] Copied Blender.app to: {dest_app}")
                else:
                    print(f"[UniRig Installer] Error: Blender.app not found in {mount_point}")
                    return False

            finally:
                # Unmount the DMG
                subprocess.run(['hdiutil', 'detach', mount_point], check=False)

        else:
            print(f"[UniRig Installer] Error: Unknown archive format: {archive_path}")
            return False

        print(f"[UniRig Installer] Extraction complete!")
        return True

    except Exception as e:
        print(f"[UniRig Installer] Error extracting: {e}")
        return False


def find_blender_executable(blender_dir):
    """Find the blender executable in the extracted directory."""
    plat, _ = get_platform_info()

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


def install_blender(target_dir=None):
    """Main installation function."""
    print("\n" + "="*60)
    print("ComfyUI-UniRig: Blender Installation")
    print("="*60 + "\n")

    # Get script directory
    if target_dir is None:
        script_dir = Path(__file__).parent.absolute()
        target_dir = script_dir / "lib" / "blender"
    else:
        target_dir = Path(target_dir)

    # Check if Blender already installed
    blender_exe = find_blender_executable(target_dir)
    if blender_exe and blender_exe.exists():
        print("[UniRig Installer] Blender already installed at:")
        print(f"[UniRig Installer]   {blender_exe}")
        print("[UniRig Installer] Skipping download.")
        return str(blender_exe)

    # Detect platform
    plat, arch = get_platform_info()
    if not plat or not arch:
        print("[UniRig Installer] Error: Could not detect platform")
        print("[UniRig Installer] Please install Blender manually from: https://www.blender.org/download/")
        return None

    print(f"[UniRig Installer] Detected platform: {plat}-{arch}")

    # Get download URL
    url, version, filename = get_blender_download_url(plat, arch)
    if not url:
        print("[UniRig Installer] Error: Could not find Blender download for your platform")
        print("[UniRig Installer] Please install Blender manually from: https://www.blender.org/download/")
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

        print("\n[UniRig Installer] ✓ Blender installation complete!")
        print(f"[UniRig Installer] Location: {target_dir}")

        # Find blender executable
        blender_exe = find_blender_executable(target_dir)

        if blender_exe:
            print(f"[UniRig Installer] Blender executable: {blender_exe}")
            return str(blender_exe)
        else:
            print("[UniRig Installer] Warning: Could not find blender executable")
            return None

    except Exception as e:
        print(f"\n[UniRig Installer] Error during installation: {e}")
        return None

    finally:
        # Cleanup temp files
        if temp_dir.exists():
            print("[UniRig Installer] Cleaning up temporary files...")
            shutil.rmtree(temp_dir, ignore_errors=True)


def main():
    """Entry point."""
    blender_path = install_blender()

    if blender_path:
        print("\n" + "="*60)
        print("Installation completed successfully!")
        print(f"Blender: {blender_path}")
        print("="*60 + "\n")
        return 0
    else:
        print("\n" + "="*60)
        print("Installation failed.")
        print("You can:")
        print("  1. Install Blender manually: https://www.blender.org/download/")
        print("  2. Try running this script again: python install_blender.py")
        print("="*60 + "\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
