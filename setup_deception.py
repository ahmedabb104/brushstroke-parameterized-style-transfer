#!/usr/bin/env python3
"""
Download and normalize Sanakoyeu et al. deception-score assets.

After running, expect:
  deception_score_vgg/split.hdf5
  deception_score_vgg/model.ckpt-<step>.*
"""

import os
import shutil
import sys
import zipfile

import requests

API_ENDPOINT = (
    "https://cloud-api.yandex.net/v1/disk/public/resources/download"
    "?public_key={}"
)
EVALUATION_DATA_URL = "https://yadi.sk/d/A2CBqSGuJ0M_XA"
ZIP_NAME = "evaluation_data.zip"
TARGET_DIR = "deception_score_vgg"


def _ssl_verify():
    """Use certifi CAs when available (fixes macOS Python SSL verify failures)."""
    try:
        import certifi
        return certifi.where()
    except ImportError:
        return True


def _direct_link(sharing_link: str) -> str:
    r = requests.get(
        API_ENDPOINT.format(sharing_link),
        timeout=60,
        verify=_ssl_verify(),
    )
    r.raise_for_status()
    data = r.json()
    if "href" not in data:
        raise RuntimeError(f"Unexpected Yandex API response: {data}")
    return data["href"]


def _download_large_file(url: str, dest_path: str, chunk_size: int = 8 * 1024 * 1024) -> None:
    verify = _ssl_verify()
    with requests.get(url, stream=True, timeout=120, verify=verify) as r:
        r.raise_for_status()
        with open(dest_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)


def _has_deception_files(root: str) -> bool:
    split_path = os.path.join(root, "split.hdf5")
    if not os.path.isfile(split_path):
        return False
    for name in os.listdir(root):
        if name.startswith("model.ckpt-") and name.endswith(".index"):
            return True
    return False


def _normalize_to_target() -> None:
    """Move known upstream folder names into deception_score_vgg/."""
    if os.path.isdir(TARGET_DIR) and _has_deception_files(TARGET_DIR):
        print(f"{TARGET_DIR}/ already contains checkpoint and split.")
        return

    if os.path.isdir("evaluation_data") and _has_deception_files("evaluation_data"):
        if os.path.isdir(TARGET_DIR):
            print(
                f"Rename skipped: {TARGET_DIR}/ exists but evaluation_data/ is present. "
                "Remove or merge manually."
            )
        else:
            print("Renaming evaluation_data/ -> deception_score_vgg/")
            shutil.move("evaluation_data", TARGET_DIR)
        return

    # Zip sometimes extracts a single top-level folder; search for split.hdf5
    for name in sorted(os.listdir(".")):
        path = os.path.join(name, "split.hdf5")
        if os.path.isfile(path) and _has_deception_files(name):
            if not os.path.isdir(TARGET_DIR):
                print(f"Moving {name}/ -> {TARGET_DIR}/")
                shutil.move(name, TARGET_DIR)
            return


def main() -> None:
    root = os.path.dirname(os.path.abspath(__file__)) or "."
    os.chdir(root)

    if os.path.isdir(TARGET_DIR) and _has_deception_files(TARGET_DIR):
        print(f"Deception data already present in ./{TARGET_DIR}/")
        return

    zip_path = os.path.join(".", ZIP_NAME)
    if os.path.isfile(zip_path) and zipfile.is_zipfile(zip_path):
        print(f"Using existing {ZIP_NAME} (delete it to re-download).")
    else:
        if os.path.isfile(zip_path):
            os.remove(zip_path)
        link = _direct_link(EVALUATION_DATA_URL)
        print("Downloading evaluation data (~1 GB). This may take a while...")
        try:
            _download_large_file(link, zip_path)
        except Exception as err:
            print(f"Download failed: {err}", file=sys.stderr)
            print(
                "Try: pip install certifi\n"
                "Or on macOS (python.org installer): open "
                "'/Applications/Python 3.11/Install Certificates.command'\n"
                "Or download the evaluation zip manually and save it as "
                f"{ZIP_NAME} in this folder, then run this script again.",
                file=sys.stderr,
            )
            sys.exit(1)

    print("Extracting...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(".")

    _normalize_to_target()

    if not (os.path.isdir(TARGET_DIR) and _has_deception_files(TARGET_DIR)):
        print(
            "Extraction finished but deception_score_vgg/ is missing split.hdf5 "
            "or model.ckpt-*. Look for evaluation_data/ or another folder and "
            "rename it to deception_score_vgg/.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Done. Assets are in ./{TARGET_DIR}/")


if __name__ == "__main__":
    main()
