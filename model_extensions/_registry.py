"""Fetch the online model registry and install / remove extensions."""

import json
import urllib.request
import urllib.error
from pathlib import Path

from model_extensions._loader import _BUNDLED_DIR

REGISTRY_URL = (
    "https://raw.githubusercontent.com/stangandaho/declas/main"
    "/model_extensions/registry.json"
)
_TIMEOUT_SEC = 10
_CHUNK_SIZE  = 256 * 1024   # 256 KB per read


def _download_with_progress(url: str, dest: Path, bytes_cb=None) -> None:
    """Stream *url* to *dest*, calling bytes_cb(downloaded, total) after each chunk."""
    with urllib.request.urlopen(url, timeout=600) as resp:
        total      = int(resp.headers.get("Content-Length") or 0)
        downloaded = 0
        with open(dest, "wb") as fout:
            while True:
                chunk = resp.read(_CHUNK_SIZE)
                if not chunk:
                    break
                fout.write(chunk)
                downloaded += len(chunk)
                if bytes_cb:
                    bytes_cb(downloaded, total)


def fetch_registry() -> dict:
    """Fetch the online registry JSON.

    Returns the parsed dict on success, or {"models": [], "error": "<msg>"} on
    failure so callers can always iterate over registry["models"] safely.
    """
    try:
        with urllib.request.urlopen(REGISTRY_URL, timeout=_TIMEOUT_SEC) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        return {"models": [], "error": f" {exc.reason}"}
    except Exception as exc:
        return {"models": [], "error": str(exc)}


def download_extension(manifest: dict,
                       progress_callback=None,
                       bytes_callback=None) -> bool:
    """Download model weights into the bundled extension directory.

    The adapter and manifest are already bundled with the app.
    Only the weights file (which can be hundreds of MB) is downloaded at
    runtime and stored alongside the bundled adapter in _internal/model_extensions/<name>/.
    """
    name    = manifest.get("name", "unknown")
    ext_dir = _BUNDLED_DIR / name
    ext_dir.mkdir(parents=True, exist_ok=True)

    def report(msg: str):
        if progress_callback:
            progress_callback(msg)

    try:
        weights_url = manifest.get("download_url", "")
        model_file  = manifest.get("model_file", "")
        if weights_url and model_file:
            dest = ext_dir / model_file
            report(f"Downloading weights ({model_file}). This may take a while …")
            _download_with_progress(weights_url, dest, bytes_cb=bytes_callback)
            report(f"Weights saved → {dest.name}")

        # Refresh the manifest so the local copy stays in sync with the registry.
        with open(ext_dir / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        report(f"✅ {name} installed successfully.")
        return True

    except Exception as exc:
        report(f"❌ Install failed: {exc}")
        return False


def remove_extension(name: str) -> bool:
    """Delete only the weights file for an extension.

    The adapter and manifest (bundled) are left intact so the extension
    shows as 'missing_weights' — ready to be re-downloaded.
    Returns True if a weights file was found and deleted.
    """
    ext_dir      = _BUNDLED_DIR / name
    manifest_path = ext_dir / "manifest.json"
    if not manifest_path.exists():
        return False
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        model_file = manifest.get("model_file", "")
        if model_file:
            weights = ext_dir / model_file
            if weights.exists():
                weights.unlink()
                return True
    except Exception:
        pass
    return False
