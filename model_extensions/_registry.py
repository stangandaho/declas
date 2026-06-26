"""Fetch the online model registry and install / remove extensions."""

import json
import urllib.request
import urllib.error
import zipfile
import io
import shutil
from pathlib import Path

# Public registry hosted in the Declas GitHub repo.
# Push model_extensions/registry.json to enable the online catalogue.
REGISTRY_URL = (
    "https://raw.githubusercontent.com/stangandaho/declas/main"
    "/model_extensions/registry.json"
)
EXTENSIONS_DIR = Path(__file__).parent
_TIMEOUT_SEC = 10


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
                       progress_callback=None) -> bool:
    """Download and install one extension from the registry.

    Steps
    -----
    1. Download the extension ZIP (manifest.json + adapter.py + auxiliaries).
    2. Extract to  model_extensions/<name>/.
    3. Download the model weights separately (potentially hundreds of MB).
    4. Write / overwrite manifest.json to ensure local copy is consistent.

    Parameters
    ----------
    manifest : registry entry for the model (dict)
    progress_callback : optional callable(message: str) for status updates

    Returns True on success.
    """
    name = manifest.get("name", "unknown")
    ext_dir = EXTENSIONS_DIR / name
    ext_dir.mkdir(parents=True, exist_ok=True)

    def report(msg: str):
        if progress_callback:
            progress_callback(msg)

    try:
        # 1 — Extension ZIP (adapter code + any auxiliaries)
        zip_url = manifest.get("zip_url", "")
        if zip_url:
            report(f"Downloading {name} package …")
            with urllib.request.urlopen(zip_url, timeout=120) as resp:
                zdata = resp.read()
            with zipfile.ZipFile(io.BytesIO(zdata)) as zf:
                zf.extractall(ext_dir)
            report("Package extracted.")
        else:
            # No ZIP: download adapter.py directly if provided
            adapter_url = manifest.get("adapter_url", "")
            if adapter_url:
                report("Downloading adapter …")
                urllib.request.urlretrieve(adapter_url,
                                           str(ext_dir / "adapter.py"))

        # 2 — Model weights (separate because they can be very large)
        weights_url = manifest.get("download_url", "")
        model_file  = manifest.get("model_file", "")
        if weights_url and model_file:
            dest = ext_dir / model_file
            report(f"Downloading weights ({model_file}). This may take a while …")
            urllib.request.urlretrieve(weights_url, str(dest))
            report(f"Weights saved → {dest.name}")

        # 3 — Write manifest locally
        with open(ext_dir / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        report(f"✅ {name} installed successfully.")
        return True

    except Exception as exc:
        report(f"❌ Install failed: {exc}")
        return False


def remove_extension(name: str) -> bool:
    """Delete an installed extension directory and all its contents.

    Returns True if the directory existed and was removed.
    """
    ext_dir = EXTENSIONS_DIR / name
    if ext_dir.exists() and ext_dir.is_dir():
        shutil.rmtree(str(ext_dir))
        return True
    return False
