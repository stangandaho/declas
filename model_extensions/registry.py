"""Fetch the online model registry and install / remove extensions."""

import json
import ssl
import urllib.request
import urllib.error
from pathlib import Path

from model_extensions.loader import BUNDLED_DIR

REGISTRY_URL = (
    "https://raw.githubusercontent.com/stangandaho/declas/main"
    "/model_extensions/registry.json"
)
TIMEOUT_SEC = 10
CHUNK_SIZE  = 256 * 1024   # 256 KB per read


def ssl_context() -> ssl.SSLContext:
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


def download_with_progress(url: str, dest: Path, bytes_cb=None) -> None:
    """Stream *url* to *dest*, calling bytes_cb(downloaded, total) after each chunk."""
    with urllib.request.urlopen(url, timeout=600, context=ssl_context()) as resp:
        total      = int(resp.headers.get("Content-Length") or 0)
        downloaded = 0
        with open(dest, "wb") as fout:
            while True:
                chunk = resp.read(CHUNK_SIZE)
                if not chunk:
                    break
                fout.write(chunk)
                downloaded += len(chunk)
                if bytes_cb:
                    bytes_cb(downloaded, total)


def load_local_registry() -> list:
    """Load models from the bundled registry.json (shipped with the app)."""
    local_path = BUNDLED_DIR / "registry.json"
    try:
        with open(local_path, "r", encoding="utf-8") as f:
            return json.load(f).get("models", [])
    except Exception:
        return []


def fetch_registry() -> dict:
    """Fetch the online registry JSON, merged with the local bundled registry.

    Online entries take precedence; local-only entries (e.g. bundled
    adapter-only extensions not yet pushed to GitHub) are appended so they
    always appear in the Available tab.

    Returns {"models": [...], "error": "<msg>"} — callers can always iterate
    over registry["models"] safely.
    """
    local_models = load_local_registry()
    error = None

    try:
        with urllib.request.urlopen(REGISTRY_URL, timeout=TIMEOUT_SEC, context=ssl_context()) as resp:
            online = json.loads(resp.read().decode("utf-8"))
            online_models = online.get("models", [])
    except urllib.error.URLError as exc:
        online_models = []
        error = f" {exc.reason}"
    except Exception as exc:
        online_models = []
        error = str(exc)

    # Merge: online first, then local entries whose name isn't already online
    online_names = {m.get("name") for m in online_models}
    extra = [m for m in local_models if m.get("name") not in online_names]
    merged = online_models + extra

    result: dict = {"models": merged}
    if error and not merged:
        result["error"] = error
    elif error:
        result["warning"] = error
    return result


def download_extension(manifest: dict,
                       progress_callback=None,
                       bytes_callback=None) -> bool:
    """Download model weights into the bundled extension directory.

    The adapter and manifest are already bundled with the app.
    Only the weights file (which can be hundreds of MB) is downloaded at
    runtime and stored alongside the bundled adapter in _internal/model_extensions/<name>/.
    """
    name    = manifest.get("name", "unknown")
    ext_dir = BUNDLED_DIR / name
    ext_dir.mkdir(parents=True, exist_ok=True)

    def report(msg: str):
        if progress_callback:
            progress_callback(msg)

    try:
        weights_url = manifest.get("download_url", "")
        model_file  = manifest.get("model_file", "")
        if weights_url and model_file:
            dest = ext_dir / model_file
            report(f"Downloading weights {model_file}")
            download_with_progress(weights_url, dest, bytes_cb=bytes_callback)
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
    ext_dir      = BUNDLED_DIR / name
    manifest_path = ext_dir / "manifest.json"
    if not manifest_path.exists():
        return False
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        model_file = manifest.get("model_file", "")
        if not model_file:
            # Adapter-only extension (weights live in external cache, e.g. HuggingFace).
            # Nothing to delete from disk — treat as success.
            return True
        weights = ext_dir / model_file
        if weights.exists():
            weights.unlink()
            return True
    except Exception:
        pass
    return False
