"""Scan model_extensions/ at startup and dynamically load adapters."""

import importlib.util
import sys
import json
from pathlib import Path

# In a packaged build sys._MEIPASS points to _internal/; adapters and weights
# both live under _internal/model_extensions/<name>/.
if getattr(sys, 'frozen', False):
    _BUNDLED_DIR = Path(sys._MEIPASS) / "model_extensions"
else:
    _BUNDLED_DIR = Path(__file__).parent

_DECLAS_ROOT = _BUNDLED_DIR.parent
if str(_DECLAS_ROOT) not in sys.path:
    sys.path.insert(0, str(_DECLAS_ROOT))


def scan_extensions() -> dict:
    """Return metadata for every installed extension.

    Scans _BUNDLED_DIR for manifest.json files. Weights are expected in the
    same directory (downloaded there by download_extension).

    Returns
    -------
    dict keyed by extension name:
        {
            "manifest":      dict,
            "adapter_path":  str,
            "weights_path":  str | None,
            "status":        "ready" | "missing_weights" | "missing_adapter" | "bad_manifest"
        }
    """
    installed = {}

    for manifest_path in sorted(_BUNDLED_DIR.glob("*/manifest.json")):
        name = manifest_path.parent.name
        if name.startswith("_"):
            continue

        ext_dir = _BUNDLED_DIR / name

        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
        except Exception:
            installed[name] = {"manifest": {}, "status": "bad_manifest"}
            continue

        adapter_name = manifest.get("adapter", "adapter.py")
        adapter_file = ext_dir / adapter_name
        if not adapter_file.exists():
            installed[name] = {"manifest": manifest, "status": "missing_adapter"}
            continue

        model_filename = manifest.get("model_file", "")
        weights_file   = None
        weights_ready  = False

        if model_filename:
            candidate = ext_dir / model_filename
            if candidate.exists():
                weights_file  = candidate
                weights_ready = True

        installed[name] = {
            "manifest":     manifest,
            "adapter_path": str(adapter_file),
            "weights_path": str(weights_file) if weights_ready else None,
            "status":       "ready" if weights_ready else "missing_weights",
        }

    return installed


def load_adapter(extension_info: dict, device: str = "cpu"):
    """Instantiate and load the adapter described by extension_info.

    Parameters
    ----------
    extension_info : one entry from scan_extensions()
    device : "cpu" or "cuda"

    Returns
    -------
    ModelAdapter instance with load() already called.
    """
    from model_extensions._base import ModelAdapter

    adapter_path = extension_info["adapter_path"]
    weights_path = extension_info["weights_path"]
    manifest     = extension_info["manifest"]

    spec   = importlib.util.spec_from_file_location("_ext_adapter", adapter_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    adapter_cls = None
    for attr_name in dir(module):
        attr = getattr(module, attr_name)
        try:
            if (isinstance(attr, type)
                    and issubclass(attr, ModelAdapter)
                    and attr is not ModelAdapter):
                adapter_cls = attr
                break
        except TypeError:
            continue

    if adapter_cls is None:
        raise RuntimeError(
            f"No ModelAdapter subclass found in {adapter_path}.\n"
            "Make sure your adapter.py defines a class that inherits from ModelAdapter."
        )

    adapter          = adapter_cls()
    adapter.manifest = manifest
    adapter.load(weights_path, device)
    return adapter
