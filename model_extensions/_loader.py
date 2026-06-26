"""Scan model_extensions/ at startup and dynamically load adapters."""

import importlib.util
import sys
import json
from pathlib import Path

EXTENSIONS_DIR = Path(__file__).parent
# Ensure the Declas root (parent of model_extensions/) is importable
_DECLAS_ROOT = EXTENSIONS_DIR.parent
if str(_DECLAS_ROOT) not in sys.path:
    sys.path.insert(0, str(_DECLAS_ROOT))


def scan_extensions() -> dict:
    """Return metadata for every installed extension.

    Returns
    -------
    dict keyed by extension name:
        {
            "manifest": dict,
            "adapter_path":  str,
            "weights_path":  str | None,   # None when weights file is missing
            "status": "ready" | "missing_weights" | "missing_adapter" | "bad_manifest"
        }
    """
    installed = {}
    for manifest_path in EXTENSIONS_DIR.glob("*/manifest.json"):
        ext_dir  = manifest_path.parent
        name     = ext_dir.name
        if name.startswith("_"):
            continue
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
        except Exception:
            installed[name] = {"manifest": {}, "status": "bad_manifest"}
            continue

        adapter_file = ext_dir / manifest.get("adapter", "adapter.py")
        if not adapter_file.exists():
            installed[name] = {"manifest": manifest, "status": "missing_adapter"}
            continue

        model_filename = manifest.get("model_file", "")
        weights_file   = ext_dir / model_filename if model_filename else None
        weights_ready  = weights_file.exists() if weights_file else False

        # Fallback: also accept weights placed in the top-level model/ directory
        # (useful while the user hasn't gone through the Extensions download flow).
        if not weights_ready and model_filename:
            legacy = _DECLAS_ROOT / "model" / model_filename
            if legacy.exists():
                weights_file  = legacy
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
    manifest = extension_info["manifest"]

    spec = importlib.util.spec_from_file_location("_ext_adapter", adapter_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Find the ModelAdapter subclass defined in the module
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
