"""Depth Anything V2 Metric Outdoor adapter for Declas distance estimation.

Requires: pip install transformers>=4.46 torch
Weights (~97 MB Small model) are downloaded to the HuggingFace cache on first use
and reused on subsequent runs.
"""

import numpy as np
from model_extensions._base import ModelAdapter

_MODEL_ID = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf"


class DepthAnythingV2Adapter(ModelAdapter):

    def load(self, model_path: str, device: str) -> None:
        try:
            from transformers import pipeline
        except ImportError:
            raise RuntimeError(
                "The 'transformers' package is required for depth estimation.\n"
                "Install it with:  pip install transformers>=4.46"
            )
        import torch
        dev = 0 if (device == "cuda" and torch.cuda.is_available()) else -1
        self._pipe = pipeline(
            task="depth-estimation",
            model=_MODEL_ID,
            device=dev,
        )

    def predict_depth(self, image_path: str) -> np.ndarray:
        """Return a (H, W) float32 array of metric depth values in metres."""
        from PIL import Image
        img = Image.open(image_path).convert("RGB")
        out = self._pipe(img)
        return np.array(out["predicted_depth"], dtype=np.float32)

    def predict_single(self, image_path: str, conf_thres: float) -> list:
        return []  # not a detection model
