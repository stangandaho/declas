"""Base class every Declas model-extension adapter must inherit from.

Extension authors subclass ModelAdapter in their adapter.py, implement the three
methods, and ship the file alongside manifest.json in their extension ZIP.
"""

from pathlib import Path


class ModelAdapter:
    """Abstract base for Declas model extensions.

    Declas calls load() once after install, then predict_single() or
    predict_batch() for each inference request.

    Mandatory output fields per detection dict
    ------------------------------------------
    species     : str          - class label (species name)
    confidence  : float        - detection confidence, 0-1
    bbox        : list | None  - [x1, y1, x2, y2] pixel coords (xyxy), or None
                                 for pure-classifier models (always include the key)
    """

    # Set by the loader before load() is called — read-only inside the adapter.
    manifest: dict = {}

    def load(self, model_path: str, device: str) -> None:
        """Load model weights from disk.

        Parameters
        ----------
        model_path : absolute path to the weights file declared in manifest["model_file"]
        device     : "cpu" or "cuda"
        """
        raise NotImplementedError(f"{type(self).__name__} must implement load()")

    def predict_single(self, image_path: str, conf_thres: float) -> list:
        """Run inference on one image.

        Returns
        -------
        list[dict] — one entry per detected/classified animal:
            {
                "species":    str,               # class label
                "confidence": float,             # 0-1
                "bbox":       [x1,y1,x2,y2]      # pixel coords, or None
            }
        Empty list when nothing is detected.
        """
        raise NotImplementedError(f"{type(self).__name__} must implement predict_single()")

    def predict_batch(self, image_dir: str, conf_thres: float,
                      extension: str = ".JPG") -> dict:
        """Run inference on all matching images in image_dir.

        Returns
        -------
        dict keyed by image stem → list[dict] as returned by predict_single().
        """
        # Default implementation — override for efficiency (e.g. batched GPU inference).
        results = {}
        image_dir = Path(image_dir)
        files = sorted(image_dir.glob(f"*{extension}"))
        if not files:
            files = sorted(image_dir.glob(
                f"*{extension.upper() if extension.islower() else extension.lower()}"
            ))
        for img in files:
            results[img.stem] = self.predict_single(str(img), conf_thres)
        return results
