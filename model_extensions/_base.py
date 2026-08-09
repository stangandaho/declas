"""Base class every Declas model-extension adapter must inherit from.

Extension authors subclass ModelAdapter in their adapter.py, implement the three
methods, and ship the file alongside manifest.json in their extension ZIP.
"""

from pathlib import Path


class ModelAdapter:
    """Abstract base for Declas model extensions."""

    # Set by the loader before load() is called.
    manifest: dict = {}

    def load(self, model_path: str, device: str) -> None:
        """Load model weights from disk."""
        raise NotImplementedError(f"{type(self).__name__} must implement load()")

    def predict_single(self, image_path: str, conf_thres: float) -> list:
        """Run inference on one image."""
        raise NotImplementedError(
            f"{type(self).__name__} must implement predict_single()"
        )

    def predict_depth(self, image_path: str):
        """
        Estimate a per-pixel metric Z-depth map.

        Returns
        -------
        numpy.ndarray
            2-D H×W array.
            Values are metric depth in metres when implemented by
            a metric-depth adapter.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support depth estimation"
        )

    def predict_depth_batch(self, image_paths: list, fov_deg: float = None) -> list:
        """Estimate depth maps for multiple images."""
        return [self.predict_depth(p) for p in image_paths]

    def predict_batch(
        self,
        image_dir: str,
        conf_thres: float,
        extension: str = ".JPG"
    ) -> dict:
        """Run inference on all matching images in image_dir."""

        results = {}

        image_dir = Path(image_dir)
        files = sorted(image_dir.glob(f"*{extension}"))

        if not files:
            files = sorted(
                image_dir.glob(
                    f"*{extension.upper()}" if extension.islower() else f"*{extension.lower()}"
                )
            )

        for img in files:

            results[img.stem] = self.predict_single(
                str(img),
                conf_thres
            )

        return results