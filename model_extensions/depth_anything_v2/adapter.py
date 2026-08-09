
import numpy as np
from model_extensions._base import ModelAdapter

_MODEL_ID = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf"


class DepthAnythingV2Adapter(ModelAdapter):
    def __init__(self):
        super().__init__()
        self._processor = None
        self._model = None
        self._device = None

    def load(self, model_path: str, device: str) -> None:
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        except ImportError as e:
            raise RuntimeError(
                "Metric Depth Anything V2 requires:\n\n"
                "    pip install torch torchvision transformers pillow numpy\n\n"
                f"Original error: {e}"
            )

        requested_device = str(device).lower().strip()
        if requested_device == "cuda" and torch.cuda.is_available():
            self._device = torch.device("cuda")
        else:
            self._device = torch.device("cpu")

        self._processor = AutoImageProcessor.from_pretrained(_MODEL_ID)
        self._model = AutoModelForDepthEstimation.from_pretrained(_MODEL_ID)
        self._model.to(self._device)
        self._model.eval()

    def predict_depth(self, image_path: str) -> np.ndarray:
        from PIL import Image
        import torch
        import torch.nn.functional as F

        if self._model is None or self._processor is None:
            raise RuntimeError("Depth model has not been loaded.")

        image = Image.open(image_path).convert("RGB")
        original_w, original_h = image.size

        inputs = self._processor(images=image, return_tensors="pt")
        inputs = {
            key: value.to(self._device)
            for key, value in inputs.items()
            if hasattr(value, "to")
        }

        with torch.inference_mode():
            outputs = self._model(**inputs)
            predicted_depth = outputs.predicted_depth
            prediction = F.interpolate(
                predicted_depth.unsqueeze(1),
                size=(original_h, original_w),
                mode="bicubic",
                align_corners=False,
            ).squeeze(1)

        depth = prediction[0].detach().float().cpu().numpy()
        depth = np.asarray(depth, dtype=np.float32)

        invalid = ~np.isfinite(depth)
        depth[invalid] = 0.0
        depth[depth < 0] = 0.0

        return depth

    def predict_depth_batch(self, image_paths: list, fov_deg: float = None) -> list:
        from PIL import Image
        import torch
        import torch.nn.functional as F

        if self._model is None or self._processor is None:
            raise RuntimeError("Depth model has not been loaded.")

        if not image_paths:
            return []

        images = []
        original_sizes = []
        for path in image_paths:
            image = Image.open(path).convert("RGB")
            original_w, original_h = image.size
            images.append(image)
            original_sizes.append((original_w, original_h))

        inputs = self._processor(images=images, return_tensors="pt")
        inputs = {
            key: value.to(self._device)
            for key, value in inputs.items()
            if hasattr(value, "to")
        }

        with torch.inference_mode():
            outputs = self._model(**inputs)
            predicted_depth = outputs.predicted_depth
            results = []

            for i, (original_w, original_h) in enumerate(original_sizes):
                depth = F.interpolate(
                    predicted_depth[i : i + 1].unsqueeze(1),
                    size=(original_h, original_w),
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()

                depth = depth.detach().float().cpu().numpy()
                depth = np.asarray(depth, dtype=np.float32)

                invalid = ~np.isfinite(depth)
                depth[invalid] = 0.0
                depth[depth < 0] = 0.0
                results.append(depth)

        return results

    def predict_single(self, image_path: str, conf_thres: float) -> list:
        return []
