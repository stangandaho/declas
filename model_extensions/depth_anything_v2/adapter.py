
import numpy as np
from model_extensions.base import ModelAdapter

MODEL_ID = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf"


class DepthAnythingV2Adapter(ModelAdapter):
    def __init__(self):
        super().__init__()
        self.processor = None
        self.model = None
        self.device = None

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
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.processor = AutoImageProcessor.from_pretrained(MODEL_ID)
        self.model = AutoModelForDepthEstimation.from_pretrained(MODEL_ID)
        self.model.to(self.device)
        self.model.eval()

    def predict_depth(self, image_path: str) -> np.ndarray:
        from PIL import Image
        import torch
        import torch.nn.functional as F

        if self.model is None or self.processor is None:
            raise RuntimeError("Depth model has not been loaded.")

        image = Image.open(image_path).convert("RGB")
        original_w, original_h = image.size

        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
            if hasattr(value, "to")
        }

        with torch.inference_mode():
            outputs = self.model(**inputs)
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
        return [self.predict_depth(p) for p in image_paths]

    def predict_single(self, image_path: str, conf_thres: float) -> list:
        return []
