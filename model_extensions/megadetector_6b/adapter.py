from model_extensions._base import ModelAdapter

_CLASSES = {0: "animal", 1: "person", 2: "vehicle"}

class MegaDetectorV6bAdapter(ModelAdapter):

    def load(self, model_path: str, device: str) -> None:
        from ultralytics import YOLO
        self._model = YOLO(model_path)
        self._device = device

    def predict_single(self, image_path: str, conf_thres: float) -> list:
        import numpy as np
        from PIL import Image as _PIL
        img = np.array(_PIL.open(image_path).convert("RGB"))
        results = self._model.predict(
            source=img, conf=conf_thres,
            device=self._device, verbose=False,
        )
        detections = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                cls_id = int(box.cls[0])
                detections.append({
                    "species":    _CLASSES.get(cls_id, str(cls_id)),
                    "confidence": float(box.conf[0]),
                    "bbox":       box.xyxy[0].tolist(),
                })
        return detections
