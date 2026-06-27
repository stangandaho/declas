from model_extensions._base import ModelAdapter


def _fix_clf_transforms(model):
    """Replace environment-specific pickled transforms with current-env ones."""
    if hasattr(model.model, "transforms"):
        try:
            from ultralytics.data.augment import classify_transforms
            imgsz = getattr(model.model, "args", {})
            imgsz = imgsz.get("imgsz", 224) if isinstance(imgsz, dict) else 224
            if isinstance(imgsz, (list, tuple)):
                imgsz = imgsz[0]
            model.model.transforms = classify_transforms(int(imgsz))
        except Exception:
            del model.model.transforms


class NamibDesertV1Adapter(ModelAdapter):

    def load(self, model_path: str, device: str) -> None:
        from ultralytics import YOLO
        self._model  = YOLO(model_path)
        self._device = device
        self._names  = self._model.names
        self._task   = self._model.task
        if self._task == "classify":
            _fix_clf_transforms(self._model)

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
            if self._task == "classify":
                if r.probs is None:
                    continue
                for i, conf in enumerate(r.probs.data.tolist()):
                    if conf >= conf_thres:
                        detections.append({
                            "species":    self._names.get(i, str(i)),
                            "confidence": float(conf),
                            "bbox":       None,
                        })
            else:
                if r.boxes is None:
                    continue
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    detections.append({
                        "species":    self._names.get(cls_id, str(cls_id)),
                        "confidence": float(box.conf[0]),
                        "bbox":       box.xyxy[0].tolist(),
                    })
        return detections
