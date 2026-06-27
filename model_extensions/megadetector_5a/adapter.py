from model_extensions._base import ModelAdapter

_CLASSES = {0: "animal", 1: "person", 2: "vehicle"}

class MegaDetectorV5aAdapter(ModelAdapter):

    def load(self, model_path: str, device: str) -> None:
        try:
            from ultralytics import YOLO
            self._model = YOLO(model_path, task="detect")
            self._use_ultralytics = True
        except Exception:
            import torch
            self._model = torch.hub.load(
                "ultralytics/yolov5", "custom",
                path=model_path, force_reload=False, trust_repo=True,
            )
            self._use_ultralytics = False
        self._device = device

    def predict_single(self, image_path: str, conf_thres: float) -> list:
        import numpy as np
        from PIL import Image as _PIL
        img = np.array(_PIL.open(image_path).convert("RGB"))

        if self._use_ultralytics:
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
        else:
            # torch-hub YOLOv5 — accepts numpy RGB
            results = self._model(img, size=640)
            detections = []
            for row in results.xyxy[0].tolist():
                x1, y1, x2, y2, conf, cls_id = row
                if conf >= conf_thres:
                    detections.append({
                        "species":    _CLASSES.get(int(cls_id), str(int(cls_id))),
                        "confidence": float(conf),
                        "bbox":       [x1, y1, x2, y2],
                    })
        return detections
