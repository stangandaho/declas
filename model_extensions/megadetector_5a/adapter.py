from model_extensions._base import ModelAdapter

_CLASSES = {0: "animal", 1: "person", 2: "vehicle"}

class MegaDetectorV5aAdapter(ModelAdapter):

    def load(self, model_path: str, device: str) -> None:
        try:
            from ultralytics import YOLO
            self._model = YOLO(model_path, task="detect")
        except Exception:
            import torch
            self._model = torch.hub.load(
                "ultralytics/yolov5", "custom",
                path=model_path, force_reload=False, trust_repo=True,
            )
        self._device = device

    def predict_single(self, image_path: str, conf_thres: float) -> list:
        results = self._model(image_path, conf=conf_thres, device=self._device, verbose=False)
        detections = []
        for r in (results if hasattr(results, "__iter__") else [results]):
            boxes = getattr(r, "boxes", None) or getattr(r, "xyxy", [None])[0]
            if boxes is None:
                continue
            if hasattr(boxes, "xyxy"):
                for box in boxes:
                    cls_id = int(box.cls[0])
                    detections.append({
                        "species":    _CLASSES.get(cls_id, str(cls_id)),
                        "confidence": float(box.conf[0]),
                        "bbox":       box.xyxy[0].tolist(),
                    })
            else:
                # torch-hub tensor format [x1, y1, x2, y2, conf, cls]
                for row in boxes:
                    cls_id = int(row[5])
                    detections.append({
                        "species":    _CLASSES.get(cls_id, str(cls_id)),
                        "confidence": float(row[4]),
                        "bbox":       row[:4].tolist(),
                    })
        return detections
