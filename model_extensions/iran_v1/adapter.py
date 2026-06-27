from model_extensions._base import ModelAdapter

class IranV1Adapter(ModelAdapter):

    def load(self, model_path: str, device: str) -> None:
        from ultralytics import YOLO
        self._model  = YOLO(model_path)
        self._device = device
        self._names  = self._model.names

    def predict_single(self, image_path: str, conf_thres: float) -> list:
        results = self._model.predict(
            source=image_path, conf=conf_thres,
            device=self._device, verbose=False,
        )
        detections = []
        for r in results:
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
