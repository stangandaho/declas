from model_extensions._base import ModelAdapter

class AI4GAmazonClfAdapter(ModelAdapter):

    def load(self, model_path: str, device: str) -> None:
        import PytorchWildlife.models as pw_models
        self._device = device
        self._clf = pw_models.classification.AI4GAmazonClassification(
            weights=model_path, device=device
        )

    def predict_single(self, image_path: str, conf_thres: float) -> list:
        from PIL import Image
        img = Image.open(image_path).convert("RGB")
        results = self._clf.single_image_classification(img)
        detections = []
        for pred in (results if isinstance(results, list) else [results]):
            label = pred.get("class", "Unknown")
            conf  = float(pred.get("confidence", 0.0))
            if conf >= conf_thres:
                detections.append({"species": label, "confidence": conf, "bbox": None})
        return detections
