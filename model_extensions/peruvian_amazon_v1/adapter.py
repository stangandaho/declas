from model_extensions._base import ModelAdapter

_IMG_SIZE = (224, 224)

class PeruvianAmazonV1Adapter(ModelAdapter):
    """TensorFlow/Keras ResNet classifier. Requires: pip install tensorflow."""

    def load(self, model_path: str, device: str) -> None:
        import tensorflow as tf
        self._model   = tf.keras.models.load_model(model_path)
        self._classes = self.manifest.get("classes", [])

    def predict_single(self, image_path: str, conf_thres: float) -> list:
        import numpy as np
        import tensorflow as tf
        from PIL import Image
        img    = Image.open(image_path).convert("RGB").resize(_IMG_SIZE)
        arr    = np.expand_dims(np.array(img) / 255.0, axis=0).astype("float32")
        probs  = self._model.predict(arr, verbose=0)[0]
        best   = int(np.argmax(probs))
        conf   = float(probs[best])
        if conf < conf_thres:
            return []
        label = self._classes[best] if best < len(self._classes) else str(best)
        return [{"species": label, "confidence": conf, "bbox": None}]
