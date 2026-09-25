import ast

import numpy as np

from model_extensions.base import ModelAdapter


class CaiSubSaharanAfricaAdapter(ModelAdapter):

    def load(self, model_path: str, device: str) -> None:
        import onnxruntime as ort

        providers = ["CPUExecutionProvider"]
        if (str(device).lower().startswith("cuda")
                and "CUDAExecutionProvider" in ort.get_available_providers()):
            providers.insert(0, "CUDAExecutionProvider")
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name

        meta = self.session.get_modelmeta().custom_metadata_map
        names = ast.literal_eval(meta.get("names", "{}"))
        self.labels = {int(k): v for k, v in names.items()}
        imgsz = ast.literal_eval(meta.get("imgsz", "[640, 640]"))
        self.input_h, self.input_w = (imgsz, imgsz) if isinstance(imgsz, int) else imgsz

    def predict_single(self, image_path: str, conf_thres: float) -> list:
        import cv2
        from PIL import Image

        img = np.array(Image.open(image_path).convert("RGB"))
        img_h, img_w = img.shape[:2]

        # Letterbox to the export size, as Ultralytics does at training time.
        scale = min(self.input_h / img_h, self.input_w / img_w)
        new_w, new_h = round(img_w * scale), round(img_h * scale)
        top = (self.input_h - new_h) // 2
        left = (self.input_w - new_w) // 2
        canvas = np.full((self.input_h, self.input_w, 3), 114, dtype=np.uint8)
        canvas[top:top + new_h, left:left + new_w] = cv2.resize(
            img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        blob = canvas.transpose(2, 0, 1)[None].astype(np.float32) / 255.0

        # YOLOv10 is NMS-free: each row is [x1, y1, x2, y2, score, class].
        rows = self.session.run(None, {self.input_name: blob})[0][0]

        detections = []
        for x1, y1, x2, y2, score, cls_id in rows:
            if score < conf_thres:
                continue
            detections.append({
                "species": self.labels.get(int(cls_id), str(int(cls_id))),
                "confidence": float(score),
                "bbox": [
                    float(np.clip((x1 - left) / scale, 0, img_w)),
                    float(np.clip((y1 - top) / scale, 0, img_h)),
                    float(np.clip((x2 - left) / scale, 0, img_w)),
                    float(np.clip((y2 - top) / scale, 0, img_h)),
                ],
            })
        return detections
