from model_extensions._base import ModelAdapter


class SouthwestUSAV3Adapter(ModelAdapter):

    def load(self, model_path, device):
        import torch
        import torch.nn as nn
        import torchvision.models as tv
        import torchvision.transforms as T

        self._device = device
        self._classes = self.manifest.get("classes", [])
        nc = len(self._classes)

        ckpt = torch.load(model_path, map_location=device, weights_only=False)
        state = {k[len("model."):]: v for k, v in ckpt["model"].items()}

        self._model = tv.efficientnet_v2_m(weights=None)
        self._model.classifier[1] = nn.Linear(self._model.classifier[1].in_features, nc)
        self._model.load_state_dict(state, strict=True)
        self._model.to(device)
        self._model.eval()

        self._transform = T.Compose([
            T.Resize(384),
            T.CenterCrop(384),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def predict_single(self, image_path, conf_thres):
        import torch
        from PIL import Image

        img = Image.open(image_path).convert("RGB")
        tensor = self._transform(img).unsqueeze(0).to(self._device)
        with torch.no_grad():
            probs = torch.softmax(self._model(tensor), dim=1)[0]

        detections = []
        for i, conf in enumerate(probs.tolist()):
            if conf >= conf_thres:
                label = self._classes[i] if i < len(self._classes) else str(i)
                detections.append({"species": label, "confidence": float(conf), "bbox": None})
        return detections
