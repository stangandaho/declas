from model_extensions._base import ModelAdapter

class DeepFauneClfV11Adapter(ModelAdapter):
    """DeepFaune ViT-Large/DINOv2 species classifier. Requires: pip install timm."""

    def load(self, model_path: str, device: str) -> None:
        import torch
        import timm
        self._device = device
        self._model  = timm.create_model(
            "vit_large_patch14_dinov2.lvd142m",
            pretrained=False,
            num_classes=len(self.manifest.get("classes", [])),
        )
        state = torch.load(model_path, map_location=device)
        if "state_dict" in state:
            state = state["state_dict"]
        self._model.load_state_dict(state, strict=False)
        self._model.to(device)
        self._model.eval()
        data_cfg = timm.data.resolve_data_config(self._model.pretrained_cfg)
        self._transform = timm.data.create_transform(**data_cfg)
        self._classes   = self.manifest.get("classes", [])

    def predict_single(self, image_path: str, conf_thres: float) -> list:
        import torch
        from PIL import Image
        img    = Image.open(image_path).convert("RGB")
        tensor = self._transform(img).unsqueeze(0).to(self._device)
        with torch.no_grad():
            logits = self._model(tensor)
            probs  = torch.softmax(logits, dim=1)[0]
        best_idx  = int(probs.argmax())
        best_conf = float(probs[best_idx])
        if best_conf < conf_thres:
            return []
        label = self._classes[best_idx] if best_idx < len(self._classes) else str(best_idx)
        return [{"species": label, "confidence": best_conf, "bbox": None}]
