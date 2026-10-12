"""DeepForestVisionV2 two-pass pipeline for African tropical-forest camera traps.

MegaDetector v5 locates animals, people and vehicles; every animal crop is then
classified into 61 taxa by a fine-tuned DINOv3 ViT-B/16. The DINOv3 backbone is
implemented below in plain PyTorch so no extra package is needed.
Reference: Magaldi et al. (2026), https://github.com/MNHN-OFVI/DeepForestVisionV2
"""

import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from model_extensions.base import ModelAdapter

MD_CLASSES = {0: "animal", 1: "person", 2: "vehicle"}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


class RopeEmbedding(nn.Module):
    """2-D rotary position embedding of DINOv3 (eval mode, separate coord normalisation)."""

    def __init__(self, head_dim: int):
        super().__init__()
        self.register_buffer("periods", torch.ones(head_dim // 4))

    def forward(self, h: int, w: int):
        device = self.periods.device
        coords_h = torch.arange(0.5, h, device=device, dtype=torch.float32) / h
        coords_w = torch.arange(0.5, w, device=device, dtype=torch.float32) / w
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"), dim=-1)
        coords = 2.0 * coords.flatten(0, 1) - 1.0
        angles = 2 * math.pi * coords[:, :, None] / self.periods[None, None, :]
        angles = angles.flatten(1, 2).tile(2)
        return torch.sin(angles), torch.cos(angles)


class MaskedBiasLinear(nn.Linear):
    """qkv projection whose key bias is zeroed by a stored mask."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features)
        self.register_buffer("bias_mask", torch.ones(out_features))

    def forward(self, x):
        return F.linear(x, self.weight, self.bias * self.bias_mask.to(x.dtype))


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.qkv = MaskedBiasLinear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, rope):
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, c // self.num_heads)
        q, k, v = qkv.unbind(2)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        sin, cos = rope
        prefix = n - sin.shape[-2]
        q = torch.cat([q[:, :, :prefix], q[:, :, prefix:] * cos + rotate_half(q[:, :, prefix:]) * sin], dim=2)
        k = torch.cat([k[:, :, :prefix], k[:, :, prefix:] * cos + rotate_half(k[:, :, prefix:]) * sin], dim=2)
        x = F.scaled_dot_product_attention(q, k, v)
        return self.proj(x.transpose(1, 2).reshape(b, n, c))


class LayerScale(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * self.gamma


class Mlp(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-5)
        self.attn = Attention(dim, num_heads)
        self.ls1 = LayerScale(dim)
        self.norm2 = nn.LayerNorm(dim, eps=1e-5)
        self.mlp = Mlp(dim, dim * 4)
        self.ls2 = LayerScale(dim)

    def forward(self, x, rope):
        x = x + self.ls1(self.attn(self.norm1(x), rope))
        return x + self.ls2(self.mlp(self.norm2(x)))


class PatchEmbed(nn.Module):
    def __init__(self, dim: int, patch: int):
        super().__init__()
        self.proj = nn.Conv2d(3, dim, kernel_size=patch, stride=patch)

    def forward(self, x):
        return self.proj(x)


class DinoV3ViTB16(nn.Module):
    """DINOv3 ViT-B/16 backbone with 4 storage (register) tokens."""

    def __init__(self, dim: int = 768, depth: int = 12, num_heads: int = 12, storage_tokens: int = 4):
        super().__init__()
        self.storage_count = storage_tokens
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.storage_tokens = nn.Parameter(torch.zeros(1, storage_tokens, dim))
        self.mask_token = nn.Parameter(torch.zeros(1, dim))
        self.patch_embed = PatchEmbed(dim, 16)
        self.rope_embed = RopeEmbedding(dim // num_heads)
        self.blocks = nn.ModuleList(Block(dim, num_heads) for i in range(depth))
        self.norm = nn.LayerNorm(dim, eps=1e-5)

    def forward(self, x):
        """Return the normalised patch tokens, as DINOv3's get_intermediate_layers(n=1)."""
        x = self.patch_embed(x)
        h, w = x.shape[-2:]
        x = x.flatten(2).transpose(1, 2)
        b = x.shape[0]
        cls = (self.cls_token + 0 * self.mask_token).expand(b, -1, -1)
        x = torch.cat([cls, self.storage_tokens.expand(b, -1, -1), x], dim=1)
        rope = self.rope_embed(h, w)
        for block in self.blocks:
            x = block(x, rope)
        return self.norm(x)[:, self.storage_count + 1:]


class DeepForestVisionV2Net(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.backbone = DinoV3ViTB16()
        self.classifier = nn.Linear(768 * 2, num_classes)

    def forward(self, x):
        # Mirrors the official DinoV3Head: the first returned token and the mean
        # of the remaining ones are concatenated before the linear head.
        seq = self.backbone(x)
        return self.classifier(torch.cat([seq[:, 0], seq[:, 1:].mean(dim=1)], dim=1))


class DeepForestVisionV2Adapter(ModelAdapter):
    """MegaDetector v5 + DINOv3 ViT-B/16 classifier for 61 African forest taxa."""

    def load(self, model_path: str, device: str) -> None:
        from torchvision import transforms as T
        self.device = device

        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        state = checkpoint.get("model_state_dict", checkpoint)
        self.classes = list(checkpoint.get("labels") or self.manifest.get("classes", []))
        self.classifier = DeepForestVisionV2Net(len(self.classes))
        self.classifier.load_state_dict(state, strict=True)
        self.classifier.to(device).eval()
        del checkpoint, state

        self.transform = T.Compose([
            T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
        self.detector = self.load_detector(device)

    def load_detector(self, device: str):
        """Reuse installed MegaDetector v5 weights, otherwise let PytorchWildlife download them."""
        from PytorchWildlife.models import detection as pw_detection
        from PytorchWildlife.data import transforms as pw_trans

        self.md_transform = pw_trans.MegaDetector_v5_Transform(
            target_size=pw_detection.MegaDetectorV5.IMAGE_SIZE,
            stride=pw_detection.MegaDetectorV5.STRIDE,
        )
        ext_root = Path(__file__).resolve().parent.parent
        hub_cache = Path(torch.hub.get_dir()) / "checkpoints"
        for local in (ext_root / "megadetector_5a" / "md_v5a.0.0.pt",
                      ext_root / "megadetector_5b" / "md_v5b.0.0.pt",
                      hub_cache / "md_v5a.0.0.pt",
                      hub_cache / "MegaDetector_v5b.0.0.pt"):
            if local.exists():
                try:
                    return pw_detection.MegaDetectorV5(weights=str(local), device=device, pretrained=False)
                except Exception:
                    continue
        try:
            return pw_detection.MegaDetectorV5(device=device, pretrained=True, version="a")
        except Exception:
            return pw_detection.MegaDetectorV5(device=device, pretrained=True, version="b")

    def predict_single(self, image_path: str, conf_thres: float) -> list:
        import numpy as np
        from PIL import Image

        img = Image.open(image_path).convert("RGB")
        arr = np.array(img)
        with torch.no_grad():
            result = self.detector.single_image_detection(
                self.md_transform(arr), img_size=arr.shape, img_path=image_path,
                conf_thres=conf_thres,
            )
        dets = result["detections"]
        w, h = img.size

        detections = []
        for xyxy, det_conf, cls_id in zip(dets.xyxy, dets.confidence, dets.class_id):
            x1, y1, x2, y2 = [float(v) for v in xyxy]
            x1, y1 = max(0.0, min(x1, w - 1.0)), max(0.0, min(y1, h - 1.0))
            x2, y2 = max(1.0, min(x2, float(w))), max(1.0, min(y2, float(h)))
            bbox = [x1, y1, x2, y2]

            if int(cls_id) != 0:
                detections.append({"species": MD_CLASSES.get(int(cls_id), str(cls_id)),
                                   "confidence": float(det_conf), "bbox": bbox})
                continue

            crop = self.transform(img.crop((x1, y1, x2, y2))).unsqueeze(0).to(self.device)
            with torch.no_grad():
                probs = torch.softmax(self.classifier(crop), dim=-1)[0]
            best = int(probs.argmax())
            detections.append({
                "species": self.classes[best],
                "confidence": float(probs[best]),
                "detection_confidence": float(det_conf),
                "bbox": bbox,
            })
        return detections
