import os
import sys
from typing import Optional

import torch
import torch.nn as nn


class DeepEncoderVisionTower(nn.Module):
    """
    A thin adapter that wraps DeepSeek-OCR's deepencoder (SAM + CLIP + Projector)
    and exposes a single vision tower to replace Qwen-VL's visual encoder.

    Requirements:
    - Ensure the external deepencoder repo is importable by appending its path.
    - Provide checkpoints if needed (SAM checkpoint supported by builder; CLIP/Projector optional).
    """

    def __init__(
        self,
        deepencoder_path: Optional[str] = None,
        sam_checkpoint: Optional[str] = None,
        clip_checkpoint: Optional[str] = None,
        projector_checkpoint: Optional[str] = None,
        freeze_sam: bool = True,
        freeze_clip: bool = False,
        projector_type: str = "mlp_gelu",
        projector_input_dim: int = 1024,
        projector_output_dim: int = 1024,
    ):
        super().__init__()

        if deepencoder_path and deepencoder_path not in sys.path:
            sys.path.insert(0, deepencoder_path)

        # Lazy import to avoid hard dependency when not used
        from deepencoder.sam_vary_sdpa import build_sam_vit_b
        from deepencoder.clip_sdpa import build_clip_l
        from deepencoder.build_linear import MlpProjector
        from easydict import EasyDict as adict

        # Build SAM encoder
        # Note: build_sam_vit_b supports a checkpoint argument in some variants.
        self.sam_encoder = build_sam_vit_b(checkpoint=sam_checkpoint)
        if freeze_sam:
            for p in self.sam_encoder.parameters():
                p.requires_grad = False

        # Build CLIP encoder
        self.clip_encoder = build_clip_l()
        if freeze_clip:
            for p in self.clip_encoder.parameters():
                p.requires_grad = False

        # Build projector
        proj_cfg = adict(
            projector_type=projector_type,
            input_dim=projector_input_dim,
            n_embed=projector_output_dim,
            depth=2,
            mlp_ratio=1,
        )
        self.projector = MlpProjector(proj_cfg)
        self.output_dim = projector_output_dim

        # Optionally load CLIP/Projector checkpoints (best-effort)
        if clip_checkpoint and os.path.isfile(clip_checkpoint):
            state = torch.load(clip_checkpoint, map_location="cpu")
            try:
                self.clip_encoder.load_state_dict(state, strict=False)
            except Exception:
                pass
        if projector_checkpoint and os.path.isfile(projector_checkpoint):
            state = torch.load(projector_checkpoint, map_location="cpu")
            try:
                self.projector.load_state_dict(state, strict=False)
            except Exception:
                pass

    @torch.no_grad()
    def _resize_for_sam(self, images: torch.Tensor) -> torch.Tensor:
        if images.shape[-2:] == (1024, 1024):
            return images
        return torch.nn.functional.interpolate(images, size=(1024, 1024), mode='bilinear', align_corners=False)

    @torch.no_grad()
    def _resize_for_clip(self, images: torch.Tensor) -> torch.Tensor:
        if images.shape[-2:] == (224, 224):
            return images
        return torch.nn.functional.interpolate(images, size=(224, 224), mode='bilinear', align_corners=False)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: [B, 3, H, W]
        Returns:
            features: [B, N, D]
        """
        # SAM path uses 1024x1024
        images_sam = self._resize_for_sam(images)
        sam_features = self.sam_encoder(images_sam)  # [B, C, H', W']

        # CLIP path uses 224x224
        images_clip = self._resize_for_clip(images)
        sam_feat_flat = sam_features.flatten(2).permute(0, 2, 1)  # [B, HW, C]
        clip_features = self.clip_encoder(images_clip, sam_feat_flat)  # [B, N, D]

        features = self.projector(clip_features)  # [B, N, D]
        return features


