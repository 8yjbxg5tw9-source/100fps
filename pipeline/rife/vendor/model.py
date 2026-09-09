"""Inference-only RIFE v4 wrapper (adapted from upstream ``model/RIFE.py``).

Upstream source: https://github.com/megvii-research/ECCV2022-RIFE
(vendored commit ``5d8adbd``, MIT license — see ``LICENSE-RIFE``).

This keeps the exact upstream inference path::

    imgs = torch.cat((img0, img1), 1)
    flow, mask, merged, *_ = flownet(imgs, scale_list, timestep=timestep)
    return merged[2]

while dropping the training stack (optimizers, losses, DDP) that upstream's
``Model`` class pulls in. ``torch`` is imported lazily so that merely
importing this module never requires torch to be installed.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


class RifeModel:
    """Minimal inference-only port of upstream ``RIFE.Model``."""

    def __init__(self, weights_path: Optional[str] = None) -> None:
        import torch  # lazy: torch is only needed on the inference path

        from pipeline.rife.vendor.upstream.IFNet import IFNet

        self.torch = torch
        self.flownet = IFNet()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if weights_path is not None:
            self.load_weights(weights_path)

    # -- lifecycle ------------------------------------------------------------
    def eval(self) -> "RifeModel":
        self.flownet.eval()
        return self

    def to(self, device: Any) -> "RifeModel":
        self.device = self.torch.device(device)
        self.flownet.to(self.device)
        return self

    def half(self) -> "RifeModel":
        """Convert to fp16 (CUDA only — mirrors upstream fp16 inference)."""
        self.flownet.half()
        return self

    def load_weights(self, weights_path: str) -> "RifeModel":
        """Load a ``flownet.pkl`` / ``rife*.pth`` state dict (strict).

        Upstream training saves DDP-wrapped keys (``module.*`` prefix); the
        prefix is stripped exactly like upstream ``Model.load_model``. Files
        whose keys don't match the v4 architecture fail loudly instead of
        silently producing garbage.
        """
        state = self.torch.load(weights_path, map_location="cpu")
        if isinstance(state, dict) and any(
            k.startswith("module.") for k in state.keys()
        ):
            state = {k.replace("module.", "", 1): v for k, v in state.items()}
        missing, unexpected = self._load_strict(state)
        if missing or unexpected:
            raise RuntimeError(
                f"Weights {weights_path} do not match the vendored RIFE v4 "
                f"architecture (missing={len(missing)}, unexpected={len(unexpected)}). "
                f"Use official v4.x weights, e.g. via automatic download "
                f"(see pipeline.rife.weights). "
                f"First missing: {sorted(missing)[:3]}; "
                f"first unexpected: {sorted(unexpected)[:3]}"
            )
        return self

    def _load_strict(self, state: Dict[str, Any]) -> Any:
        result = self.flownet.load_state_dict(state, strict=False)
        return result.missing_keys, result.unexpected_keys

    # -- inference (upstream logic, batch-safe) -------------------------------
    def inference(
        self,
        img0: Any,  # (B,3,H,W) float tensor in [0,1], already padded
        img1: Any,
        scale_list: Optional[List[float]] = None,
        timestep: float = 0.5,
    ) -> Any:
        if scale_list is None:
            scale_list = [4.0, 2.0, 1.0]
        imgs = self.torch.cat((img0, img1), 1)
        _, _, merged, *_ = self.flownet(imgs, scale_list, timestep=timestep)
        return merged[2]
