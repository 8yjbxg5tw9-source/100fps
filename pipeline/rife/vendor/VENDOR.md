# Vendored RIFE inference code — provenance & patch notes

## Source

- Project: **RIFE — Real-Time Intermediate Flow Estimation for Video Frame
  Interpolation** ([arXiv:2011.06294](https://arxiv.org/abs/2011.06294))
- Repository: <https://github.com/megvii-research/ECCV2022-RIFE>
  (mirror of <https://github.com/hzwer/arXiv2020-RIFE>)
- Vendored commit: `5d8adbd` (file `model/{IFNet,refine,warplayer}.py`)
- License: **MIT** (© Megvii Inc. / hzwer) — full text in `LICENSE-RIFE`.
  Kept verbatim as required by the license.

## What is vendored (and why)

`upstream/` contains the exact upstream inference path for the RIFE v4
architecture, byte-identical except for the import patch below:

| File | Role |
|------|------|
| `upstream/IFNet.py` | `IFNet` — coarse-to-fine flow estimator + refinement UNet |
| `upstream/refine.py` | `Contextnet`, `Unet` used by `IFNet` |
| `upstream/warplayer.py` | `warp()` backwarping helper |

Deliberately **not** vendored:

- `model/RIFE.py` — the training-oriented `Model` class (imports the whole
  training stack: `loss.py`, `laplacian.py`, `IFNet_m.py`, DDP, optimizers).
  Our inference-only equivalent is `model.py` (`RifeModel`), adapted from
  upstream's `Model.inference()` logic (concat → `flownet(imgs, scale_list,
  timestep)` → `merged[2]`).
- `model/IFNet_m.py`, `model/loss.py`, `model/laplacian.py`,
  `model/pytorch_msssim/` — training / arbitrary-time / metric code, unused
  for fixed `t=0.5` midpoint interpolation.

## Patch applied to vendored files

Upstream files use absolute imports (`from model.warplayer import warp`)
which assume the repository root is on `sys.path`. The only modification is
mechanical, applied with `sed 's/^from model\./from ./'`:

```diff
-from model.warplayer import warp
-from model.refine import *
+from .warplayer import warp
+from .refine import *
```

No logic was changed. To re-vendor from a newer upstream commit, re-copy the
three files and re-apply the one-line import patch, then update the commit
hash above.

## Weight compatibility

This code implements the **RIFE v4** architecture (`IFNet` with
`block0/1/2 + block_tea + contextnet + unet`). Compatible weights:

- Official ECCV paper v4 model and HD v4.x models (Google Drive links in the
  upstream README; see `pipeline/rife/weights.py`).
- Community `rife4N.pth` files whose state dict matches these module names
  (keys carry a `module.` prefix from DDP training — stripped on load).

Newer Practical-RIFE versions (4.15+, changed block layout) are **not**
compatible with this vendored code by design; the loader fails loudly
(`strict=True` state-dict load) instead of silently misbehaving.
