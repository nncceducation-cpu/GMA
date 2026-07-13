"""Export ViTPose-H to a self-contained TorchScript file.

WHY

The Docker image needs mmpose, which needs mmengine, mmcv and mmdet. mmcv has no
wheel for torch 2.11/cu128 and compiles from source — which needs a C++ toolchain.
Asking a clinical centre to install Visual Studio Build Tools is not a one-click
install; it is how a deployment dies.

Everything mmpose does at INFERENCE time is: crop the person box to 256x192,
run a ViT and a deconv head, and decode the heatmaps. The crop and the decode are
ordinary array code (see pipeline/pose_native.py). The network is the only part
worth keeping exactly as-is — so we export it, once, here, in the container that
already works, and ship the artefact.

The result runs on torch + torchvision alone. No mmcv, no mmengine, no compiler,
no Docker.

fp16 halves the file to ~1.3 GB, which fits under GitHub's 2 GB release-asset
limit, and we already established that fp16 does not move a keypoint (the model
runs in fp16 autocast in the container anyway).
"""
import sys
sys.path.insert(0, "/app")

import torch
from mmpose.apis.inferencers import Pose2DInferencer

OUT = "/app/models/vitpose_h.ts"
ALIAS = "td-hm_ViTPose-huge_8xb64-210e_coco-256x192"


class ViTPoseNet(torch.nn.Module):
    """backbone + head, nothing else. In: [B,3,256,192] normalised. Out: [B,17,64,48]."""

    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        return self.head(self.backbone(x))


m = Pose2DInferencer(model=ALIAS, device="cuda").model
m.eval()

net = ViTPoseNet(m.backbone, m.head).eval().cuda().half()

dummy = torch.randn(2, 3, 256, 192, device="cuda", dtype=torch.half)
with torch.no_grad():
    ref = net(dummy)
print("output shape:", list(ref.shape), "(expect [2, 17, 64, 48])")

with torch.no_grad():
    ts = torch.jit.trace(net, dummy, strict=False)
    ts = torch.jit.freeze(ts)
    out = ts(dummy)

err = (out.float() - ref.float()).abs().max().item()
print("traced-vs-eager max abs diff: %.2e" % err)
assert err < 1e-2, "trace does not reproduce the model"

torch.jit.save(ts, OUT)

import os
print("saved %s  (%.2f GB)" % (OUT, os.path.getsize(OUT) / 1e9))

# Also record the pre/post-processing constants the native runner must match.
cfg = m.cfg
print("\nconstants for pose_native.py:")
print("  input_size :", cfg.codec["input_size"])      # (W, H) = (192, 256)
print("  heatmap    :", cfg.codec["heatmap_size"])    # (48, 64)
print("  codec      :", cfg.codec["type"])            # UDPHeatmap
pp = m.data_preprocessor
print("  mean       :", pp.mean.flatten().tolist())
print("  std        :", pp.std.flatten().tolist())
print("  bgr_to_rgb :", getattr(pp, "_channel_conversion", None))
