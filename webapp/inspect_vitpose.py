"""What is actually inside the ViTPose-H checkpoint? Measure, don't guess."""
import sys
sys.path.insert(0, "/app")
import torch
from collections import OrderedDict

CKPT = "/app/.torch/hub/checkpoints/td-hm_ViTPose-huge_8xb64-210e_coco-256x192-e32adcd4_20230314.pth"
sd = torch.load(CKPT, map_location="cpu", weights_only=False)
sd = sd.get("state_dict", sd)

print("total tensors:", len(sd))
print("\n--- top-level prefixes ---")
pref = OrderedDict()
for k in sd:
    p = k.split(".")[0]
    pref[p] = pref.get(p, 0) + 1
for p, n in pref.items():
    print("  %-12s %d tensors" % (p, n))

print("\n--- backbone: non-layer keys (structure) ---")
for k, v in sd.items():
    if k.startswith("backbone.") and ".layers." not in k:
        print("  %-46s %s" % (k, list(v.shape)))

print("\n--- backbone.layers.0 (one transformer block) ---")
for k, v in sd.items():
    if k.startswith("backbone.layers.0."):
        print("  %-46s %s" % (k, list(v.shape)))

n_layers = 1 + max(int(k.split(".")[2]) for k in sd if k.startswith("backbone.layers."))
print("\ndepth (blocks):", n_layers)

print("\n--- head ---")
for k, v in sd.items():
    if k.startswith("head.") or k.startswith("keypoint_head."):
        print("  %-46s %s" % (k, list(v.shape)))

# the live model, for the exact config
from mmpose.apis.inferencers import Pose2DInferencer
m = Pose2DInferencer(model="td-hm_ViTPose-huge_8xb64-210e_coco-256x192",
                     device="cpu").model
b = m.cfg.model["backbone"]
h = m.cfg.model["head"]
print("\n--- config: backbone ---")
for k in ("type", "arch", "img_size", "patch_size", "embed_dims", "num_layers",
          "num_heads", "feedforward_channels", "qkv_bias", "with_cls_token",
          "out_type", "final_norm", "drop_path_rate", "ratio"):
    if k in b:
        print("  %-24s %s" % (k, b[k]))
print("--- config: head ---")
for k, v in h.items():
    if k not in ("decoder", "loss"):
        print("  %-24s %s" % (k, v))
print("  codec:", m.cfg.get("codec", {}))
