"""Does the mm*-free runner reproduce mmpose exactly?

If it does not agree to sub-pixel on a real clip, it does not ship. A pose
pipeline that is "close" is a pipeline that measures something slightly different
from the one that produced the training data — and every feature downstream is a
derivative, which amplifies exactly that kind of small, systematic offset.
"""
import sys, time
sys.path.insert(0, "/app")
from pathlib import Path

import cv2
import numpy as np

from pipeline.pose_extract import DET_EVERY, PAD_FRAC, PoseExtractor
from pipeline.pose_native import NativeViTPose
from pipeline.normalise import COCO, normalise

RID = "914f311ddd18"
D = Path("/app/webapp/data_runtime/raw/recordings") / RID

# --- ground truth: mmpose ViTPose-H, as stored by the app -------------------
z = np.load(D / "pose_raw.npz")
gt_xy, gt_conf, src_fps = z["xy"], z["conf"], float(z["fps"])
print("mmpose ViTPose-H : %d frames" % len(gt_xy))

# --- same boxes, so we compare the POSE PATH, not the detector --------------
pe = PoseExtractor(device="cuda")
det = pe._detector()
dev = next(det.parameters()).device

cap = cv2.VideoCapture(str(D / "video.mp4"))
frames, boxes, last, i = [], [], None, 0
while True:
    ok, f = cap.read()
    if not ok:
        break
    h, w = f.shape[:2]
    if i % DET_EVERY == 0 or last is None:
        bb = pe._detect_one(f, det, dev)
        if bb is not None:
            last = pe._pad_box(bb, w, h, PAD_FRAC)
    frames.append(f); boxes.append(last); i += 1
cap.release()

# --- candidate: torch-only native runner -----------------------------------
net = NativeViTPose("/app/models/vitpose_h.ts", device="cuda")
xs, cs = [], []
t0 = time.time()
for s in range(0, len(frames), 24):
    a, b = net(frames[s:s + 24], boxes[s:s + 24])
    xs.append(a); cs.append(b)
dt = time.time() - t0
nx = pe._fill_gaps(np.concatenate(xs))
nc = np.concatenate(cs)
print("native torch     : %d frames in %.0fs (%.1f fps)" % (len(nx), dt, len(nx) / dt))

n = min(len(gt_xy), len(nx))
d = np.linalg.norm(gt_xy[:n] - nx[:n], axis=2)          # PIXELS

print("\nkeypoint agreement, in PIXELS of the original frame")
print("  median      %.4f" % np.median(d))
print("  p95         %.4f" % np.percentile(d, 95))
print("  max         %.4f" % d.max())
print("  confidence  mmpose %.3f   native %.3f"
      % (np.nanmean(gt_conf[:n]), np.nanmean(nc[:n])))

W = [COCO[j] for j in ("left_wrist", "right_wrist")]
A = [COCO[j] for j in ("left_ankle", "right_ankle")]
print("  wrists      median %.4f px" % np.median(d[:, W]))
print("  ankles      median %.4f px" % np.median(d[:, A]))

# --- and does the FEATURE that matters come out the same? ------------------
from pipeline.series import compute_series
a = normalise(gt_xy[:n], gt_conf[:n], src_fps, 30.0)
b = normalise(nx[:n], nc[:n], src_fps, 30.0)
sa = compute_series(a.xy, a.fps)["summary"]
sb = compute_series(b.xy, b.fps)["summary"]
print("\n%-24s %10s %10s %9s" % ("summary measure", "mmpose", "native", "diff"))
for k in ("distal_speed_mean", "small_amp_fraction", "direction_change_mean",
          "fidgety_power_mean", "lr_balance"):
    va, vb = sa.get(k), sb.get(k)
    if va is None or vb is None:
        continue
    print("%-24s %10.4f %10.4f %8.2f%%" % (k, va, vb, 100 * (vb - va) / (abs(va) + 1e-9)))

ok = np.median(d) < 1.0 and np.percentile(d, 95) < 3.0
print("\n%s" % ("NATIVE RUNNER REPRODUCES MMPOSE — safe to ship."
                if ok else "!!! DOES NOT MATCH — DO NOT SHIP !!!"))
sys.exit(0 if ok else 1)
