"""Which pose model can a BROWSER run, without wrecking the GMA signal?

ViTPose-H (637M, 2.4 GB) cannot run in a browser. So: which smaller model keeps
the wrist and ankle accuracy that fidgety-movement analysis depends on?

Ground truth = the ViTPose-H keypoints already stored for a real clip.
Error is in TORSO UNITS (torso length = 1), the unit the features use.
"""
import sys, time
sys.path.insert(0, "/app")
from pathlib import Path

import numpy as np
import torch, cv2
from mmengine.dataset import Compose, pseudo_collate
from mmengine.registry import init_default_scope
from mmpose.apis.inferencers import Pose2DInferencer

from pipeline.normalise import COCO, normalise
from pipeline.pose_extract import DET_EVERY, PAD_FRAC, PoseExtractor
from pipeline.series import compute_series

RID = "914f311ddd18"
D = Path("/app/webapp/data_runtime/raw/recordings") / RID

z = np.load(D / "pose_raw.npz")
gt_xy, gt_conf, src_fps = z["xy"], z["conf"], float(z["fps"])

pe = PoseExtractor(device="cuda")
det = pe._detector()
dev = next(det.parameters()).device

# Boxes computed ONCE and reused for every model, so we compare MODELS, not boxes.
cap = cv2.VideoCapture(str(D / "video.mp4"))
frames, boxes, last = [], [], None
i = 0
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
print("clip: %d frames @ %.0f fps  (ground truth = ViTPose-H, 637 M params)\n"
      % (len(frames), src_fps))


def run(alias):
    p2d = Pose2DInferencer(model=alias, device="cuda")
    m = p2d.model
    npar = sum(x.numel() for x in m.parameters()) / 1e6
    init_default_scope(m.cfg.get("default_scope", "mmpose"))
    pipe = Compose(m.cfg.test_dataloader.dataset.pipeline)
    xs, cs = [], []
    t0 = time.time()
    for s in range(0, len(frames), 24):
        bf, bb = frames[s:s + 24], boxes[s:s + 24]
        data, keep = [], []
        for k, (f, b) in enumerate(zip(bf, bb)):
            if b is None:
                continue
            info = dict(img=f, bbox=b[None, :], bbox_score=np.ones(1, dtype=np.float32))
            info.update(m.dataset_meta)
            data.append(pipe(info)); keep.append(k)
        out = [None] * len(bf)
        if data:
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                res = m.test_step(pseudo_collate(data))
            for k, r in zip(keep, res):
                out[k] = r.pred_instances
        for pi in out:
            if pi is None:
                xs.append(np.full((17, 2), np.nan, dtype=np.float32))
                cs.append(np.zeros(17, dtype=np.float32))
            else:
                xs.append(np.asarray(pi.keypoints[0], dtype=np.float32)[:17])
                cs.append(np.asarray(pi.keypoint_scores[0], dtype=np.float32)[:17])
    dt = time.time() - t0
    return pe._fill_gaps(np.stack(xs)), np.stack(cs), npar, len(xs) / dt


CANDIDATES = [
    ("RTMPose-m",  "rtmpose-m_8xb256-420e_coco-256x192"),
    ("ViTPose-S",  "td-hm_ViTPose-small_8xb64-210e_coco-256x192"),
    ("ViTPose-B",  "td-hm_ViTPose-base_8xb64-210e_coco-256x192"),
]

n = len(gt_xy)
a = normalise(gt_xy, gt_conf, src_fps, target_fps=30.0)
sa = compute_series(a.xy, a.fps)["summary"]

W = [COCO[j] for j in ("left_wrist", "right_wrist")]
A = [COCO[j] for j in ("left_ankle", "right_ankle")]

print("%-11s %6s %7s | %-13s %-13s | %-9s %-9s | %s" %
      ("model", "params", "fps", "WRIST err", "ANKLE err", "wrist", "ankle", "distal_speed"))
print("%-11s %6s %7s | %-13s %-13s | %-9s %-9s | %s" %
      ("", "(M)", "", "med / p90", "med / p90", "conf", "conf", "vs ViTPose-H"))
print("-" * 108)
print("%-11s %6d %7s | %-13s %-13s | %-9.2f %-9.2f | %.3f (truth)" %
      ("ViTPose-H", 637, "4.7", "—", "—",
       np.nanmean(gt_conf[:, W]), np.nanmean(gt_conf[:, A]), sa["distal_speed_mean"]))

for name, alias in CANDIDATES:
    try:
        xy, cf, npar, fps_run = run(alias)
    except Exception as e:
        print("%-11s  FAILED: %s" % (name, str(e)[:60]))
        continue
    b = normalise(xy[:n], cf[:n], src_fps, target_fps=30.0)
    m_ = min(len(a.xy), len(b.xy))
    err = np.linalg.norm(a.xy[:m_] - b.xy[:m_], axis=2)
    ew, ea = err[:, W].ravel(), err[:, A].ravel()
    sb = compute_series(b.xy, b.fps)["summary"]
    d = sb["distal_speed_mean"]
    print("%-11s %6.0f %7.1f | %5.3f / %-5.3f %5.3f / %-5.3f | %-9.2f %-9.2f | %.3f (%+.0f%%)" %
          (name, npar, fps_run,
           np.median(ew), np.percentile(ew, 90),
           np.median(ea), np.percentile(ea, 90),
           np.nanmean(cf[:, W]), np.nanmean(cf[:, A]),
           d, 100 * (d - sa["distal_speed_mean"]) / sa["distal_speed_mean"]))

print("""
Error is in TORSO UNITS. 0.10 = a tenth of the infant's torso length.
Fidgety movements are SMALL and DISTAL: an ankle error of 0.25 torso units is
larger than the movement being measured.""")
