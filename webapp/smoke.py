"""End-to-end smoke test on a real stored clip. Proves pose actually runs."""
import sys, time
sys.path.insert(0, "/app")

import numpy as np
from pipeline.pose_extract import PoseExtractor
from pipeline.normalise import normalise
from pipeline.quality import pose_quality
from pipeline.features_gma import extract_windows

VIDEO = "/app/webapp/data_runtime/raw/recordings/906a25d86d75/video.mp4"

t0 = time.time()
p = PoseExtractor(device="cuda")
print("backend:", p.backend)

xy, conf, src_fps = p.extract(VIDEO)
t_pose = time.time() - t0
print("pose      : %s frames, src_fps %.1f, %.0fs (%.1f fps)"
      % (len(xy), src_fps, t_pose, len(xy) / max(t_pose, 1e-9)))
print("keypoint confidence: mean %.2f  (wrists %.2f, ankles %.2f)"
      % (float(np.nanmean(conf)),
         float(np.nanmean(conf[:, [9, 10]])),
         float(np.nanmean(conf[:, [15, 16]]))))

npose = normalise(xy, conf, src_fps, target_fps=30.0)
qc = pose_quality(npose)
print("normalised: %d frames @ %.0f fps, torso %.0f px"
      % (npose.meta["n_frames"], npose.fps, npose.torso_px))
print("QC usable :", qc["usable"], "| wingspan ratio %.2f" % qc["wingspan_ratio"])
for i in qc["issues"]:
    print("   ", i["severity"], "-", i["detail"][:80])

rows = extract_windows(npose.xy, npose.fps)
print("features  : %d windows x %d features" % (len(rows), len(rows[0]) if rows else 0))
print("\nSMOKE TEST PASSED")
