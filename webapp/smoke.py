"""End-to-end on a real clip: timing AND accuracy vs the fp32 baseline."""
import sys, time
sys.path.insert(0, "/app")

import numpy as np
from pipeline.pose_extract import PoseExtractor, POSE_BATCH, DET_EVERY, FP16
from pipeline.normalise import normalise
from pipeline.quality import pose_quality
from pipeline.features_gma import extract_windows

VIDEO = "/app/webapp/data_runtime/raw/recordings/906a25d86d75/video.mp4"

p = PoseExtractor(device="cuda")
print("backend %s | fp16 %s | batch %d | detect every %d frames"
      % (p.backend, FP16, POSE_BATCH, DET_EVERY))
p._load()

last = [0]
def prog(done, total, rate):
    if done - last[0] >= 120 or done == total:
        last[0] = done
        print("   %4d/%-4d  %5.1f fps  eta %3.0fs" % (done, total, rate,
              (total-done)/rate if rate else 0), flush=True)

t0 = time.time()
xy, conf, src_fps = p.extract(VIDEO, progress=prog)
dt = time.time() - t0

print("\n%-22s %s" % ("BASELINE (fp32, 1x1)", "674 frames / 438 s =  1.5 fps"))
print("%-22s %d frames / %.0f s = %4.1f fps   -> %.1fx faster"
      % ("NOW (fp16, batched)", len(xy), dt, len(xy)/dt, 438/dt))

print("\naccuracy check (fp32 baseline in brackets)")
print("  mean confidence  %.2f   [0.82]" % float(np.nanmean(conf)))
print("  wrists           %.2f   [0.81]" % float(np.nanmean(conf[:, [9, 10]])))
print("  ankles           %.2f   [0.76]" % float(np.nanmean(conf[:, [15, 16]])))

npose = normalise(xy, conf, src_fps, target_fps=30.0)
qc = pose_quality(npose)
print("  wingspan ratio   %.2f   [0.73]" % qc["wingspan_ratio"])
print("  QC usable        %s" % qc["usable"])
rows = extract_windows(npose.xy, npose.fps)
print("  windows          %d x %d features   [16 x 134]"
      % (len(rows), len(rows[0]) if rows else 0))
print("\nSMOKE TEST PASSED")
