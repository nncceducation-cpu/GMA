"""Rebuild series + dashboards for recordings processed before this feature, and
smoke-test the whole analysis/export chain."""
import sys, json
sys.path.insert(0, "/app")
from pathlib import Path

import pandas as pd
from pipeline.series import compute_series, series_frame_table
from webapp.app import STORE, LEARNER, _json_safe
from webapp.figures import dashboard
from webapp.mlexport import build_bundle, all_data_csv, _dq

man = STORE.manifest()
print("recordings stored:", len(man))

for _, r in man.iterrows():
    rid = r["recording_id"]
    got = STORE.load_pose(rid, "L2")
    if got is None:
        print("  %s  no pose yet — skipped" % rid)
        continue
    xy, conf, fps = got
    s = compute_series(xy, fps)
    d = Path(STORE.root) / "recordings" / rid
    (d / "series.json").write_text(json.dumps(_json_safe(s)))
    dashboard(s, {"subject_id": r["subject_id"],
                  "corrected_age_weeks": r.get("corrected_age_weeks")},
              d / "dashboard.png", label=r.get("gma_label"))
    u = s["summary"]
    print("  %s  %s  %5.1fs  distal %.3f  fidgety-amp %3.0f%%  L/R %.2f  -> series.json + dashboard.png"
          % (rid, r["subject_id"], s["duration_s"], u["distal_speed_mean"],
             u["small_amp_fraction"] * 100, u["lr_balance"]))

print("\n--- frame table ---")
s = json.loads((Path(STORE.root) / "recordings" / man.iloc[-1]["recording_id"] / "series.json").read_text())
ft = series_frame_table(s)
print(ft.head(3).to_string(index=False))
print("frames: %d rows x %d cols" % ft.shape)

print("\n--- ML bundle ---")
out = build_bundle(STORE, LEARNER, Path("/tmp/neogma_dataset.zip"))
import zipfile
with zipfile.ZipFile(out) as z:
    for n in z.namelist():
        print("  %-28s %8d bytes" % (n, z.getinfo(n).file_size))
    print("\nsummary.json:")
    print(z.read("summary.json").decode())

print("--- all_data.csv ---")
csv = all_data_csv(STORE)
print("%d rows, %d cols" % (csv.count("\n") - 1, csv.split("\n")[0].count(",") + 1))
print("first cols:", ", ".join(csv.split("\n")[0].split(",")[:8]))

print("\n--- leaky-column flags ---")
w = pd.concat([pd.read_parquet(Path(STORE.root)/"recordings"/r["recording_id"]/"features.parquet")
               for _, r in man.iterrows()
               if (Path(STORE.root)/"recordings"/r["recording_id"]/"features.parquet").exists()])
dq = _dq(w)
print(dq[dq.leaky | dq.constant][["column", "leaky", "constant", "missing_frac"]].to_string(index=False))

print("\nBACKFILL + EXPORT CHAIN OK")
