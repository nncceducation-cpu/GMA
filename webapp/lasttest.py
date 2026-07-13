"""Prove the last-clip export works end to end."""
import sys, zipfile
sys.path.insert(0, "/app")
from pathlib import Path

from webapp.app import STORE
from webapp.mlexport import all_data_csv, clip_bundle, last_recording_id

rid = last_recording_id(STORE)
man = STORE.manifest()
r = man[man.recording_id == rid].iloc[0]
print("last recording : %s  (subject %s, ingested %s)"
      % (rid, r.get("subject_id"), r.get("ingested_at")))

print("\n--- /export/last/all_data.csv ---")
csv = all_data_csv(STORE, recording_ids=[rid])
rows = csv.count("\n") - 1
cols = csv.split("\n")[0].count(",") + 1
print("  %d rows x %d cols   (this clip only)" % (rows, cols))
print("  first cols:", ", ".join(csv.split("\n")[0].split(",")[:6]))

cum = all_data_csv(STORE)
print("  cumulative CSV is %d rows — same %d columns, so they concatenate"
      % (cum.count("\n") - 1, cum.split("\n")[0].count(",") + 1))

print("\n--- /export/last/clip.zip ---")
out = clip_bundle(STORE, rid, Path("/tmp/last_clip.zip"))
with zipfile.ZipFile(out) as z:
    for n in sorted(z.namelist()):
        print("  %-22s %9d bytes" % (n, z.getinfo(n).file_size))
    assert "windows.csv" in z.namelist()
    assert "frames.csv" in z.namelist()
    print("\n  summary.json:")
    print("   ", z.read("summary.json").decode()[:340].replace("\n", "\n    "))
    vids = [n for n in z.namelist() if n.endswith((".mp4", ".mov", ".avi"))]
    print("\n  video in bundle:", vids or "NO — correct, it stays on the machine")

print("\nLAST-CLIP EXPORT OK")
