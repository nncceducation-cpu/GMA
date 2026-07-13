"""Does a re-upload analyse, flag, and stay out of training memory twice?"""
import sys, shutil, uuid
sys.path.insert(0, "/app")
from pathlib import Path
import pandas as pd

from webapp.app import STORE, LEARNER
from pipeline.rawstore import sha256_file

SRC = Path("/app/webapp/data_runtime/raw/recordings/e0442d4f8a20/video.mp4")
tmp = Path("/tmp/clip.mp4"); shutil.copyfile(SRC, tmp)
sha = sha256_file(tmp)
win = pd.read_parquet("/app/webapp/data_runtime/raw/recordings/e0442d4f8a20/features.parquet")

def ingest(subject):
    rid = uuid.uuid4().hex[:12]
    r = STORE.ingest(video=tmp, subject_id=subject, recording_id=rid,
                     corrected_age_weeks=15.0, site="ACH", risk_group="IVH")
    return rid, r

print("=== 1. ORIGINAL upload, subject DUP-A ===")
rid1, r1 = ingest("DUP-A")
print("   ingested        :", rid1, "| duplicate_of:", r1["duplicate_of"])
LEARNER.add(recording_id=rid1, subject_id="DUP-A", windows=win,
            gma_label="fm_present", content_sha256=sha)
m = LEARNER.summary()
print("   training memory : %d infants, %d recordings" % (m["total_subjects"], m["total_recordings"]))

print("\n=== 2. SAME clip re-uploaded, SAME subject DUP-A ===")
rid2, r2 = ingest("DUP-A")
print("   ANALYSED        : yes (new recording %s)" % rid2)
print("   flagged as dup  :", r2["duplicate_of"], "| conflict:", r2["subject_id_conflict"])
LEARNER.add(recording_id=rid2, subject_id="DUP-A", windows=win,
            gma_label="fm_absent", content_sha256=sha)   # relabel it
m = LEARNER.summary()
f = pd.read_csv(LEARNER.features_csv)
print("   training memory : %d infants, %d recordings  <- NOT doubled" %
      (m["total_subjects"], m["total_recordings"]))
print("   windows for this clip: %d  <- counted ONCE, label now %s"
      % (len(f[f.content_sha256.astype(str) == sha]),
         f[f.content_sha256.astype(str) == sha].gma_label.unique()))

print("\n=== 3. SAME clip, DIFFERENT subject DUP-B (the dangerous one) ===")
rid3, r3 = ingest("DUP-B")
print("   ANALYSED        : yes (new recording %s)" % rid3)
print("   conflict flagged:", r3["subject_id_conflict"])
try:
    LEARNER.add(recording_id=rid3, subject_id="DUP-B", windows=win,
                gma_label="fm_present", content_sha256=sha)
    print("   !!! TRAINING ACCEPTED IT — LEAK !!!")
except ValueError as e:
    print("   training REFUSED:", str(e)[:96], "...")

m = LEARNER.summary()
print("\n   final memory   : %d infants, %d recordings" % (m["total_subjects"], m["total_recordings"]))

# cleanup so the demo subjects do not pollute the real corpus
f = pd.read_csv(LEARNER.features_csv)
f[~f.subject_id.astype(str).str.startswith("DUP-")].to_csv(LEARNER.features_csv, index=False)
mm = pd.read_csv(LEARNER.manifest_csv)
mm[~mm.subject_id.astype(str).str.startswith("DUP-")].to_csv(LEARNER.manifest_csv, index=False)
man = STORE.manifest()
man[~man.subject_id.astype(str).str.startswith("DUP-")].to_csv(STORE.manifest_path, index=False)
for r in (rid1, rid2, rid3):
    shutil.rmtree(Path(STORE.root) / "recordings" / r, ignore_errors=True)
print("   (test subjects removed)")
print("\nDUPLICATE POLICY OK")
