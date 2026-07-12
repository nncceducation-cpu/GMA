"""NeoGMA — local web app. Upload an infant video, get a GMA assessment.

Pipeline: protocol gate -> ViTPose-H -> normalise -> QC -> windowed features
          -> clinician label -> grouped-CV retrain -> exports (raw retained).
"""
from __future__ import annotations

import hashlib, json, logging, math, os, shutil, sys, threading, time, traceback, uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from fastapi import Body, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.features_gma import extract_windows                       # noqa: E402
from pipeline.normalise import normalise                                # noqa: E402
from pipeline.quality import pose_quality, protocol_gate               # noqa: E402
from pipeline.rawstore import RawStore, sha256_file                    # noqa: E402
from webapp.learning import CP_LABELS, GMA_LABELS, Learner             # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("neogma.web")

DEVICE = os.environ.get("NEOGMA_DEVICE", "cuda")
TARGET_FPS = float(os.environ.get("NEOGMA_TARGET_FPS", "30"))
DATA_DIR = Path(os.environ.get("NEOGMA_DATA_DIR", ROOT / "webapp" / "data_runtime"))
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = Path(os.environ.get("NEOGMA_MODEL_PATH", ROOT / "models" / "neogma_model.joblib"))

STORE = RawStore(DATA_DIR / "raw")
LEARNER = Learner(DATA_DIR / "memory", MODEL_PATH)

app = FastAPI(title="NeoGMA", version="0.1")
_POSE = None
_POSE_LOCK = threading.Lock()


def _json_safe(o):
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, dict):
        return {k: _json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_safe(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return _json_safe(float(o))
    return o


@dataclass
class Job:
    id: str
    status: str = "queued"
    stage: str = "queued"
    percent: float = 0.0
    message: str = ""
    error: str = ""
    subject_id: str = ""
    recording_id: str = ""
    corrected_age_weeks: Optional[float] = None
    site: str = ""
    sha: str = ""
    pose_backend: str = ""
    n_windows: int = 0
    qc: Dict = field(default_factory=dict)
    gate: Dict = field(default_factory=dict)
    started_at: float = field(default_factory=time.time)


JOBS: Dict[str, Job] = {}


POSE_BACKEND = os.getenv("NEOGMA_POSE_BACKEND", "auto")

POSE_LABEL = {
    "vitpose": "ViTPose-H",
    "keypointrcnn": "Keypoint R-CNN (FALLBACK - dev only)",
}


def _get_pose():
    global _POSE
    with _POSE_LOCK:
        if _POSE is None:
            from pipeline.pose_extract import PoseExtractor
            _POSE = PoseExtractor(backend=POSE_BACKEND, device=DEVICE)
        return _POSE


def _process(job: Job, video: Path):
    try:
        job.status = "running"
        job.stage = "model"; job.percent = 5
        # ViTPose-H is ~2.4 GB and is fetched on first use. Without this message
        # the UI sits at 10% for several minutes and looks hung — which is
        # exactly what it looked like the first time.
        job.message = ("Loading pose model. First run downloads ViTPose-H "
                       "(~2.4 GB) — this happens once, then it is cached.")
        pose = _get_pose()
        job.pose_backend = pose.backend
        pose._load()

        job.stage = "pose"; job.percent = 10
        job.message = f"Estimating infant pose ({POSE_LABEL[pose.backend]})..."
        xy, conf, src_fps = pose.extract(video)
        STORE.save_pose(job.recording_id, xy, conf, src_fps, level="L1")
        # The backend is provenance. Mixing backends within a cohort is a site
        # effect under another name, and probes.py will flag it as one.
        STORE.set_label(job.recording_id, pose_backend=pose.backend)

        job.stage = "normalise"; job.percent = 55
        job.message = "Normalising (frame rate, scale, rotation, jitter)..."
        npose = normalise(xy, conf, src_fps, target_fps=TARGET_FPS)
        STORE.save_pose(job.recording_id, npose.xy, npose.conf, npose.fps, level="L2")

        job.stage = "qc"; job.percent = 70
        job.qc = _json_safe(pose_quality(npose))
        if not job.qc["usable"]:
            job.status = "error"; job.stage = "error"
            job.error = ("Pose tracking failed quality control:\n" +
                         "\n".join(i["detail"] for i in job.qc["issues"]
                                   if i["severity"] == "ERROR"))
            return

        job.stage = "features"; job.percent = 85
        job.message = "Extracting windowed movement features..."
        rows = extract_windows(npose.xy, npose.fps)
        df = pd.DataFrame(rows)
        df["source_fps"] = src_fps
        df["torso_px"] = npose.torso_px
        df["site"] = job.site
        df["corrected_age_weeks"] = job.corrected_age_weeks
        STORE.save_features(job.recording_id, df)
        job.n_windows = len(df)

        job.stage = "done"; job.percent = 100
        job.message = f"Complete — {len(df)} windows."
        job.status = "done"
    except Exception as exc:
        logger.exception("job %s failed", job.id)
        job.status = "error"; job.stage = "error"
        job.error = f"{exc}\n\n{traceback.format_exc()}"


@app.post("/upload")
async def upload(file: UploadFile = File(...),
                 subject_id: str = Form(...),
                 corrected_age_weeks: float = Form(...),
                 site: str = Form(""),
                 risk_group: str = Form("")) -> JSONResponse:
    ext = Path(file.filename or "").suffix.lower()
    if ext not in {".mp4", ".mov", ".avi", ".mkv", ".m4v"}:
        raise HTTPException(400, f"Unsupported video type '{ext}'.")

    rid = uuid.uuid4().hex[:12]
    tmp_dir = DATA_DIR / "_incoming"; tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp = tmp_dir / f"{rid}{ext}"
    with tmp.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    # duration, for the protocol gate
    import cv2
    cap = cv2.VideoCapture(str(tmp))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    cap.release()
    duration = n / fps if fps else 0.0

    gate = protocol_gate(corrected_age_weeks, duration)
    if not gate["pass"]:
        tmp.unlink(missing_ok=True)
        raise HTTPException(422, {"blocking": gate["blocking"],
                                  "warnings": gate["warnings"]})

    sha = sha256_file(tmp)
    dup = LEARNER.find_duplicate(sha)

    rec = STORE.ingest(video=tmp, subject_id=subject_id, recording_id=rid,
                       corrected_age_weeks=corrected_age_weeks, site=site,
                       risk_group=risk_group,
                       extra={"duration_s": gate["duration_s"],
                              "protocol_compliant": gate["protocol_compliant"]})
    tmp.unlink(missing_ok=True)

    job = Job(id=rid, recording_id=rid, subject_id=subject_id, sha=sha,
              site=site, corrected_age_weeks=corrected_age_weeks, gate=gate)
    JOBS[rid] = job
    threading.Thread(target=_process, args=(job, Path(STORE.root) / "recordings" / rid / rec["video_file"]),
                     daemon=True).start()
    return JSONResponse(_json_safe({"job_id": rid, "duplicate_of": dup,
                                    "gate": gate}))


@app.get("/status/{job_id}")
def status(job_id: str) -> JSONResponse:
    j = JOBS.get(job_id)
    if not j:
        raise HTTPException(404, "Unknown job")
    p = {"status": j.status, "stage": j.stage, "percent": round(j.percent, 1),
         "message": j.message, "error": j.error, "n_windows": j.n_windows,
         "elapsed": round(time.time() - j.started_at, 1)}
    if j.status == "done":
        p["qc"] = j.qc
        p["exports"] = {
            "windows": f"/export/{job_id}/features.parquet",
            "pose_norm": f"/export/{job_id}/pose_norm.npz",
            "pose_raw": f"/export/{job_id}/pose_raw.npz",
        }
    return JSONResponse(_json_safe(p))


@app.post("/label")
def label(payload: dict = Body(...)) -> JSONResponse:
    rid = str(payload.get("job_id", "")).strip()
    gma = payload.get("gma_label")
    cp = payload.get("cp_status")
    job = JOBS.get(rid)
    if not job or job.status != "done":
        raise HTTPException(404, "Unknown or unfinished job.")
    if gma and gma not in GMA_LABELS:
        raise HTTPException(400, f"Unknown GMA label. Allowed: {list(GMA_LABELS)}")
    if cp and cp not in CP_LABELS:
        raise HTTPException(400, f"Unknown CP status. Allowed: {list(CP_LABELS)}")

    df = pd.read_parquet(STORE.root / "recordings" / rid / "features.parquet")
    try:
        stored = LEARNER.add(recording_id=rid, subject_id=job.subject_id,
                             windows=df, gma_label=gma, cp_status=cp,
                             content_sha256=job.sha,
                             meta={"site": job.site,
                                   "corrected_age_weeks": job.corrected_age_weeks})
    except ValueError as e:
        raise HTTPException(409, str(e))
    STORE.set_label(rid, gma_label=gma, cp_status=cp)
    training = LEARNER.retrain("gma")
    return JSONResponse(_json_safe({"ok": True, "stored": stored,
                                    "training": training,
                                    "memory": LEARNER.summary()}))


@app.post("/outcome")
def outcome(payload: dict = Body(...)) -> JSONResponse:
    """Join the CP outcome in later — possibly years later. This is the endpoint
    that turns a GMA-surrogate model into a real CP model."""
    sid = str(payload.get("subject_id", "")).strip()
    cp = payload.get("cp_status")
    if cp not in CP_LABELS:
        raise HTTPException(400, f"Allowed: {list(CP_LABELS)}")
    n = LEARNER.set_outcome(sid, cp, cp_gmfcs=payload.get("gmfcs"))
    return JSONResponse(_json_safe({"ok": True, "rows_updated": n,
                                    "training_cp": LEARNER.retrain("cp")}))


@app.get("/memory")
def memory() -> JSONResponse:
    return JSONResponse(_json_safe(LEARNER.summary()))


@app.get("/export/{rid}/{fname}")
def export(rid: str, fname: str):
    allowed = {"features.parquet": "application/octet-stream",
               "pose_norm.npz": "application/octet-stream",
               "pose_raw.npz": "application/octet-stream"}
    if fname not in allowed:
        raise HTTPException(404, "Unknown export.")
    p = STORE.root / "recordings" / rid / fname
    if not p.exists():
        raise HTTPException(404, "Not available.")
    return FileResponse(str(p), media_type=allowed[fname],
                        filename=f"neogma_{rid}_{fname}")


@app.get("/health")
def health() -> dict:
    from pipeline.pose_extract import resolve_backend
    b = resolve_backend(POSE_BACKEND)
    return {"ok": True, "device": DEVICE, "target_fps": TARGET_FPS,
            "pose_backend": b, "pose_backend_label": POSE_LABEL[b],
            "pose_is_fallback": b != "vitpose",
            "gma_labels": GMA_LABELS, "cp_labels": CP_LABELS,
            "model_exists": MODEL_PATH.exists()}


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return INDEX_HTML


INDEX_HTML = """<!doctype html><html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>NeoGMA — automated General Movements Assessment</title>
<style>
:root{--bg:#0f172a;--card:#1e293b;--fg:#e2e8f0;--mut:#94a3b8;--acc:#38bdf8;--bad:#f87171;--ok:#34d399}
*{box-sizing:border-box}body{margin:0;font-family:system-ui,Segoe UI,Roboto,sans-serif;background:var(--bg);color:var(--fg)}
.wrap{max-width:940px;margin:0 auto;padding:30px 20px 80px}
h1{font-size:26px;margin:0 0 4px}.sub{color:var(--mut);margin:0 0 22px}
.card{background:var(--card);border:1px solid #334155;border-radius:14px;padding:22px;margin-bottom:20px}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:14px}
label{display:block;font-size:12px;color:var(--mut);margin-bottom:5px}
input,select{width:100%;background:#0b1220;border:1px solid #334155;color:var(--fg);border-radius:9px;padding:10px 12px;font-size:14px}
.btn{background:var(--acc);color:#04283a;border:0;border-radius:9px;padding:12px 22px;font-weight:700;cursor:pointer;font-size:15px}
.btn:disabled{opacity:.5;cursor:not-allowed}
.drop{border:2px dashed #475569;border-radius:12px;padding:34px;text-align:center;cursor:pointer;margin:14px 0}
.drop:hover{border-color:var(--acc);background:#0b1220}
.bar{height:10px;background:#0b1220;border-radius:6px;overflow:hidden;border:1px solid #334155;margin-top:10px}
.bar>i{display:block;height:100%;width:0;background:linear-gradient(90deg,#38bdf8,#818cf8);transition:width .4s}
.lbl{display:flex;gap:10px;flex-wrap:wrap;margin:6px 0}
.lb{flex:1;min-width:170px;border:1px solid #334155;background:#0b1220;color:var(--fg);border-radius:11px;padding:13px;font-weight:600;cursor:pointer}
.lb.present:hover{border-color:var(--ok)}.lb.abnormal:hover{border-color:#fbbf24}.lb.absent:hover{border-color:var(--bad)}
.chip{display:inline-flex;gap:6px;background:#0b1220;border:1px solid #334155;border-radius:20px;padding:3px 11px;font-size:12px;margin-right:6px}
.chip .n{color:var(--acc);font-weight:700}
.warn{background:#3b1d1d;border:1px solid #7f1d1d;color:#fecaca;padding:11px 13px;border-radius:9px;font-size:12.5px;line-height:1.5;margin-top:12px}
.note{color:var(--mut);font-size:12px;line-height:1.5;margin-top:10px}
.err{color:#fda4af;white-space:pre-wrap;font-size:12px}
.metrics{background:#0b1220;border:1px solid #334155;border-radius:10px;padding:12px;margin-top:12px;font-size:12.5px;display:none}
.mrow{display:flex;justify-content:space-between;color:var(--mut);margin:4px 0}.mrow b{color:var(--acc)}
</style></head><body><div class="wrap">
<h1>NeoGMA</h1>
<p class="sub">Automated General Movements Assessment — fidgety movements, 9–20 weeks corrected age.</p>
<div id="mem" class="card" style="padding:12px 16px;font-size:13px">Training memory: loading…</div>

<div class="card">
  <div class="grid2">
    <div><label>Subject ID (one per infant — never reuse)</label><input id="sid" placeholder="e.g. NEO-0042"></div>
    <div><label>Corrected age (weeks) — must be 9–20</label><input id="age" type="number" step="0.1" placeholder="14.6"></div>
    <div><label>Site</label><input id="site" placeholder="e.g. RVH"></div>
    <div><label>Risk group</label>
      <select id="risk"><option value="">—</option><option>HIE / cooled</option>
      <option>Preterm</option><option>IVH</option><option>Other high-risk</option></select></div>
  </div>
  <div id="drop" class="drop">Drop the infant video here or <u>click to choose</u>
    <div class="note">supine · top-down · nappy only · no pacifier/toys/interaction · 60–120 s</div>
    <input id="file" type="file" accept="video/*" style="display:none"></div>
  <button id="go" class="btn" disabled>Analyse</button> <span id="fn" class="note"></span>
  <div id="prog" style="display:none"><div class="bar"><i id="pb"></i></div>
    <div id="pmsg" class="note"></div></div>
  <div id="msg"></div>
</div>

<div id="res" class="card" style="display:none">
  <h2 style="margin-top:0;font-size:18px">Assessment</h2>
  <div id="qc"></div>
  <h3 style="font-size:14px;margin:18px 0 6px">GMA score (certified assessor)</h3>
  <div class="lbl">
    <button class="lb present" data-l="fm_present">Fidgety PRESENT<br><span class="note">normal</span></button>
    <button class="lb abnormal" data-l="fm_abnormal">Fidgety ABNORMAL<br><span class="note">not collapsed into the binary</span></button>
    <button class="lb absent" data-l="fm_absent">Fidgety ABSENT<br><span class="note">high CP risk</span></button>
  </div>
  <div id="lmsg" class="note"></div>
  <div id="metrics" class="metrics"></div>
  <div id="ex" class="note" style="margin-top:14px"></div>
  <div class="warn"><b>Research tool — not a medical device.</b> Predicts the GMA score, which is a
    <i>surrogate</i> for CP, not CP itself. Even expert GMA has a positive predictive value of ~33% at
    10% prevalence: most abnormal results are children who will not develop CP. Its strength is the
    ~97% negative predictive value. Never use this to make a clinical decision.</div>
</div>
<script>
const $=id=>document.getElementById(id);
let chosen=null,jid=null;
$('drop').onclick=()=>$('file').click();
$('file').onchange=()=>{if($('file').files[0]){chosen=$('file').files[0];$('fn').textContent=chosen.name;check();}};
['dragover','drop'].forEach(e=>$('drop').addEventListener(e,ev=>{ev.preventDefault();
  if(e==='drop'&&ev.dataTransfer.files[0]){chosen=ev.dataTransfer.files[0];$('fn').textContent=chosen.name;check();}}));
['sid','age'].forEach(i=>$(i).oninput=check);
function check(){$('go').disabled=!(chosen&&$('sid').value.trim()&&$('age').value);}
async function mem(){const m=await (await fetch('/memory')).json();
  let c='';for(const [k,v] of Object.entries(m.per_class))c+=`<span class="chip">${v.name.split('(')[0]}<span class="n">${v.subjects}</span></span>`;
  $('mem').innerHTML=`<b>Training memory:</b> ${c} <span style="float:right;color:#94a3b8">${m.total_subjects} infants · ${m.total_recordings} recordings · CP outcome known for ${m.cp_known}</span>`;}
mem();
$('go').onclick=async()=>{
  $('go').disabled=true;$('res').style.display='none';$('msg').innerHTML='';
  $('prog').style.display='block';$('pb').style.width='5%';$('pmsg').textContent='Uploading…';
  const fd=new FormData();fd.append('file',chosen);fd.append('subject_id',$('sid').value.trim());
  fd.append('corrected_age_weeks',$('age').value);fd.append('site',$('site').value);
  fd.append('risk_group',$('risk').value);
  const r=await fetch('/upload',{method:'POST',body:fd});
  const j=await r.json();
  if(!r.ok){const d=j.detail;
    $('msg').innerHTML='<div class="warn"><b>Refused.</b><br>'+
      (d.blocking?d.blocking.join('<br>'):JSON.stringify(d))+'</div>';
    $('prog').style.display='none';$('go').disabled=false;return;}
  let pre='';
  if(j.gate&&j.gate.warnings&&j.gate.warnings.length)
    pre+='<div class="warn"><b>Accepted with warnings.</b><br>'+j.gate.warnings.join('<br>')+'</div>';
  if(j.duplicate_of)
    pre+='<div class="warn">This video is byte-identical to a recording already stored for subject <b>'+j.duplicate_of.subject_id+'</b>.</div>';
  $('msg').innerHTML=pre;
  jid=j.job_id;poll();
};
async function poll(){
  const s=await (await fetch('/status/'+jid)).json();
  $('pb').style.width=(s.percent||0)+'%';$('pmsg').textContent=s.message||s.stage;
  if(s.status==='error'){$('prog').style.display='none';$('go').disabled=false;
    $('msg').innerHTML='<div class="warn"><b>Failed.</b><br><span class="err">'+s.error+'</span></div>';return;}
  if(s.status==='done'){
    $('prog').style.display='none';$('go').disabled=false;
    let q='<div class="note">'+s.n_windows+' windows · mean keypoint confidence '+
      (s.qc.mean_confidence||0).toFixed(2)+' · wingspan ratio '+(s.qc.wingspan_ratio||0).toFixed(2)+'</div>';
    if(s.qc.issues&&s.qc.issues.length)q+='<div class="warn">'+s.qc.issues.map(i=>'<b>'+i.severity+'</b> '+i.detail).join('<br>')+'</div>';
    $('qc').innerHTML=q;
    const e=s.exports||{};
    $('ex').innerHTML='Raw data retained: <a style="color:#38bdf8" href="'+e.windows+'">window features</a> · '+
      '<a style="color:#38bdf8" href="'+e.pose_norm+'">normalised pose</a> · '+
      '<a style="color:#38bdf8" href="'+e.pose_raw+'">raw keypoints</a> — kept for unsupervised re-analysis.';
    $('res').style.display='block';return;}
  setTimeout(poll,900);
}
document.querySelectorAll('.lb').forEach(b=>b.onclick=async()=>{
  document.querySelectorAll('.lb').forEach(x=>x.disabled=true);
  $('lmsg').textContent='Saving and retraining…';
  const r=await fetch('/label',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({job_id:jid,gma_label:b.dataset.l})});
  const j=await r.json();
  if(!r.ok){$('lmsg').innerHTML='<span class="err">'+(j.detail||'failed')+'</span>';
    document.querySelectorAll('.lb').forEach(x=>x.disabled=false);return;}
  const t=j.training||{};
  $('lmsg').innerHTML='✓ Saved '+j.stored.n_windows+' windows. '+(t.trained?'Model refit.':'<span style="color:#94a3b8">'+(t.reason||'')+'</span>');
  const M=$('metrics');
  if(t.trained&&t.cv&&t.cv.ok){
    const c=t.cv,o=c.operating_point_youden,l=t.leakage_selftest||{};
    M.innerHTML='<b>Grouped cross-validation (split by INFANT)</b>'+
      '<div class="mrow"><span>Infants / positives</span><b>'+c.n_infants+' / '+c.n_positive+'</b></div>'+
      '<div class="mrow"><span>ROC-AUC</span><b>'+c.roc_auc.toFixed(2)+'</b></div>'+
      '<div class="mrow"><span>PR-AUC (chance '+c.pr_auc_baseline.toFixed(2)+')</span><b>'+c.pr_auc.toFixed(2)+'</b></div>'+
      '<div class="mrow"><span>Sens / Spec</span><b>'+(o.sensitivity*100).toFixed(0)+'% / '+(o.specificity*100).toFixed(0)+'%</b></div>'+
      '<div class="mrow"><span>PPV / NPV</span><b>'+(o.ppv*100).toFixed(0)+'% / '+(o.npv*100).toFixed(0)+'%</b></div>'+
      (l.ok?'<div class="mrow" style="margin-top:8px"><span>Leakage self-test (window-split vs infant-split)</span><b>'+
        l.window_level_auc_LEAKY.toFixed(2)+' vs '+l.subject_level_auc_HONEST.toFixed(2)+'</b></div>':'')+
      '<div class="note">'+(c.warnings||[]).join(' ')+'</div>';
    M.style.display='block';
  } else if(t.reason){M.innerHTML='<b>No honest score yet.</b><div class="note">'+t.reason+'</div>';M.style.display='block';}
  mem();
});
</script></div></body></html>"""
