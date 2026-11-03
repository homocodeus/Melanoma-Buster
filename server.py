"""
Live web interface for melanoma.py (no extra dependencies).

  python3 server.py            -> open http://localhost:8000
  python3 server.py --port 9000

Upload / drag-drop an image or capture from the webcam; results appear instantly.
EDUCATIONAL TOOL ONLY - NOT A MEDICAL DIAGNOSIS.
"""
import argparse
import base64
import csv
import json
import os
import tempfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

from melanoma import NoLesionError, analyze

MAX_BYTES = 20 * 1024 * 1024

PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Melanoma Busters</title>
<style>
:root{--bg:#0f1115;--card:#181b22;--line:#2a2f3a;--txt:#e8eaf0;--mut:#8b93a5;--acc:#4ea1ff;
      --ok:#3ecf8e;--warn:#f5b041;--bad:#ff5c5c}
*{box-sizing:border-box}
body{margin:0;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",system-ui,sans-serif;
     background:var(--bg);color:var(--txt)}
.wrap{max-width:1200px;margin:0 auto;padding:28px 24px 60px}
header{text-align:center;margin-bottom:24px}
header h1{margin:0;font-size:28px;letter-spacing:.3px}
header p{margin:6px 0 0;color:var(--mut)}
.disclaimer{display:inline-block;margin-top:10px;padding:6px 14px;border-radius:999px;
     background:#3a1d1d;color:#ffb3b3;font-size:13px}
.grid{display:grid;grid-template-columns:minmax(320px,420px) 1fr;gap:20px;align-items:start}
@media(max-width:860px){.grid{grid-template-columns:1fr}}
.card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:20px}
.card h2{margin:0 0 14px;font-size:15px;text-transform:uppercase;letter-spacing:1px;color:var(--mut)}
#drop{border:2px dashed var(--line);border-radius:12px;padding:34px 16px;text-align:center;
      cursor:pointer;transition:.15s}
#drop:hover,#drop.over{border-color:var(--acc);background:#16202e}
#drop .big{font-size:34px}#drop small{color:var(--mut)}
.btns{display:flex;gap:8px;margin-top:14px}
button{flex:1;background:var(--acc);border:0;color:#04111f;font-weight:600;padding:10px;
       border-radius:8px;cursor:pointer}
button.sec{background:#262b36;color:var(--txt)}
button:disabled{opacity:.4;cursor:default}
.opts{display:flex;justify-content:space-between;align-items:center;margin-top:14px;
      color:var(--mut);font-size:14px}
.opts input[type=number]{width:70px;background:#0f1115;border:1px solid var(--line);
      color:var(--txt);border-radius:6px;padding:5px}
video{width:100%;border-radius:10px;margin-top:14px;display:none;background:#000}
.gallery{display:grid;grid-template-columns:repeat(4,1fr);gap:8px}
.gallery figure{margin:0;cursor:pointer;position:relative}
.gallery img{width:100%;aspect-ratio:1;object-fit:cover;border-radius:8px;
      border:2px solid transparent;transition:.15s}
.gallery img:hover{border-color:var(--acc)}
.gallery span{position:absolute;left:4px;bottom:4px;font-size:10px;padding:1px 5px;
      border-radius:4px;background:#000a}
.m{color:var(--bad)}.b{color:var(--ok)}
.empty{text-align:center;color:var(--mut);padding:70px 10px}
.verdict{display:flex;align-items:center;justify-content:space-between;padding:16px 18px;
      border-radius:12px;margin-bottom:16px;background:#11141a;border-left:6px solid var(--mut)}
.verdict .lbl{font-size:22px;font-weight:700}
.verdict .tds{text-align:right;color:var(--mut);font-size:13px}
.verdict .tds b{display:block;font-size:26px;color:var(--txt)}
.scores{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-bottom:16px}
@media(max-width:560px){.scores{grid-template-columns:repeat(2,1fr)}}
.sc{background:#11141a;border-radius:10px;padding:12px;text-align:center}
.sc .k{font-size:12px;color:var(--mut)}
.sc .v{font-size:24px;font-weight:700;margin:4px 0}
.sc .d{font-size:11px;color:var(--mut);min-height:28px}
.bar{height:4px;background:#262b36;border-radius:2px;overflow:hidden}
.bar i{display:block;height:100%;background:var(--acc)}
.overlay{width:100%;border-radius:10px;display:block}
.legend{display:flex;flex-wrap:wrap;gap:14px;justify-content:center;font-size:12px;
      color:var(--mut);margin-top:8px}
.legend i{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:4px;vertical-align:-1px}
.steps{margin-top:20px;text-align:center}.steps img{width:100%;border-radius:10px}
.notdet{text-align:center;padding:30px 10px}
.notdet .ic{font-size:44px}.notdet h3{color:var(--warn);margin:6px 0 12px}
.notdet ul{display:inline-block;text-align:left;color:var(--mut);margin:0 0 12px}
#status{margin-top:12px;text-align:center;color:var(--mut);min-height:20px}
.spin{display:inline-block;width:14px;height:14px;border:2px solid var(--mut);
      border-top-color:transparent;border-radius:50%;animation:s 1s linear infinite;vertical-align:-2px}
@keyframes s{to{transform:rotate(360deg)}}
</style></head><body><div class="wrap">

<header>
  <h1>Melanoma Busters</h1>
  <p>OpenCV lesion segmentation &middot; Asymmetry &middot; Border &middot; Color &middot; Diameter</p>
  <div class="disclaimer">Educational tool &mdash; not a medical diagnosis. Consult a dermatologist.</div>
</header>

<div class="grid">
  <div>
    <div class="card">
      <h2>Input</h2>
      <div id="drop"><div class="big">&#128247;</div>
        <div>Drop an image here or <b>click to browse</b></div>
        <small>JPG / PNG, close-up of a single mole</small>
        <input id="file" type="file" accept="image/*" hidden></div>
      <div class="btns">
        <button id="camBtn" class="sec">Start webcam</button>
        <button id="snapBtn" disabled>Capture</button>
      </div>
      <video id="video" autoplay playsinline muted></video>
      <div class="opts">
        <label><input id="live" type="checkbox"> Live (every 2 s)</label>
        <label>Scale px/mm <input id="scale" type="number" step="0.1" placeholder="opt."></label>
      </div>
    </div>
    <div class="card" style="margin-top:20px" id="galCard">
      <h2>Sample images (ISIC archive)</h2>
      <div class="gallery" id="gallery"></div>
    </div>
  </div>

  <div class="card">
    <h2>Result</h2>
    <div id="result"><div class="empty">Upload an image or choose a sample to start.</div></div>
    <div id="status"></div>
  </div>
</div>

<div class="card steps" id="stepsCard" style="display:none;margin-top:20px">
  <h2>Processing pipeline</h2><img id="steps">
</div>

</div><script>
const $=id=>document.getElementById(id);
let busy=false,timer=null;
const COLORS={ok:'var(--ok)',warn:'var(--warn)',bad:'var(--bad)'};

function scoreCard(k,v,max,d){
  return `<div class="sc"><div class="k">${k}</div><div class="v">${v}<small style="font-size:13px;color:var(--mut)"> / ${max}</small></div>
          <div class="bar"><i style="width:${Math.min(100,v/max*100)}%"></i></div><div class="d">${d}</div></div>`;
}

function render(d){
  if(!d.detected){
    $('result').innerHTML=`<div class="notdet"><div class="ic">&#128269;</div><h3>Mole not detected</h3>
      <ul>${d.reasons.map(r=>`<li>${r}</li>`).join('')}</ul>
      <div style="color:var(--mut)">Try a sharp, well-lit close-up of a single mole on skin.</div></div>`;
    $('stepsCard').style.display='none'; return;
  }
  const m=d.metrics, lvl=m.TDS<4.75?'ok':m.TDS<=5.45?'warn':'bad';
  const diam=m.diameter_mm?m.diameter_mm.toFixed(1)+' mm':Math.round(m.diameter_px)+' px';
  $('result').innerHTML=`
    <div class="verdict" style="border-left-color:${COLORS[lvl]}">
      <div><div style="font-size:12px;color:var(--mut)">ASSESSMENT</div>
           <div class="lbl" style="color:${COLORS[lvl]}">${m.verdict}</div></div>
      <div class="tds">TDS score<b>${m.TDS.toFixed(2)}</b>&lt;4.75 benign &middot; &gt;5.45 suspicious</div>
    </div>
    <div class="scores">
      ${scoreCard('A &middot; Asymmetry',m.A,2,'asymmetric axes')}
      ${scoreCard('B &middot; Border',m.B,8,'compact '+m.compactness.toFixed(2)+'<br>solidity '+m.solidity.toFixed(2))}
      ${scoreCard('C &middot; Color',m.C,6,Object.keys(m.colors).join(', ')||'-')}
      ${scoreCard('D &middot; Diameter',m.D,5,diam)}
    </div>
    <img class="overlay" src="data:image/png;base64,${d.overlay}">
    <div class="legend"><span><i style="background:#0f0"></i>border</span><span><i style="background:#0cf"></i>convex hull</span>
      <span><i style="background:#f0f"></i>fitted ellipse</span><span><i style="background:#ff0"></i>symmetry axes</span>
      <span><i style="background:#f00"></i>max diameter</span></div>`;
  $('steps').src='data:image/png;base64,'+d.steps; $('stepsCard').style.display='block';
}

async function send(blob){
  if(busy)return; busy=true; $('status').innerHTML='<span class="spin"></span> Analyzing&hellip;';
  const s=$('scale').value;
  try{
    const r=await fetch('/analyze'+(s?'?px_per_mm='+encodeURIComponent(s):''),{method:'POST',body:blob});
    const d=await r.json(); if(!r.ok) throw new Error(d.error);
    render(d); $('status').textContent='';
  }catch(e){$('status').textContent='Error: '+e.message}
  busy=false;
}

$('drop').onclick=()=>$('file').click();
$('file').onchange=e=>{e.target.files[0]&&send(e.target.files[0]);e.target.value=''};
$('drop').ondragover=e=>{e.preventDefault();$('drop').classList.add('over')};
$('drop').ondragleave=()=>$('drop').classList.remove('over');
$('drop').ondrop=e=>{e.preventDefault();$('drop').classList.remove('over');e.dataTransfer.files[0]&&send(e.dataTransfer.files[0])};

function snap(){const v=$('video'),c=document.createElement('canvas');
  c.width=v.videoWidth;c.height=v.videoHeight;c.getContext('2d').drawImage(v,0,0);c.toBlob(send,'image/jpeg',0.9)}
$('camBtn').onclick=async()=>{
  const v=$('video');
  if(v.srcObject){v.srcObject.getTracks().forEach(t=>t.stop());v.srcObject=null;v.style.display='none';
    $('snapBtn').disabled=true;$('camBtn').textContent='Start webcam';return}
  try{v.srcObject=await navigator.mediaDevices.getUserMedia({video:true});v.style.display='block';
      $('snapBtn').disabled=false;$('camBtn').textContent='Stop webcam'}
  catch(e){$('status').textContent='Webcam error: '+e.message}};
$('snapBtn').onclick=snap;
$('live').onchange=e=>{clearInterval(timer);if(e.target.checked)timer=setInterval(()=>{if($('video').srcObject)snap()},2000)};

fetch('/samples').then(r=>r.json()).then(list=>{
  if(!list.length){$('galCard').style.display='none';return}
  $('gallery').innerHTML=list.map(s=>`<figure data-f="${s.file}"><img src="/samples/${s.file}" loading="lazy">
     <span class="${s.label[0]}">${s.label}</span></figure>`).join('');
  document.querySelectorAll('.gallery figure').forEach(f=>f.onclick=async()=>
     send(await (await fetch('/samples/'+f.dataset.f)).blob()));
});
</script></body></html>"""


SAMPLES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "samples")


def list_samples():
    """Sample images listed in samples/labels.csv (file,label,diagnosis,license)."""
    try:
        with open(os.path.join(SAMPLES_DIR, "labels.csv")) as f:
            rows = [r for r in csv.DictReader(f)]
    except FileNotFoundError:
        return []
    return [{"file": r["file"], "label": r["label"]} for r in rows
            if os.path.isfile(os.path.join(SAMPLES_DIR, os.path.basename(r["file"])))]


def b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype):
        data = body.encode() if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _json(self, code, obj):
        conv = lambda o: o.item() if isinstance(o, np.generic) else str(o)
        self._send(code, json.dumps(obj, default=conv), "application/json")

    def do_GET(self):
        path = self.path.split("?")[0]
        if path in ("/", "/index.html"):
            self._send(200, PAGE, "text/html; charset=utf-8")
        elif path == "/samples":
            self._json(200, list_samples())
        elif path.startswith("/samples/"):
            name = os.path.basename(path)  # no directory traversal
            if name in {s["file"] for s in list_samples()}:
                with open(os.path.join(SAMPLES_DIR, name), "rb") as f:
                    self._send(200, f.read(), "image/jpeg")
            else:
                self._send(404, "not found", "text/plain")
        else:
            self._send(404, "not found", "text/plain")

    def do_POST(self):
        if not self.path.startswith("/analyze"):
            return self._send(404, "not found", "text/plain")
        n = int(self.headers.get("Content-Length", 0))
        if not 0 < n <= MAX_BYTES:
            return self._json(400, {"error": "empty or too large (max 20MB)"})
        raw = self.rfile.read(n)

        px = None
        if "px_per_mm=" in self.path:
            try:
                px = float(self.path.split("px_per_mm=")[1].split("&")[0]) or None
            except ValueError:
                pass

        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "upload.png")
            img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                return self._json(400, {"error": "not a valid image"})
            cv2.imwrite(src, img)
            try:
                m = analyze(src, px, tmp, show=False)
            except NoLesionError as e:
                return self._json(200, {"detected": False, "reasons": e.reasons})
            except Exception as e:
                return self._json(422, {"error": str(e)})
            self._json(200, {
                "detected": True,
                "metrics": m,
                "overlay": b64(os.path.join(tmp, "upload_overlay.png")),
                "steps": b64(os.path.join(tmp, "upload_steps.png")),
            })


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    a = ap.parse_args()
    print(f"Melanoma Busters running at http://{a.host}:{a.port}  (Ctrl+C to stop)")
    ThreadingHTTPServer((a.host, a.port), Handler).serve_forever()
