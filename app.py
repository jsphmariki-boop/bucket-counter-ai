import os, json, base64, traceback, mimetypes, io, zipfile, random
from urllib.parse import urlparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import psycopg2
from psycopg2.extras import RealDictCursor

HOST = '0.0.0.0'
PORT = int(os.environ.get('PORT', '8080'))
DATABASE_URL = os.environ.get('DATABASE_URL') or os.environ.get('SUPABASE_DB_URL') or os.environ.get('POSTGRES_URL')
MAX_JSON_BYTES = 12 * 1024 * 1024
CLASSES = {'BUCKET_LOADED':'count','BUCKET_EMPTY':'no count','PEOPLE':'no count','EQUIPMENT':'no count'}
CLASS_IDS = {'BUCKET_LOADED':0,'BUCKET_EMPTY':1,'PEOPLE':2,'EQUIPMENT':3}


def db():
    if not DATABASE_URL:
        raise RuntimeError('DATABASE_URL is not configured in Render Environment Variables.')
    return psycopg2.connect(DATABASE_URL, connect_timeout=10)


def init_db():
    conn = db()
    try:
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS buckets (
            id BIGSERIAL PRIMARY KEY, bucket_code TEXT, bucket_name TEXT NOT NULL,
            location TEXT, status TEXT DEFAULT 'ACTIVE', created_at TIMESTAMPTZ DEFAULT NOW())''')
        c.execute('''CREATE TABLE IF NOT EXISTS dataset_images (
            id BIGSERIAL PRIMARY KEY, filename TEXT NOT NULL, image_data BYTEA NOT NULL,
            labeled BOOLEAN DEFAULT FALSE, created_at TIMESTAMPTZ DEFAULT NOW())''')
        c.execute('ALTER TABLE dataset_images ADD COLUMN IF NOT EXISTS labeled BOOLEAN DEFAULT FALSE')
        c.execute('ALTER TABLE dataset_images ALTER COLUMN created_at SET DEFAULT NOW()')
        c.execute('''CREATE TABLE IF NOT EXISTS annotations (
            id BIGSERIAL PRIMARY KEY, image_id BIGINT NOT NULL REFERENCES dataset_images(id) ON DELETE CASCADE,
            class_name TEXT NOT NULL, x_center DOUBLE PRECISION NOT NULL, y_center DOUBLE PRECISION NOT NULL,
            width DOUBLE PRECISION NOT NULL, height DOUBLE PRECISION NOT NULL, created_at TIMESTAMPTZ DEFAULT NOW())''')
        c.execute('UPDATE dataset_images SET labeled=TRUE WHERE id IN (SELECT DISTINCT image_id FROM annotations)')
        c.execute('ALTER TABLE annotations ADD COLUMN IF NOT EXISTS width DOUBLE PRECISION')
        c.execute('ALTER TABLE annotations ADD COLUMN IF NOT EXISTS height DOUBLE PRECISION')
        c.execute('ALTER TABLE annotations ADD COLUMN IF NOT EXISTS class_name TEXT')
        c.execute('ALTER TABLE annotations ADD COLUMN IF NOT EXISTS x_center DOUBLE PRECISION')
        c.execute('ALTER TABLE annotations ADD COLUMN IF NOT EXISTS y_center DOUBLE PRECISION')
        c.execute('ALTER TABLE annotations ADD COLUMN IF NOT EXISTS image_id BIGINT')
        c.execute('ALTER TABLE annotations ADD COLUMN IF NOT EXISTS box_width DOUBLE PRECISION DEFAULT 0')
        c.execute('ALTER TABLE annotations ADD COLUMN IF NOT EXISTS box_height DOUBLE PRECISION DEFAULT 0')
        c.execute('ALTER TABLE annotations ALTER COLUMN created_at SET DEFAULT NOW()')
        c.execute('ALTER TABLE annotations ALTER COLUMN class_id DROP NOT NULL') if 'class_id' in annotation_columns() else None
        c.execute('ALTER TABLE annotations ALTER COLUMN box_width DROP NOT NULL')
        c.execute('ALTER TABLE annotations ALTER COLUMN box_height DROP NOT NULL')
        c.execute('UPDATE annotations SET created_at=NOW() WHERE created_at IS NULL')
        c.execute('''CREATE TABLE IF NOT EXISTS training_state (
            id INTEGER PRIMARY KEY, status TEXT NOT NULL DEFAULT 'WAITING', progress INTEGER NOT NULL DEFAULT 0,
            message TEXT, updated_at TIMESTAMPTZ DEFAULT NOW())''')
        c.execute("""INSERT INTO training_state(id,status,progress,message)
                     VALUES(1,'WAITING',0,'Ready for external YOLO training')
                     ON CONFLICT(id) DO NOTHING""")
        c.execute('''CREATE TABLE IF NOT EXISTS trained_model (
            id INTEGER PRIMARY KEY, model_name TEXT, model_data BYTEA, uploaded_at TIMESTAMPTZ DEFAULT NOW())''')
        c.execute('''CREATE TABLE IF NOT EXISTS daily_counts (
            id BIGSERIAL PRIMARY KEY, count_date DATE NOT NULL DEFAULT CURRENT_DATE,
            loaded_count INTEGER NOT NULL DEFAULT 0, empty_count INTEGER NOT NULL DEFAULT 0,
            notes TEXT, created_at TIMESTAMPTZ DEFAULT NOW())''')
        conn.commit()
    finally:
        conn.close()


def esc(v):
    return str(v if v is not None else '').replace('&','&amp;').replace('<','&lt;').replace('>','&gt;').replace('"','&quot;')


def send_json(h, data, status=200):
    raw = json.dumps(data, ensure_ascii=False).encode('utf-8')
    try:
        h.send_response(status)
        h.send_header('Content-Type','application/json; charset=utf-8')
        h.send_header('Content-Length',str(len(raw)))
        h.send_header('Cache-Control','no-store')
        h.end_headers(); h.wfile.write(raw)
    except (BrokenPipeError, ConnectionResetError):
        pass


def send_html(h, html, status=200):
    raw = html.encode('utf-8')
    try:
        h.send_response(status)
        h.send_header('Content-Type','text/html; charset=utf-8')
        h.send_header('Content-Length',str(len(raw)))
        h.send_header('Cache-Control','no-store')
        h.end_headers(); h.wfile.write(raw)
    except (BrokenPipeError, ConnectionResetError):
        pass


def read_json(h):
    n = int(h.headers.get('Content-Length','0') or 0)
    if n > MAX_JSON_BYTES:
        raise ValueError('Request is too large. Use a smaller image.')
    if not n:
        return {}
    return json.loads(h.rfile.read(n).decode('utf-8'))


def summary():
    conn=db()
    try:
        c=conn.cursor()
        c.execute('SELECT COUNT(*) FROM dataset_images'); total=c.fetchone()[0]
        c.execute('SELECT COUNT(*) FROM dataset_images WHERE labeled=TRUE'); labeled=c.fetchone()[0]
        c.execute('SELECT COUNT(*) FROM annotations'); annotations=c.fetchone()[0]
        return {'total':total,'labeled':labeled,'annotations':annotations}
    finally: conn.close()


def training_state():
    conn=db()
    try:
        c=conn.cursor(cursor_factory=RealDictCursor)
        c.execute('SELECT status,progress,message,updated_at FROM training_state WHERE id=1')
        r=c.fetchone()
        if not r:
            return {'status':'WAITING','progress':0,'message':'Ready for external YOLO training'}
        r=dict(r)
        if r.get('updated_at'): r['updated_at']=str(r['updated_at'])
        return r
    finally: conn.close()


def model_ready():
    conn=db()
    try:
        c=conn.cursor(); c.execute('SELECT COUNT(*) FROM trained_model WHERE id=1 AND model_data IS NOT NULL')
        return c.fetchone()[0] > 0
    finally: conn.close()


def layout(title, body, active):
    links=[('Dashboard','/'),('Camera','/camera'),('Buckets','/buckets'),('AI Training','/training'),('History','/history'),('Settings','/settings')]
    nav=''.join('<a class="%s" href="%s">%s</a>' % ('active' if x==active else '',u,x) for x,u in links)
    return '''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>%s - Bucket Counter AI</title><style>
*{box-sizing:border-box}body{margin:0;font-family:Arial,sans-serif;background:#f4f6f8;color:#17202a}header{background:#111827;color:#fff;padding:16px;position:sticky;top:0;z-index:5}.brand{font-size:21px;font-weight:800}.sub{font-size:12px;color:#cbd5e1;margin-top:4px}nav{display:flex;gap:6px;overflow:auto;margin-top:12px}nav a{color:#dbeafe;text-decoration:none;padding:9px 11px;border-radius:9px;white-space:nowrap}nav a.active,nav a:hover{background:#2563eb;color:#fff}main{max-width:1100px;margin:auto;padding:18px}.card{background:#fff;border-radius:14px;padding:18px;margin-bottom:16px;box-shadow:0 2px 9px #00000012}h1{margin:0 0 15px;font-size:25px}h2{margin:0 0 12px;font-size:19px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px}.stat{padding:15px;border:1px solid #e2e8f0;border-radius:12px;background:#f8fafc}.stat b{display:block;font-size:28px;margin-top:5px}button{border:0;border-radius:9px;padding:11px 15px;background:#2563eb;color:#fff;font-weight:700;cursor:pointer}button.secondary{background:#475569}button.danger{background:#dc2626}button.green{background:#16a34a}input,select{width:100%%;padding:11px;border:1px solid #cbd5e1;border-radius:9px;margin:6px 0 12px;background:#fff}label{font-weight:700;font-size:14px}.muted{color:#64748b}.badge{padding:5px 9px;border-radius:20px;background:#e2e8f0;font-size:12px;font-weight:700}.classbox{padding:10px;border:1px solid #e2e8f0;border-radius:10px;margin:8px 0}.row{display:flex;gap:10px;flex-wrap:wrap;align-items:center}.camera-wrap{background:#0f172a;border-radius:12px;padding:10px;text-align:center}video{width:100%%;max-height:500px;background:#000;border-radius:8px}canvas{max-width:100%%;border:2px solid #334155;border-radius:10px;touch-action:none}#canvasWrap{overflow:auto;text-align:center;background:#0f172a;padding:10px;border-radius:12px}.notice{padding:12px;border-radius:10px;background:#fff7ed;border:1px solid #fed7aa;margin-top:12px}.hidden{display:none!important}table{width:100%%;border-collapse:collapse}th,td{padding:10px;border-bottom:1px solid #e5e7eb;text-align:left}footer{text-align:center;padding:25px;color:#64748b;font-size:12px}</style></head><body><header><div class="brand">🪣 BUCKET COUNTER AI</div><div class="sub">Underground production monitoring</div><nav>%s</nav></header><main>%s</main><footer>Geology &amp; Mining Services</footer></body></html>''' % (esc(title),nav,body)


def dashboard_page():
    s=summary(); ready='READY' if model_ready() else 'NOT READY'
    return layout('Dashboard','''<div class="card"><h1>📊 Dashboard</h1><p class="muted">Underground bucket production monitoring system.</p></div><div class="grid"><div class="stat">Dataset images<b>%s</b></div><div class="stat">Labeled<b>%s</b></div><div class="stat">Annotations<b>%s</b></div><div class="stat">AI Model<b>%s</b></div></div><div class="card"><h2>System status</h2><p>Database: <span class="badge">Supabase PostgreSQL</span></p><p>YOLO: <span class="badge">AVAILABLE FOR MODEL USE</span></p><p>Model: <span class="badge">%s</span></p></div>''' % (s['total'],s['labeled'],s['annotations'],ready,ready),'Dashboard')


def camera_page():
    return layout('Camera','''<div class="card"><h1>📷 Camera</h1><p class="muted">Camera preview. Detection will be enabled after a trained model is uploaded.</p><div class="camera-wrap"><video id="v" autoplay playsinline></video></div><div class="row" style="margin-top:10px"><button onclick="startCamera()">📷 START CAMERA</button><button class="secondary" onclick="stopCamera()">STOP</button></div><p id="m" class="notice">Model is not ready.</p></div><script>let stream=null;async function startCamera(){try{stream=await navigator.mediaDevices.getUserMedia({video:{facingMode:{ideal:'environment'}},audio:false});v.srcObject=stream;m.textContent='Camera is running.'}catch(e){m.textContent=e.message}}function stopCamera(){if(stream)stream.getTracks().forEach(x=>x.stop());stream=null;v.srcObject=null}</script>''','Camera')


def buckets_page():
    conn=db()
    try:
        c=conn.cursor(cursor_factory=RealDictCursor); c.execute('SELECT id,bucket_code,bucket_name,location,status FROM buckets ORDER BY id DESC'); rows=c.fetchall()
    finally: conn.close()
    tr=''.join('<tr><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td></tr>'%(r['id'],esc(r['bucket_code']),esc(r['bucket_name']),esc(r['location']),esc(r['status'])) for r in rows)
    body='''<div class="card"><h1>🪣 Buckets</h1><div class="grid"><div><label>Bucket code</label><input id="code"></div><div><label>Bucket name</label><input id="name"></div><div><label>Location</label><input id="loc"></div></div><button onclick="addBucket()">➕ ADD BUCKET</button><p id="msg" class="muted"></p></div><div class="card"><h2>Registered buckets</h2><div style="overflow:auto"><table><thead><tr><th>ID</th><th>Code</th><th>Name</th><th>Location</th><th>Status</th></tr></thead><tbody>%s</tbody></table></div></div><script>async function addBucket(){let d={bucket_code:code.value.trim(),bucket_name:name.value.trim(),location:loc.value.trim()};let r=await fetch('/api/buckets',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(d)});let j=await r.json();if(j.ok)location.reload();else msg.textContent=j.error||'Failed.'}</script>'''%(tr or '<tr><td colspan="5">No buckets yet.</td></tr>')
    return layout('Buckets',body,'Buckets')


def training_page():
    s=summary(); st=training_state(); ready=model_ready(); prog=int(st.get('progress') or 0); message=esc(st.get('message') or st.get('status'))
    body='''<div class="card"><h1>🤖 AI Training</h1><p>Database: <span class="badge">Supabase PostgreSQL</span> | YOLO: <span class="badge">AVAILABLE FOR MODEL USE</span> | Model: <span class="badge">__MODEL__</span></p></div>
<div class="card"><h2>Classes</h2><div class="classbox"><b>BUCKET_LOADED</b> = count</div><div class="classbox"><b>BUCKET_EMPTY</b> = no count</div><div class="classbox"><b>PEOPLE</b> = no count</div><div class="classbox"><b>EQUIPMENT</b> = no count</div></div>
<div class="card"><h2>➕ Add Training Image</h2><p class="muted">Piga picha au chagua picha kutoka kwenye simu/computer. Kisha chora box kuzunguka object unayotaka AI ijifunze.</p><p class="muted">Unaweza kuweka objects nyingi kwenye picha moja.</p>
<input id="fileInput" type="file" accept="image/*" style="display:none" onchange="handleFile(this)">
<div class="row"><button type="button" onclick="document.getElementById('fileInput').click()">📁 CHOOSE IMAGE</button><button type="button" class="secondary" onclick="openCamera()">📷 OPEN CAMERA</button><button type="button" class="danger" onclick="clearBoxes()">🗑️ CLEAR ALL BOXES</button></div>
<p id="fileName" class="muted">No file chosen</p><div id="cameraBox" class="camera-wrap hidden"><video id="trainVideo" autoplay playsinline></video><div class="row" style="margin-top:10px"><button onclick="takePhoto()">📸 TAKE PHOTO</button><button class="secondary" onclick="closeCamera()">CLOSE CAMERA</button></div></div>
<div id="editor" class="hidden"><label>Class:</label><select id="classSelect"><option value="BUCKET_LOADED">BUCKET_LOADED — COUNT</option><option value="BUCKET_EMPTY">BUCKET_EMPTY — NO COUNT</option><option value="PEOPLE">PEOPLE — NO COUNT</option><option value="EQUIPMENT">EQUIPMENT — NO COUNT</option></select><p class="muted">Chagua class kisha drag kwenye picha kuchora box.</p><div id="canvasWrap"><canvas id="canvas"></canvas></div><div id="boxList"></div><button class="green" onclick="saveImage()">💾 SAVE TRAINING IMAGE</button> <button class="secondary" onclick="cancelImage()">CANCEL</button><p id="saveMsg" class="muted"></p></div></div>
<div class="card"><h2>Dataset</h2><div class="grid"><div class="stat">Total images<b id="total">__TOTAL__</b></div><div class="stat">Labeled<b id="labeled">__LABELED__</b></div><div class="stat">Annotations<b id="ann">__ANN__</b></div></div></div>
<div class="card"><h2>Training status</h2><p>__MESSAGE__</p><div style="height:18px;background:#e2e8f0;border-radius:20px;overflow:hidden"><div style="height:100%;width:__PROGRESS__%;background:#2563eb"></div></div><p>__PROGRESS__%</p><div class="notice">⚠️ YOLO training is not executed inside the Render Web Service. Images and labels are safely stored in Supabase PostgreSQL. After a trained model is uploaded, the system will use it for bucket detection.</div></div>
<script>
let currentImage=null,currentFilename='',boxes=[],drawing=false,sx=0,sy=0,tempBox=null,cameraStream=null;const canvas=document.getElementById('canvas'),ctx=canvas.getContext('2d');
function handleFile(input){const file=input.files&&input.files[0];if(!file)return;currentFilename=file.name;document.getElementById('fileName').textContent=file.name;const reader=new FileReader();reader.onload=e=>loadImage(e.target.result,file.name);reader.readAsDataURL(file)}
function loadImage(src,name){const img=new Image();img.onload=function(){currentImage=img;currentFilename=name||'training.jpg';boxes=[];const scale=Math.min(1,(window.innerWidth-50)/img.naturalWidth);canvas.width=Math.round(img.naturalWidth*scale);canvas.height=Math.round(img.naturalHeight*scale);document.getElementById('editor').classList.remove('hidden');drawCanvas();updateList()};img.src=src}
function getPoint(e){const r=canvas.getBoundingClientRect(),q=e.touches&&e.touches[0]?e.touches[0]:e;return{x:(q.clientX-r.left)*canvas.width/r.width,y:(q.clientY-r.top)*canvas.height/r.height}}
function beginDraw(e){if(!currentImage)return;e.preventDefault();const p=getPoint(e);drawing=true;sx=p.x;sy=p.y;tempBox={x:sx,y:sy,w:0,h:0}}
function moveDraw(e){if(!drawing)return;e.preventDefault();const p=getPoint(e);tempBox={x:Math.min(sx,p.x),y:Math.min(sy,p.y),w:Math.abs(p.x-sx),h:Math.abs(p.y-sy)};drawCanvas()}
function endDraw(e){if(!drawing)return;e.preventDefault();drawing=false;if(tempBox&&tempBox.w>=8&&tempBox.h>=8)boxes.push({x:tempBox.x,y:tempBox.y,w:tempBox.w,h:tempBox.h,className:document.getElementById('classSelect').value});tempBox=null;drawCanvas();updateList()}
canvas.addEventListener('mousedown',beginDraw);canvas.addEventListener('mousemove',moveDraw);canvas.addEventListener('mouseup',endDraw);canvas.addEventListener('mouseleave',endDraw);canvas.addEventListener('touchstart',beginDraw,{passive:false});canvas.addEventListener('touchmove',moveDraw,{passive:false});canvas.addEventListener('touchend',endDraw,{passive:false});
function drawCanvas(){if(!currentImage)return;ctx.clearRect(0,0,canvas.width,canvas.height);ctx.drawImage(currentImage,0,0,canvas.width,canvas.height);boxes.forEach((b,i)=>{ctx.lineWidth=3;ctx.strokeStyle='#00ff66';ctx.strokeRect(b.x,b.y,b.w,b.h);ctx.fillStyle='#000';ctx.fillRect(b.x,Math.max(0,b.y-21),170,21);ctx.fillStyle='#fff';ctx.font='12px Arial';ctx.fillText((i+1)+' '+b.className,b.x+4,Math.max(14,b.y-7))});if(tempBox){ctx.strokeStyle='#ffcc00';ctx.setLineDash([6,5]);ctx.strokeRect(tempBox.x,tempBox.y,tempBox.w,tempBox.h);ctx.setLineDash([])}}
function updateList(){const el=document.getElementById('boxList');if(!boxes.length){el.innerHTML='<p class="muted">No boxes yet.</p>';return}el.innerHTML=boxes.map((b,i)=>'<div class="classbox"><b>'+(i+1)+'. '+b.className+'</b> <button class="danger" onclick="removeBox('+i+')">DELETE</button></div>').join('')}
function removeBox(i){boxes.splice(i,1);drawCanvas();updateList()}function clearBoxes(){boxes=[];drawCanvas();updateList()}
async function openCamera(){try{cameraStream=await navigator.mediaDevices.getUserMedia({video:{facingMode:{ideal:'environment'}},audio:false});document.getElementById('trainVideo').srcObject=cameraStream;document.getElementById('cameraBox').classList.remove('hidden')}catch(e){alert('Camera could not be opened: '+e.message)}}
function closeCamera(){if(cameraStream)cameraStream.getTracks().forEach(t=>t.stop());cameraStream=null;document.getElementById('trainVideo').srcObject=null;document.getElementById('cameraBox').classList.add('hidden')}
function takePhoto(){const v=document.getElementById('trainVideo');if(!v.videoWidth){alert('Camera is not ready.');return}const c=document.createElement('canvas'),scale=Math.min(1,1280/v.videoWidth);c.width=Math.round(v.videoWidth*scale);c.height=Math.round(v.videoHeight*scale);c.getContext('2d').drawImage(v,0,0,c.width,c.height);loadImage(c.toDataURL('image/jpeg',0.82),'camera_'+Date.now()+'.jpg');document.getElementById('fileName').textContent='Camera photo';closeCamera()}
function cancelImage(){currentImage=null;currentFilename='';boxes=[];document.getElementById('editor').classList.add('hidden');document.getElementById('fileInput').value='';document.getElementById('fileName').textContent='No file chosen';closeCamera()}
function imageData(){const c=document.createElement('canvas'),scale=Math.min(1,1280/currentImage.naturalWidth);c.width=Math.round(currentImage.naturalWidth*scale);c.height=Math.round(currentImage.naturalHeight*scale);c.getContext('2d').drawImage(currentImage,0,0,c.width,c.height);return c.toDataURL('image/jpeg',0.82)}
async function saveImage(){const msg=document.getElementById('saveMsg');if(!currentImage){msg.textContent='Choose an image first.';return}if(!boxes.length){msg.textContent='Draw at least one box.';return}msg.textContent='Saving...';const annotations=boxes.map(b=>({class_name:b.className,x_center:(b.x+b.w/2)/canvas.width,y_center:(b.y+b.h/2)/canvas.height,width:b.w/canvas.width,height:b.h/canvas.height}));try{const r=await fetch('/api/training/image',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({filename:currentFilename||'training.jpg',image_data:imageData(),annotations})});const j=await r.json();if(!j.ok){msg.textContent=j.error||'Save failed.';return}msg.textContent='✅ Saved successfully.';document.getElementById('total').textContent=j.summary.total;document.getElementById('labeled').textContent=j.summary.labeled;document.getElementById('ann').textContent=j.summary.annotations;setTimeout(cancelImage,1000)}catch(e){msg.textContent='Save error: '+e.message}}
</script>
<div class="card">
<h2>📦 YOLO Dataset</h2>
<p class="muted">Export images na annotations zako kuwa dataset ya YOLO. Mfumo utagawanya dataset kuwa <b>train</b> na <b>val</b> automatically.</p>
<div class="classbox"><b>Class IDs</b><br>0 = BUCKET_LOADED (count)<br>1 = BUCKET_EMPTY (no count)<br>2 = PEOPLE (no count)<br>3 = EQUIPMENT (no count)</div>
<button type="button" class="green" onclick="exportYolo()">📦 EXPORT YOLO DATASET</button>
<p id="exportMsg" class="muted"></p>
</div>
<div class="card">
<h2>🖼️ Training Images Gallery</h2>
<p class="muted">Review images zilizohifadhiwa, badilisha class ya box, au futa image.</p>
<div id="gallery" class="grid"><p class="muted">Loading...</p></div>
</div>
<div id="reviewBox" class="card hidden">
<h2>🔍 Review Training Image</h2>
<p id="reviewName" class="muted"></p>
<div style="text-align:center;background:#0f172a;padding:10px;border-radius:12px"><img id="reviewImage" style="max-width:100%;max-height:600px;border-radius:8px"></div>
<div id="reviewAnnotations" style="margin-top:12px"></div>
<div class="row" style="margin-top:12px"><button type="button" class="secondary" onclick="closeReview()">CLOSE</button><button type="button" class="danger" onclick="deleteReviewedImage()">🗑️ DELETE IMAGE</button></div>
<p id="reviewMsg" class="notice"></p>
</div>
<script>
let reviewedId=null;
function escapeHtml(s){return String(s??'').replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;').replaceAll('"','&quot;')}
async function loadGallery(){const el=document.getElementById('gallery');try{const r=await fetch('/api/training/images');const j=await r.json();if(!j.ok){el.innerHTML='<p class="muted">'+escapeHtml(j.error)+'</p>';return}if(!j.images.length){el.innerHTML='<p class="muted">No training images yet.</p>';return}el.innerHTML=j.images.map(x=>'<div class="card" style="margin:0;border:1px solid #e2e8f0;box-shadow:none"><img src="/api/training/image/'+x.id+'" style="width:100%;height:180px;object-fit:cover;border-radius:9px"><b>'+escapeHtml(x.filename)+'</b><p class="muted">Boxes: '+x.annotation_count+' | '+escapeHtml(x.created_at||'')+'</p><div class="row"><button type="button" onclick="reviewImage('+x.id+')">🔍 REVIEW</button><button type="button" class="danger" onclick="deleteImage('+x.id+')">🗑️ DELETE</button></div></div>').join('')}catch(e){el.innerHTML='<p class="muted">Gallery error: '+escapeHtml(e.message)+'</p>'}}
async function refreshGallery(){loadGallery()}
async function reviewImage(id){reviewedId=id;const box=document.getElementById('reviewBox');box.classList.remove('hidden');box.scrollIntoView({behavior:'smooth',block:'start'});document.getElementById('reviewImage').src='';document.getElementById('reviewMsg').textContent='Loading...';try{const r=await fetch('/api/training/image/'+id+'/detail');const j=await r.json();if(!j.ok){document.getElementById('reviewMsg').textContent=j.error;return}document.getElementById('reviewName').textContent=j.image.filename;document.getElementById('reviewImage').src=j.image.image_data||('/api/training/image/'+id);document.getElementById('reviewImage').style.display='block';document.getElementById('reviewAnnotations').innerHTML=j.annotations.map((a,i)=>'<div class="classbox"><b>'+(i+1)+'. '+escapeHtml(a.class_name)+'</b><select id="editClass'+a.id+'"><option value="BUCKET_LOADED">BUCKET_LOADED</option><option value="BUCKET_EMPTY">BUCKET_EMPTY</option><option value="PEOPLE">PEOPLE</option><option value="EQUIPMENT">EQUIPMENT</option></select><button type="button" class="green" onclick="updateAnnotation('+a.id+')">SAVE CLASS</button></div>').join('');j.annotations.forEach(a=>{document.getElementById('editClass'+a.id).value=a.class_name});document.getElementById('reviewMsg').textContent=''}catch(e){document.getElementById('reviewMsg').textContent=e.message}}
function closeReview(){document.getElementById('reviewBox').classList.add('hidden');reviewedId=null}
async function updateAnnotation(id){const cls=document.getElementById('editClass'+id).value;const r=await fetch('/api/training/annotation/'+id,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({class_name:cls})});const j=await r.json();document.getElementById('reviewMsg').textContent=j.ok?'✅ Class updated.':(j.error||'Update failed.');if(j.ok)loadGallery()}
async function deleteImage(id){if(!confirm('Delete this training image and all its annotations?'))return;const r=await fetch('/api/training/image/'+id,{method:'DELETE'});const j=await r.json();if(!j.ok){alert(j.error||'Delete failed.');return}document.getElementById('reviewBox').classList.add('hidden');reviewedId=null;loadGallery();location.reload()}
async function deleteReviewedImage(){if(reviewedId)await deleteImage(reviewedId)}
async function exportYolo(){const msg=document.getElementById('exportMsg');msg.textContent='Preparing YOLO dataset...';try{const r=await fetch('/api/training/export-yolo');if(!r.ok){let t=await r.text();msg.textContent='Export failed: '+t;return}const blob=await r.blob();const url=URL.createObjectURL(blob);const a=document.createElement('a');a.href=url;a.download='bucket_counter_yolo_dataset.zip';document.body.appendChild(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(url),2000);msg.textContent='✅ YOLO dataset exported and downloaded.'}catch(e){msg.textContent='Export error: '+e.message}}
loadGallery();
</script>'''.replace('__MODEL__', 'READY' if ready else 'NOT READY').replace('__TOTAL__', str(s['total'])).replace('__LABELED__', str(s['labeled'])).replace('__ANN__', str(s['annotations'])).replace('__MESSAGE__', message).replace('__PROGRESS__', str(prog))

    return layout('AI Training',body,'AI Training')


def history_page():
    return layout('History','<div class="card"><h1>📜 History</h1><p class="muted">Production history will appear here after detection/counting is connected.</p></div>','History')


def settings_page():
    return layout('Settings','<div class="card"><h1>⚙️ Settings</h1><div class="classbox"><b>Counting rule</b><br>Only BUCKET_LOADED is counted.</div><div class="classbox"><b>Not counted</b><br>BUCKET_EMPTY, PEOPLE and EQUIPMENT.</div><div class="classbox"><b>Database</b><br>Supabase PostgreSQL</div></div>','Settings')


def class_counts():
    conn=db()
    try:
        c=conn.cursor()
        c.execute('SELECT class_name, COUNT(*) FROM annotations GROUP BY class_name')
        out={k:0 for k in CLASSES}
        for k,v in c.fetchall():
            if k in out: out[k]=int(v)
        return out
    finally:
        conn.close()


def training_images():
    conn=db()
    try:
        c=conn.cursor(cursor_factory=RealDictCursor)
        c.execute('''SELECT d.id,d.filename,d.created_at,d.labeled,COUNT(a.id) AS annotation_count
                     FROM dataset_images d LEFT JOIN annotations a ON a.image_id=d.id
                     GROUP BY d.id ORDER BY d.id DESC''')
        rows=[]
        for r in c.fetchall():
            r=dict(r)
            if r.get('created_at'): r['created_at']=str(r['created_at'])
            r['annotation_count']=int(r.get('annotation_count') or 0)
            rows.append(r)
        return rows
    finally:
        conn.close()


def annotation_columns():
    conn=db()
    try:
        c=conn.cursor()
        c.execute("SELECT column_name FROM information_schema.columns WHERE table_schema='public' AND table_name='annotations'")
        return {r[0] for r in c.fetchall()}
    finally:
        conn.close()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args): print('%s - %s' % (self.address_string(), fmt % args))
    def do_GET(self):
        try:
            p=urlparse(self.path).path
            pages={'/':dashboard_page,'/camera':camera_page,'/buckets':buckets_page,'/training':training_page,'/history':history_page,'/settings':settings_page}
            if p in pages: send_html(self,pages[p]()); return
            if p=='/api/status': send_json(self,{'ok':True,'database':'Supabase PostgreSQL','model_ready':model_ready(),'training':training_state()}); return
            if p=='/api/training/summary': send_json(self,{'ok':True,'summary':summary(),'classes':class_counts(),'state':training_state(),'model_ready':model_ready()}); return
            if p=='/api/model/status': send_json(self,{'ok':True,'ready':model_ready()}); return
            if p=='/api/training/export-yolo': self.export_yolo(); return
            if p=='/api/training/images': send_json(self,{'ok':True,'images':training_images()}); return
            if p.startswith('/api/training/image/') and p.endswith('/detail'):
                iid=int(p.split('/')[4])
                conn=db()
                try:
                    c=conn.cursor(cursor_factory=RealDictCursor)
                    c.execute('SELECT id,filename,image_data,created_at,labeled FROM dataset_images WHERE id=%s',(iid,)); img=c.fetchone()
                    if not img: send_json(self,{'ok':False,'error':'Image not found.'},404); return
                    c.execute('SELECT id,class_name,x_center,y_center,width,height FROM annotations WHERE image_id=%s ORDER BY id',(iid,)); anns=[dict(x) for x in c.fetchall()]
                    img=dict(img)
                    raw=img.pop('image_data',None)
                    if raw is not None:
                        if isinstance(raw,str):
                            if raw.startswith('data:image/'):
                                image_data=raw
                            else:
                                try: image_data='data:image/jpeg;base64,'+raw
                                except Exception: image_data=''
                        else:
                            b=bytes(raw)
                            ext=str(img.get('filename','')).lower()
                            mime='image/jpeg'
                            if ext.endswith('.png'): mime='image/png'
                            elif ext.endswith('.webp'): mime='image/webp'
                            image_data='data:'+mime+';base64,'+base64.b64encode(b).decode('ascii')
                        img['image_data']=image_data
                    if img.get('created_at'): img['created_at']=str(img['created_at'])
                    send_json(self,{'ok':True,'image':img,'annotations':anns})
                finally: conn.close()
                return
            if p.startswith('/api/training/image/'):
                iid=int(p.rsplit('/',1)[1])
                conn=db()
                try:
                    c=conn.cursor(); c.execute('SELECT image_data,filename FROM dataset_images WHERE id=%s',(iid,)); row=c.fetchone()
                    if not row: send_json(self,{'ok':False,'error':'Image not found.'},404); return
                    raw,filename=row; ext=str(filename).lower(); mime='image/jpeg'
                    if ext.endswith('.png'): mime='image/png'
                    elif ext.endswith('.webp'): mime='image/webp'
                    if isinstance(raw,str):
                        if raw.startswith('data:image/') and ',' in raw:
                            raw=base64.b64decode(raw.split(',',1)[1])
                        else:
                            raw=base64.b64decode(raw)
                    raw=bytes(raw)
                    try:
                        self.send_response(200); self.send_header('Content-Type',mime); self.send_header('Content-Length',str(len(raw))); self.send_header('Cache-Control','no-store'); self.send_header('Content-Disposition','inline'); self.end_headers(); self.wfile.write(raw)
                    except (BrokenPipeError,ConnectionResetError): pass
                finally: conn.close()
                return
            send_json(self,{'ok':False,'error':'Not found'},404)
        except Exception as e: traceback.print_exc(); send_json(self,{'ok':False,'error':str(e)},500)
    def do_POST(self):
        try:
            p=urlparse(self.path).path; d=read_json(self)
            if p=='/api/buckets': self.add_bucket(d)
            elif p=='/api/training/image': self.add_training_image(d)
            elif p=='/api/training/state': self.update_training_state(d)
            elif p.startswith('/api/training/annotation/'):
                self.update_annotation(int(p.rsplit('/',1)[1]),d)
            elif p=='/api/train': send_json(self,{'ok':True,'message':'Ready for external YOLO training.','summary':summary()})
            else: send_json(self,{'ok':False,'error':'Not found'},404)
        except ValueError as e: send_json(self,{'ok':False,'error':str(e)},400)
        except Exception as e: traceback.print_exc(); send_json(self,{'ok':False,'error':str(e)},500)
    def do_DELETE(self):
        try:
            p=urlparse(self.path).path
            if p.startswith('/api/training/image/'):
                self.delete_training_image(int(p.rsplit('/',1)[1])); return
            send_json(self,{'ok':False,'error':'Not found'},404)
        except ValueError as e: send_json(self,{'ok':False,'error':str(e)},400)
        except Exception as e: traceback.print_exc(); send_json(self,{'ok':False,'error':str(e)},500)

    def export_yolo(self):
        conn=db()
        try:
            c=conn.cursor(cursor_factory=RealDictCursor)
            c.execute("SELECT id,filename,image_data FROM dataset_images ORDER BY id ASC")
            images=[dict(x) for x in c.fetchall()]
            if not images:
                send_json(self,{'ok':False,'error':'No training images found.'},400); return
            rows=[]
            for img in images:
                c.execute("SELECT class_name,x_center,y_center,width,height FROM annotations WHERE image_id=%s ORDER BY id",(img['id'],))
                anns=[dict(x) for x in c.fetchall()]
                if anns: rows.append((img,anns))
        finally: conn.close()
        if not rows:
            send_json(self,{'ok':False,'error':'No labeled training images found.'},400); return
        rng=random.Random(42); rng.shuffle(rows)
        if len(rows)==1: train_rows,val_rows=rows,[]
        else:
            val_n=max(1,round(len(rows)*0.2)); val_n=min(val_n,len(rows)-1)
            val_rows=rows[:val_n]; train_rows=rows[val_n:]
        def safe_filename(name,iid):
            name=os.path.basename(str(name or 'image.jpg')); root,ext=os.path.splitext(name); ext=ext.lower()
            if ext not in ('.jpg','.jpeg','.png','.webp','.bmp'): ext='.jpg'
            clean=''.join(ch if ch.isalnum() or ch in ('-','_') else '_' for ch in root).strip('_') or 'image'
            return f'{iid}_{clean}{ext}'
        def image_bytes(raw):
            if raw is None:
                raise ValueError('Image data is missing.')
            if isinstance(raw,str):
                raw=raw.strip()
                if raw.startswith('data:image/') and ',' in raw:
                    raw=raw.split(',',1)[1].strip()
                # PostgreSQL encode(..., 'base64') may contain line breaks.
                raw=''.join(raw.split())
                if not raw:
                    raise ValueError('Image data is empty.')
                # Add harmless padding when needed. A remainder of 1 is never
                # valid Base64, so fail with a useful message instead.
                rem=len(raw)%4
                if rem==1:
                    raise ValueError('Invalid image Base64 data.')
                if rem:
                    raw += '='*(4-rem)
                try:
                    return base64.b64decode(raw,validate=True)
                except Exception as e:
                    raise ValueError('Invalid image Base64 data.') from e
            if isinstance(raw,(bytes,bytearray,memoryview)):
                b=bytes(raw)
                # Normal PostgreSQL BYTEA: the value is already the original
                # image bytes. Do NOT Base64-decode real JPEG/PNG/WebP data.
                if b.startswith(b'\xff\xd8\xff') or b.startswith(b'\x89PNG\r\n\x1a\n') or b.startswith(b'RIFF') or b.startswith(b'GIF8') or b.startswith(b'BM'):
                    return b
                # Some older records may contain a data URI or Base64 text
                # inside the BYTEA column. Handle those records too.
                try:
                    text=b.decode('utf-8').strip()
                except UnicodeDecodeError:
                    return b
                if text.startswith('data:image/') and ',' in text:
                    text=text.split(',',1)[1].strip()
                compact=''.join(text.split())
                if compact:
                    rem=len(compact)%4
                    if rem==1:
                        # It is not valid Base64; preserve the original bytes
                        # rather than incorrectly rejecting a valid image format.
                        return b
                    if rem:
                        compact += '='*(4-rem)
                    try:
                        decoded=base64.b64decode(compact,validate=True)
                        if decoded.startswith((b'\xff\xd8\xff',b'\x89PNG\r\n\x1a\n',b'RIFF',b'GIF8',b'BM')):
                            return decoded
                    except Exception:
                        pass
                return b
            return bytes(raw)
        def make_label(anns):
            lines=[]
            for a in anns:
                cls=a.get('class_name')
                if cls not in CLASS_IDS: continue
                try: x,y,w,h=[float(a[k]) for k in ('x_center','y_center','width','height')]
                except Exception: continue
                if not (0<=x<=1 and 0<=y<=1 and 0<w<=1 and 0<h<=1): continue
                lines.append(f"{CLASS_IDS[cls]} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
            return '\n'.join(lines)+'\n' if lines else ''
        mem=io.BytesIO()
        with zipfile.ZipFile(mem,'w',compression=zipfile.ZIP_DEFLATED) as z:
            data_yaml="""path: .
train: images/train
val: images/val
nc: 4
names:
  0: BUCKET_LOADED
  1: BUCKET_EMPTY
  2: PEOPLE
  3: EQUIPMENT
"""
            z.writestr('data.yaml',data_yaml)
            z.writestr('README.txt','BUCKET COUNTER AI - YOLO DATASET\n\nClasses:\n0 BUCKET_LOADED (count)\n1 BUCKET_EMPTY (no count)\n2 PEOPLE (no count)\n3 EQUIPMENT (no count)\n')
            manifest=[]
            for split,items in (('train',train_rows),('val',val_rows)):
                for img,anns in items:
                    fname=safe_filename(img.get('filename'),img.get('id'))
                    z.writestr(f'images/{split}/{fname}',image_bytes(img.get('image_data')))
                    z.writestr(f'labels/{split}/{os.path.splitext(fname)[0]}.txt',make_label(anns))
                    manifest.append({'image_id':img.get('id'),'filename':img.get('filename'),'split':split,'annotations':len(anns)})
            z.writestr('manifest.json',json.dumps(manifest,ensure_ascii=False,indent=2))
        raw=mem.getvalue()
        try:
            self.send_response(200); self.send_header('Content-Type','application/zip'); self.send_header('Content-Length',str(len(raw))); self.send_header('Content-Disposition','attachment; filename="bucket_counter_yolo_dataset.zip"'); self.send_header('Cache-Control','no-store'); self.end_headers(); self.wfile.write(raw)
        except (BrokenPipeError,ConnectionResetError): pass

    def add_bucket(self,d):
        name=str(d.get('bucket_name','')).strip()
        if not name: send_json(self,{'ok':False,'error':'Bucket name is required.'},400); return
        conn=db()
        try:
            c=conn.cursor(); c.execute('INSERT INTO buckets(bucket_code,bucket_name,location) VALUES(%s,%s,%s) RETURNING id',(str(d.get('bucket_code','')),name,str(d.get('location','')))); i=c.fetchone()[0]; conn.commit(); send_json(self,{'ok':True,'id':i})
        finally: conn.close()
    def add_training_image(self,d):
        src=str(d.get('image_data','')); anns=d.get('annotations') or []
        if not src.startswith('data:image/'): send_json(self,{'ok':False,'error':'Invalid image.'},400); return
        if not anns: send_json(self,{'ok':False,'error':'At least one box is required.'},400); return
        try: raw=base64.b64decode(src.split(',',1)[1],validate=True)
        except Exception: send_json(self,{'ok':False,'error':'Could not decode image.'},400); return
        if len(raw)>8*1024*1024: send_json(self,{'ok':False,'error':'Image is too large. Maximum 8 MB.'},400); return
        clean=[]
        for a in anns:
            cls=str(a.get('class_name',''))
            if cls not in CLASSES: send_json(self,{'ok':False,'error':'Invalid class.'},400); return
            try: x,y,w,h=[float(a[k]) for k in ('x_center','y_center','width','height')]
            except Exception: send_json(self,{'ok':False,'error':'Invalid box.'},400); return
            if not(0<=x<=1 and 0<=y<=1 and 0<w<=1 and 0<h<=1): send_json(self,{'ok':False,'error':'Box values must be normalized 0..1.'},400); return
            clean.append((cls,x,y,w,h))
        conn=db()
        try:
            c=conn.cursor(); c.execute('INSERT INTO dataset_images(filename,image_data,labeled,created_at) VALUES(%s,%s,TRUE,NOW()) RETURNING id',(str(d.get('filename') or 'training.jpg'),psycopg2.Binary(raw))); iid=c.fetchone()[0]
            cols=annotation_columns()
            for cls,x,y,w,h in clean:
                names=['image_id','class_name','x_center','y_center','width','height']; vals=[iid,cls,x,y,w,h]
                if 'class_id' in cols: names.insert(1,'class_id'); vals.insert(1,None)
                if 'box_width' in cols: names.append('box_width'); vals.append(w)
                if 'box_height' in cols: names.append('box_height'); vals.append(h)
                if 'created_at' in cols: names.append('created_at')
                placeholders=[]; final_vals=[]
                for n,v in zip(names,vals):
                    if n=='created_at': placeholders.append('NOW()')
                    else: placeholders.append('%s'); final_vals.append(v)
                c.execute('INSERT INTO annotations('+','.join(names)+') VALUES('+','.join(placeholders)+')',tuple(final_vals))
            conn.commit(); send_json(self,{'ok':True,'image_id':iid,'summary':summary(),'classes':class_counts()})
        except Exception: conn.rollback(); raise
        finally: conn.close()

    def update_annotation(self,aid,d):
        cls=str(d.get('class_name',''))
        if cls not in CLASSES: send_json(self,{'ok':False,'error':'Invalid class.'},400); return
        conn=db()
        try:
            c=conn.cursor(); c.execute('UPDATE annotations SET class_name=%s WHERE id=%s',(cls,aid))
            if c.rowcount==0: conn.rollback(); send_json(self,{'ok':False,'error':'Annotation not found.'},404); return
            conn.commit(); send_json(self,{'ok':True,'classes':class_counts()})
        finally: conn.close()

    def delete_training_image(self,iid):
        conn=db()
        try:
            c=conn.cursor(); c.execute('DELETE FROM dataset_images WHERE id=%s',(iid,))
            if c.rowcount==0: conn.rollback(); send_json(self,{'ok':False,'error':'Image not found.'},404); return
            conn.commit(); send_json(self,{'ok':True,'summary':summary(),'classes':class_counts()})
        finally: conn.close()

    def update_training_state(self,d):
        conn=db()
        try:
            c=conn.cursor(); progress=max(0,min(100,int(d.get('progress',0)))); c.execute('UPDATE training_state SET status=%s,progress=%s,message=%s,updated_at=NOW() WHERE id=1',(str(d.get('status','WAITING')),progress,str(d.get('message','')))); conn.commit(); send_json(self,{'ok':True})
        finally: conn.close()


if __name__=='__main__':
    print('==============================================')
    print(' BUCKET COUNTER AI')
    print('==============================================')
    init_db()
    print('Database initialization: OK')
    print('Server running on port',PORT)
    ThreadingHTTPServer((HOST,PORT),Handler).serve_forever()
