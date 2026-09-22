import os
import io
import csv
import json
import base64
import traceback
import tempfile
import shutil
import threading
import time
import uuid
from datetime import datetime
from urllib.parse import urlparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/Ultralytics")

import psycopg2
from psycopg2.extras import RealDictCursor
from ultralytics import YOLO
from PIL import Image
import numpy as np

# ============================================================
# NEERIKA BUCKET AI
# Mining Production Bucket Counter
# ============================================================

HOST = "0.0.0.0"
PORT = int(os.environ.get("PORT", "8080"))

DATABASE_URL = (
    os.environ.get("DATABASE_URL")
    or os.environ.get("SUPABASE_DB_URL")
    or os.environ.get("POSTGRES_URL")
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "best.pt")

MODEL_BACKUP_PATH = os.path.join(
    BASE_DIR,
    "ai",
    "models",
    "best.pt",
)

YOLO_CONFIDENCE = float(
    os.environ.get("YOLO_CONFIDENCE", "0.25")
)

YOLO_IMAGE_SIZE = int(
    os.environ.get("YOLO_IMAGE_SIZE", "640")
)

TRAIN_EPOCHS = int(
    os.environ.get("TRAIN_EPOCHS", "20")
)

MAX_JSON_BYTES = 15 * 1024 * 1024
MAX_UPLOAD_BYTES = 15 * 1024 * 1024

COUNT_COOLDOWN_SECONDS = float(
    os.environ.get(
        "COUNT_COOLDOWN_SECONDS",
        "4"
    )
)

# ============================================================
# GLOBAL STATE
# ============================================================

MODEL = None
MODEL_ERROR = ""

MODEL_LOCK = threading.Lock()

TRAINING = False
TRAINING_RUN_ID = None
TRAINING_LOCK_CONN = None

TRAINING_LOCK_KEY = 918273645

LAST_COUNT_TIME = 0.0

SERVER_INSTANCE_ID = uuid.uuid4().hex

CLASS_IDS = {
    0: "BUCKET_LOADED",
    1: "BUCKET_EMPTY",
    2: "PEOPLE",
    3: "EQUIPMENT",
}

DAILY_COUNTS_DATE_TYPE = "date"

# ============================================================
# DATABASE
# ============================================================

def db():
    if not DATABASE_URL:
        raise RuntimeError(
            "DATABASE_URL haijawekwa kwenye Render Environment Variables."
        )

    return psycopg2.connect(
        DATABASE_URL,
        cursor_factory=RealDictCursor,
        connect_timeout=15,
    )

def get_today_string():
    return datetime.now().strftime("%Y-%m-%d")

def get_today_date():
    return datetime.now().date()

def init_db():
    global DAILY_COUNTS_DATE_TYPE

    c = db()
    x = c.cursor()

    try:
        x.execute("""
            CREATE TABLE IF NOT EXISTS buckets (
                id SERIAL PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT DEFAULT '',
                active BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)

        x.execute("""
            CREATE TABLE IF NOT EXISTS dataset_images (
                id SERIAL PRIMARY KEY,
                filename TEXT NOT NULL,
                image_data BYTEA NOT NULL,
                mime_type TEXT DEFAULT 'image/jpeg',
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)

        x.execute("""
            CREATE TABLE IF NOT EXISTS annotations (
                id SERIAL PRIMARY KEY,
                image_id INTEGER
                    REFERENCES dataset_images(id)
                    ON DELETE CASCADE,
                class_name TEXT NOT NULL,
                x_center DOUBLE PRECISION NOT NULL,
                y_center DOUBLE PRECISION NOT NULL,
                box_width DOUBLE PRECISION NOT NULL,
                box_height DOUBLE PRECISION NOT NULL,
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)

        x.execute("""
            CREATE TABLE IF NOT EXISTS training_state (
                id INTEGER PRIMARY KEY,
                status TEXT DEFAULT 'idle',
                message TEXT DEFAULT '',
                progress INTEGER DEFAULT 0,
                updated_at TIMESTAMPTZ DEFAULT NOW(),
                run_id TEXT,
                server_id TEXT,
                started_at TIMESTAMPTZ,
                finished_at TIMESTAMPTZ
            )
        """)

        x.execute("""
            CREATE TABLE IF NOT EXISTS trained_model (
                id INTEGER PRIMARY KEY,
                model_data BYTEA NOT NULL,
                filename TEXT DEFAULT 'best.pt',
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)

        x.execute("""
            CREATE TABLE IF NOT EXISTS daily_counts (
                id SERIAL PRIMARY KEY,
                count_date DATE UNIQUE NOT NULL,
                bucket_count INTEGER DEFAULT 0,
                updated_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)

        x.execute("""
            CREATE TABLE IF NOT EXISTS detection_events (
                id SERIAL PRIMARY KEY,
                class_name TEXT NOT NULL,
                confidence DOUBLE PRECISION DEFAULT 0,
                counted BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)

        x.execute("""
            INSERT INTO training_state
                (id, status, message, progress)
            VALUES
                (
                    1,
                    'idle',
                    'Ready',
                    0
                )
            ON CONFLICT (id) DO NOTHING
        """)

        c.commit()

    except Exception:
        c.rollback()
        raise

    finally:
        x.close()
        c.close()

    try:
        DAILY_COUNTS_DATE_TYPE = get_daily_counts_column_type()
    except Exception:
        DAILY_COUNTS_DATE_TYPE = "date"

def get_daily_counts_column_type():
    c = db()
    x = c.cursor()
    try:
        x.execute("""
            SELECT data_type
            FROM information_schema.columns
            WHERE table_schema = 'public'
              AND table_name = 'daily_counts'
              AND column_name = 'count_date'
            LIMIT 1
        """)
        r = x.fetchone()
        if not r:
            return "date"
        return str(r.get("data_type") or "date").lower()
    finally:
        x.close()
        c.close()

def daily_count_date_expression():
    if DAILY_COUNTS_DATE_TYPE in ("text", "character varying", "character"):
        return "CURRENT_DATE::text"
    return "CURRENT_DATE"

def find_today_daily_count(x):
    expression = daily_count_date_expression()
    x.execute(f"SELECT id, bucket_count, count_date FROM daily_counts WHERE count_date = {expression} LIMIT 1")
    return x.fetchone()

def insert_today_daily_count(x):
    expression = daily_count_date_expression()
    x.execute(f"INSERT INTO daily_counts (count_date, bucket_count, updated_at) VALUES ({expression}, 1, NOW()) RETURNING id")
    return x.fetchone()

# ============================================================
# HTML / JSON HELPERS
# ============================================================

def esc(v):
    return (
        "" if v is None else str(v)
    ).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;").replace("'", "&#039;")

def json_out(h, data, status=200):
    b = json.dumps(data, ensure_ascii=False, default=str).encode("utf-8")
    h.send_response(status)
    h.send_header("Content-Type", "application/json; charset=utf-8")
    h.send_header("Content-Length", str(len(b)))
    h.send_header("Cache-Control", "no-store")
    h.end_headers()
    h.wfile.write(b)

def html_out(h, text, status=200):
    b = text.encode("utf-8")
    h.send_response(status)
    h.send_header("Content-Type", "text/html; charset=utf-8")
    h.send_header("Content-Length", str(len(b)))
    h.send_header("Cache-Control", "no-store")
    h.end_headers()
    h.wfile.write(b)

def error_out(h, msg, status=500):
    return json_out(h, {"error": str(msg)}, status)

# ============================================================
# PAGE LAYOUT
# ============================================================

def layout(title, body, active):
    items = [
        ("Dashboard", "/"),
        ("Camera", "/camera"),
        ("Buckets", "/buckets"),
        ("Training", "/training"),
        ("History", "/history"),
        ("Settings", "/settings"),
    ]

    nav = "".join(
        '<a class="nav-item {}" href="{}">{}</a>'.format(
            "active" if n == active else "",
            p,
            n,
        )
        for n, p in items
    )

    page = r"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__ - NEERIKA BUCKET AI</title>
<style>
*{box-sizing:border-box}
body{margin:0;font-family:Arial,sans-serif;background:#f4f6f8;color:#17202a}
header{background:#111827;color:#fff;padding:16px}
.brand{font-size:22px;font-weight:800}
.sub{font-size:13px;opacity:.75;margin-top:4px}
.nav{display:flex;gap:7px;overflow:auto;background:#1f2937;padding:8px}
.nav-item{color:#fff;text-decoration:none;padding:10px 13px;border-radius:8px;white-space:nowrap}
.nav-item.active,.nav-item:hover{background:#374151}
main{max-width:1200px;margin:auto;padding:16px}
.card,.stat{background:#fff;border-radius:14px;padding:18px;margin-bottom:16px;box-shadow:0 2px 10px rgba(0,0,0,.07)}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:14px}
.num{font-size:38px;font-weight:800;margin-top:7px}
button{border:0;border-radius:9px;padding:10px 15px;background:#111827;color:#fff;cursor:pointer;margin:3px}
button.secondary{background:#6b7280}
button.success{background:#166534}
button.danger{background:#991b1b}
button:disabled{opacity:.55;cursor:not-allowed}
input,select{width:100%;padding:10px;border:1px solid #d1d5db;border-radius:8px}
label{font-weight:700;display:block;margin-bottom:5px}
.row{margin-bottom:13px}
.status{padding:10px;background:#f3f4f6;border-radius:8px;margin-top:10px}
table{width:100%;border-collapse:collapse}
th,td{text-align:left;padding:9px;border-bottom:1px solid #e5e7eb}
.footer{text-align:center;color:#6b7280;font-size:12px;padding:25px}
.progress{height:24px;background:#e5e7eb;border-radius:12px;overflow:hidden}
.bar{height:100%;background:#111827;width:0%;transition:width .3s;text-align:center;color:#fff;font-size:12px;line-height:24px}
.epoch{font-size:28px;font-weight:800;margin:8px 0}
@media(max-width:600px){main{padding:10px}.num{font-size:30px}table{font-size:13px}}
</style>
</head>
<body>
<header>
<div class="brand">NEERIKA BUCKET AI</div>
<div class="sub">Mining Production Bucket Counter</div>
</header>
<nav class="nav">__NAV__</nav>
<main>__BODY__</main>
<div class="footer">Geology &amp; Mining Services</div>
</body>
</html>
"""
    return page.replace("__TITLE__", esc(title)).replace("__NAV__", nav).replace("__BODY__", body)

# ============================================================
# IMAGE HELPERS
# ============================================================

def _image_bytes(value):
    if value is None:
        return b""
    if isinstance(value, bytes):
        return value
    if isinstance(value, memoryview):
        return value.tobytes()
    if isinstance(value, bytearray):
        return bytes(value)
    if isinstance(value, str):
        q = value.strip()
        if q.startswith("\\x"):
            try:
                return bytes.fromhex(q[2:])
            except Exception:
                pass
        try:
            return base64.b64decode(q, validate=True)
        except Exception:
            return q.encode("utf-8")
    try:
        return bytes(value)
    except Exception:
        return b""

# ============================================================
# MODEL
# ============================================================

def restore_model():
    if os.path.exists(MODEL_PATH):
        return True
    try:
        c = db()
        x = c.cursor()
        try:
            x.execute("SELECT model_data FROM trained_model WHERE id=1")
            r = x.fetchone()
        finally:
            x.close()
            c.close()

        if r and r["model_data"]:
            os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
            with open(MODEL_PATH, "wb") as f:
                f.write(_image_bytes(r["model_data"]))
            return True
    except Exception:
        traceback.print_exc()
    return False

def load_model():
    global MODEL, MODEL_ERROR
    with MODEL_LOCK:
        if MODEL is not None:
            return MODEL
        try:
            restore_model()
            if os.path.exists(MODEL_PATH):
                p = MODEL_PATH
            elif os.path.exists(MODEL_BACKUP_PATH):
                p = MODEL_BACKUP_PATH
            else:
                p = None

            if p:
                MODEL = YOLO(p)
                MODEL_ERROR = ""
                return MODEL

            MODEL = YOLO("yolo11n.pt")
            MODEL_ERROR = "Fallback yolo11n.pt loaded."
            return MODEL
        except Exception as e:
            MODEL_ERROR = str(e)
            return None

def save_model_db(path):
    with open(path, "rb") as f:
        b = f.read()
    c = db()
    x = c.cursor()
    try:
        x.execute(
            "INSERT INTO trained_model (id, model_data, filename) VALUES (1, %s, 'best.pt') ON CONFLICT(id) DO UPDATE SET model_data=EXCLUDED.model_data, created_at=NOW()",
            (psycopg2.Binary(b),)
        )
        c.commit()
    finally:
        x.close()
        c.close()

def normalize(cid, name=""):
    if cid in CLASS_IDS:
        return CLASS_IDS[cid]
    s = str(name).upper()
    if "EMPTY" in s:
        return "BUCKET_EMPTY"
    if "PEOPLE" in s or "PERSON" in s:
        return "PEOPLE"
    if "EQUIPMENT" in s:
        return "EQUIPMENT"
    return "BUCKET_LOADED"

def detect(data):
    model = load_model()
    if model is None:
        raise RuntimeError("YOLO model haijapatikana: " + MODEL_ERROR)
    im = Image.open(io.BytesIO(data)).convert("RGB")
    arr = np.array(im)
    results = model.predict(source=arr, conf=YOLO_CONFIDENCE, imgsz=YOLO_IMAGE_SIZE, verbose=False)
    out = []
    if not results or results[0].boxes is None:
        return out
    names = getattr(results[0], "names", {})
    for box in results[0].boxes:
        try:
            xy = box.xyxy[0].cpu().numpy().tolist()
            conf = float(box.conf[0].cpu().item())
            cid = int(box.cls[0].cpu().item())
            raw = names.get(cid, "")
            out.append({
                "class_id": cid,
                "class_name": normalize(cid, raw),
                "confidence": round(conf, 4),
                "x1": round(float(xy[0]), 2),
                "y1": round(float(xy[1]), 2),
                "x2": round(float(xy[2]), 2),
                "y2": round(float(xy[3]), 2),
            })
        except Exception:
            pass
    return out

def count_loaded(dets):
    global LAST_COUNT_TIME
    loaded = [d for d in dets if d["class_name"] == "BUCKET_LOADED"]
    if not loaded:
        return {"counted": False, "reason": "No loaded bucket detected.", "count": 0}
    if time.time() - LAST_COUNT_TIME < COUNT_COOLDOWN_SECONDS:
        return {"counted": False, "reason": "Cooldown active.", "count": 0}

    best = max(loaded, key=lambda d: d["confidence"])
    conf = float(best["confidence"])

    c = db()
    x = c.cursor()
    try:
        existing = find_today_daily_count(x)
        if existing:
            x.execute("UPDATE daily_counts SET bucket_count = COALESCE(bucket_count, 0) + 1, updated_at = NOW() WHERE id=%s", (existing["id"],))
        else:
            insert_today_daily_count(x)
        x.execute("INSERT INTO detection_events (class_name, confidence, counted) VALUES (%s, %s, TRUE)", ("BUCKET_LOADED", conf))
        c.commit()
    finally:
        x.close()
        c.close()

    LAST_COUNT_TIME = time.time()
    return {"counted": True, "reason": "Loaded bucket counted.", "count": 1, "confidence": conf}

def today():
    c = db()
    x = c.cursor()
    try:
        r = find_today_daily_count(x)
        return int(r.get("bucket_count", 0) or 0) if r else 0
    finally:
        x.close()
        c.close()

def dataset_rows():
    c = db()
    x = c.cursor()
    try:
        x.execute("""
            SELECT di.id, di.filename, di.mime_type, di.created_at, COUNT(a.id) AS annotation_count
            FROM dataset_images di
            LEFT JOIN annotations a ON a.image_id = di.id
            GROUP BY di.id, di.filename, di.mime_type, di.created_at
            ORDER BY di.id DESC
        """)
        return x.fetchall()
    finally:
        x.close()
        c.close()

def save_image(filename, mime, data):
    c = db()
    x = c.cursor()
    try:
        x.execute("INSERT INTO dataset_images (filename, image_data, mime_type) VALUES (%s, %s, %s) RETURNING id", (filename, psycopg2.Binary(data), mime))
        i = x.fetchone()["id"]
        c.commit()
        return i
    finally:
        x.close()
        c.close()

def save_annotations(image_id, anns):
    c = db()
    x = c.cursor()
    try:
        x.execute("DELETE FROM annotations WHERE image_id=%s", (image_id,))
        for a in anns:
            x.execute(
                "INSERT INTO annotations (image_id, class_name, x_center, y_center, box_width, box_height) VALUES (%s, %s, %s, %s, %s, %s)",
                (image_id, str(a.get("class_name", "BUCKET_LOADED")), float(a["x_center"]), float(a["y_center"]), float(a["box_width"]), float(a["box_height"]))
            )
        c.commit()
    finally:
        x.close()
        c.close()

def delete_image(i):
    c = db()
    x = c.cursor()
    try:
        x.execute("DELETE FROM dataset_images WHERE id=%s", (i,))
        ok = x.rowcount > 0
        c.commit()
        return ok
    finally:
        x.close()
        c.close()

def set_training(status, msg, progress, run_id=None, finished=False):
    progress = max(0, min(100, int(progress)))
    c = db()
    x = c.cursor()
    try:
        if run_id:
            if finished:
                x.execute("UPDATE training_state SET status=%s, message=%s, progress=%s, updated_at=NOW(), finished_at=NOW() WHERE id=1 AND run_id=%s", (status, msg, progress, str(run_id)))
            else:
                x.execute("UPDATE training_state SET status=%s, message=%s, progress=%s, updated_at=NOW() WHERE id=1 AND run_id=%s", (status, msg, progress, str(run_id)))
        else:
            x.execute("UPDATE training_state SET status=%s, message=%s, progress=%s, updated_at=NOW() WHERE id=1", (status, msg, progress))
        c.commit()
    finally:
        x.close()
        c.close()

def acquire_training_lock():
    global TRAINING_LOCK_CONN
    conn = psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor, connect_timeout=15)
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute("SELECT pg_try_advisory_lock(%s) AS locked", (TRAINING_LOCK_KEY,))
    row = cur.fetchone()
    cur.close()
    if not row or not row["locked"]:
        conn.close()
        return None
    TRAINING_LOCK_CONN = conn
    return conn

def release_training_lock(conn=None):
    global TRAINING_LOCK_CONN
    conn = conn or TRAINING_LOCK_CONN
    if not conn:
        return
    try:
        cur = conn.cursor()
        cur.execute("SELECT pg_advisory_unlock(%s)", (TRAINING_LOCK_KEY,))
        cur.close()
    except Exception:
        pass
    try:
        conn.close()
    except Exception:
        pass
    if conn is TRAINING_LOCK_CONN:
        TRAINING_LOCK_CONN = None

def claim_training_run():
    global TRAINING, TRAINING_RUN_ID
    lock_conn = acquire_training_lock()
    if not lock_conn:
        return None
    run_id = uuid.uuid4().hex
    c = db()
    x = c.cursor()
    try:
        x.execute(
            "UPDATE training_state SET status='preparing', message='Preparing dataset...', progress=1, updated_at=NOW(), run_id=%s, server_id=%s, started_at=NOW(), finished_at=NULL WHERE id=1 AND status NOT IN ('preparing', 'training', 'saving') RETURNING id",
            (run_id, SERVER_INSTANCE_ID)
        )
        if not x.fetchone():
            c.rollback()
            release_training_lock(lock_conn)
            return None
        c.commit()
        TRAINING = True
        TRAINING_RUN_ID = run_id
        return run_id
    finally:
        x.close()
        c.close()

def build_dataset():
    rows = dataset_rows()
    if len(rows) < 1:
        raise RuntimeError("Training inahitaji angalau picha 1 yenye annotations.")

    root = tempfile.mkdtemp(prefix="neerika_train_")
    imgs = os.path.join(root, "images")
    labs = os.path.join(root, "labels")
    os.makedirs(imgs, exist_ok=True)
    os.makedirs(labs, exist_ok=True)

    usable = 0
    c = db()
    x = c.cursor()
    try:
        for r in rows:
            x.execute("SELECT image_data, mime_type, filename FROM dataset_images WHERE id=%s", (r["id"],))
            ir = x.fetchone()
            if not ir:
                continue

            x.execute("SELECT class_name, x_center, y_center, box_width, box_height FROM annotations WHERE image_id=%s", (r["id"],))
            anns = x.fetchall()
            if not anns:
                continue

            raw = _image_bytes(ir.get("image_data"))
            if not raw:
                continue

            ext = ".jpg"
            filename = str(ir.get("filename") or "").lower()
            if "png" in filename or "png" in str(ir.get("mime_type")):
                ext = ".png"

            stem = "image_" + str(r["id"])
            with open(os.path.join(imgs, stem + ext), "wb") as f:
                f.write(raw)

            with open(os.path.join(labs, stem + ".txt"), "w", encoding="utf-8") as f:
                for a in anns:
                    f.write(f"0 {a['x_center']} {a['y_center']} {a['box_width']} {a['box_height']}\n")
            usable += 1
    finally:
        x.close()
        c.close()

    if usable < 1:
        shutil.rmtree(root, ignore_errors=True)
        raise RuntimeError("Hakuna picha zenye annotations zilizopatikana.")

    yaml_path = os.path.join(root, "data.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(f"path: {root.replace(os.sep, '/')}\ntrain: images\nval: images\nnames:\n  0: BUCKET_LOADED\n")

    return root, yaml_path, usable

def training_worker(run_id, lock_conn):
    global TRAINING, TRAINING_RUN_ID, MODEL, MODEL_ERROR
    root = None
    try:
        set_training("preparing", "Preparing dataset...", 5, run_id)
        root, yaml_path, usable = build_dataset()

        set_training("training", f"Starting YOLO training ({TRAIN_EPOCHS} Epochs)...", 20, run_id)
        model = YOLO("yolo11n.pt")
        out_dir = os.path.join(root, "runs")

        def on_epoch_end(trainer):
            try:
                epoch = int(getattr(trainer, "epoch", 0)) + 1
                progress = min(90, 20 + int((epoch / TRAIN_EPOCHS) * 70))
                set_training("training", f"Training YOLO... Epoch {epoch}/{TRAIN_EPOCHS}", progress, run_id)
            except Exception:
                pass

        try:
            model.add_callback("on_fit_epoch_end", on_epoch_end)
        except Exception:
            pass

        model.train(data=yaml_path, epochs=TRAIN_EPOCHS, imgsz=YOLO_IMAGE_SIZE, project=out_dir, name="neerika", exist_ok=True, verbose=False)

        set_training("saving", "Training complete. Saving model...", 95, run_id)
        best = os.path.join(out_dir, "neerika", "weights", "best.pt")
        if not os.path.exists(best):
            raise RuntimeError("best.pt haikupatikana baada ya training.")

        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        shutil.copy2(best, MODEL_PATH)
        save_model_db(MODEL_PATH)

        with MODEL_LOCK:
            MODEL = YOLO(MODEL_PATH)
            MODEL_ERROR = ""

        set_training("completed", f"Training completed successfully! {usable} images trained.", 100, run_id, finished=True)
    except Exception as e:
        traceback.print_exc()
        set_training("error", f"Error: {str(e)}", 0, run_id, finished=True)
    finally:
        if root:
            shutil.rmtree(root, ignore_errors=True)
        TRAINING = False
        TRAINING_RUN_ID = None
        release_training_lock(lock_conn)

def start_training_request():
    run_id = claim_training_run()
    if not run_id:
        return None
    threading.Thread(target=training_worker, args=(run_id, TRAINING_LOCK_CONN), daemon=True).start()
    return run_id

# ============================================================
# PAGES & UI
# ============================================================

def dashboard():
    body = f"""
<div class="card">
    <h2>Mining Production Dashboard</h2>
    <p>NEERIKA BUCKET AI counts loaded ore/material buckets coming from the mine shaft.</p>
</div>
<div class="grid">
    <div class="stat">
        <div class="small">Today's Loaded Buckets</div>
        <div class="num">{today()}</div>
    </div>
    <div class="stat">
        <div class="small">AI System</div>
        <div class="num">READY</div>
    </div>
</div>
<div class="card">
    <a href="/camera"><button>Open Camera</button></a>
    <a href="/training"><button class="secondary">Dataset & Training</button></a>
</div>
"""
    return layout("Dashboard", body, "Dashboard")

def camera():
    body = r"""
<div class="card">
    <h2>Bucket Camera</h2>
    <p class="small">Take a camera frame and send it to YOLO to count loaded buckets.</p>
    <video id="v" autoplay playsinline style="width:100%;border-radius:12px;background:#000"></video>
    <canvas id="c" style="display:none"></canvas>
    <p>
        <button onclick="start()">Start Camera</button>
        <button class="secondary" onclick="stop()">Stop</button>
        <button class="success" onclick="detectNow()">Detect & Count</button>
    </p>
    <div id="r" class="status">Camera not started.</div>
    <div class="num" id="n">0</div>
</div>
<script>
let stream = null;
const video = document.getElementById('v');
const canvas = document.getElementById('c');
const resultBox = document.getElementById('r');
const numberBox = document.getElementById('n');

async function start(){
    try{
        stream = await navigator.mediaDevices.getUserMedia({video:{facingMode:'environment'},audio:false});
        video.srcObject = stream;
        resultBox.innerText = 'Camera started.';
    }catch(e){
        resultBox.innerText = 'Camera error: ' + e.message;
    }
}
function stop(){
    if(stream) stream.getTracks().forEach(t => t.stop());
    stream = null;
    resultBox.innerText = 'Camera stopped.';
}
async function detectNow(){
    if(!video.videoWidth) { alert('Start camera first.'); return; }
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext('2d').drawImage(video, 0, 0);
    resultBox.innerText = 'Detecting...';
    try{
        const q = await fetch('/api/detect', {
            method:'POST',
            headers:{'Content-Type':'application/json'},
            body:JSON.stringify({image:canvas.toDataURL('image/jpeg', .85), count:true})
        });
        const d = await q.json();
        const loaded = d.detections.filter(x => x.class_name === 'BUCKET_LOADED').length;
        resultBox.innerText = 'Detections: ' + d.detections.length + ' | Loaded: ' + loaded + ' | ' + (d.count_result ? d.count_result.reason : '');
        refresh();
    }catch(e){
        resultBox.innerText = 'Error: ' + e.message;
    }
}
async function refresh(){
    try{
        const q = await fetch('/api/history');
        const d = await q.json();
        numberBox.innerText = d.today_count || 0;
    }catch(e){}
}
refresh();
</script>
"""
    return layout("Camera", body, "Camera")

def buckets():
    body = r"""
<div class="card">
    <h2>Bucket Registration</h2>
    <div class="row"><label>Bucket Name</label><input id="name" placeholder="NEERIKA Loaded Bucket"></div>
    <div class="row"><label>Description</label><input id="desc" placeholder="Description"></div>
    <button onclick="add()">Register Bucket</button>
    <div id="msg" class="status"></div>
</div>
<div class="card"><h3>Registered Buckets</h3><div id="list">Loading...</div></div>
<script>
async function load(){
    const q = await fetch('/api/buckets');
    const d = await q.json();
    if(!d.buckets.length){ list.innerHTML='<p>No buckets.</p>'; return; }
    let h='<table><tr><th>Name</th><th>Status</th></tr>';
    d.buckets.forEach(b=>{ h+='<tr><td>'+b.name+'</td><td>'+(b.active?'ACTIVE':'')+'</td></tr>'; });
    list.innerHTML=h+'</table>';
}
async function add(){
    await fetch('/api/buckets', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({name:document.getElementById('name').value, description:document.getElementById('desc').value})});
    load();
}
load();
</script>
"""
    return layout("Buckets", body, "Buckets")

def training_page():
    rows = dataset_rows()
    tr = "".join([f"<tr><td>{r['id']}</td><td>{esc(r['filename'])}</td><td>{r['annotation_count']}</td><td><a href='/annotate?id={r['id']}'><button class='secondary'>Annotate</button></a> <button class='danger' onclick='delimg({r['id']})'>Delete</button></td></tr>" for r in rows]) or "<tr><td colspan='4'>No images.</td></tr>"

    body = r"""
<div class="card">
    <h2>YOLO Training Dataset</h2>
    <div class="row"><label>Upload Image</label><input id="file" type="file" accept="image/*"></div>
    <button onclick="upload()">Upload & Annotate</button>
    <div id="up" class="status"></div>
</div>
<div class="card">
    <h3>Training Status</h3>
    <p>Status: <b id="st">idle</b></p>
    <p id="ms">Ready</p>
    <div class="epoch" id="epoch">Epoch 0/__EPOCHS__</div>
    <div class="progress"><div id="bar" class="bar" style="width:0%">0%</div></div>
    <br>
    <button id="startBtn" class="success" onclick="startTrain()">Start Training</button>
    <button class="secondary" onclick="poll()">Refresh Status</button>
</div>
<div class="card">
    <h3>Dataset Images</h3>
    <table><tr><th>ID</th><th>Filename</th><th>Annotations</th><th>Action</th></tr>__ROWS__</table>
</div>
<script>
async function upload(){
    const f = document.getElementById('file').files[0];
    if(!f) return alert('Choose image');
    const rd = new FileReader();
    rd.onload = async () => {
        const q = await fetch('/api/dataset/upload', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({filename:f.name, mime_type:f.type, data:rd.result})});
        const d = await q.json();
        if(d.id) window.location.href = '/annotate?id=' + d.id;
    };
    rd.readAsDataURL(f);
}
async function delimg(id){
    if(confirm('Delete?')) { await fetch('/api/dataset/delete', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({id:id})}); location.reload(); }
}
async function startTrain(){
    await fetch('/api/training/start', {method:'POST'});
    poll();
}
async function poll(){
    const q = await fetch('/api/training/status');
    const d = await q.json();
    document.getElementById('st').innerText = d.status;
    document.getElementById('ms').innerText = d.message;
    const p = d.progress || 0;
    document.getElementById('bar').style.width = p + '%';
    document.getElementById('bar').innerText = p + '%';
    if(['preparing','training','saving'].includes(d.status)) setTimeout(poll, 2000);
}
poll();
</script>
""".replace("__ROWS__", tr).replace("__EPOCHS__", str(TRAIN_EPOCHS))
    return layout("Training", body, "Training")

def annotate_page(image_id):
    body = r"""
<div class="card">
    <h2>Image Annotation (Draw Bounding Box)</h2>
    <p class="small">Click na buruta picha kuweka box la BUCKET_LOADED.</p>
    <div style="position:relative;display:inline-block;max-width:100%">
        <canvas id="canvas" style="cursor:crosshair;max-width:100%;border:1px solid #ccc"></canvas>
    </div>
    <p>
        <button class="success" onclick="saveAnns()">Save Annotations</button>
        <a href="/training"><button class="secondary">Back to Training</button></a>
    </p>
    <div id="status" class="status"></div>
</div>
<script>
const imageId = __IMAGE_ID__;
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
let img = new Image();
let boxes = [];
let drawing = false;
let startX, startY;

img.onload = () => {
    canvas.width = img.naturalWidth;
    canvas.height = img.naturalHeight;
    draw();
    loadExisting();
};
img.src = '/api/dataset/image/' + imageId;

async function loadExisting(){
    const q = await fetch('/api/dataset/annotations/' + imageId);
    const d = await q.json();
    boxes = (d.annotations || []).map(a => {
        return {
            x: (a.x_center - a.box_width / 2) * canvas.width,
            y: (a.y_center - a.box_height / 2) * canvas.height,
            w: a.box_width * canvas.width,
            h: a.box_height * canvas.height
        };
    });
    draw();
}

canvas.onmousedown = (e) => {
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    startX = (e.clientX - rect.left) * scaleX;
    startY = (e.clientY - rect.top) * scaleY;
    drawing = true;
};

canvas.onmousemove = (e) => {
    if(!drawing) return;
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    const curX = (e.clientX - rect.left) * scaleX;
    const curY = (e.clientY - rect.top) * scaleY;
    draw();
    ctx.strokeStyle = '#ef4444';
    ctx.lineWidth = 3;
    ctx.strokeRect(startX, startY, curX - startX, curY - startY);
};

canvas.onmouseup = (e) => {
    if(!drawing) return;
    drawing = false;
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    const endX = (e.clientX - rect.left) * scaleX;
    const endY = (e.clientY - rect.top) * scaleY;
    const w = endX - startX;
    const h = endY - startY;
    if(Math.abs(w) > 5 && Math.abs(h) > 5){
        boxes.push({x: w < 0 ? endX : startX, y: h < 0 ? endY : startY, w: Math.abs(w), h: Math.abs(h)});
    }
    draw();
};

function draw(){
    ctx.clearRect(0,0,canvas.width,canvas.height);
    ctx.drawImage(img,0,0);
    ctx.strokeStyle = '#22c55e';
    ctx.lineWidth = 3;
    boxes.forEach(b => {
        ctx.strokeRect(b.x, b.y, b.w, b.h);
        ctx.fillStyle = 'rgba(34, 197, 94, 0.2)';
        ctx.fillRect(b.x, b.y, b.w, b.h);
    });
}

async function saveAnns(){
    const formatted = boxes.map(b => {
        return {
            class_name: 'BUCKET_LOADED',
            x_center: (b.x + b.w / 2) / canvas.width,
            y_center: (b.y + b.h / 2) / canvas.height,
            box_width: b.w / canvas.width,
            box_height: b.h / canvas.height
        };
    });
    const q = await fetch('/api/dataset/annotations', {
        method:'POST',
        headers:{'Content-Type':'application/json'},
        body:JSON.stringify({image_id: imageId, annotations: formatted})
    });
    const d = await q.json();
    document.getElementById('status').innerText = d.message || 'Saved!';
}
</script>
""".replace("__IMAGE_ID__", str(image_id))
    return layout("Annotate", body, "Training")

def history():
    body = r"""
<div class="card">
    <h2>Production History</h2>
    <a href="/api/history.csv"><button class="secondary">Export CSV</button></a>
</div>
<div class="card" id="table">Loading...</div>
<script>
async function load(){
    const q = await fetch('/api/history');
    const d = await q.json();
    let h = '<table><tr><th>Date</th><th>Loaded Buckets</th></tr>';
    (d.history || []).forEach(r => { h += '<tr><td>' + r.count_date + '</td><td>' + r.bucket_count + '</td></tr>'; });
    document.getElementById('table').innerHTML = h + '</table>';
}
load();
</script>
"""
    return layout("History", body, "History")

def settings():
    body = f"""
<div class="card">
    <h2>Settings</h2>
    <p>Training Epochs: <b>{TRAIN_EPOCHS}</b></p>
    <p>Model Status: <b>{'Loaded' if MODEL is not None else 'Not Loaded'}</b></p>
</div>
"""
    return layout("Settings", body, "Settings")

# ============================================================
# HTTP HANDLER
# ============================================================

class Handler(BaseHTTPRequestHandler):
    def body_json(self):
        n = int(self.headers.get("Content-Length", "0"))
        return json.loads(self.rfile.read(n).decode("utf-8")) if n else {}

    def do_GET(self):
        p = urlparse(self.path)
        path = p.path
        query = p.query

        if path == "/": return html_out(self, dashboard())
        if path == "/camera": return html_out(self, camera())
        if path == "/buckets": return html_out(self, buckets())
        if path == "/training": return html_out(self, training_page())
        if path == "/history": return html_out(self, history())
        if path == "/settings": return html_out(self, settings())

        if path == "/annotate":
            q_id = 0
            for part in query.split("&"):
                if part.startswith("id="):
                    try: q_id = int(part.split("=")[1])
                    except: pass
            return html_out(self, annotate_page(q_id))

        if path == "/api/buckets":
            c = db()
            x = c.cursor()
            try: x.execute("SELECT * FROM buckets ORDER BY id DESC"); r = x.fetchall()
            finally: x.close(); c.close()
            return json_out(self, {"buckets": r})

        if path == "/api/training/status":
            c = db()
            x = c.cursor()
            try: x.execute("SELECT * FROM training_state WHERE id=1"); r = x.fetchone()
            finally: x.close(); c.close()
            return json_out(self, r or {})

        if path == "/api/history":
            c = db()
            x = c.cursor()
            try: x.execute("SELECT count_date::text AS count_date, bucket_count FROM daily_counts ORDER BY count_date DESC LIMIT 50"); r = x.fetchall()
            finally: x.close(); c.close()
            return json_out(self, {"history": r, "today_count": today()})

        if path.startswith("/api/dataset/image/"):
            i = int(path.rsplit("/", 1)[1])
            c = db()
            x = c.cursor()
            try: x.execute("SELECT image_data, mime_type FROM dataset_images WHERE id=%s", (i,)); r = x.fetchone()
            finally: x.close(); c.close()
            if not r: return error_out(self, "Not found", 404)
            b = _image_bytes(r["image_data"])
            self.send_response(200)
            self.send_header("Content-Type", r["mime_type"] or "image/jpeg")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)
            return

        if path.startswith("/api/dataset/annotations/"):
            i = int(path.rsplit("/", 1)[1])
            c = db()
            x = c.cursor()
            try: x.execute("SELECT * FROM annotations WHERE image_id=%s", (i,)); r = x.fetchall()
            finally: x.close(); c.close()
            return json_out(self, {"annotations": r})

        return error_out(self, "Not found", 404)

    def do_POST(self):
        p = urlparse(self.path).path
        d = self.body_json()

        if p == "/api/detect":
            s = d.get("image", "")
            if "," in s: s = s.split(",", 1)[1]
            b = base64.b64decode(s)
            det = detect(b)
            cr = count_loaded(det) if d.get("count", True) else None
            return json_out(self, {"detections": det, "count_result": cr})

        if p == "/api/buckets":
            c = db()
            x = c.cursor()
            try:
                x.execute("INSERT INTO buckets (name, description) VALUES (%s, %s)", (d.get("name"), d.get("description")))
                c.commit()
            finally: x.close(); c.close()
            return json_out(self, {"message": "Success"})

        if p == "/api/dataset/upload":
            s = d.get("data", "")
            if "," in s: s = s.split(",", 1)[1]
            b = base64.b64decode(s)
            i = save_image(d.get("filename", "img.jpg"), d.get("mime_type", "image/jpeg"), b)
            return json_out(self, {"id": i})

        if p == "/api/dataset/delete":
            delete_image(int(d["id"]))
            return json_out(self, {"message": "Deleted"})

        if p == "/api/dataset/annotations":
            save_annotations(int(d["image_id"]), d.get("annotations", []))
            return json_out(self, {"message": "Annotations saved successfully."})

        if p == "/api/training/start":
            run_id = start_training_request()
            if not run_id: return error_out(self, "Training tayari inaendelea.", 409)
            return json_out(self, {"run_id": run_id})

        return error_out(self, "Not found", 404)

# ============================================================
# MAIN
# ============================================================

def main():
    if DATABASE_URL:
        try: init_db()
        except: traceback.print_exc()
    load_model()
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print("Server running on port", PORT)
    server.serve_forever()

if __name__ == "__main__":
    main()
