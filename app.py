import os
import io
import csv
import json
import time
import base64
import shutil
import threading
import traceback
from datetime import datetime, date
from urllib.parse import urlparse

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# =========================================================
# SERVER
# =========================================================

HOST = "0.0.0.0"
PORT = int(os.environ.get("PORT", "8080"))

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

AI_DIR = os.path.join(BASE_DIR, "ai")
DATASET_DIR = os.path.join(AI_DIR, "dataset")
TRAIN_DIR = os.path.join(AI_DIR, "training")
MODEL_DIR = os.path.join(AI_DIR, "models")

MODEL_FILE = os.path.join(MODEL_DIR, "best.pt")
DATA_YAML = os.path.join(DATASET_DIR, "data.yaml")

os.makedirs(AI_DIR, exist_ok=True)
os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# =========================================================
# DATABASE
# =========================================================

DATABASE_URL = (
    os.environ.get("DATABASE_URL")
    or os.environ.get("SUPABASE_DB_URL")
    or os.environ.get("POSTGRES_URL")
)

if not DATABASE_URL:
    raise RuntimeError(
        "DATABASE_URL is not configured. "
        "Add your Supabase PostgreSQL connection string "
        "to Render Environment Variables."
    )

if "sslmode=" not in DATABASE_URL:
    separator = "&" if "?" in DATABASE_URL else "?"
    DATABASE_URL += separator + "sslmode=require"

import psycopg2
from psycopg2.extras import RealDictCursor

# =========================================================
# OPTIONAL AI IMPORTS
# =========================================================

YOLO = None

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except Exception:
    YOLO_AVAILABLE = False

try:
    import cv2
    import numpy as np
    CV_AVAILABLE = True
except Exception:
    CV_AVAILABLE = False

# =========================================================
# CLASSES
# =========================================================

CLASSES = [
    "BUCKET_LOADED",
    "BUCKET_EMPTY",
    "PEOPLE",
    "EQUIPMENT",
]

BUCKET_LOADED = 0
BUCKET_EMPTY = 1
PEOPLE = 2
EQUIPMENT = 3

# =========================================================
# TRAINING STATE
# =========================================================

training_lock = threading.Lock()

DEFAULT_TRAINING = {
    "status": "Not started",
    "progress": 0,
    "message": "",
    "error": "",
}

# =========================================================
# TRACKING
# =========================================================

COUNTED_TRACKS = set()
LAST_TRACK_Y = {}
LAST_TRACK_TIME = {}

# =========================================================
# DATABASE HELPERS
# =========================================================

def db():
    conn = psycopg2.connect(
        DATABASE_URL,
        connect_timeout=15,
        cursor_factory=RealDictCursor,
    )
    conn.autocommit = True
    return conn


def execute(sql, params=None, fetchone=False, fetchall=False):
    conn = db()

    try:
        with conn.cursor() as cur:
            cur.execute(sql, params or ())

            if fetchone:
                return cur.fetchone()

            if fetchall:
                return cur.fetchall()

            return None

    finally:
        conn.close()


def init_db():
    statements = [

        """
        CREATE TABLE IF NOT EXISTS buckets (
            id BIGSERIAL PRIMARY KEY,
            name TEXT NOT NULL,
            capacity DOUBLE PRECISION DEFAULT 0,
            active INTEGER DEFAULT 1,
            created_at TIMESTAMPTZ DEFAULT NOW()
        )
        """,

        """
        CREATE TABLE IF NOT EXISTS bucket_images (
            id BIGSERIAL PRIMARY KEY,
            bucket_id BIGINT,
            filename TEXT NOT NULL,
            image_data TEXT NOT NULL,
            created_at TIMESTAMPTZ DEFAULT NOW()
        )
        """,

        """
        CREATE TABLE IF NOT EXISTS dataset_images (
            id BIGSERIAL PRIMARY KEY,
            filename TEXT NOT NULL,
            image_data TEXT NOT NULL,
            width INTEGER DEFAULT 0,
            height INTEGER DEFAULT 0,
            created_at TIMESTAMPTZ DEFAULT NOW()
        )
        """,

        """
        CREATE TABLE IF NOT EXISTS annotations (
            id BIGSERIAL PRIMARY KEY,
            image_id BIGINT NOT NULL,
            class_id INTEGER NOT NULL,
            class_name TEXT NOT NULL,
            x_center DOUBLE PRECISION NOT NULL,
            y_center DOUBLE PRECISION NOT NULL,
            box_width DOUBLE PRECISION NOT NULL,
            box_height DOUBLE PRECISION NOT NULL,
            created_at TIMESTAMPTZ DEFAULT NOW()
        )
        """,

        """
        CREATE TABLE IF NOT EXISTS detections (
            id BIGSERIAL PRIMARY KEY,
            detection_time TIMESTAMPTZ DEFAULT NOW(),
            bucket_id BIGINT,
            bucket_name TEXT,
            status TEXT,
            counted INTEGER DEFAULT 0,
            confidence DOUBLE PRECISION DEFAULT 0,
            track_id INTEGER,
            note TEXT
        )
        """,

        """
        CREATE TABLE IF NOT EXISTS daily_counts (
            id BIGSERIAL PRIMARY KEY,
            count_date DATE UNIQUE NOT NULL,
            loaded INTEGER DEFAULT 0,
            empty INTEGER DEFAULT 0,
            people INTEGER DEFAULT 0,
            equipment INTEGER DEFAULT 0,
            unknown INTEGER DEFAULT 0
        )
        """,

        """
        CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY,
            value TEXT
        )
        """,

        """
        CREATE TABLE IF NOT EXISTS training_state (
            id INTEGER PRIMARY KEY,
            status TEXT NOT NULL DEFAULT 'Not started',
            progress INTEGER DEFAULT 0,
            message TEXT DEFAULT '',
            error TEXT DEFAULT '',
            updated_at TIMESTAMPTZ DEFAULT NOW()
        )
        """,

        """
        CREATE TABLE IF NOT EXISTS trained_model (
            id INTEGER PRIMARY KEY,
            model_data BYTEA,
            filename TEXT,
            created_at TIMESTAMPTZ DEFAULT NOW()
        )
        """,

        """
        CREATE TABLE IF NOT EXISTS bucket_counts (
            id BIGSERIAL PRIMARY KEY,
            count_date DATE NOT NULL DEFAULT CURRENT_DATE,
            count_time TIME NOT NULL DEFAULT CURRENT_TIME,
            bucket_count INTEGER NOT NULL DEFAULT 0,
            loaded_count INTEGER NOT NULL DEFAULT 0,
            empty_count INTEGER NOT NULL DEFAULT 0,
            notes TEXT,
            created_at TIMESTAMPTZ DEFAULT NOW()
        )
        """,

        """
        INSERT INTO settings(key, value)
        VALUES ('line_position', '55')
        ON CONFLICT(key) DO NOTHING
        """,

        """
        INSERT INTO training_state(id, status, progress, message, error)
        VALUES (1, 'Not started', 0, '', '')
        ON CONFLICT(id) DO NOTHING
        """
    ]

    conn = db()

    try:
        with conn.cursor() as cur:
            for statement in statements:
                cur.execute(statement)

    finally:
        conn.close()


# =========================================================
# SETTINGS
# =========================================================

def setting(key, default=None):
    row = execute(
        "SELECT value FROM settings WHERE key=%s",
        (key,),
        fetchone=True
    )

    if row:
        return row["value"]

    return default


def set_setting(key, value):
    execute(
        """
        INSERT INTO settings(key, value)
        VALUES(%s, %s)
        ON CONFLICT(key)
        DO UPDATE SET value=EXCLUDED.value
        """,
        (key, str(value))
    )


# =========================================================
# TRAINING STATUS
# =========================================================

def get_training_status():
    row = execute(
        """
        SELECT status, progress, message, error, updated_at
        FROM training_state
        WHERE id=1
        """,
        fetchone=True
    )

    if not row:
        return DEFAULT_TRAINING.copy()

    return {
        "status": row["status"],
        "progress": int(row["progress"] or 0),
        "message": row["message"] or "",
        "error": row["error"] or "",
        "updated_at": str(row["updated_at"]) if row["updated_at"] else "",
    }


def set_training_status(
    status,
    progress=0,
    message="",
    error=""
):
    execute(
        """
        INSERT INTO training_state
        (id, status, progress, message, error, updated_at)
        VALUES(1, %s, %s, %s, %s, NOW())
        ON CONFLICT(id)
        DO UPDATE SET
            status=EXCLUDED.status,
            progress=EXCLUDED.progress,
            message=EXCLUDED.message,
            error=EXCLUDED.error,
            updated_at=NOW()
        """,
        (
            status,
            int(progress),
            message,
            error
        )
    )


# =========================================================
# DATA URL
# =========================================================

def data_url_to_bytes(data_url):
    if not data_url:
        return b""

    if "," in data_url:
        data_url = data_url.split(",", 1)[1]

    return base64.b64decode(data_url)


# =========================================================
# DAILY COUNTS
# =========================================================

def update_daily(column, amount=1):
    allowed = {
        "loaded",
        "empty",
        "people",
        "equipment",
        "unknown"
    }

    if column not in allowed:
        return

    today = date.today().isoformat()

    execute(
        """
        INSERT INTO daily_counts
        (count_date, loaded, empty, people, equipment, unknown)
        VALUES(%s, 0, 0, 0, 0, 0)
        ON CONFLICT(count_date) DO NOTHING
        """,
        (today,)
    )

    execute(
        f"""
        UPDATE daily_counts
        SET {column} = COALESCE({column}, 0) + %s
        WHERE count_date=%s
        """,
        (amount, today)
    )


# =========================================================
# DETECTIONS
# =========================================================

def save_detection(
    status,
    counted=0,
    confidence=0,
    track_id=None,
    bucket_id=None,
    bucket_name=None,
    note=""
):
    execute(
        """
        INSERT INTO detections
        (
            detection_time,
            bucket_id,
            bucket_name,
            status,
            counted,
            confidence,
            track_id,
            note
        )
        VALUES(
            NOW(), %s, %s, %s, %s, %s, %s, %s
        )
        """,
        (
            bucket_id,
            bucket_name,
            status,
            int(counted),
            float(confidence),
            track_id,
            note
        )
    )


# =========================================================
# DATASET
# =========================================================

def get_dataset_summary():
    row = execute(
        """
        SELECT
            COUNT(*) AS total_images,
            COUNT(
                CASE
                    WHEN EXISTS(
                        SELECT 1
                        FROM annotations a
                        WHERE a.image_id=dataset_images.id
                    )
                    THEN 1
                END
            ) AS labeled_images
        FROM dataset_images
        """,
        fetchone=True
    )

    annotations = execute(
        "SELECT COUNT(*) AS n FROM annotations",
        fetchone=True
    )

    return {
        "total_images": int(row["total_images"] or 0),
        "labeled_images": int(row["labeled_images"] or 0),
        "annotations": int(annotations["n"] or 0),
    }


def prepare_dataset():
    summary = get_dataset_summary()

    if summary["labeled_images"] < 5:
        raise RuntimeError(
            "At least 5 labeled images are required."
        )

    if os.path.exists(DATASET_DIR):
        shutil.rmtree(DATASET_DIR)

    os.makedirs(os.path.join(DATASET_DIR, "images", "train"), exist_ok=True)
    os.makedirs(os.path.join(DATASET_DIR, "images", "val"), exist_ok=True)
    os.makedirs(os.path.join(DATASET_DIR, "labels", "train"), exist_ok=True)
    os.makedirs(os.path.join(DATASET_DIR, "labels", "val"), exist_ok=True)

    images = execute(
        """
        SELECT *
        FROM dataset_images
        ORDER BY id
        """,
        fetchall=True
    )

    if not images:
        raise RuntimeError("No dataset images found.")

    # 80/20 split
    split_index = max(1, int(len(images) * 0.8))

    for index, image in enumerate(images):

        image_id = image["id"]

        raw = data_url_to_bytes(image["image_data"])

        ext = ".jpg"

        filename = image["filename"] or f"image_{image_id}.jpg"

        if "." in filename:
            ext = "." + filename.rsplit(".", 1)[1]

        stem = f"image_{image_id}"

        split = "train" if index < split_index else "val"

        image_path = os.path.join(
            DATASET_DIR,
            "images",
            split,
            stem + ext
        )

        label_path = os.path.join(
            DATASET_DIR,
            "labels",
            split,
            stem + ".txt"
        )

        with open(image_path, "wb") as f:
            f.write(raw)

        annotations = execute(
            """
            SELECT *
            FROM annotations
            WHERE image_id=%s
            ORDER BY id
            """,
            (image_id,),
            fetchall=True
        )

        with open(label_path, "w", encoding="utf-8") as f:
            for a in annotations:
                f.write(
                    "{} {:.6f} {:.6f} {:.6f} {:.6f}\n".format(
                        int(a["class_id"]),
                        float(a["x_center"]),
                        float(a["y_center"]),
                        float(a["box_width"]),
                        float(a["box_height"])
                    )
                )

    yaml_content = f"""
path: {DATASET_DIR}
train: images/train
val: images/val

names:
  0: BUCKET_LOADED
  1: BUCKET_EMPTY
  2: PEOPLE
  3: EQUIPMENT
"""

    with open(DATA_YAML, "w", encoding="utf-8") as f:
        f.write(yaml_content.strip())

    return summary


# =========================================================
# MODEL STORAGE
# =========================================================

def save_model_to_database(path):
    if not os.path.exists(path):
        raise RuntimeError("best.pt was not created.")

    with open(path, "rb") as f:
        model_data = f.read()

    execute(
        """
        INSERT INTO trained_model(id, model_data, filename, created_at)
        VALUES(1, %s, %s, NOW())
        ON CONFLICT(id)
        DO UPDATE SET
            model_data=EXCLUDED.model_data,
            filename=EXCLUDED.filename,
            created_at=NOW()
        """,
        (
            psycopg2.Binary(model_data),
            "best.pt"
        )
    )


def restore_model_from_database():
    if os.path.exists(MODEL_FILE):
        return True

    row = execute(
        """
        SELECT model_data
        FROM trained_model
        WHERE id=1
        """,
        fetchone=True
    )

    if not row or not row["model_data"]:
        return False

    try:
        with open(MODEL_FILE, "wb") as f:
            f.write(bytes(row["model_data"]))

        return True

    except Exception:
        return False


def model_ready():
    return os.path.exists(MODEL_FILE)


# =========================================================
# TRAINING
# =========================================================

def training_callback(trainer):
    try:
        epoch = int(trainer.epoch) + 1
        total = int(trainer.epochs)

        progress = int((epoch / total) * 100)

        set_training_status(
            "Training",
            progress,
            f"Epoch {epoch}/{total}",
            ""
        )

    except Exception:
        pass


def run_training():
    if not YOLO_AVAILABLE:
        set_training_status(
            "Failed",
            0,
            "",
            "Ultralytics YOLO is not installed."
        )
        return

    try:
        set_training_status(
            "Preparing",
            5,
            "Preparing dataset...",
            ""
        )

        prepare_dataset()

        set_training_status(
            "Starting",
            10,
            "Loading YOLO model...",
            ""
        )

        model = YOLO("yolo11n.pt")

        try:
            model.add_callback(
                "on_fit_epoch_end",
                training_callback
            )
        except Exception:
            pass

        set_training_status(
            "Training",
            15,
            "Training YOLO...",
            ""
        )

        # =================================================
        # RENDER FREE FRIENDLY SETTINGS
        # =================================================

        model.train(
            data=DATA_YAML,

            # LIGHTWEIGHT SETTINGS
            epochs=10,
            imgsz=416,
            batch=1,
            workers=0,

            # CPU
            device="cpu",

            # Reduce RAM/CPU usage
            cache=False,
            amp=False,

            # Stable settings
            pretrained=True,
            project=TRAIN_DIR,
            name="bucket_ai",
            exist_ok=True,

            # Disable unnecessary heavy augmentation
            mosaic=0.0,
            mixup=0.0,
            copy_paste=0.0,

            verbose=True,
            plots=False,
            save=True,
            val=True,
        )

        result_dir = os.path.join(
            TRAIN_DIR,
            "bucket_ai"
        )

        best = os.path.join(
            result_dir,
            "weights",
            "best.pt"
        )

        if not os.path.exists(best):
            raise RuntimeError(
                "Training finished but best.pt was not found."
            )

        shutil.copy2(
            best,
            MODEL_FILE
        )

        set_training_status(
            "Saving model",
            95,
            "Saving trained model to Supabase...",
            ""
        )

        save_model_to_database(MODEL_FILE)

        set_training_status(
            "Completed",
            100,
            "AI model trained successfully.",
            ""
        )

    except Exception as e:

        error_text = (
            str(e)
            + "\n\n"
            + traceback.format_exc()
        )

        print(error_text)

        set_training_status(
            "Failed",
            0,
            "",
            error_text[-8000:]
        )


def start_training():
    with training_lock:

        current = get_training_status()

        if current["status"] in (
            "Preparing",
            "Starting",
            "Training",
            "Saving model"
        ):
            return False

        set_training_status(
            "Preparing",
            1,
            "Training queued...",
            ""
        )

        thread = threading.Thread(
            target=run_training,
            daemon=True
        )

        thread.start()

        return True


# =========================================================
# BUCKETS
# =========================================================

def active_bucket():
    return execute(
        """
        SELECT *
        FROM buckets
        WHERE active=1
        ORDER BY id
        LIMIT 1
        """,
        fetchone=True
    )


# =========================================================
# HTTP HELPERS
# =========================================================

def send_json(handler, data, status=200):
    body = json.dumps(
        data,
        ensure_ascii=False
    ).encode("utf-8")

    handler.send_response(status)
    handler.send_header(
        "Content-Type",
        "application/json; charset=utf-8"
    )
    handler.send_header(
        "Content-Length",
        str(len(body))
    )
    handler.end_headers()

    handler.wfile.write(body)


def send_html(handler, html, status=200):
    body = html.encode("utf-8")

    handler.send_response(status)
    handler.send_header(
        "Content-Type",
        "text/html; charset=utf-8"
    )
    handler.send_header(
        "Content-Length",
        str(len(body))
    )
    handler.end_headers()

    handler.wfile.write(body)


def read_json(handler):
    length = int(
        handler.headers.get(
            "Content-Length",
            "0"
        )
    )

    raw = handler.rfile.read(length)

    if not raw:
        return {}

    return json.loads(
        raw.decode("utf-8")
    )


# =========================================================
# HTML
# =========================================================

STYLE = """
<style>
*{
    box-sizing:border-box;
}

body{
    margin:0;
    font-family:Arial,sans-serif;
    background:#f4f6f8;
    color:#17202a;
}

header{
    background:#111827;
    color:white;
    padding:18px;
}

header h1{
    margin:0;
    font-size:23px;
}

header p{
    margin:5px 0 0;
    opacity:.75;
}

nav{
    background:white;
    padding:10px;
    display:flex;
    gap:8px;
    flex-wrap:wrap;
    border-bottom:1px solid #ddd;
}

nav a{
    text-decoration:none;
    padding:9px 13px;
    background:#eef2f7;
    color:#17202a;
    border-radius:7px;
}

main{
    max-width:1100px;
    margin:auto;
    padding:20px;
}

.card{
    background:white;
    padding:20px;
    margin-bottom:18px;
    border-radius:12px;
    box-shadow:0 2px 8px rgba(0,0,0,.07);
}

button{
    border:0;
    padding:12px 18px;
    border-radius:8px;
    background:#111827;
    color:white;
    cursor:pointer;
    font-weight:bold;
}

button:disabled{
    opacity:.5;
}

input,select{
    width:100%;
    padding:11px;
    border:1px solid #ccc;
    border-radius:7px;
    margin:6px 0 12px;
}

.grid{
    display:grid;
    grid-template-columns:repeat(auto-fit,minmax(220px,1fr));
    gap:15px;
}

.stat{
    background:white;
    padding:20px;
    border-radius:12px;
    box-shadow:0 2px 8px rgba(0,0,0,.06);
}

.stat strong{
    font-size:30px;
    display:block;
}

.success{
    background:#dcfce7;
    padding:12px;
    border-radius:8px;
}

.warning{
    background:#fef3c7;
    padding:12px;
    border-radius:8px;
}

.error{
    background:#fee2e2;
    padding:12px;
    border-radius:8px;
    white-space:pre-wrap;
}

#preview{
    max-width:100%;
    display:none;
    margin-top:15px;
    border:1px solid #ddd;
}

#canvas{
    max-width:100%;
    border:1px solid #999;
    display:none;
    touch-action:none;
}

.progress{
    height:24px;
    background:#ddd;
    border-radius:20px;
    overflow:hidden;
}

.progressbar{
    height:100%;
    width:0%;
    background:#111827;
    color:white;
    text-align:center;
    line-height:24px;
}

video{
    width:100%;
    max-width:800px;
    background:black;
    border-radius:10px;
}

table{
    width:100%;
    border-collapse:collapse;
}

td,th{
    padding:10px;
    border-bottom:1px solid #ddd;
    text-align:left;
}

.small{
    font-size:13px;
    opacity:.7;
}
</style>
"""


def layout(title, content):
    return f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport"
content="width=device-width,initial-scale=1">
<title>{title}</title>
{STYLE}
</head>

<body>

<header>
<h1>🪣 BUCKET COUNTER AI</h1>
<p>Underground production monitoring</p>
</header>

<nav>
<a href="/">Dashboard</a>
<a href="/camera">Camera</a>
<a href="/buckets">Buckets</a>
<a href="/training">AI Training</a>
<a href="/history">History</a>
<a href="/settings">Settings</a>
</nav>

<main>
{content}
</main>

</body>
</html>
"""


# =========================================================
# DASHBOARD
# =========================================================

def dashboard_page():
    content = """
<h2>📊 Dashboard</h2>

<div class="grid">

<div class="stat">
Loaded buckets
<strong id="loaded">0</strong>
</div>

<div class="stat">
Empty buckets
<strong id="empty">0</strong>
</div>

<div class="stat">
People
<strong id="people">0</strong>
</div>

<div class="stat">
Equipment
<strong id="equipment">0</strong>
</div>

</div>

<br>

<div class="card">
<h3>AI Status</h3>
<div id="status">Loading...</div>
</div>

<script>
async function refresh(){

    const r = await fetch('/api/dashboard');
    const d = await r.json();

    document.getElementById('loaded').textContent =
        d.loaded;

    document.getElementById('empty').textContent =
        d.empty;

    document.getElementById('people').textContent =
        d.people;

    document.getElementById('equipment').textContent =
        d.equipment;

    document.getElementById('status').innerHTML =
        'Database: <b>' + d.database +
        '</b><br>YOLO: <b>' + d.yolo +
        '</b><br>Model: <b>' + d.model +
        '</b>';
}

refresh();
setInterval(refresh,3000);
</script>
"""

    return layout(
        "Dashboard - Bucket Counter AI",
        content
    )


# =========================================================
# TRAINING PAGE
# =========================================================

def training_page():
    content = """
<h2>🤖 AI Training</h2>

<div class="card">

<div id="systemStatus">
Loading...
</div>

<p>
<b>Classes:</b><br>
BUCKET_LOADED = count<br>
BUCKET_EMPTY = no count<br>
PEOPLE = no count<br>
EQUIPMENT = no count
</p>

</div>

<div class="card">

<h3>1. Upload and label one image</h3>

<input
type="file"
id="imageInput"
accept="image/*"
>

<select id="classSelect">
<option value="0">BUCKET_LOADED</option>
<option value="1">BUCKET_EMPTY</option>
<option value="2">PEOPLE</option>
<option value="3">EQUIPMENT</option>
</select>

<p>
Drag on the image to draw one box around
the selected object.
</p>

<img id="preview">

<canvas id="canvas"></canvas>

<br><br>

<button onclick="saveImage()">
💾 SAVE IMAGE + LABEL
</button>

<div id="saveResult"></div>

</div>

<div class="card">

<h3>2. Train model</h3>

<p>
At least 5 labeled images are required.
More varied images normally improve detection.
</p>

<button id="trainButton"
onclick="trainAI()">
🚀 TRAIN AI
</button>

<br><br>

<div class="progress">
<div class="progressbar"
id="progressbar">0%</div>
</div>

<p id="trainingMessage">
Not started
</p>

<div id="trainingError"></div>

</div>

<div class="card">

<h3>Saved dataset</h3>

<div id="summary">
Loading...
</div>

</div>

<script>

let imgData = null;
let imageFileName = "";
let startX = 0;
let startY = 0;
let endX = 0;
let endY = 0;
let drawing = false;

const input =
document.getElementById("imageInput");

const preview =
document.getElementById("preview");

const canvas =
document.getElementById("canvas");

const ctx =
canvas.getContext("2d");

input.addEventListener("change",function(){

    const file = this.files[0];

    if(!file) return;

    imageFileName = file.name;

    const reader = new FileReader();

    reader.onload = function(e){

        imgData = e.target.result;

        preview.src = imgData;
        preview.style.display = "block";

        preview.onload = function(){

            canvas.width = preview.naturalWidth;
            canvas.height = preview.naturalHeight;

            canvas.style.width =
                Math.min(
                    preview.naturalWidth,
                    900
                ) + "px";

            canvas.style.height =
                "auto";

            ctx.drawImage(
                preview,
                0,
                0
            );

            canvas.style.display = "block";
            preview.style.display = "none";
        };
    };

    reader.readAsDataURL(file);
});


function getPosition(e){

    const rect =
        canvas.getBoundingClientRect();

    let clientX;
    let clientY;

    if(e.touches && e.touches.length){

        clientX = e.touches[0].clientX;
        clientY = e.touches[0].clientY;

    }else{

        clientX = e.clientX;
        clientY = e.clientY;
    }

    return {
        x:
            (clientX - rect.left)
            * canvas.width
            / rect.width,

        y:
            (clientY - rect.top)
            * canvas.height
            / rect.height
    };
}


function startDraw(e){

    e.preventDefault();

    const p = getPosition(e);

    startX = p.x;
    startY = p.y;

    drawing = true;
}


function drawBox(e){

    if(!drawing) return;

    e.preventDefault();

    const p = getPosition(e);

    endX = p.x;
    endY = p.y;

    ctx.drawImage(
        preview,
        0,
        0
    );

    ctx.strokeStyle = "red";
    ctx.lineWidth = 4;

    ctx.strokeRect(
        startX,
        startY,
        endX - startX,
        endY - startY
    );
}


function stopDraw(e){

    if(!drawing) return;

    drawing = false;

    if(e && e.preventDefault)
        e.preventDefault();

    const p = getPosition(e);

    endX = p.x;
    endY = p.y;

    ctx.drawImage(
        preview,
        0,
        0
    );

    ctx.strokeStyle = "red";
    ctx.lineWidth = 4;

    ctx.strokeRect(
        startX,
        startY,
        endX - startX,
        endY - startY
    );
}


canvas.addEventListener(
    "mousedown",
    startDraw
);

canvas.addEventListener(
    "mousemove",
    drawBox
);

canvas.addEventListener(
    "mouseup",
    stopDraw
);

canvas.addEventListener(
    "touchstart",
    startDraw,
    {passive:false}
);

canvas.addEventListener(
    "touchmove",
    drawBox,
    {passive:false}
);

canvas.addEventListener(
    "touchend",
    stopDraw,
    {passive:false}
);


async function saveImage(){

    if(!imgData){

        alert("Choose an image first.");
        return;
    }

    const x =
        Math.min(startX,endX);

    const y =
        Math.min(startY,endY);

    const w =
        Math.abs(endX-startX);

    const h =
        Math.abs(endY-startY);

    if(w < 5 || h < 5){

        alert(
            "Draw a box around the object first."
        );

        return;
    }

    const payload = {

        filename:
            imageFileName || "image.jpg",

        image_data:
            imgData,

        width:
            canvas.width,

        height:
            canvas.height,

        class_id:
            parseInt(
                document.getElementById(
                    "classSelect"
                ).value
            ),

        class_name:
            document.getElementById(
                "classSelect"
            ).selectedOptions[0].text,

        x_center:
            (x + w/2) / canvas.width,

        y_center:
            (y + h/2) / canvas.height,

        box_width:
            w / canvas.width,

        box_height:
            h / canvas.height
    };

    const r = await fetch(
        "/api/training/image",
        {
            method:"POST",
            headers:{
                "Content-Type":
                    "application/json"
            },
            body:JSON.stringify(payload)
        }
    );

    const d = await r.json();

    if(d.ok){

        document.getElementById(
            "saveResult"
        ).innerHTML =
            '<div class="success">' +
            '✅ IMAGE + LABEL SAVED SUCCESSFULLY. ID ' +
            d.image_id +
            '</div>';

        refreshSummary();

    }else{

        document.getElementById(
            "saveResult"
        ).innerHTML =
            '<div class="error">' +
            d.error +
            '</div>';
    }
}


async function trainAI(){

    const button =
        document.getElementById(
            "trainButton"
        );

    button.disabled = true;

    const r = await fetch(
        "/api/train",
        {
            method:"POST"
        }
    );

    const d = await r.json();

    if(!d.ok){

        alert(d.error);
        button.disabled = false;
        return;
    }

    pollTraining();
}


async function pollTraining(){

    const r =
        await fetch("/api/status");

    const d =
        await r.json();

    const p =
        d.training.progress || 0;

    document.getElementById(
        "progressbar"
    ).style.width = p + "%";

    document.getElementById(
        "progressbar"
    ).textContent = p + "%";

    document.getElementById(
        "trainingMessage"
    ).textContent =
        d.training.status +
        " - " +
        d.training.message;

    if(d.training.error){

        document.getElementById(
            "trainingError"
        ).innerHTML =
            '<div class="error">' +
            d.training.error +
            '</div>';
    }

    if(
        d.training.status === "Completed"
        ||
        d.training.status === "Failed"
    ){

        document.getElementById(
            "trainButton"
        ).disabled = false;

        refreshSystem();

        return;
    }

    setTimeout(
        pollTraining,
        2000
    );
}


async function refreshSummary(){

    const r =
        await fetch(
            "/api/training/summary"
        );

    const d =
        await r.json();

    document.getElementById(
        "summary"
    ).innerHTML =
        "Total images: <b>" +
        d.total_images +
        "</b> | Labeled: <b>" +
        d.labeled_images +
        "</b> | Annotations: <b>" +
        d.annotations +
        "</b>";
}


async function refreshSystem(){

    const r =
        await fetch("/api/status");

    const d =
        await r.json();

    document.getElementById(
        "systemStatus"
    ).innerHTML =
        "Database: <b>" +
        d.database +
        "</b> | YOLO: <b>" +
        d.yolo +
        "</b> | Model: <b>" +
        d.model +
        "</b>";

    if(
        d.training.status !==
        "Not started"
    ){

        document.getElementById(
            "progressbar"
        ).style.width =
            d.training.progress + "%";

        document.getElementById(
            "progressbar"
        ).textContent =
            d.training.progress + "%";

        document.getElementById(
            "trainingMessage"
        ).textContent =
            d.training.status +
            " - " +
            d.training.message;
    }
}


refreshSummary();
refreshSystem();

setInterval(
    refreshSystem,
    3000
);

</script>
"""

    return layout(
        "AI Training",
        content
    )


# =========================================================
# CAMERA PAGE
# =========================================================

def camera_page():
    content = """
<h2>📷 Camera</h2>

<div class="card">

<p>
Camera detection will identify:
</p>

<ul>
<li>🪣 BUCKET_LOADED → COUNT</li>
<li>🪣 BUCKET_EMPTY → NO COUNT</li>
<li>👷 PEOPLE → NO COUNT</li>
<li>🔧 EQUIPMENT → NO COUNT</li>
</ul>

<p>
Counting line:
<strong id="line">55%</strong>
</p>

<video
id="video"
autoplay
playsinline>
</video>

<br><br>

<button onclick="startCamera()">
📷 START CAMERA
</button>

<button onclick="stopCamera()">
⛔ STOP
</button>

<br><br>

<button onclick="resetTracker()">
🔄 RESET TRACKER
</button>

</div>

<script>

let stream = null;

async function startCamera(){

    try{

        stream =
            await navigator.mediaDevices
            .getUserMedia({
                video:{
                    facingMode:"environment"
                },
                audio:false
            });

        document.getElementById(
            "video"
        ).srcObject = stream;

    }catch(e){

        alert(
            "Camera error: " + e
        );
    }
}


function stopCamera(){

    if(!stream) return;

    stream.getTracks().forEach(
        track => track.stop()
    );

    stream = null;
}


async function resetTracker(){

    await fetch(
        "/api/reset-tracker",
        {
            method:"POST"
        }
    );

    alert(
        "Tracker reset successfully."
    );
}


async function load(){

    const r =
        await fetch("/api/settings");

    const d =
        await r.json();

    document.getElementById(
        "line"
    ).textContent =
        d.line_position + "%";
}

load();

</script>
"""

    return layout(
        "Camera",
        content
    )


# =========================================================
# BUCKETS PAGE
# =========================================================

def buckets_page():
    buckets = execute(
        """
        SELECT *
        FROM buckets
        ORDER BY id DESC
        """,
        fetchall=True
    )

    rows = ""

    for b in buckets:
        rows += f"""
<tr>
<td>{b["id"]}</td>
<td>{b["name"]}</td>
<td>{b["capacity"]}</td>
<td>{"YES" if b["active"] else "NO"}</td>
</tr>
"""

    content = """
<h2>🪣 Buckets</h2>

<div class="card">

<h3>Add Bucket</h3>

<input
id="name"
placeholder="Bucket name">

<input
id="capacity"
type="number"
placeholder="Capacity">

<button onclick="addBucket()">
➕ ADD BUCKET
</button>

</div>

<div class="card">

<table>
<tr>
<th>ID</th>
<th>Name</th>
<th>Capacity</th>
<th>Active</th>
</tr>

{rows}

</table>

</div>

<script>

async function addBucket() {{

    const name = document.getElementById("name").value;

    const capacity = document.getElementById("capacity").value;

    const r = await fetch("/api/buckets", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({
            name: name,
            capacity: capacity
        })
    });

    const name = document.getElementById("name").value;

    const capacity = document.getElementById("capacity").value;

    const r = await fetch("/api/buckets", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({
            name: name,
            capacity: capacity
        })
    });

    const d = await r.json();

    if (d.ok) {
        window.location.reload();
    } else {
        alert(d.error);
    }
}

    const r = await fetch(
            "/api/buckets",
            {
                method:"POST",
                headers:{
                    "Content-Type":
                        "application/json"
                },
                body:JSON.stringify({
                    name:name,
                    capacity:capacity
                })
            }
        );

    const d =
        await r.json();

    if(d.ok)
        location.reload();
    else
        alert(d.error);
}

</script>
"""

    return layout(
        "Buckets",
        content
    )


# =========================================================
# HISTORY
# =========================================================

def history_page():
    rows = execute(
        """
        SELECT
            detection_time,
            status,
            counted,
            confidence,
            track_id
        FROM detections
        ORDER BY id DESC
        LIMIT 100
        """,
        fetchall=True
    )

    html_rows = ""

    for r in rows:

        html_rows += f"""
<tr>
<td>{r["detection_time"]}</td>
<td>{r["status"]}</td>
<td>{r["counted"]}</td>
<td>{float(r["confidence"] or 0):.2f}</td>
<td>{r["track_id"] or ""}</td>
</tr>
"""

    content = f"""
<h2>📜 History</h2>

<div class="card">

<a href="/api/export.csv">
<button>📥 EXPORT CSV</button>
</a>

<br><br>

<table>

<tr>
<th>Time</th>
<th>Status</th>
<th>Counted</th>
<th>Confidence</th>
<th>Track ID</th>
</tr>

{html_rows}

</table>

</div>
"""

    return layout(
        "History",
        content
    )


# =========================================================
# SETTINGS
# =========================================================

def settings_page():
    line = setting(
        "line_position",
        "55"
    )

    content = """
<h2>⚙️ Settings</h2>

<div class="card">

<h3>Counting Line</h3>

<label>
Horizontal line position (%)
</label>

<input
id="line"
type="number"
min="1"
max="99"
value="{line}">

<button onclick="saveSettings()">
💾 SAVE SETTINGS
</button>

<div id="result"></div>

</div>

<script>

async function saveSettings(){

    const line = document.getElementById(
            "line"
        ).value;

    const r =
        await fetch(
            "/api/settings",
            {
                method:"POST",
                headers:{
                    "Content-Type":
                        "application/json"
                },
                body:JSON.stringify({
                    line_position:line
                })
            }
        );

    const d =
        await r.json();

    document.getElementById(
        "result"
    ).innerHTML =
        d.ok
        ? '<div class="success">Saved successfully.</div>'
        : '<div class="error">' +
          d.error +
          '</div>';
}

</script>
"""

    return layout(
        "Settings",
        content
    )


# =========================================================
# API STATUS
# =========================================================

def api_status():
    return {
        "ok": True,
        "database": "Supabase PostgreSQL",
        "database_connected": True,
        "yolo": "INSTALLED"
        if YOLO_AVAILABLE
        else "NOT INSTALLED",
        "model": "READY"
        if model_ready()
        else "NOT READY",
        "training": get_training_status(),
        "dataset": get_dataset_summary(),
    }


# =========================================================
# API DASHBOARD
# =========================================================

def api_dashboard():

    today = date.today().isoformat()

    row = execute(
        """
        SELECT *
        FROM daily_counts
        WHERE count_date=%s
        """,
        (today,),
        fetchone=True
    )

    if not row:

        loaded = 0
        empty = 0
        people = 0
        equipment = 0

    else:

        loaded = int(row["loaded"] or 0)
        empty = int(row["empty"] or 0)
        people = int(row["people"] or 0)
        equipment = int(row["equipment"] or 0)

    return {
        "loaded": loaded,
        "empty": empty,
        "people": people,
        "equipment": equipment,
        "database": "Supabase PostgreSQL",
        "yolo": "INSTALLED"
        if YOLO_AVAILABLE
        else "NOT INSTALLED",
        "model": "READY"
        if model_ready()
        else "NOT READY",
    }


# =========================================================
# HANDLER
# =========================================================

class Handler(BaseHTTPRequestHandler):

    def log_message(self, format, *args):
        print(
            "%s - %s"
            % (
                self.address_string(),
                format % args
            )
        )

    def do_GET(self):

        try:

            parsed = urlparse(
                self.path
            )

            path = parsed.path

            # -----------------------------
            # PAGES
            # -----------------------------

            if path == "/":
                send_html(
                    self,
                    dashboard_page()
                )
                return

            if path == "/camera":
                send_html(
                    self,
                    camera_page()
                )
                return

            if path == "/buckets":
                send_html(
                    self,
                    buckets_page()
                )
                return

            if path == "/training":
                send_html(
                    self,
                    training_page()
                )
                return

            if path == "/history":
                send_html(
                    self,
                    history_page()
                )
                return

            if path == "/settings":
                send_html(
                    self,
                    settings_page()
                )
                return

            # -----------------------------
            # STATUS
            # -----------------------------

            if path == "/api/status":

                send_json(
                    self,
                    api_status()
                )
                return

            # -----------------------------
            # DASHBOARD
            # -----------------------------

            if path == "/api/dashboard":

                send_json(
                    self,
                    api_dashboard()
                )
                return

            # -----------------------------
            # TRAINING SUMMARY
            # -----------------------------

            if path == "/api/training/summary":

                send_json(
                    self,
                    get_dataset_summary()
                )
                return

            # -----------------------------
            # SETTINGS
            # -----------------------------

            if path == "/api/settings":

                send_json(
                    self,
                    {
                        "line_position":
                            int(
                                setting(
                                    "line_position",
                                    "55"
                                )
                            )
                    }
                )
                return

            # -----------------------------
            # BUCKETS
            # -----------------------------

            if path == "/api/buckets":

                rows = execute(
                    """
                    SELECT *
                    FROM buckets
                    ORDER BY id DESC
                    """,
                    fetchall=True
                )

                send_json(
                    self,
                    {
                        "buckets": rows
                    }
                )

                return

            # -----------------------------
            # CSV
            # -----------------------------

            if path == "/api/export.csv":

                rows = execute(
                    """
                    SELECT
                        detection_time,
                        status,
                        counted,
                        confidence,
                        track_id,
                        bucket_name,
                        note
                    FROM detections
                    ORDER BY id DESC
                    """,
                    fetchall=True
                )

                output = io.StringIO()

                writer = csv.writer(output)

                writer.writerow([
                    "detection_time",
                    "status",
                    "counted",
                    "confidence",
                    "track_id",
                    "bucket_name",
                    "note"
                ])

                for r in rows:

                    writer.writerow([
                        r["detection_time"],
                        r["status"],
                        r["counted"],
                        r["confidence"],
                        r["track_id"],
                        r["bucket_name"],
                        r["note"]
                    ])

                body = output.getvalue().encode(
                    "utf-8"
                )

                self.send_response(200)

                self.send_header(
                    "Content-Type",
                    "text/csv"
                )

                self.send_header(
                    "Content-Disposition",
                    "attachment; filename=bucket_history.csv"
                )

                self.send_header(
                    "Content-Length",
                    str(len(body))
                )

                self.end_headers()

                self.wfile.write(body)

                return

            send_json(
                self,
                {
                    "error": "Not found"
                },
                404
            )

        except Exception as e:

            print(traceback.format_exc())

            send_json(
                self,
                {
                    "ok": False,
                    "error": str(e)
                },
                500
            )


    def do_POST(self):

        try:

            parsed = urlparse(
                self.path
            )

            path = parsed.path

            data = read_json(self)

            # =================================================
            # RESET TRACKER
            # =================================================

            if path == "/api/reset-tracker":

                COUNTED_TRACKS.clear()
                LAST_TRACK_Y.clear()
                LAST_TRACK_TIME.clear()

                send_json(
                    self,
                    {
                        "ok": True
                    }
                )

                return

            # =================================================
            # SETTINGS
            # =================================================

            if path == "/api/settings":

                line = int(
                    data.get(
                        "line_position",
                        55
                    )
                )

                line = max(
                    1,
                    min(99, line)
                )

                set_setting(
                    "line_position",
                    line
                )

                send_json(
                    self,
                    {
                        "ok": True,
                        "line_position": line
                    }
                )

                return

            # =================================================
            # ADD BUCKET
            # =================================================

            if path == "/api/buckets":

                name = str(
                    data.get(
                        "name",
                        ""
                    )
                ).strip()

                capacity = float(
                    data.get(
                        "capacity",
                        0
                    ) or 0
                )

                if not name:

                    send_json(
                        self,
                        {
                            "ok": False,
                            "error":
                                "Bucket name is required."
                        },
                        400
                    )

                    return

                row = execute(
                    """
                    INSERT INTO buckets
                    (name, capacity, active)
                    VALUES(%s, %s, 1)
                    RETURNING id
                    """,
                    (
                        name,
                        capacity
                    ),
                    fetchone=True
                )

                send_json(
                    self,
                    {
                        "ok": True,
                        "id": row["id"]
                    }
                )

                return

            # =================================================
            # ACTIVATE BUCKET
            # =================================================

            if path == "/api/buckets/activate":

                bucket_id = int(
                    data.get("bucket_id")
                )

                execute(
                    """
                    UPDATE buckets
                    SET active=0
                    """
                )

                execute(
                    """
                    UPDATE buckets
                    SET active=1
                    WHERE id=%s
                    """,
                    (bucket_id,)
                )

                send_json(
                    self,
                    {
                        "ok": True
                    }
                )

                return

            # =================================================
            # SAVE TRAINING IMAGE
            # =================================================

            if path == "/api/training/image":

                image_data = data.get(
                    "image_data"
                )

                filename = data.get(
                    "filename",
                    "image.jpg"
                )

                width = int(
                    data.get(
                        "width",
                        0
                    )
                )

                height = int(
                    data.get(
                        "height",
                        0
                    )
                )

                class_id = int(
                    data.get(
                        "class_id",
                        0
                    )
                )

                class_name = data.get(
                    "class_name",
                    CLASSES[class_id]
                    if 0 <= class_id < len(CLASSES)
                    else "UNKNOWN"
                )

                x_center = float(
                    data.get(
                        "x_center",
                        0
                    )
                )

                y_center = float(
                    data.get(
                        "y_center",
                        0
                    )
                )

                box_width = float(
                    data.get(
                        "box_width",
                        0
                    )
                )

                box_height = float(
                    data.get(
                        "box_height",
                        0
                    )
                )

                if not image_data:

                    send_json(
                        self,
                        {
                            "ok": False,
                            "error":
                                "Image is required."
                        },
                        400
                    )

                    return

                if not (
                    0 <= class_id < len(CLASSES)
                ):

                    send_json(
                        self,
                        {
                            "ok": False,
                            "error":
                                "Invalid class."
                        },
                        400
                    )

                    return

                image_row = execute(
                    """
                    INSERT INTO dataset_images
                    (
                        filename,
                        image_data,
                        width,
                        height
                    )
                    VALUES(%s, %s, %s, %s)
                    RETURNING id
                    """,
                    (
                        filename,
                        image_data,
                        width,
                        height
                    ),
                    fetchone=True
                )

                image_id = image_row["id"]

                execute(
                    """
                    INSERT INTO annotations
                    (
                        image_id,
                        class_id,
                        class_name,
                        x_center,
                        y_center,
                        box_width,
                        box_height
                    )
                    VALUES(
                        %s, %s, %s,
                        %s, %s, %s, %s
                    )
                    """,
                    (
                        image_id,
                        class_id,
                        class_name,
                        x_center,
                        y_center,
                        box_width,
                        box_height
                    )
                )

                send_json(
                    self,
                    {
                        "ok": True,
                        "image_id": image_id
                    }
                )

                return

            # =================================================
            # TRAIN
            # =================================================

            if path == "/api/train":

                summary =  get_dataset_summary()

                if summary["labeled_images"] < 5:

                    send_json(
                        self,
                        {
                            "ok": False,
                            "error":
                                "At least 5 labeled images are required."
                        },
                        400
                    )

                    return

                started = start_training()

                if not started:

                    send_json(
                        self,
                        {
                            "ok": False,
                            "error":
                                "Training is already running."
                        },
                        409
                    )

                    return

                send_json(
                    self,
                    {
                        "ok": True,
                        "message":
                            "Training started."
                    }
                )

                return

            # =================================================
            # DETECTION
            # =================================================

            if path == "/api/detect":

                if not YOLO_AVAILABLE:

                    send_json(
                        self,
                        {
                            "ok": False,
                            "error":
                                "YOLO is not installed."
                        },
                        500
                    )

                    return

                if not model_ready():

                    restore_model_from_database()

                if not model_ready():

                    send_json(
                        self,
                        {
                            "ok": False,
                            "error":
                                "AI model is not ready."
                        },
                        400
                    )

                    return

                if not CV_AVAILABLE:

                    send_json(
                        self,
                        {
                            "ok": False,
                            "error":
                                "OpenCV is not installed."
                        },
                        500
                    )

                    return

                image_data = data.get(
                    "image_data"
                )

                if not image_data:

                    send_json(
                        self,
                        {
                            "ok": False,
                            "error":
                                "image_data is required."
                        },
                        400
                    )

                    return

                raw = data_url_to_bytes(
                    image_data
                )

                array = np.frombuffer(
                    raw,
                    dtype=np.uint8
                )

                frame = cv2.imdecode(
                    array,
                    cv2.IMREAD_COLOR
                )

                if frame is None:

                    send_json(
                        self,
                        {
                            "ok": False,
                            "error":
                                "Could not decode image."
                        },
                        400
                    )

                    return

                model = YOLO(
                    MODEL_FILE
                )

                results = model.predict(
                    source=frame,
                    conf=0.35,
                    iou=0.50,
                    imgsz=416,
                    device="cpu",
                    verbose=False
                )

                detections = []

                for result in results:

                    boxes = result.boxes

                    if boxes is None:
                        continue

                    for box in boxes:

                        cls_id = int(
                            box.cls[0].item()
                        )

                        confidence = float(
                            box.conf[0].item()
                        )

                        if (
                            cls_id < 0
                            or cls_id >= len(CLASSES)
                        ):
                            continue

                        class_name = CLASSES[cls_id]

                        x1,y1,x2,y2 = box.xyxy[0].tolist()

                        center_y = (y1+y2)/2

                        line_position = int(
                                setting(
                                    "line_position",
                                    "55"
                                )
                            )

                        line_y = frame.shape[0] * (
                                line_position / 100
                            )

                        counted = 0

                        if class_name == \
                            "BUCKET_LOADED":

                            if center_y >= line_y:

                                counted = 1

                                update_daily(
                                    "loaded"
                                )

                        elif class_name == \
                            "BUCKET_EMPTY":

                            update_daily(
                                "empty"
                            )

                        elif class_name == \
                            "PEOPLE":

                            update_daily(
                                "people"
                            )

                        elif class_name == \
                            "EQUIPMENT":

                            update_daily(
                                "equipment"
                            )

                        save_detection(
                            status=class_name,
                            counted=counted,
                            confidence=confidence
                        )

                        detections.append({
                            "class_id": cls_id,
                            "class_name": class_name,
                            "confidence": confidence,
                            "x1": x1,
                            "y1": y1,
                            "x2": x2,
                            "y2": y2,
                            "counted": counted
                        })

                send_json(
                    self,
                    {
                        "ok": True,
                        "detections": detections
                    }
                )

                return

            send_json(
                self,
                {
                    "ok": False,
                    "error": "Not found"
                },
                404
            )

        except Exception as e:

            print(traceback.format_exc())

            send_json(
                self,
                {
                    "ok": False,
                    "error": str(e)
                },
                500
            )


# =========================================================
# STARTUP
# =========================================================

def main():

    print("====================================")
    print(" BUCKET COUNTER AI")
    print(" Underground production monitoring")
    print("====================================")

    print(
        "Database: Supabase PostgreSQL"
    )

    init_db()

    restored = restore_model_from_database()

    print(
        "YOLO:",
        "INSTALLED"
        if YOLO_AVAILABLE
        else "NOT INSTALLED"
    )

    print(
        "Model:",
        "READY"
        if restored or model_ready()
        else "NOT READY"
    )

    server = ThreadingHTTPServer(
        (HOST, PORT),
        Handler
    )

    print(
        f"Server running on {HOST}:{PORT}"
    )

    server.serve_forever()


if __name__ == "__main__":
    main()
