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
import random
from datetime import datetime
from urllib.parse import urlparse, parse_qs
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# ============================================================
# NEERIKA BUCKET AI
# Mining Production Bucket Counter
# FULL REPLACEMENT app.py
# ============================================================

# ------------------------------------------------------------
# ULTRALYTICS CONFIG
# ------------------------------------------------------------

os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/Ultralytics")

# ------------------------------------------------------------
# OPTIONAL DEPENDENCIES
# ------------------------------------------------------------

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
except Exception:
    psycopg2 = None
    RealDictCursor = None

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None


# ============================================================
# BASIC SETTINGS
# ============================================================

APP_NAME = "NEERIKA BUCKET AI"
APP_SUBTITLE = "Mining Production Bucket Counter"

PORT = int(os.environ.get("PORT", "8080"))

DATABASE_URL = os.environ.get("DATABASE_URL", "").strip()

# ------------------------------------------------------------
# DIRECTORY STRUCTURE
# ------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

AI_DIR = os.path.join(BASE_DIR, "ai")

DATASET_DIR = os.path.join(AI_DIR, "dataset")

IMAGES_DIR = os.path.join(DATASET_DIR, "images")
LABELS_DIR = os.path.join(DATASET_DIR, "labels")

TRAIN_IMAGES_DIR = os.path.join(IMAGES_DIR, "train")
VAL_IMAGES_DIR = os.path.join(IMAGES_DIR, "val")

TRAIN_LABELS_DIR = os.path.join(LABELS_DIR, "train")
VAL_LABELS_DIR = os.path.join(LABELS_DIR, "val")

MODELS_DIR = os.path.join(AI_DIR, "models")

UPLOAD_DIR = os.path.join(BASE_DIR, "bucket_images")

REFERENCE_DIR = os.path.join(BASE_DIR, "reference_images")

TEMP_DIR = os.path.join(BASE_DIR, "tmp")

YOLO_DATASET_YAML = os.path.join(DATASET_DIR, "dataset.yaml")

# ------------------------------------------------------------
# MODEL
# ------------------------------------------------------------

DEFAULT_MODEL = os.environ.get(
    "YOLO_MODEL",
    "yolo11n.pt"
)

TRAINED_MODEL = os.path.join(
    MODELS_DIR,
    "bucket_best.pt"
)

# ============================================================
# CREATE ALL REQUIRED DIRECTORIES
# ============================================================

REQUIRED_DIRECTORIES = [
    AI_DIR,
    DATASET_DIR,
    IMAGES_DIR,
    LABELS_DIR,
    TRAIN_IMAGES_DIR,
    VAL_IMAGES_DIR,
    TRAIN_LABELS_DIR,
    VAL_LABELS_DIR,
    MODELS_DIR,
    UPLOAD_DIR,
    REFERENCE_DIR,
    TEMP_DIR,
]

for directory in REQUIRED_DIRECTORIES:
    try:
        os.makedirs(directory, exist_ok=True)
    except Exception as e:
        print("DIRECTORY ERROR:", directory, e)


# ============================================================
# GLOBAL TRAINING STATE
# ============================================================

TRAINING_LOCK = threading.Lock()

TRAINING_STATE = {
    "status": "idle",
    "message": "Training has not started.",
    "progress": 0,
    "epoch": 0,
    "total_epochs": 0,
    "images": 0,
    "labels": 0,
    "started_at": None,
    "finished_at": None,
    "error": None
}


# ============================================================
# DATABASE
# ============================================================

def db_connection():
    if not DATABASE_URL:
        return None

    if psycopg2 is None:
        return None

    try:
        return psycopg2.connect(
            DATABASE_URL,
            connect_timeout=10
        )
    except Exception as e:
        print("DATABASE CONNECTION ERROR:", e)
        return None


def init_database():
    conn = db_connection()

    if conn is None:
        print("DATABASE_URL not available. Running without database.")
        return

    try:
        cur = conn.cursor()

        cur.execute("""
            CREATE TABLE IF NOT EXISTS bucket_counts (
                id BIGSERIAL PRIMARY KEY,
                bucket_type TEXT DEFAULT 'default',
                count INTEGER DEFAULT 1,
                shift TEXT DEFAULT '',
                operator_name TEXT DEFAULT '',
                source TEXT DEFAULT 'camera',
                confidence DOUBLE PRECISION DEFAULT 0,
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)

        conn.commit()
        cur.close()
        conn.close()

        print("Database initialized successfully.")

    except Exception as e:
        print("DATABASE INITIALIZATION ERROR:", e)
        try:
            conn.rollback()
            conn.close()
        except Exception:
            pass


# ============================================================
# DATABASE FUNCTIONS
# ============================================================

def save_bucket_count(
    bucket_type="default",
    count=1,
    shift="",
    operator_name="",
    source="camera",
    confidence=0
):
    conn = db_connection()

    if conn is None:
        return False

    try:
        cur = conn.cursor()

        cur.execute("""
            INSERT INTO bucket_counts
            (
                bucket_type,
                count,
                shift,
                operator_name,
                source,
                confidence
            )
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (
            bucket_type,
            int(count),
            shift,
            operator_name,
            source,
            float(confidence)
        ))

        conn.commit()

        cur.close()
        conn.close()

        return True

    except Exception as e:
        print("SAVE COUNT ERROR:", e)

        try:
            conn.rollback()
            conn.close()
        except Exception:
            pass

        return False


def get_bucket_history(limit=200):
    conn = db_connection()

    if conn is None:
        return []

    try:
        cur = conn.cursor(cursor_factory=RealDictCursor)

        cur.execute("""
            SELECT
                id,
                bucket_type,
                count,
                shift,
                operator_name,
                source,
                confidence,
                created_at
            FROM bucket_counts
            ORDER BY created_at DESC
            LIMIT %s
        """, (int(limit),))

        rows = cur.fetchall()

        result = []

        for row in rows:
            item = dict(row)

            if item.get("created_at"):
                item["created_at"] = item["created_at"].isoformat()

            result.append(item)

        cur.close()
        conn.close()

        return result

    except Exception as e:
        print("HISTORY ERROR:", e)

        try:
            conn.close()
        except Exception:
            pass

        return []


def get_total_count():
    conn = db_connection()

    if conn is None:
        return 0

    try:
        cur = conn.cursor()

        cur.execute("""
            SELECT COALESCE(SUM(count), 0)
            FROM bucket_counts
        """)

        value = cur.fetchone()[0]

        cur.close()
        conn.close()

        return int(value or 0)

    except Exception as e:
        print("TOTAL COUNT ERROR:", e)

        try:
            conn.close()
        except Exception:
            pass

        return 0


# ============================================================
# DATASET FUNCTIONS
# ============================================================

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp"
}


def is_image(filename):
    extension = os.path.splitext(filename)[1].lower()
    return extension in IMAGE_EXTENSIONS


def list_images(directory):
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        return []

    result = []

    for filename in os.listdir(directory):
        path = os.path.join(directory, filename)

        if os.path.isfile(path) and is_image(filename):
            result.append(path)

    return result


def list_labels(directory):
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        return []

    result = []

    for filename in os.listdir(directory):
        if filename.lower().endswith(".txt"):
            path = os.path.join(directory, filename)

            if os.path.isfile(path):
                result.append(path)

    return result


def dataset_statistics():

    train_images = list_images(TRAIN_IMAGES_DIR)
    val_images = list_images(VAL_IMAGES_DIR)

    train_labels = list_labels(TRAIN_LABELS_DIR)
    val_labels = list_labels(VAL_LABELS_DIR)

    all_images = train_images + val_images
    all_labels = train_labels + val_labels

    return {
        "train_images": len(train_images),
        "val_images": len(val_images),
        "train_labels": len(train_labels),
        "val_labels": len(val_labels),
        "images": len(all_images),
        "labels": len(all_labels)
    }


def make_dataset_yaml():

    os.makedirs(DATASET_DIR, exist_ok=True)

    # Use forward slashes because YOLO handles them reliably.
    train_path = TRAIN_IMAGES_DIR.replace("\\", "/")
    val_path = VAL_IMAGES_DIR.replace("\\", "/")

    yaml_text = f"""path: {DATASET_DIR.replace("\\", "/")}
train: {train_path}
val: {val_path}

names:
  0: BUCKET_LOADED
"""

    with open(
        YOLO_DATASET_YAML,
        "w",
        encoding="utf-8"
    ) as f:
        f.write(yaml_text)

    return YOLO_DATASET_YAML


def validate_dataset():

    stats = dataset_statistics()

    if stats["images"] == 0:
        return False, (
            "No training images found. "
            "Upload at least one bucket image first."
        ), stats

    if stats["labels"] == 0:
        return False, (
            "No YOLO labels found. "
            "Images must be annotated before training."
        ), stats

    # Check matching label/image names.
    image_names = set()

    for directory in [TRAIN_IMAGES_DIR, VAL_IMAGES_DIR]:
        for path in list_images(directory):
            image_names.add(
                os.path.splitext(
                    os.path.basename(path)
                )[0]
            )

    label_names = set()

    for directory in [TRAIN_LABELS_DIR, VAL_LABELS_DIR]:
        for path in list_labels(directory):
            label_names.add(
                os.path.splitext(
                    os.path.basename(path)
                )[0]
            )

    matched = image_names.intersection(label_names)

    if len(matched) == 0:
        return False, (
            "Images were found, but no image has a matching YOLO "
            "label file. Example: bucket001.jpg must have "
            "bucket001.txt."
        ), stats

    return True, "Dataset is ready for training.", stats


# ============================================================
# SAVE UPLOADED TRAINING IMAGE
# ============================================================

def save_training_image(filename, data):

    if not filename:
        filename = "image.jpg"

    original_name = os.path.basename(filename)

    extension = os.path.splitext(original_name)[1].lower()

    if extension not in IMAGE_EXTENSIONS:
        extension = ".jpg"

    safe_name = (
        datetime.now().strftime("%Y%m%d_%H%M%S")
        + "_"
        + uuid.uuid4().hex[:8]
        + extension
    )

    path = os.path.join(
        TRAIN_IMAGES_DIR,
        safe_name
    )

    with open(path, "wb") as f:
        f.write(data)

    return safe_name, path


# ============================================================
# REFERENCE BUCKET IMAGES
# ============================================================

def save_reference_image(filename, data):

    extension = os.path.splitext(
        os.path.basename(filename)
    )[1].lower()

    if extension not in IMAGE_EXTENSIONS:
        extension = ".jpg"

    safe_name = (
        datetime.now().strftime("%Y%m%d_%H%M%S")
        + "_"
        + uuid.uuid4().hex[:8]
        + extension
    )

    path = os.path.join(
        REFERENCE_DIR,
        safe_name
    )

    with open(path, "wb") as f:
        f.write(data)

    return safe_name


def list_reference_images():

    if not os.path.exists(REFERENCE_DIR):
        os.makedirs(REFERENCE_DIR, exist_ok=True)

    result = []

    for filename in os.listdir(REFERENCE_DIR):

        path = os.path.join(
            REFERENCE_DIR,
            filename
        )

        if os.path.isfile(path) and is_image(filename):

            result.append({
                "name": filename,
                "url": "/reference/" + filename
            })

    return result


# ============================================================
# YOLO TRAINING
# ============================================================

def update_training_state(**kwargs):

    with TRAINING_LOCK:
        TRAINING_STATE.update(kwargs)


def run_yolo_training(epochs=20):

    try:

        update_training_state(
            status="preparing",
            message="Preparing YOLO dataset...",
            progress=1,
            epoch=0,
            total_epochs=epochs,
            error=None
        )

        valid, message, stats = validate_dataset()

        update_training_state(
            images=stats["images"],
            labels=stats["labels"]
        )

        if not valid:
            update_training_state(
                status="error",
                message=message,
                progress=0,
                error=message,
                finished_at=datetime.now().isoformat()
            )
            return

        if YOLO is None:
            message = (
                "Ultralytics YOLO is not installed. "
                "Add ultralytics to requirements.txt."
            )

            update_training_state(
                status="error",
                message=message,
                progress=0,
                error=message,
                finished_at=datetime.now().isoformat()
            )
            return

        make_dataset_yaml()

        update_training_state(
            status="loading_model",
            message="Loading YOLO model...",
            progress=2
        )

        model_path = DEFAULT_MODEL

        # If trained model already exists, use it.
        if os.path.exists(TRAINED_MODEL):
            model_path = TRAINED_MODEL

        print("Loading model:", model_path)

        model = YOLO(model_path)

        update_training_state(
            status="training",
            message=f"Training YOLO for {epochs} epochs...",
            progress=5
        )

        # ----------------------------------------------------
        # TRAIN
        # ----------------------------------------------------

        results = model.train(
            data=YOLO_DATASET_YAML,
            epochs=int(epochs),
            imgsz=640,
            batch=4,
            workers=0,
            project=MODELS_DIR,
            name="bucket_training",
            exist_ok=True,
            pretrained=True,
            verbose=True
        )

        # ----------------------------------------------------
        # FIND BEST MODEL
        # ----------------------------------------------------

        possible_best = os.path.join(
            MODELS_DIR,
            "bucket_training",
            "weights",
            "best.pt"
        )

        if os.path.exists(possible_best):

            shutil.copy2(
                possible_best,
                TRAINED_MODEL
            )

        update_training_state(
            status="completed",
            message="YOLO training completed successfully.",
            progress=100,
            epoch=epochs,
            total_epochs=epochs,
            finished_at=datetime.now().isoformat(),
            error=None
        )

        print("YOLO TRAINING COMPLETED.")

    except Exception as e:

        error_text = traceback.format_exc()

        print("YOLO TRAINING ERROR:")
        print(error_text)

        update_training_state(
            status="error",
            message=str(e),
            progress=0,
            error=str(e),
            finished_at=datetime.now().isoformat()
        )


def start_training(epochs=20):

    with TRAINING_LOCK:

        if TRAINING_STATE["status"] in [
            "preparing",
            "loading_model",
            "training"
        ]:

            return False, "Training is already running."

        TRAINING_STATE.update({
            "status": "preparing",
            "message": "Training started.",
            "progress": 0,
            "epoch": 0,
            "total_epochs": int(epochs),
            "started_at": datetime.now().isoformat(),
            "finished_at": None,
            "error": None
        })

    thread = threading.Thread(
        target=run_yolo_training,
        args=(int(epochs),),
        daemon=True
    )

    thread.start()

    return True, "Training started."


# ============================================================
# HTTP HELPERS
# ============================================================

def json_response(handler, data, status=200):

    body = json.dumps(
        data,
        ensure_ascii=False,
        default=str
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

    handler.send_header(
        "Cache-Control",
        "no-cache"
    )

    handler.end_headers()

    handler.wfile.write(body)


def html_response(handler, html, status=200):

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


def text_response(handler, text, status=200):

    body = text.encode("utf-8")

    handler.send_response(status)

    handler.send_header(
        "Content-Type",
        "text/plain; charset=utf-8"
    )

    handler.send_header(
        "Content-Length",
        str(len(body))
    )

    handler.end_headers()

    handler.wfile.write(body)


def not_found(handler):

    json_response(
        handler,
        {
            "ok": False,
            "error": "Not found"
        },
        404
    )


# ============================================================
# MULTIPART PARSER
# ============================================================

def parse_multipart(handler):

    content_type = handler.headers.get(
        "Content-Type",
        ""
    )

    if "multipart/form-data" not in content_type:
        return {}

    boundary = None

    parts = content_type.split(";")

    for part in parts:

        part = part.strip()

        if part.startswith("boundary="):

            boundary = part.split(
                "=",
                1
            )[1]

            boundary = boundary.strip('"')

    if not boundary:
        return {}

    length = int(
        handler.headers.get(
            "Content-Length",
            "0"
        )
    )

    raw = handler.rfile.read(length)

    boundary_bytes = (
        b"--" + boundary.encode()
    )

    result = {}

    for block in raw.split(boundary_bytes):

        if not block:
            continue

        if block in [b"--", b"--\r\n"]:
            continue

        block = block.strip(b"\r\n-")

        header_end = block.find(
            b"\r\n\r\n"
        )

        if header_end == -1:
            continue

        header_data = block[
            :header_end
        ].decode(
            "utf-8",
            errors="ignore"
        )

        file_data = block[
            header_end + 4:
        ]

        disposition = ""

        for line in header_data.split(
            "\r\n"
        ):

            if line.lower().startswith(
                "content-disposition:"
            ):

                disposition = line

        name = None
        filename = None

        if 'name="' in disposition:

            name = disposition.split(
                'name="',
                1
            )[1].split(
                '"',
                1
            )[0]

        if 'filename="' in disposition:

            filename = disposition.split(
                'filename="',
                1
            )[1].split(
                '"',
                1
            )[0]

        if name:

            result[name] = {
                "filename": filename,
                "data": file_data
            }

    return result


# ============================================================
# HTML PAGE
# ============================================================

HTML_PAGE = r"""
<!DOCTYPE html>
<html lang="en">
<head>

<meta charset="UTF-8">

<meta name="viewport"
      content="width=device-width, initial-scale=1.0">

<title>NEERIKA BUCKET AI</title>

<style>

* {
    box-sizing: border-box;
}

body {
    margin: 0;
    font-family: Arial, sans-serif;
    background: #f4f6f8;
    color: #222;
}

header {
    background: #111827;
    color: white;
    padding: 18px;
}

header h1 {
    margin: 0;
    font-size: 24px;
}

header p {
    margin: 5px 0 0;
    color: #cbd5e1;
}

nav {
    display: flex;
    overflow-x: auto;
    background: white;
    border-bottom: 1px solid #ddd;
}

nav button {
    border: 0;
    background: white;
    padding: 14px 18px;
    cursor: pointer;
    font-weight: bold;
}

nav button.active {
    background: #111827;
    color: white;
}

main {
    padding: 18px;
    max-width: 1200px;
    margin: auto;
}

.section {
    display: none;
}

.section.active {
    display: block;
}

.card {
    background: white;
    padding: 18px;
    margin-bottom: 18px;
    border-radius: 12px;
    box-shadow: 0 2px 8px rgba(0,0,0,.08);
}

.stat-grid {
    display: grid;
    grid-template-columns:
        repeat(auto-fit, minmax(180px, 1fr));
    gap: 15px;
}

.stat {
    background: white;
    border-radius: 12px;
    padding: 20px;
    box-shadow: 0 2px 8px rgba(0,0,0,.08);
}

.stat h3 {
    margin: 0;
    color: #64748b;
}

.stat strong {
    display: block;
    font-size: 32px;
    margin-top: 8px;
}

button,
input,
select {
    font-size: 16px;
}

button.primary {
    background: #111827;
    color: white;
    border: 0;
    border-radius: 8px;
    padding: 12px 18px;
    cursor: pointer;
}

button.primary:hover {
    opacity: .9;
}

input[type="file"] {
    margin: 10px 0;
}

.status {
    padding: 14px;
    border-radius: 8px;
    background: #f1f5f9;
    margin-top: 12px;
}

.progress {
    width: 100%;
    height: 20px;
    background: #e5e7eb;
    border-radius: 10px;
    overflow: hidden;
    margin-top: 10px;
}

.progress-bar {
    height: 100%;
    width: 0%;
    background: #111827;
    transition: width .3s;
}

table {
    width: 100%;
    border-collapse: collapse;
}

th,
td {
    padding: 10px;
    border-bottom: 1px solid #ddd;
    text-align: left;
}

.preview-grid {
    display: grid;
    grid-template-columns:
        repeat(auto-fill, minmax(160px, 1fr));
    gap: 12px;
}

.preview {
    background: #f8fafc;
    border-radius: 8px;
    padding: 8px;
}

.preview img {
    width: 100%;
    height: 130px;
    object-fit: cover;
    border-radius: 6px;
}

.success {
    color: #166534;
}

.error {
    color: #b91c1c;
}

.warning {
    color: #92400e;
}

footer {
    text-align: center;
    color: #64748b;
    padding: 30px;
}

</style>

</head>

<body>

<header>

<h1>NEERIKA BUCKET AI</h1>

<p>Mining Production Bucket Counter</p>

</header>

<nav>

<button onclick="showSection('dashboard', this)"
        class="active">
Dashboard
</button>

<button onclick="showSection('camera', this)">
Camera
</button>

<button onclick="showSection('buckets', this)">
Buckets
</button>

<button onclick="showSection('training', this)">
Training
</button>

<button onclick="showSection('history', this)">
History
</button>

<button onclick="showSection('settings', this)">
Settings
</button>

</nav>

<main>

<!-- ===================================================== -->
<!-- DASHBOARD -->
<!-- ===================================================== -->

<section id="dashboard"
         class="section active">

<div class="stat-grid">

<div class="stat">
<h3>Total Buckets</h3>
<strong id="totalCount">0</strong>
</div>

<div class="stat">
<h3>Training Images</h3>
<strong id="imageCount">0</strong>
</div>

<div class="stat">
<h3>YOLO Labels</h3>
<strong id="labelCount">0</strong>
</div>

<div class="stat">
<h3>Training Status</h3>
<strong id="dashboardStatus">
Idle
</strong>
</div>

</div>

<div class="card">

<h2>System Status</h2>

<p id="systemMessage">
Loading...
</p>

</div>

</section>


<!-- ===================================================== -->
<!-- CAMERA -->
<!-- ===================================================== -->

<section id="camera"
         class="section">

<div class="card">

<h2>Camera</h2>

<video id="cameraVideo"
       autoplay
       playsinline
       style="width:100%;max-width:700px;">
</video>

<br><br>

<button class="primary"
        onclick="startCamera()">
Start Camera
</button>

</div>

</section>


<!-- ===================================================== -->
<!-- BUCKETS -->
<!-- ===================================================== -->

<section id="buckets"
         class="section">

<div class="card">

<h2>Bucket Reference Images</h2>

<p>
Upload reference photos of the loaded bucket type.
</p>

<input type="file"
       id="referenceFile"
       accept="image/*">

<br>

<button class="primary"
        onclick="uploadReference()">
Upload Reference Image
</button>

<div id="referenceMessage"
     class="status">
</div>

</div>

<div class="card">

<h3>Reference Images</h3>

<div id="referenceGrid"
     class="preview-grid">
</div>

</div>

</section>


<!-- ===================================================== -->
<!-- TRAINING -->
<!-- ===================================================== -->

<section id="training"
         class="section">

<div class="card">

<h2>YOLO Training Dataset</h2>

<p>
Upload images for training.
</p>

<input type="file"
       id="trainingFile"
       accept="image/*">

<br>

<button class="primary"
        onclick="uploadTrainingImage()">
Upload Image
</button>

<div id="uploadMessage"
     class="status">
</div>

</div>


<div class="card">

<h2>Dataset Information</h2>

<table>

<tr>
<th>Training Images</th>
<td id="trainImages">0</td>
</tr>

<tr>
<th>Validation Images</th>
<td id="valImages">0</td>
</tr>

<tr>
<th>Training Labels</th>
<td id="trainLabels">0</td>
</tr>

<tr>
<th>Validation Labels</th>
<td id="valLabels">0</td>
</tr>

<tr>
<th>Total Images</th>
<td id="allImages">0</td>
</tr>

<tr>
<th>Total Labels</th>
<td id="allLabels">0</td>
</tr>

</table>

</div>


<div class="card">

<h2>Training Status</h2>

<div id="trainingStatus"
     class="status">

Status: idle

</div>

<div class="progress">

<div id="progressBar"
     class="progress-bar">
</div>

</div>

<p id="progressText">
0%
</p>

</div>


<div class="card">

<h2>Start YOLO Training</h2>

<label>
Epochs:
</label>

<select id="epochs">

<option value="10">10</option>
<option value="20" selected>20</option>
<option value="30">30</option>
<option value="50">50</option>

</select>

<br><br>

<button class="primary"
        onclick="startTraining()">

Start YOLO Training

</button>

</div>

</section>


<!-- ===================================================== -->
<!-- HISTORY -->
<!-- ===================================================== -->

<section id="history"
         class="section">

<div class="card">

<h2>Bucket History</h2>

<button class="primary"
        onclick="loadHistory()">

Refresh History

</button>

<br><br>

<div style="overflow-x:auto;">

<table>

<thead>

<tr>
<th>ID</th>
<th>Bucket Type</th>
<th>Count</th>
<th>Shift</th>
<th>Operator</th>
<th>Confidence</th>
<th>Date</th>
</tr>

</thead>

<tbody id="historyBody">
</tbody>

</table>

</div>

</div>

</section>


<!-- ===================================================== -->
<!-- SETTINGS -->
<!-- ===================================================== -->

<section id="settings"
         class="section">

<div class="card">

<h2>Settings</h2>

<p>
Application: NEERIKA BUCKET AI
</p>

<p>
Model: YOLO
</p>

<p>
Detection class: BUCKET_LOADED
</p>

<p>
Database:
<span id="databaseStatus">
Checking...
</span>
</p>

</div>

</section>

</main>

<footer>

Geology & Mining Services

</footer>


<script>

let cameraStream = null;


/* ========================================================
   NAVIGATION
======================================================== */

function showSection(id, button) {

    document
        .querySelectorAll(".section")
        .forEach(
            section => section.classList.remove("active")
        );

    document
        .getElementById(id)
        .classList.add("active");

    document
        .querySelectorAll("nav button")
        .forEach(
            b => b.classList.remove("active")
        );

    if (button) {
        button.classList.add("active");
    }

    if (id === "training") {
        loadDataset();
        loadTrainingStatus();
    }

    if (id === "history") {
        loadHistory();
    }

    if (id === "buckets") {
        loadReferences();
    }
}


/* ========================================================
   DASHBOARD
======================================================== */

async function loadDashboard() {

    try {

        const response =
            await fetch("/api/dashboard");

        const data =
            await response.json();

        document.getElementById(
            "totalCount"
        ).textContent = data.total_count;

        document.getElementById(
            "imageCount"
        ).textContent = data.dataset.images;

        document.getElementById(
            "labelCount"
        ).textContent = data.dataset.labels;

        document.getElementById(
            "dashboardStatus"
        ).textContent =
            data.training.status;

        document.getElementById(
            "systemMessage"
        ).textContent =
            data.message;

    } catch (error) {

        document.getElementById(
            "systemMessage"
        ).textContent =
            "Could not connect to server.";

    }

}


/* ========================================================
   DATASET
======================================================== */

async function loadDataset() {

    try {

        const response =
            await fetch("/api/dataset");

        const data =
            await response.json();

        document.getElementById(
            "trainImages"
        ).textContent =
            data.train_images;

        document.getElementById(
            "valImages"
        ).textContent =
            data.val_images;

        document.getElementById(
            "trainLabels"
        ).textContent =
            data.train_labels;

        document.getElementById(
            "valLabels"
        ).textContent =
            data.val_labels;

        document.getElementById(
            "allImages"
        ).textContent =
            data.images;

        document.getElementById(
            "allLabels"
        ).textContent =
            data.labels;

        document.getElementById(
            "imageCount"
        ).textContent =
            data.images;

        document.getElementById(
            "labelCount"
        ).textContent =
            data.labels;

    } catch (error) {

        console.error(error);

    }

}


/* ========================================================
   UPLOAD TRAINING IMAGE
======================================================== */

async function uploadTrainingImage() {

    const fileInput =
        document.getElementById(
            "trainingFile"
        );

    const message =
        document.getElementById(
            "uploadMessage"
        );

    if (!fileInput.files.length) {

        message.innerHTML =
            '<span class="error">' +
            'Please choose an image first.' +
            '</span>';

        return;
    }

    const formData =
        new FormData();

    formData.append(
        "image",
        fileInput.files[0]
    );

    message.textContent =
        "Uploading image...";

    try {

        const response =
            await fetch(
                "/api/training/upload",
                {
                    method: "POST",
                    body: formData
                }
            );

        const data =
            await response.json();

        if (!data.ok) {

            message.innerHTML =
                '<span class="error">' +
                data.error +
                '</span>';

            return;
        }

        message.innerHTML =
            '<span class="success">' +
            "Image uploaded successfully: " +
            data.filename +
            '</span>';

        fileInput.value = "";

        loadDataset();
        loadDashboard();

    } catch (error) {

        message.innerHTML =
            '<span class="error">' +
            error.message +
            '</span>';

    }

}


/* ========================================================
   TRAINING STATUS
======================================================== */

async function loadTrainingStatus() {

    try {

        const response =
            await fetch("/api/training/status");

        const data =
            await response.json();

        const status =
            document.getElementById(
                "trainingStatus"
            );

        status.innerHTML =
            "<strong>Status:</strong> " +
            data.status +
            "<br><br>" +
            "<strong>Message:</strong> " +
            data.message;

        if (data.error) {

            status.innerHTML +=
                "<br><br><span class='error'>" +
                data.error +
                "</span>";

        }

        const progress =
            Number(data.progress || 0);

        document.getElementById(
            "progressBar"
        ).style.width =
            progress + "%";

        document.getElementById(
            "progressText"
        ).textContent =
            progress + "%";

        document.getElementById(
            "dashboardStatus"
        ).textContent =
            data.status;

    } catch (error) {

        console.error(error);

    }

}


/* ========================================================
   START TRAINING
======================================================== */

async function startTraining() {

    const epochs =
        document.getElementById(
            "epochs"
        ).value;

    const status =
        document.getElementById(
            "trainingStatus"
        );

    status.textContent =
        "Checking dataset...";

    try {

        const check =
            await fetch(
                "/api/training/check"
            );

        const checkData =
            await check.json();

        if (!checkData.ok) {

            status.innerHTML =
                "<span class='error'>" +
                checkData.message +
                "</span>";

            loadDataset();

            return;
        }

        status.textContent =
            "Starting YOLO training...";

        const response =
            await fetch(
                "/api/training/start",
                {
                    method: "POST",
                    headers: {
                        "Content-Type":
                            "application/json"
                    },
                    body: JSON.stringify({
                        epochs:
                            Number(epochs)
                    })
                }
            );

        const data =
            await response.json();

        status.textContent =
            data.message;

        loadTrainingStatus();

    } catch (error) {

        status.innerHTML =
            "<span class='error'>" +
            error.message +
            "</span>";

    }

}


/* ========================================================
   REFERENCE IMAGES
======================================================== */

async function uploadReference() {

    const fileInput =
        document.getElementById(
            "referenceFile"
        );

    const message =
        document.getElementById(
            "referenceMessage"
        );

    if (!fileInput.files.length) {

        message.textContent =
            "Choose an image first.";

        return;
    }

    const formData =
        new FormData();

    formData.append(
        "image",
        fileInput.files[0]
    );

    message.textContent =
        "Uploading...";

    try {

        const response =
            await fetch(
                "/api/reference/upload",
                {
                    method: "POST",
                    body: formData
                }
            );

        const data =
            await response.json();

        if (!data.ok) {

            message.textContent =
                data.error;

            return;
        }

        message.textContent =
            "Reference image uploaded.";

        fileInput.value = "";

        loadReferences();

    } catch (error) {

        message.textContent =
            error.message;

    }

}


async function loadReferences() {

    try {

        const response =
            await fetch(
                "/api/reference/list"
            );

        const data =
            await response.json();

        const grid =
            document.getElementById(
                "referenceGrid"
            );

        grid.innerHTML = "";

        data.images.forEach(
            image => {

                const div =
                    document.createElement(
                        "div"
                    );

                div.className =
                    "preview";

                div.innerHTML =
                    `
                    <img src="${image.url}">
                    <div>${image.name}</div>
                    `;

                grid.appendChild(div);

            }
        );

    } catch (error) {

        console.error(error);

    }

}


/* ========================================================
   HISTORY
======================================================== */

async function loadHistory() {

    try {

        const response =
            await fetch(
                "/api/history"
            );

        const data =
            await response.json();

        const body =
            document.getElementById(
                "historyBody"
            );

        body.innerHTML = "";

        data.history.forEach(
            item => {

                const row =
                    document.createElement(
                        "tr"
                    );

                row.innerHTML =
                    `
                    <td>${item.id}</td>
                    <td>${item.bucket_type || ""}</td>
                    <td>${item.count || 0}</td>
                    <td>${item.shift || ""}</td>
                    <td>${item.operator_name || ""}</td>
                    <td>${Number(item.confidence || 0).toFixed(3)}</td>
                    <td>${item.created_at || ""}</td>
                    `;

                body.appendChild(row);

            }
        );

    } catch (error) {

        console.error(error);

    }

}


/* ========================================================
   CAMERA
======================================================== */

async function startCamera() {

    try {

        if (cameraStream) {

            cameraStream
                .getTracks()
                .forEach(
                    track => track.stop()
                );

        }

        cameraStream =
            await navigator.mediaDevices
                .getUserMedia({
                    video: {
                        facingMode:
                            "environment"
                    },
                    audio: false
                });

        document.getElementById(
            "cameraVideo"
        ).srcObject =
            cameraStream;

    } catch (error) {

        alert(
            "Camera error: " +
            error.message
        );

    }

}


/* ========================================================
   AUTO REFRESH
======================================================== */

setInterval(
    loadTrainingStatus,
    3000
);

setInterval(
    loadDashboard,
    5000
);


/* ========================================================
   INITIAL LOAD
======================================================== */

loadDashboard();
loadDataset();
loadTrainingStatus();

</script>

</body>
</html>
"""


# ============================================================
# HTTP HANDLER
# ============================================================

class Handler(BaseHTTPRequestHandler):

    server_version = "NEERIKA-BUCKET-AI/1.0"

    def log_message(self, format, *args):
        print(
            "%s - %s"
            % (
                self.address_string(),
                format % args
            )
        )

    # --------------------------------------------------------
    # GET
    # --------------------------------------------------------

    def do_GET(self):

        parsed = urlparse(
            self.path
        )

        path = parsed.path

        # Main page
        if path == "/":

            html_response(
                self,
                HTML_PAGE
            )

            return

        # Dashboard
        if path == "/api/dashboard":

            stats = dataset_statistics()

            json_response(
                self,
                {
                    "ok": True,
                    "total_count":
                        get_total_count(),
                    "dataset":
                        stats,
                    "training":
                        dict(TRAINING_STATE),
                    "message":
                        "NEERIKA BUCKET AI is running."
                }
            )

            return

        # Dataset
        if path == "/api/dataset":

            json_response(
                self,
                dataset_statistics()
            )

            return

        # Training status
        if path == "/api/training/status":

            with TRAINING_LOCK:
                state = dict(
                    TRAINING_STATE
                )

            json_response(
                self,
                state
            )

            return

        # Training check
        if path == "/api/training/check":

            valid, message, stats = (
                validate_dataset()
            )

            json_response(
                self,
                {
                    "ok": valid,
                    "message": message,
                    "dataset": stats
                }
            )

            return

        # History
        if path == "/api/history":

            json_response(
                self,
                {
                    "ok": True,
                    "history":
                        get_bucket_history()
                }
            )

            return

        # Reference list
        if path == "/api/reference/list":

            json_response(
                self,
                {
                    "ok": True,
                    "images":
                        list_reference_images()
                }
            )

            return

        # Reference image
        if path.startswith(
            "/reference/"
        ):

            filename = os.path.basename(
                path[len("/reference/"):]
            )

            file_path = os.path.join(
                REFERENCE_DIR,
                filename
            )

            if not os.path.isfile(
                file_path
            ):

                not_found(self)
                return

            self.serve_file(
                file_path
            )

            return

        # Model download/test
        if path == "/api/model/status":

            json_response(
                self,
                {
                    "ok": True,
                    "ultralytics_installed":
                        YOLO is not None,
                    "trained_model_exists":
                        os.path.exists(
                            TRAINED_MODEL
                        ),
                    "trained_model":
                        TRAINED_MODEL
                }
            )

            return

        not_found(self)

    # --------------------------------------------------------
    # POST
    # --------------------------------------------------------

    def do_POST(self):

        parsed = urlparse(
            self.path
        )

        path = parsed.path

        # ----------------------------------------------------
        # TRAINING IMAGE UPLOAD
        # ----------------------------------------------------

        if path == "/api/training/upload":

            try:

                fields = parse_multipart(
                    self
                )

                image = fields.get(
                    "image"
                )

                if not image:
                    json_response(
                        self,
                        {
                            "ok": False,
                            "error":
                                "No image uploaded."
                        },
                        400
                    )
                    return

                filename = (
                    image.get("filename")
                    or "image.jpg"
                )

                data = image.get(
                    "data",
                    b""
                )

                if not is_image(filename):

                    json_response(
                        self,
                        {
                            "ok": False,
                            "error":
                                "Only image files are allowed."
                        },
                        400
                    )
                    return

                if not data:

                    json_response(
                        self,
                        {
                            "ok": False,
                            "error":
                                "Uploaded image is empty."
                        },
                        400
                    )
                    return

                saved_name, saved_path = (
                    save_training_image(
                        filename,
                        data
                    )
                )

                # Automatically ensure dataset.yaml exists.
                make_dataset_yaml()

                json_response(
                    self,
                    {
                        "ok": True,
                        "filename":
                            saved_name,
                        "path":
                            saved_path,
                        "message":
                            "Training image uploaded successfully.",
                        "dataset":
                            dataset_statistics()
                    }
                )

                return

            except Exception as e:

                print(
                    traceback.format_exc()
                )

                json_response(
                    self,
                    {
                        "ok": False,
                        "error": str(e)
                    },
                    500
                )

                return

        # ----------------------------------------------------
        # REFERENCE IMAGE UPLOAD
        # ----------------------------------------------------

        if path == "/api/reference/upload":

            try:

                fields = parse_multipart(
                    self
                )

                image = fields.get(
                    "image"
                )

                if not image:

                    json_response(
                        self,
                        {
                            "ok": False,
                            "error":
                                "No image uploaded."
                        },
                        400
                    )

                    return

                filename = (
                    image.get("filename")
                    or "reference.jpg"
                )

                data = image.get(
                    "data",
                    b""
                )

                if not is_image(filename):

                    json_response(
                        self,
                        {
                            "ok": False,
                            "error":
                                "Only image files are allowed."
                        },
                        400
                    )

                    return

                saved_name = (
                    save_reference_image(
                        filename,
                        data
                    )
                )

                json_response(
                    self,
                    {
                        "ok": True,
                        "filename":
                            saved_name
                    }
                )

                return

            except Exception as e:

                json_response(
                    self,
                    {
                        "ok": False,
                        "error": str(e)
                    },
                    500
                )

                return

        # ----------------------------------------------------
        # START TRAINING
        # ----------------------------------------------------

        if path == "/api/training/start":

            try:

                content_length = int(
                    self.headers.get(
                        "Content-Length",
                        "0"
                    )
                )

                raw = self.rfile.read(
                    content_length
                )

                payload = {}

                if raw:

                    try:
                        payload = json.loads(
                            raw.decode(
                                "utf-8"
                            )
                        )
                    except Exception:
                        payload = {}

                epochs = int(
                    payload.get(
                        "epochs",
                        20
                    )
                )

                if epochs < 1:
                    epochs = 1

                if epochs > 100:
                    epochs = 100

                # IMPORTANT:
                # Validate before saying training started.
                valid, message, stats = (
                    validate_dataset()
                )

                if not valid:

                    update_training_state(
                        status="error",
                        message=message,
                        progress=0,
                        error=message
                    )

                    json_response(
                        self,
                        {
                            "ok": False,
                            "message": message,
                            "dataset": stats
                        },
                        400
                    )

                    return

                started, start_message = (
                    start_training(
                        epochs
                    )
                )

                if not started:

                    json_response(
                        self,
                        {
                            "ok": False,
                            "message":
                                start_message
                        },
                        409
                    )

                    return

                json_response(
                    self,
                    {
                        "ok": True,
                        "message":
                            start_message,
                        "epochs":
                            epochs
                    }
                )

                return

            except Exception as e:

                print(
                    traceback.format_exc()
                )

                json_response(
                    self,
                    {
                        "ok": False,
                        "message":
                            str(e)
                    },
                    500
                )

                return

        # ----------------------------------------------------
        # SAVE BUCKET COUNT
        # ----------------------------------------------------

        if path == "/api/count":

            try:

                content_length = int(
                    self.headers.get(
                        "Content-Length",
                        "0"
                    )
                )

                raw = self.rfile.read(
                    content_length
                )

                payload = json.loads(
                    raw.decode("utf-8")
                )

                bucket_type = payload.get(
                    "bucket_type",
                    "default"
                )

                count = int(
                    payload.get(
                        "count",
                        1
                    )
                )

                shift = payload.get(
                    "shift",
                    ""
                )

                operator_name = payload.get(
                    "operator_name",
                    ""
                )

                source = payload.get(
                    "source",
                    "camera"
                )

                confidence = float(
                    payload.get(
                        "confidence",
                        0
                    )
                )

                success = save_bucket_count(
                    bucket_type=
                        bucket_type,
                    count=count,
                    shift=shift,
                    operator_name=
                        operator_name,
                    source=source,
                    confidence=
                        confidence
                )

                json_response(
                    self,
                    {
                        "ok": success,
                        "message":
                            (
                                "Count saved."
                                if success
                                else
                                "Could not save count."
                            )
                    }
                )

                return

            except Exception as e:

                json_response(
                    self,
                    {
                        "ok": False,
                        "error": str(e)
                    },
                    500
                )

                return

        not_found(self)

    # --------------------------------------------------------
    # SERVE FILE
    # --------------------------------------------------------

    def serve_file(self, file_path):

        extension = os.path.splitext(
            file_path
        )[1].lower()

        content_types = {
            ".jpg":
                "image/jpeg",
            ".jpeg":
                "image/jpeg",
            ".png":
                "image/png",
            ".webp":
                "image/webp",
            ".bmp":
                "image/bmp",
            ".pt":
                "application/octet-stream"
        }

        content_type = content_types.get(
            extension,
            "application/octet-stream"
        )

        try:

            with open(
                file_path,
                "rb"
            ) as f:

                data = f.read()

            self.send_response(200)

            self.send_header(
                "Content-Type",
                content_type
            )

            self.send_header(
                "Content-Length",
                str(len(data))
            )

            self.end_headers()

            self.wfile.write(data)

        except Exception:

            not_found(self)


# ============================================================
# START SERVER
# ============================================================

def main():

    print("=" * 60)
    print("NEERIKA BUCKET AI")
    print("Mining Production Bucket Counter")
    print("=" * 60)

    print("BASE_DIR:", BASE_DIR)

    print("AI_DIR:", AI_DIR)

    print("DATASET_DIR:", DATASET_DIR)

    print(
        "TRAIN_IMAGES_DIR:",
        TRAIN_IMAGES_DIR
    )

    print(
        "TRAIN_LABELS_DIR:",
        TRAIN_LABELS_DIR
    )

    print(
        "VAL_IMAGES_DIR:",
        VAL_IMAGES_DIR
    )

    print(
        "VAL_LABELS_DIR:",
        VAL_LABELS_DIR
    )

    print(
        "YOLO available:",
        YOLO is not None
    )

    print(
        "Database configured:",
        bool(DATABASE_URL)
    )

    # Make sure directories exist again
    # before server starts.
    for directory in REQUIRED_DIRECTORIES:

        os.makedirs(
            directory,
            exist_ok=True
        )

    # Always create dataset.yaml.
    try:
        make_dataset_yaml()
        print(
            "YOLO dataset.yaml created:",
            YOLO_DATASET_YAML
        )
    except Exception as e:
        print(
            "Could not create dataset.yaml:",
            e
        )

    # Database
    init_database()

    # Server
    server = ThreadingHTTPServer(
        ("0.0.0.0", PORT),
        Handler
    )

    print(
        "Server running on port:",
        PORT
    )

    print("=" * 60)

    try:

        server.serve_forever()

    except KeyboardInterrupt:

        print(
            "Server stopped."
        )

    finally:

        server.server_close()


if __name__ == "__main__":
    main()
