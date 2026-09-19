import os
import io
import csv
import json
import time
import base64
import shutil
import sqlite3
import threading
from datetime import datetime, date
from urllib.parse import urlparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# =========================================================
# SERVER
# =========================================================

HOST = "0.0.0.0"
PORT = int(os.environ.get("PORT", "8080"))

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DB_FILE = os.path.join(BASE_DIR, "bucket_counter.db")

AI_DIR = os.path.join(BASE_DIR, "ai")
DATASET_DIR = os.path.join(AI_DIR, "dataset")
TRAIN_DIR = os.path.join(AI_DIR, "training")
MODEL_DIR = os.path.join(AI_DIR, "models")

MODEL_FILE = os.path.join(MODEL_DIR, "best.pt")
DATA_YAML = os.path.join(DATASET_DIR, "data.yaml")

for folder in [
    AI_DIR,
    DATASET_DIR,
    TRAIN_DIR,
    MODEL_DIR
]:
    os.makedirs(folder, exist_ok=True)


# =========================================================
# OPTIONAL AI IMPORTS
# =========================================================

try:
    from ultralytics import YOLO

    YOLO_AVAILABLE = True

except Exception as e:

    YOLO_AVAILABLE = False
    YOLO_IMPORT_ERROR = str(e)
    YOLO = None


try:
    import cv2
    import numpy as np

    CV_AVAILABLE = True

except Exception as e:

    CV_AVAILABLE = False
    CV_IMPORT_ERROR = str(e)

    cv2 = None
    np = None


# =========================================================
# AI CLASSES
# =========================================================

CLASS_NAMES = [
    "BUCKET_LOADED",
    "BUCKET_EMPTY",
    "PEOPLE",
    "EQUIPMENT"
]


# =========================================================
# TRAINING STATUS
# =========================================================

TRAIN_STATUS = {
    "running": False,
    "message": "Not started",
    "progress": 0,
    "error": "",
    "finished": False
}

TRAIN_LOCK = threading.Lock()


# =========================================================
# MODEL
# =========================================================

MODEL = None
MODEL_LOCK = threading.Lock()


# =========================================================
# TRACKING
# =========================================================

COUNTED_TRACKS = set()

LAST_TRACK_Y = {}

LAST_TRACK_TIME = {}


# =========================================================
# DATABASE
# =========================================================

def db():

    con = sqlite3.connect(
        DB_FILE,
        timeout=20
    )

    con.row_factory = sqlite3.Row

    return con


def init_db():

    con = db()

    con.executescript("""

    CREATE TABLE IF NOT EXISTS buckets(

        id INTEGER PRIMARY KEY AUTOINCREMENT,

        name TEXT NOT NULL,

        capacity REAL DEFAULT 0,

        active INTEGER DEFAULT 1,

        created_at TEXT NOT NULL
    );


    CREATE TABLE IF NOT EXISTS bucket_images(

        id INTEGER PRIMARY KEY AUTOINCREMENT,

        bucket_id INTEGER NOT NULL,

        filename TEXT NOT NULL,

        image_data TEXT NOT NULL,

        created_at TEXT NOT NULL
    );


    CREATE TABLE IF NOT EXISTS dataset_images(

        id INTEGER PRIMARY KEY AUTOINCREMENT,

        filename TEXT NOT NULL,

        image_data TEXT NOT NULL,

        width INTEGER DEFAULT 0,

        height INTEGER DEFAULT 0,

        created_at TEXT NOT NULL
    );


    CREATE TABLE IF NOT EXISTS annotations(

        id INTEGER PRIMARY KEY AUTOINCREMENT,

        image_id INTEGER NOT NULL,

        class_id INTEGER NOT NULL,

        class_name TEXT NOT NULL,

        x_center REAL NOT NULL,

        y_center REAL NOT NULL,

        box_width REAL NOT NULL,

        box_height REAL NOT NULL,

        created_at TEXT NOT NULL
    );


    CREATE TABLE IF NOT EXISTS detections(

        id INTEGER PRIMARY KEY AUTOINCREMENT,

        detection_time TEXT NOT NULL,

        bucket_id INTEGER,

        bucket_name TEXT,

        status TEXT NOT NULL,

        counted INTEGER DEFAULT 0,

        confidence REAL DEFAULT 0,

        track_id INTEGER,

        note TEXT DEFAULT ''
    );


    CREATE TABLE IF NOT EXISTS daily_counts(

        id INTEGER PRIMARY KEY AUTOINCREMENT,

        count_date TEXT UNIQUE NOT NULL,

        loaded INTEGER DEFAULT 0,

        empty INTEGER DEFAULT 0,

        people INTEGER DEFAULT 0,

        equipment INTEGER DEFAULT 0,

        unknown INTEGER DEFAULT 0
    );


    CREATE TABLE IF NOT EXISTS settings(

        key TEXT PRIMARY KEY,

        value TEXT
    );


    INSERT OR IGNORE INTO settings
    (key,value)
    VALUES
    ('line_position','55');

    """)

    con.commit()

    con.close()


# =========================================================
# SETTINGS
# =========================================================

def setting(key, default=None):

    con = db()

    row = con.execute(
        "SELECT value FROM settings WHERE key=?",
        (key,)
    ).fetchone()

    con.close()

    if row:
        return row["value"]

    return default


def set_setting(key, value):

    con = db()

    con.execute(
        """
        INSERT OR REPLACE INTO settings
        (key,value)
        VALUES (?,?)
        """,
        (key, str(value))
    )

    con.commit()

    con.close()


# =========================================================
# TIME
# =========================================================

def now():

    return datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )


# =========================================================
# DATA URL
# =========================================================

def data_url_to_bytes(data_url):

    if not isinstance(data_url, str):
        raise ValueError("Invalid image data")

    if "," not in data_url:
        raise ValueError("Invalid image data")

    head, payload = data_url.split(",", 1)

    return base64.b64decode(payload)


# =========================================================
# JSON RESPONSE
# =========================================================

def json_response(handler, obj, code=200):

    raw = json.dumps(
        obj,
        ensure_ascii=False
    ).encode("utf-8")

    handler.send_response(code)

    handler.send_header(
        "Content-Type",
        "application/json; charset=utf-8"
    )

    handler.send_header(
        "Content-Length",
        str(len(raw))
    )

    handler.send_header(
        "Cache-Control",
        "no-store"
    )

    handler.end_headers()

    handler.wfile.write(raw)


# =========================================================
# HTML RESPONSE
# =========================================================

def html_response(handler, html, code=200):

    raw = html.encode("utf-8")

    handler.send_response(code)

    handler.send_header(
        "Content-Type",
        "text/html; charset=utf-8"
    )

    handler.send_header(
        "Content-Length",
        str(len(raw))
    )

    handler.end_headers()

    handler.wfile.write(raw)


# =========================================================
# READ JSON
# =========================================================

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
# ACTIVE BUCKET
# =========================================================

def active_bucket():

    con = db()

    row = con.execute(
        """
        SELECT *
        FROM buckets
        WHERE active=1
        ORDER BY id DESC
        LIMIT 1
        """
    ).fetchone()

    con.close()

    if row:
        return dict(row)

    return None


# =========================================================
# DAILY COUNT
# =========================================================

def update_daily(status):

    d = date.today().isoformat()

    con = db()

    con.execute(
        """
        INSERT OR IGNORE INTO
        daily_counts(count_date)
        VALUES(?)
        """,
        (d,)
    )

    column = {
        "BUCKET_LOADED": "loaded",
        "BUCKET_EMPTY": "empty",
        "PEOPLE": "people",
        "EQUIPMENT": "equipment"
    }.get(
        status,
        "unknown"
    )

    con.execute(
        f"""
        UPDATE daily_counts

        SET {column}={column}+1

        WHERE count_date=?
        """,
        (d,)
    )

    con.commit()

    con.close()


# =========================================================
# SAVE DETECTION
# =========================================================

def save_detection(
    status,
    confidence=0,
    track_id=None,
    counted=False,
    note=""
):

    bucket = active_bucket()

    con = db()

    con.execute(
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

        VALUES
        (?,?,?,?,?,?,?,?)
        """,
        (
            now(),

            bucket["id"]
            if bucket
            else None,

            bucket["name"]
            if bucket
            else "",

            status,

            int(counted),

            float(confidence),

            track_id,

            note
        )
    )

    con.commit()

    con.close()

    update_daily(status)


# =========================================================
# RESET TRACKER
# =========================================================

def reset_tracker():

    global COUNTED_TRACKS
    global LAST_TRACK_Y
    global LAST_TRACK_TIME

    COUNTED_TRACKS = set()

    LAST_TRACK_Y = {}

    LAST_TRACK_TIME = {}


# =========================================================
# LOAD MODEL
# =========================================================

def get_model():

    global MODEL

    if not YOLO_AVAILABLE:
        return None

    if not os.path.exists(MODEL_FILE):
        return None

    with MODEL_LOCK:

        if MODEL is None:

            MODEL = YOLO(
                MODEL_FILE
            )

    return MODEL


# =========================================================
# PREPARE YOLO DATASET
# =========================================================

def prepare_dataset():

    con = db()

    rows = con.execute(
        """
        SELECT *
        FROM dataset_images
        ORDER BY id
        """
    ).fetchall()

    labeled = []

    for row in rows:

        annotations = con.execute(
            """
            SELECT *
            FROM annotations
            WHERE image_id=?
            ORDER BY id
            """,
            (row["id"],)
        ).fetchall()

        if annotations:

            labeled.append(
                (row, annotations)
            )

    con.close()

    if len(labeled) < 5:

        raise ValueError(
            "Need at least 5 labeled images. "
            f"Current: {len(labeled)}"
        )

    if os.path.exists(DATASET_DIR):

        shutil.rmtree(
            DATASET_DIR
        )

    os.makedirs(
        os.path.join(
            DATASET_DIR,
            "images",
            "train"
        ),
        exist_ok=True
    )

    os.makedirs(
        os.path.join(
            DATASET_DIR,
            "images",
            "val"
        ),
        exist_ok=True
    )

    os.makedirs(
        os.path.join(
            DATASET_DIR,
            "labels",
            "train"
        ),
        exist_ok=True
    )

    os.makedirs(
        os.path.join(
            DATASET_DIR,
            "labels",
            "val"
        ),
        exist_ok=True
    )

    split_at = max(
        1,
        int(len(labeled) * 0.8)
    )

    if split_at >= len(labeled):

        split_at = len(labeled) - 1

    for index, item in enumerate(labeled):

        row, annotations = item

        split = (
            "train"
            if index < split_at
            else "val"
        )

        filename = (
            f"image_{row['id']}.jpg"
        )

        raw = data_url_to_bytes(
            row["image_data"]
        )

        image_path = os.path.join(
            DATASET_DIR,
            "images",
            split,
            filename
        )

        with open(
            image_path,
            "wb"
        ) as f:

            f.write(raw)

        label_path = os.path.join(
            DATASET_DIR,
            "labels",
            split,
            os.path.splitext(
                filename
            )[0] + ".txt"
        )

        with open(
            label_path,
            "w",
            encoding="utf-8"
        ) as f:

            for ann in annotations:

                f.write(
                    f"{int(ann['class_id'])} "
                    f"{ann['x_center']:.6f} "
                    f"{ann['y_center']:.6f} "
                    f"{ann['box_width']:.6f} "
                    f"{ann['box_height']:.6f}\n"
                )

    yaml_text = (
        f"path: "
        f"{DATASET_DIR.replace(chr(92), '/')}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"names:\n"
    )

    for i, name in enumerate(
        CLASS_NAMES
    ):

        yaml_text += (
            f"  {i}: {name}\n"
        )

    with open(
        DATA_YAML,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(yaml_text)

    return len(labeled)


# =========================================================
# TRAINING
# =========================================================

def train_worker():

    global MODEL

    with TRAIN_LOCK:

        TRAIN_STATUS.update(
            running=True,
            message="Preparing dataset...",
            progress=5,
            error="",
            finished=False
        )

    try:

        if not YOLO_AVAILABLE:

            raise RuntimeError(
                "Ultralytics is not installed. "
                "Deploy with requirements.txt on Render first."
            )

        number = prepare_dataset()

        with TRAIN_LOCK:

            TRAIN_STATUS.update(
                message=(
                    f"Dataset ready: "
                    f"{number} labeled images"
                ),
                progress=15
            )

        base_model = YOLO(
            "yolo11n.pt"
        )

        with TRAIN_LOCK:

            TRAIN_STATUS.update(
                message="Training YOLO...",
                progress=20
            )

        base_model.train(
            data=DATA_YAML,
            epochs=20,
            imgsz=640,
            batch=4,
            project=TRAIN_DIR,
            name="bucket_ai",
            exist_ok=True,
            verbose=False
        )

        candidates = []

        for root, dirs, files in os.walk(
            TRAIN_DIR
        ):

            if "best.pt" in files:

                candidates.append(
                    os.path.join(
                        root,
                        "best.pt"
                    )
                )

        if not candidates:

            raise RuntimeError(
                "Training finished but best.pt "
                "was not found"
            )

        source = max(
            candidates,
            key=os.path.getmtime
        )

        shutil.copy2(
            source,
            MODEL_FILE
        )

        with MODEL_LOCK:

            MODEL = YOLO(
                MODEL_FILE
            )

        with TRAIN_LOCK:

            TRAIN_STATUS.update(
                running=False,
                message=(
                    "Training complete. "
                    "Model is ready."
                ),
                progress=100,
                finished=True
            )

    except Exception as e:

        with TRAIN_LOCK:

            TRAIN_STATUS.update(
                running=False,
                message="Training failed",
                progress=0,
                error=str(e),
                finished=False
            )


# =========================================================
# CLASS NORMALIZATION
# =========================================================

def normalize_name(name):

    text = str(name).upper()

    text = text.replace(
        "-",
        "_"
    )

    text = text.replace(
        " ",
        "_"
    )

    aliases = {

        "LOADED":
            "BUCKET_LOADED",

        "FULL_BUCKET":
            "BUCKET_LOADED",

        "EMPTY":
            "BUCKET_EMPTY",

        "PERSON":
            "PEOPLE",

        "PEOPLE":
            "PEOPLE",

        "EQUIPMENT":
            "EQUIPMENT",

        "MACHINE":
            "EQUIPMENT"
    }

    return aliases.get(
        text,
        text
    )


# =========================================================
# PROCESS CAMERA FRAME
# =========================================================

def process_frame(data_url):

    if not CV_AVAILABLE:

        raise RuntimeError(
            "OpenCV/numpy are not installed"
        )

    model = get_model()

    if model is None:

        return {

            "ok": False,

            "model_ready": False,

            "message":
                "YOLO model not ready. "
                "Train a model first.",

            "detections": []
        }

    raw = data_url_to_bytes(
        data_url
    )

    array = np.frombuffer(
        raw,
        np.uint8
    )

    frame = cv2.imdecode(
        array,
        cv2.IMREAD_COLOR
    )

    if frame is None:

        raise ValueError(
            "Could not decode image"
        )

    height, width = frame.shape[:2]

    line_position = float(
        setting(
            "line_position",
            "55"
        )
    ) / 100.0

    line_y = int(
        height * line_position
    )

    results = model.track(
        source=frame,
        persist=True,
        conf=0.35,
        iou=0.5,
        verbose=False
    )

    output = []

    for result in results:

        boxes = getattr(
            result,
            "boxes",
            None
        )

        if boxes is None:
            continue

        if getattr(
            boxes,
            "id",
            None
        ) is not None:

            ids = (
                boxes.id
                .int()
                .cpu()
                .tolist()
            )

        else:

            ids = [
                None
                for _ in boxes
            ]

        coordinates = (
            boxes.xyxy
            .cpu()
            .tolist()
        )

        confidences = (
            boxes.conf
            .cpu()
            .tolist()
        )

        classes = (
            boxes.cls
            .int()
            .cpu()
            .tolist()
        )

        names = (
            result.names
            if hasattr(result, "names")
            else {}
        )

        for box, confidence, class_id, track_id in zip(
            coordinates,
            confidences,
            classes,
            ids
        ):

            class_name = normalize_name(
                names.get(
                    int(class_id),
                    CLASS_NAMES[int(class_id)]
                    if int(class_id)
                    < len(CLASS_NAMES)
                    else "UNKNOWN"
                )
            )

            x1, y1, x2, y2 = map(
                int,
                box
            )

            center_y = (
                y1 + y2
            ) / 2

            crossed = False

            if track_id is not None:

                previous_y = LAST_TRACK_Y.get(
                    track_id
                )

                current_time = time.time()

                previous_time = LAST_TRACK_TIME.get(
                    track_id,
                    0
                )

                if (
                    previous_y is not None
                    and
                    (previous_y - line_y)
                    *
                    (center_y - line_y)
                    <= 0
                    and
                    abs(center_y - previous_y) > 3
                    and
                    current_time - previous_time > 0.2
                ):

                    crossed = True

                LAST_TRACK_Y[
                    track_id
                ] = center_y

                LAST_TRACK_TIME[
                    track_id
                ] = current_time

            counted = False

            if (
                crossed
                and
                track_id is not None
                and
                track_id not in COUNTED_TRACKS
            ):

                COUNTED_TRACKS.add(
                    track_id
                )

                if class_name == "BUCKET_LOADED":

                    counted = True

                    save_detection(
                        class_name,
                        confidence,
                        track_id,
                        True,
                        "Loaded bucket crossed counting line"
                    )

                else:

                    save_detection(
                        class_name,
                        confidence,
                        track_id,
                        False,
                        "Crossed line but not counted"
                    )

            output.append({

                "class":
                    class_name,

                "confidence":
                    round(
                        float(confidence),
                        3
                    ),

                "track_id":
                    track_id,

                "x1":
                    x1,

                "y1":
                    y1,

                "x2":
                    x2,

                "y2":
                    y2,

                "counted":
                    counted
            })

    return {

        "ok": True,

        "model_ready": True,

        "line_y": line_y,

        "width": width,

        "height": height,

        "detections": output
    }


# =========================================================
# STATUS
# =========================================================

def status_obj():

    with TRAIN_LOCK:

        training = dict(
            TRAIN_STATUS
        )

    return {

        "yolo_installed":
            YOLO_AVAILABLE,

        "opencv_installed":
            CV_AVAILABLE,

        "model_ready":
            os.path.exists(
                MODEL_FILE
            ),

        "train":
            training,

        "line_position":
            float(
                setting(
                    "line_position",
                    "55"
                )
            )
    }


# =========================================================
# DASHBOARD DATA
# =========================================================

def dashboard_data():

    today_date = (
        date.today()
        .isoformat()
    )

    con = db()

    today = con.execute(
        """
        SELECT *
        FROM daily_counts
        WHERE count_date=?
        """,
        (today_date,)
    ).fetchone()

    recent = con.execute(
        """
        SELECT *
        FROM daily_counts
        ORDER BY count_date DESC
        LIMIT 7
        """
    ).fetchall()

    detections = con.execute(
        """
        SELECT *
        FROM detections
        ORDER BY id DESC
        LIMIT 30
        """
    ).fetchall()

    con.close()

    if today:

        today_data = dict(
            today
        )

    else:

        today_data = {

            "loaded": 0,
            "empty": 0,
            "people": 0,
            "equipment": 0,
            "unknown": 0
        }

    return {

        "today":
            today_data,

        "recent":
            [
                dict(row)
                for row in recent
            ],

        "detections":
            [
                dict(row)
                for row in detections
            ]
    }


# =========================================================
# CSS
# =========================================================

CSS = """

*{
    box-sizing:border-box
}

body{
    margin:0;
    font-family:Arial,sans-serif;
    background:#0b1220;
    color:#eef4ff
}

header{
    background:#111b2e;
    padding:14px 16px;
    position:sticky;
    top:0;
    z-index:5;
    border-bottom:1px solid #24334e
}

header h1{
    margin:0;
    font-size:20px
}

header small{
    color:#9eb0c9
}

.nav{
    display:flex;
    gap:7px;
    overflow:auto;
    padding:10px 12px;
    background:#0f1829
}

.nav a{
    color:#dce7f8;
    text-decoration:none;
    background:#1a2840;
    padding:9px 11px;
    border-radius:9px;
    white-space:nowrap;
    font-size:13px
}

.wrap{
    max-width:1100px;
    margin:auto;
    padding:15px
}

.card{
    background:#111b2e;
    border:1px solid #24334e;
    border-radius:14px;
    padding:15px;
    margin:12px 0
}

.grid{
    display:grid;
    grid-template-columns:
    repeat(
        auto-fit,
        minmax(180px,1fr)
    );
    gap:12px
}

.stat{
    font-size:28px;
    font-weight:bold;
    margin-top:7px
}

.muted{
    color:#9eb0c9
}

.btn{
    border:0;
    border-radius:9px;
    padding:10px 14px;
    background:#2878ff;
    color:#fff;
    font-weight:bold;
    cursor:pointer
}

.btn.alt{
    background:#263750
}

.btn.danger{
    background:#b9364f
}

input,
select{
    width:100%;
    padding:11px;
    border-radius:9px;
    border:1px solid #344766;
    background:#0a1220;
    color:#fff;
    margin:6px 0 10px
}

label{
    font-size:13px;
    color:#b9c8dc
}

.table{
    width:100%;
    border-collapse:collapse
}

.table th,
.table td{
    padding:8px;
    border-bottom:1px solid #263650;
    text-align:left;
    font-size:13px
}

.ok{
    color:#54e38e
}

.bad{
    color:#ff7188
}

.warn{
    color:#ffd166
}

.camera{
    position:relative;
    background:#000;
    border-radius:14px;
    overflow:hidden
}

.camera video{
    display:block;
    width:100%;
    max-height:65vh;
    object-fit:contain
}

.line{
    position:absolute;
    left:0;
    right:0;
    border-top:3px solid #ffdf4d;
    pointer-events:none
}

.boxes{
    position:absolute;
    inset:0;
    pointer-events:none
}

.box{
    position:absolute;
    border:2px solid #46e6a1;
    color:#fff;
    background:#0008;
    font-size:12px
}

.trainstage{
    position:relative;
    display:inline-block;
    max-width:100%;
    touch-action:none
}

.trainstage img{
    display:block;
    max-width:100%;
    max-height:55vh
}

.drawbox{
    position:absolute;
    border:2px dashed #ffdf4d;
    display:none
}

.hint{
    padding:9px;
    background:#17253c;
    border-radius:9px;
    margin:8px 0
}

.pill{
    display:inline-block;
    padding:5px 8px;
    border-radius:999px;
    background:#263750;
    margin:2px;
    font-size:12px
}

@media(max-width:600px){

    .wrap{
        padding:9px
    }

    .card{
        padding:11px
    }

    .stat{
        font-size:23px
    }
}

"""


# =========================================================
# PAGE
# =========================================================

def page(title, body):

    navigation = """

    <a href="/">Dashboard</a>

    <a href="/camera">Camera</a>

    <a href="/buckets">Buckets</a>

    <a href="/training">AI Training</a>

    <a href="/history">History</a>

    <a href="/settings">Settings</a>

    """

    return f"""

<!doctype html>

<html>

<head>

<meta charset="utf-8">

<meta name="viewport"
content="width=device-width,initial-scale=1">

<title>{title}</title>

<style>
{CSS}
</style>

</head>

<body>

<header>

<h1>
🪣 BUCKET COUNTER AI
</h1>

<small>
Underground production monitoring
</small>

</header>

<div class="nav">

{navigation}

</div>

<main class="wrap">

{body}

</main>

</body>

</html>

"""


# =========================================================
# DASHBOARD PAGE
# =========================================================

def dashboard_page():

    body = """

<div class="card">

<h2>
Dashboard
</h2>

<div id="status">
Loading...
</div>

</div>


<div class="grid">


<div class="card">

<div class="muted">
Loaded buckets today
</div>

<div id="loaded"
class="stat">
0
</div>

</div>


<div class="card">

<div class="muted">
Empty buckets detected
</div>

<div id="empty"
class="stat">
0
</div>

</div>


<div class="card">

<div class="muted">
People
</div>

<div id="people"
class="stat">
0
</div>

</div>


<div class="card">

<div class="muted">
Equipment
</div>

<div id="equipment"
class="stat">
0
</div>

</div>


</div>


<div class="card">

<h3>
Recent detections
</h3>

<div id="det">
</div>

</div>


<script>

async function load(){

    let s =
        await (
            await fetch('/api/status')
        ).json();

    document.getElementById(
        'status'
    ).innerHTML =
        `YOLO:
        <b class="${
            s.yolo_installed
            ? 'ok'
            : 'bad'
        }">
        ${
            s.yolo_installed
            ? 'INSTALLED'
            : 'NOT INSTALLED'
        }
        </b>

        &nbsp;

        Model:
        <b class="${
            s.model_ready
            ? 'ok'
            : 'bad'
        }">

        ${
            s.model_ready
            ? 'READY'
            : 'NOT READY'
        }

        </b>`;


    let d =
        await (
            await fetch('/api/dashboard')
        ).json();


    [
        'loaded',
        'empty',
        'people',
        'equipment'
    ].forEach(

        k => {

            document.getElementById(
                k
            ).textContent =
                d.today[k] || 0;

        }

    );


    document.getElementById(
        'det'
    ).innerHTML =

        d.detections.map(

            x => `

            <div class="hint">

            ${x.detection_time}

            —

            <b>
            ${x.status}
            </b>

            —

            ${(x.confidence * 100).toFixed(1)}%

            ${
                x.counted
                ? '✅ COUNTED'
                : ''
            }

            </div>

            `

        ).join('')

        ||

        'No detections yet';

}


load();

setInterval(
    load,
    5000
);

</script>

"""

    return page(
        "Dashboard",
        body
    )


# =========================================================
# CAMERA PAGE
# =========================================================

def camera_page():

    body = """

<div class="card">

<h2>
📷 Camera Counter
</h2>


<div class="hint">

Only

<b>
BUCKET_LOADED
</b>

crossing the line is counted.

Empty bucket,
people and equipment
are not counted.

</div>


<div class="camera">

<video
id="video"
autoplay
playsinline
muted>
</video>


<div
id="line"
class="line">
</div>


<div
id="boxes"
class="boxes">
</div>

</div>


<br>


<button
class="btn"
onclick="start()">

START CAMERA

</button>


<button
class="btn alt"
onclick="stop()">

STOP

</button>


<button
class="btn alt"
onclick="reset()">

RESET TRACKER

</button>


<p
id="msg"
class="muted">
</p>

</div>


<div class="card">

<h3>
Live result
</h3>

<div id="result">
Waiting...
</div>

</div>


<script>

let stream = null;

let timer = null;

let busy = false;

const video =
    document.getElementById(
        'video'
    );


async function start(){

    try{

        stream =
            await navigator.mediaDevices
            .getUserMedia({

                video:{
                    facingMode:{
                        ideal:'environment'
                    }
                },

                audio:false

            });


        video.srcObject =
            stream;


        document.getElementById(
            'msg'
        ).textContent =
            'Camera running';


        clearInterval(timer);


        timer =
            setInterval(
                sendFrame,
                1200
            );


    }catch(e){

        document.getElementById(
            'msg'
        ).textContent =
            'Camera error: '
            + e.message;

    }

}


function stop(){

    clearInterval(
        timer
    );

    timer = null;


    if(stream){

        stream
        .getTracks()
        .forEach(
            t => t.stop()
        );

    }

    stream = null;

}


function reset(){

    fetch(
        '/api/reset-tracker',
        {
            method:'POST'
        }
    );

    document.getElementById(
        'result'
    ).textContent =
        'Tracker reset';

}


async function sendFrame(){

    if(
        busy ||
        !video.videoWidth
    ){

        return;

    }


    busy = true;


    let canvas =
        document.createElement(
            'canvas'
        );


    canvas.width =
        video.videoWidth;

    canvas.height =
        video.videoHeight;


    canvas
    .getContext('2d')
    .drawImage(
        video,
        0,
        0
    );


    try{

        let response =
            await fetch(
                '/api/detect',
                {

                    method:'POST',

                    headers:{
                        'Content-Type':
                            'application/json'
                    },

                    body:
                        JSON.stringify({

                            image:
                                canvas.toDataURL(
                                    'image/jpeg',
                                    .75
                                )

                        })

                }
            );


        let data =
            await response.json();


        if(data.line_y){

            document
            .getElementById(
                'line'
            )
            .style.top =

                (
                    data.line_y /
                    data.height *
                    100
                )
                + '%';

        }


        if(data.detections){

            document
            .getElementById(
                'result'
            ).innerHTML =

                data.detections.map(

                    x => `

                    <span class="pill">

                    ${x.class}

                    ${(x.confidence*100)
                    .toFixed(0)}%

                    ${
                        x.counted
                        ? '✅ COUNTED'
                        : ''
                    }

                    </span>

                    `

                ).join('')

                ||

                'Nothing detected';


            let html = '';


            data.detections.forEach(

                x => {

                    html += `

                    <div
                    class="box"

                    style="
                    left:${x.x1/data.width*100}%;
                    top:${x.y1/data.height*100}%;
                    width:${(x.x2-x.x1)/data.width*100}%;
                    height:${(x.y2-x.y1)/data.height*100}%;
                    ">

                    ${x.class}

                    ${(x.confidence*100)
                    .toFixed(0)}%

                    </div>

                    `;

                }

            );


            document.getElementById(
                'boxes'
            ).innerHTML =
                html;

        }


    }catch(e){

        document.getElementById(
            'result'
        ).textContent =
            'Detection error: '
            + e.message;

    }


    busy = false;

}

</script>

"""

    return page(
        "Camera",
        body
    )


# =========================================================
# BUCKET PAGE
# =========================================================

def buckets_page():

    body = """

<div class="card">

<h2>
🪣 Buckets
</h2>


<p class="muted">

Add the actual bucket type and
upload several reference photos.

These photos are stored as references.

Labeled training images are still
needed for YOLO training.

</p>


<input
id="name"
placeholder="Bucket name e.g. Production Bucket">


<input
id="capacity"
type="number"
step="0.1"
placeholder="Capacity / size">


<button
class="btn"
onclick="addBucket()">

SAVE BUCKET

</button>


<div id="buckets">
</div>

</div>


<div class="card">

<h3>
Upload reference photos
</h3>


<select
id="bucketSelect">
</select>


<input
id="photos"
type="file"
accept="image/*"
multiple>


<button
class="btn"
onclick="uploadPhotos()">

UPLOAD PHOTOS

</button>


<div id="uploadMsg">
</div>

</div>


<script>

async function load(){

    let b =
        await (
            await fetch(
                '/api/buckets'
            )
        ).json();


    document.getElementById(
        'buckets'
    ).innerHTML =

        b.map(

            x => `

            <div class="hint">

            <b>
            ${x.name}
            </b>

            —

            capacity
            ${x.capacity}

            —

            ${
                x.active
                ? 'ACTIVE'
                : 'inactive'
            }

            <br>


            <button
            class="btn alt"
            onclick="activate(${x.id})">

            ${
                x.active
                ? 'ACTIVE'
                : 'ACTIVATE'
            }

            </button>

            </div>

            `

        ).join('')

        ||

        'No buckets yet';


    document.getElementById(
        'bucketSelect'
    ).innerHTML =

        b.map(

            x => `

            <option
            value="${x.id}">

            ${x.name}

            </option>

            `

        ).join('');

}


async function addBucket(){

    let response =
        await fetch(
            '/api/buckets',
            {

                method:'POST',

                headers:{
                    'Content-Type':
                        'application/json'
                },

                body:
                    JSON.stringify({

                        name:
                            document
                            .getElementById(
                                'name'
                            ).value,

                        capacity:
                            document
                            .getElementById(
                                'capacity'
                            ).value

                    })

            }
        );


    let data =
        await response.json();


    alert(
        data.message ||
        data.error
    );


    load();

}


async function activate(id){

    await fetch(
        '/api/buckets/activate',
        {

            method:'POST',

            headers:{
                'Content-Type':
                    'application/json'
            },

            body:
                JSON.stringify({
                    id:id
                })

        }
    );


    load();

}


async function uploadPhotos(){

    let bucketId =
        document
        .getElementById(
            'bucketSelect'
        ).value;


    let files =
        [
            ...
            document
            .getElementById(
                'photos'
            ).files
        ];


    if(
        !bucketId ||
        !files.length
    ){

        return alert(
            'Select bucket and photos'
        );

    }


    let success = 0;


    for(
        let file of files
    ){

        let image =
            await new Promise(
                (resolve,reject)=>{

                    let reader =
                        new FileReader();

                    reader.onload =
                        () =>
                            resolve(
                                reader.result
                            );

                    reader.onerror =
                        reject;

                    reader.readAsDataURL(
                        file
                    );

                }
            );


        let response =
            await fetch(
                '/api/buckets/photo',
                {

                    method:'POST',

                    headers:{
                        'Content-Type':
                            'application/json'
                    },

                    body:
                        JSON.stringify({

                            bucket_id:
                                bucketId,

                            filename:
                                file.name,

                            image:
                                image

                        })

                }
            );


        if(response.ok){
            success++;
        }

    }


    document.getElementById(
        'uploadMsg'
    ).textContent =
        success +
        ' photo(s) uploaded';

}


load();

</script>

"""

    return page(
        "Buckets",
        body
    )


# =========================================================
# TRAINING PAGE
# =========================================================

def training_page():

    body = """

<div class="card">

<h2>
🤖 AI Training
</h2>


<div
id="status"
class="hint">

Loading...

</div>


<div class="hint">

<b>
Classes:
</b>

BUCKET_LOADED = count

BUCKET_EMPTY = no count

PEOPLE = no count

EQUIPMENT = no count

</div>

</div>


<div class="card">

<h3>
1. Upload and label one image
</h3>


<input
id="file"
type="file"
accept="image/*">


<select id="cls">

<option value="0">
BUCKET_LOADED
</option>

<option value="1">
BUCKET_EMPTY
</option>

<option value="2">
PEOPLE
</option>

<option value="3">
EQUIPMENT
</option>

</select>


<div
id="stage"
class="trainstage">


<img
id="img">


<div
id="draw"
class="drawbox">
</div>

</div>


<p class="muted">

Drag on the image to draw one
box around the selected object.

</p>


<button
id="saveTrainingBtn"
type="button"
class="btn"
onclick="saveTrainingImage()">

💾 SAVE IMAGE + LABEL

</button>


<div id="saveMsg">
</div>

</div>


<div class="card">

<h3>
2. Train model
</h3>


<p>

At least 5 labeled images
are required.

More varied images normally
improve detection.

</p>


<button
id="trainBtn"
class="btn"
onclick="train()">

🚀 TRAIN AI

</button>


<div
id="trainMsg"
class="hint">
</div>

</div>


<div class="card">

<h3>
Saved dataset
</h3>


<div id="summary">
</div>

</div>


<script>

let selectedFile = null;

let box = null;

let drawing = false;

let startX = 0;

let startY = 0;


const image =
    document.getElementById(
        'img'
    );


const stage =
    document.getElementById(
        'stage'
    );


const draw =
    document.getElementById(
        'draw'
    );


document
.getElementById(
    'file'
)
.onchange = function(e){

    selectedFile =
        e.target.files[0];


    if(!selectedFile){
        return;
    }


    let reader =
        new FileReader();


    reader.onload =
        function(){

            image.src =
                reader.result;

            box = null;

            draw.style.display =
                'none';

        };


    reader.readAsDataURL(
        selectedFile
    );

};


function position(event){

    let rect =
        image.getBoundingClientRect();


    let x =
        event.clientX -
        rect.left;


    let y =
        event.clientY -
        rect.top;


    x =
        Math.max(
            0,
            Math.min(
                rect.width,
                x
            )
        );


    y =
        Math.max(
            0,
            Math.min(
                rect.height,
                y
            )
        );


    return {
        x:x,
        y:y
    };

}


stage.addEventListener(
    'pointerdown',
    function(event){

        if(!image.src){
            return;
        }


        stage.setPointerCapture(
            event.pointerId
        );


        let p =
            position(event);


        drawing = true;

        startX = p.x;

        startY = p.y;


        draw.style.left =
            startX + 'px';

        draw.style.top =
            startY + 'px';

        draw.style.width =
            '0px';

        draw.style.height =
            '0px';

        draw.style.display =
            'block';

    }
);


stage.addEventListener(
    'pointermove',
    function(event){

        if(!drawing){
            return;
        }


        let p =
            position(event);


        let x =
            Math.min(
                startX,
                p.x
            );


        let y =
            Math.min(
                startY,
                p.y
            );


        let width =
            Math.abs(
                p.x - startX
            );


        let height =
            Math.abs(
                p.y - startY
            );


        draw.style.left =
            x + 'px';

        draw.style.top =
            y + 'px';

        draw.style.width =
            width + 'px';

        draw.style.height =
            height + 'px';


        box = {
            x:x,
            y:y,
            w:width,
            h:height
        };

    }
);


stage.addEventListener(
    'pointerup',
    function(){

        drawing = false;

    }
);


async function saveTrainingImage(){

    const button =
        document.getElementById(
            'saveTrainingBtn'
        );


    const message =
        document.getElementById(
            'saveMsg'
        );


    try{

        if(!selectedFile){

            throw new Error(
                'Select an image first'
            );

        }


        if(
            !box ||
            box.w < 10 ||
            box.h < 10
        ){

            throw new Error(
                'Draw a box around the object first'
            );

        }


        if(
            !image.naturalWidth ||
            !image.naturalHeight
        ){

            throw new Error(
                'Image is not ready'
            );

        }


        let rect =
            image.getBoundingClientRect();


        let scaleX =
            image.naturalWidth /
            rect.width;


        let scaleY =
            image.naturalHeight /
            rect.height;


        let x =
            box.x *
            scaleX;


        let y =
            box.y *
            scaleY;


        let width =
            box.w *
            scaleX;


        let height =
            box.h *
            scaleY;


        let xCenter =
            (
                x +
                width / 2
            )
            /
            image.naturalWidth;


        let yCenter =
            (
                y +
                height / 2
            )
            /
            image.naturalHeight;


        let boxWidth =
            width /
            image.naturalWidth;


        let boxHeight =
            height /
            image.naturalHeight;


        xCenter =
            Math.max(
                0,
                Math.min(
                    1,
                    xCenter
                )
            );


        yCenter =
            Math.max(
                0,
                Math.min(
                    1,
                    yCenter
                )
            );


        boxWidth =
            Math.max(
                0.001,
                Math.min(
                    1,
                    boxWidth
                )
            );


        boxHeight =
            Math.max(
                0.001,
                Math.min(
                    1,
                    boxHeight
                )
            );


        button.disabled = true;

        button.textContent =
            '⏳ SAVING...';


        let imageData =
            await new Promise(
                function(
                    resolve,
                    reject
                ){

                    let reader =
                        new FileReader();


                    reader.onload =
                        function(){

                            resolve(
                                reader.result
                            );

                        };


                    reader.onerror =
                        function(){

                            reject(
                                new Error(
                                    'Could not read image'
                                )
                            );

                        };


                    reader.readAsDataURL(
                        selectedFile
                    );

                }
            );


        let response =
            await fetch(
                '/api/training/image',
                {

                    method:'POST',

                    headers:{
                        'Content-Type':
                            'application/json'
                    },

                    body:
                        JSON.stringify({

                            filename:
                                selectedFile.name,

                            image:
                                imageData,

                            width:
                                image.naturalWidth,

                            height:
                                image.naturalHeight,

                            class_id:
                                Number(
                                    document
                                    .getElementById(
                                        'cls'
                                    ).value
                                ),

                            x_center:
                                xCenter,

                            y_center:
                                yCenter,

                            box_width:
                                boxWidth,

                            box_height:
                                boxHeight

                        })

                }
            );


        let raw =
            await response.text();


        let data;


        try{

            data =
                JSON.parse(raw);

        }catch(e){

            throw new Error(
                'Server returned invalid response: '
                +
                raw.slice(
                    0,
                    200
                )
            );

        }


        if(
            !response.ok ||
            !data.ok
        ){

            throw new Error(
                data.error ||
                data.message ||
                'HTTP '
                +
                response.status
            );

        }


        message.innerHTML =

            '<span class="ok">' +

            '✅ IMAGE + LABEL SAVED SUCCESSFULLY. ID '

            +

            data.id

            +

            '</span>';


        box = null;

        draw.style.display =
            'none';


        document.getElementById(
            'file'
        ).value = '';


        selectedFile = null;


        loadSummary();


    }catch(error){

        console.error(
            error
        );


        message.innerHTML =

            '<span class="bad">❌ '

            +

            error.message

            +

            '</span>';

    }finally{

        button.disabled =
            false;

        button.textContent =
            '💾 SAVE IMAGE + LABEL';

    }

}


async function loadStatus(){

    let s =
        await (
            await fetch(
                '/api/status'
            )
        ).json();


    document.getElementById(
        'status'
    ).innerHTML =

        `YOLO:

        <b class="${
            s.yolo_installed
            ? 'ok'
            : 'bad'
        }">

        ${
            s.yolo_installed
            ? 'INSTALLED'
            : 'NOT INSTALLED'
        }

        </b>

        |

        Model:

        <b class="${
            s.model_ready
            ? 'ok'
            : 'bad'
        }">

        ${
            s.model_ready
            ? 'READY'
            : 'NOT READY'
        }

        </b>

        `;


    document.getElementById(
        'trainMsg'
    ).textContent =

        s.train.message

        +

        (
            s.train.error
            ? ' — '
              +
              s.train.error
            : ''
        )

        +

        ' ('

        +

        s.train.progress

        +

        '%)';

}


async function loadSummary(){

    let s =
        await (
            await fetch(
                '/api/training/summary'
            )
        ).json();


    document.getElementById(
        'summary'
    ).innerHTML =

        'Total images: <b>'

        +

        s.images

        +

        '</b> | Labeled: <b>'

        +

        s.labeled

        +

        '</b> | Annotations: <b>'

        +

        s.annotations

        +

        '</b>';

}


async function train(){

    let button =
        document.getElementById(
            'trainBtn'
        );


    button.disabled =
        true;


    button.textContent =
        '⏳ TRAINING...';


    await fetch(
        '/api/train',
        {
            method:'POST'
        }
    );


    poll();

}


async function poll(){

    let s =
        await (
            await fetch(
                '/api/status'
            )
        ).json();


    document.getElementById(
        'trainMsg'
    ).textContent =

        s.train.message

        +

        (
            s.train.error
            ? ' — '
              +
              s.train.error
            : ''
        )

        +

        ' ('

        +

        s.train.progress

        +

        '%)';


    if(
        s.train.running
    ){

        setTimeout(
            poll,
            2000
        );

    }else{

        let button =
            document.getElementById(
                'trainBtn'
            );


        button.disabled =
            false;


        button.textContent =
            '🚀 TRAIN AI';


        loadStatus();

    }

}


loadStatus();

loadSummary();

setInterval(
    loadStatus,
    2500
);

</script>

"""

    return page(
        "AI Training",
        body
    )


# =========================================================
# HISTORY PAGE
# =========================================================

def history_page():

    body = """

<div class="card">

<h2>
📜 Detection History
</h2>


<button
class="btn"
onclick="location='/api/export.csv'">

⬇️ EXPORT CSV

</button>


<div id="table">
</div>

</div>


<script>

async function load(){

    let data =
        await (
            await fetch(
                '/api/dashboard'
            )
        ).json();


    document.getElementById(
        'table'
    ).innerHTML =

        '<table class="table">'

        +

        '<tr>'

        +

        '<th>Time</th>'

        +

        '<th>Status</th>'

        +

        '<th>Confidence</th>'

        +

        '<th>Counted</th>'

        +

        '</tr>'

        +

        data.detections.map(

            x => `

            <tr>

            <td>
            ${x.detection_time}
            </td>

            <td>
            ${x.status}
            </td>

            <td>
            ${(x.confidence*100)
            .toFixed(1)}%
            </td>

            <td>
            ${
                x.counted
                ? 'YES'
                : 'NO'
            }
            </td>

            </tr>

            `

        ).join('')

        +

        '</table>';

}


load();

</script>

"""

    return page(
        "History",
        body
    )


# =========================================================
# SETTINGS PAGE
# =========================================================

def settings_page():

    body = """

<div class="card">

<h2>
⚙️ Settings
</h2>


<label>
Counting line position (%)
</label>


<input
id="line"
type="number"
min="5"
max="95"
value="55">


<p class="muted">

Example:

55 means the line is
55% down the camera image.

</p>


<button
class="btn"
onclick="save()">

SAVE SETTING

</button>


<div id="msg">
</div>

</div>


<script>

async function load(){

    let s =
        await (
            await fetch(
                '/api/status'
            )
        ).json();


    document.getElementById(
        'line'
    ).value =
        s.line_position;

}


async function save(){

    let value =
        document.getElementById(
            'line'
        ).value;


    let response =
        await fetch(
            '/api/settings',
            {

                method:'POST',

                headers:{
                    'Content-Type':
                        'application/json'
                },

                body:
                    JSON.stringify({

                        line_position:
                            value

                    })

            }
        );


    let data =
        await response.json();


    document.getElementById(
        'msg'
    ).textContent =
        data.message ||
        data.error;

}


load();

</script>

"""

    return page(
        "Settings",
        body
    )


# =========================================================
# HTTP HANDLER
# =========================================================

class Handler(
    BaseHTTPRequestHandler
):


    def log_message(
        self,
        fmt,
        *args
    ):

        print(
            "%s - %s"
            %
            (
                self.address_string(),
                fmt % args
            )
        )


    # =====================================================
    # GET
    # =====================================================

    def do_GET(self):

        try:

            path =urlparse(
                    self.path
                ).path


            if path in [
                "/",
                "/dashboard"
            ]:

                return html_response(
                    self,
                    dashboard_page()
                )


            if path == "/camera":

                return html_response(
                    self,
                    camera_page()
                )


            if path == "/buckets":

                return html_response(
                    self,
                    buckets_page()
                )


            if path == "/training":

                return html_response(
                    self,
                    training_page()
                )


            if path == "/history":

                return html_response(
                    self,
                    history_page()
                )


            if path == "/settings":

                return html_response(
                    self,
                    settings_page()
                )


            if path == "/api/status":

                return json_response(
                    self,
                    status_obj()
                )


            if path == "/api/dashboard":

                return json_response(
                    self,
                    dashboard_data()
                )


            if path == "/api/buckets":

                con = db()

                rows =con.execute(
                        """
                        SELECT *
                        FROM buckets
                        ORDER BY id DESC
                        """
                    ).fetchall()

                con.close()


                return json_response(

                    self,

                    [
                        dict(row)
                        for row in rows
                    ]

                )


            if path == "/api/training/summary":

                con = db()


                images =con.execute(
                        """
                        SELECT COUNT(*) c
                        FROM dataset_images
                        """
                    ).fetchone()["c"]


                labeled = con.execute(
                        """
                        SELECT
                        COUNT(DISTINCT image_id) c
                        FROM annotations
                        """
                    ).fetchone()["c"]


                annotations = con.execute(
                        """
                        SELECT COUNT(*) c
                        FROM annotations
                        """
                    ).fetchone()["c"]


                con.close()


                return json_response(

                    self,

                    {
                        "images":
                            images,

                        "labeled":
                            labeled,

                        "annotations":
                            annotations
                    }

                )


            if path == "/api/export.csv":

                con = db()


                rows =con.execute(
                        """
                        SELECT
                        detection_time,
                        bucket_name,
                        status,
                        counted,
                        confidence,
                        track_id,
                        note

                        FROM detections

                        ORDER BY id DESC
                        """
                    ).fetchall()


                con.close()


                output = io.StringIO()


                writer = csv.writer(
                        output
                    )


                writer.writerow([
                    "time",
                    "bucket",
                    "status",
                    "counted",
                    "confidence",
                    "track_id",
                    "note"
                ])


                for row in rows:

                    writer.writerow(
                        list(row)
                    )


                raw = output.getvalue().encode()


                self.send_response(
                    200
                )


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
                    str(len(raw))
                )


                self.end_headers()


                self.wfile.write(
                    raw
                )

                return


            self.send_error(
                404,
                "Not found"
            )


        except Exception as e:

            json_response(
                self,
                {
                    "error":
                        str(e)
                },
                500
            )


    # =====================================================
    # POST
    # =====================================================

    def do_POST(self):

        try:

            path = urlparse(
                    self.path
                ).path


            data = read_json(
                    self
                )


            # =============================================
            # RESET TRACKER
            # =============================================

            if path == "/api/reset-tracker":

                reset_tracker()


                return json_response(
                    self,
                    {
                        "ok": True,
                        "message":
                            "Tracker reset"
                    }
                )


            # =============================================
            # DETECT
            # =============================================

            if path == "/api/detect":

                result =process_frame(
                        data.get(
                            "image",
                            ""
                        )
                    )


                return json_response(
                    self,
                    result
                )


            # =============================================
            # SETTINGS
            # =============================================

            if path == "/api/settings":

                value =float(
                        data.get(
                            "line_position",
                            55
                        )
                    )


                value = max(
                        5,
                        min(
                            95,
                            value
                        )
                    )


                set_setting(
                    "line_position",
                    value
                )


                return json_response(
                    self,
                    {
                        "ok": True,
                        "message":
                            "Line position saved"
                    }
                )


            # =============================================
            # ADD BUCKET
            # =============================================

            if path == "/api/buckets":

                name = str(
                        data.get(
                            "name",
                            ""
                        )
                    ).strip()


                capacity =float(
                        data.get(
                            "capacity",
                            0
                        )
                        or 0
                    )


                if not name:

                    return json_response(
                        self,
                        {
                            "error":
                                "Bucket name is required"
                        },
                        400
                    )


                con = db()


                cursor =con.execute(
                        """
                        INSERT INTO buckets
                        (
                            name,
                            capacity,
                            active,
                            created_at
                        )

                        VALUES
                        (?,?,1,?)
                        """,
                        (
                            name,
                            capacity,
                            now()
                        )
                    )


                con.commit()


                bucket_id = cursor.lastrowid


                con.close()


                return json_response(
                    self,
                    {
                        "ok": True,
                        "id":
                            bucket_id,
                        "message":
                            "Bucket saved"
                    }
                )


            # =============================================
            # ACTIVATE BUCKET
            # =============================================

            if path == "/api/buckets/activate":

                bucket_id =int(
                        data.get(
                            "id"
                        )
                    )


                con = db()


                con.execute(
                    """
                    UPDATE buckets
                    SET active=0
                    """
                )


                con.execute(
                    """
                    UPDATE buckets
                    SET active=1
                    WHERE id=?
                    """,
                    (bucket_id,)
                )


                con.commit()

                con.close()


                return json_response(
                    self,
                    {
                        "ok": True
                    }
                )


            # =============================================
            # BUCKET PHOTO
            # =============================================

            if path == "/api/buckets/photo":

                bucket_id = int(
                        data.get(
                            "bucket_id"
                        )
                    )


                image =data.get(
                        "image",
                        ""
                    )


                raw = data_url_to_bytes(
                        image
                    )


                if len(raw) > 8 * 1024 * 1024:

                    return json_response(
                        self,
                        {
                            "error":
                                "Image too large"
                        },
                        400
                    )


                con = db()


                con.execute(
                    """
                    INSERT INTO bucket_images
                    (
                        bucket_id,
                        filename,
                        image_data,
                        created_at
                    )

                    VALUES
                    (?,?,?,?)
                    """,
                    (
                        bucket_id,

                        str(
                            data.get(
                                "filename",
                                "photo.jpg"
                            )
                        ),

                        image,

                        now()
                    )
                )


                con.commit()

                con.close()


                return json_response(
                    self,
                    {
                        "ok": True
                    }
                )


            # =============================================
            # SAVE TRAINING IMAGE
            # =============================================

            if path == "/api/training/image":

                image = data.get(
                        "image",
                        ""
                    )


                raw = data_url_to_bytes(
                        image
                    )


                if len(raw) > 12 * 1024 * 1024:

                    return json_response(
                        self,
                        {
                            "error":
                                "Image too large"
                        },
                        400
                    )


                class_id =int(
                        data.get(
                            "class_id",
                            0
                        )
                    )


                if (
                    class_id < 0
                    or
                    class_id >= len(
                        CLASS_NAMES
                    )
                ):

                    return json_response(
                        self,
                        {
                            "error":
                                "Invalid class"
                        },
                        400
                    )


                x_center =float(
                        data.get(
                            "x_center",
                            0
                        )
                    )


                y_center =float(
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


                box_height =float(
                        data.get(
                            "box_height",
                            0
                        )
                    )


                values = [

                    x_center,
                    y_center,
                    box_width,
                    box_height

                ]


                if any(
                    value < 0
                    or value > 1
                    for value in values
                ):

                    return json_response(
                        self,
                        {
                            "error":
                                "Invalid normalized box values"
                        },
                        400
                    )


                con = db()


                cursor = con.execute(
                        """
                        INSERT INTO dataset_images
                        (
                            filename,
                            image_data,
                            width,
                            height,
                            created_at
                        )

                        VALUES
                        (?,?,?,?,?)
                        """,
                        (
                            str(
                                data.get(
                                    "filename",
                                    "image.jpg"
                                )
                            ),

                            image,

                            int(
                                data.get(
                                    "width",
                                    0
                                )
                            ),

                            int(
                                data.get(
                                    "height",
                                    0
                                )
                            ),

                            now()
                        )
                    )


                image_id =cursor.lastrowid


                con.execute(
                    """
                    INSERT INTO annotations
                    (
                        image_id,
                        class_id,
                        class_name,
                        x_center,
                        y_center,
                        box_width,
                        box_height,
                        created_at
                    )

                    VALUES
                    (?,?,?,?,?,?,?,?)
                    """,
                    (
                        image_id,

                        class_id,

                        CLASS_NAMES[
                            class_id
                        ],

                        x_center,

                        y_center,

                        box_width,

                        box_height,

                        now()
                    )
                )


                con.commit()

                con.close()


                return json_response(
                    self,
                    {
                        "ok": True,
                        "id":
                            image_id,
                        "message":
                            "Image and label saved"
                    }
                )


            # =============================================
            # START TRAINING
            # =============================================

            if path == "/api/train":

                with TRAIN_LOCK:

                    if TRAIN_STATUS[
                        "running"
                    ]:

                        return json_response(
                            self,
                            {
                                "error":
                                    "Training already running"
                            },
                            409
                        )


                    TRAIN_STATUS.update(

                        running=True,

                        message="Starting...",

                        progress=1,

                        error="",

                        finished=False

                    )


                threading.Thread(

                    target=train_worker,

                    daemon=True

                ).start()


                return json_response(
                    self,
                    {
                        "ok": True,
                        "message":
                            "Training started"
                    }
                )


            self.send_error(
                404,
                "Not found"
            )


        except Exception as e:

            json_response(
                self,
                {
                    "error":
                        str(e)
                },
                500
            )


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    init_db()


    print("=" * 60)

    print(
        "BUCKET COUNTER AI"
    )

    print(
        f"Server: http://127.0.0.1:{PORT}"
    )

    print(
        f"YOLO installed: {YOLO_AVAILABLE}"
    )

    print(
        f"Model ready: "
        f"{os.path.exists(MODEL_FILE)}"
    )

    print("=" * 60)


    server =ThreadingHTTPServer(
            (HOST, PORT),
            Handler
        )


    server.serve_forever()
