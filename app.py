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

# ============================================================
# NEERIKA BUCKET AI
# Mining Production Bucket Counter
# Render Free / CPU optimized version
# ============================================================

os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/Ultralytics")

import psycopg2
from psycopg2.extras import RealDictCursor
from ultralytics import YOLO
from PIL import Image
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

HOST = "0.0.0.0"
PORT = int(os.environ.get("PORT", "8080"))

DATABASE_URL = (
    os.environ.get("DATABASE_URL")
    or os.environ.get("SUPABASE_DB_URL")
    or os.environ.get("POSTGRES_URL")
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

AI_DIR = os.path.join(BASE_DIR, "ai")
DATASET_DIR = os.path.join(AI_DIR, "dataset")
MODEL_DIR = os.path.join(AI_DIR, "models")

TRAIN_IMAGES_DIR = os.path.join(DATASET_DIR, "images", "train")
VAL_IMAGES_DIR = os.path.join(DATASET_DIR, "images", "val")

TRAIN_LABELS_DIR = os.path.join(DATASET_DIR, "labels", "train")
VAL_LABELS_DIR = os.path.join(DATASET_DIR, "labels", "val")

MODEL_OUTPUT = os.path.join(MODEL_DIR, "bucket_best.pt")
MODEL_BACKUP = os.path.join(BASE_DIR, "best.pt")

DATASET_YAML = os.path.join(AI_DIR, "bucket_dataset.yaml")

YOLO_CONFIDENCE = float(
    os.environ.get("YOLO_CONFIDENCE", "0.25")
)

# Render Free friendly settings
YOLO_IMAGE_SIZE = int(
    os.environ.get("YOLO_IMAGE_SIZE", "320")
)

TRAIN_EPOCHS = int(
    os.environ.get("TRAIN_EPOCHS", "20")
)

TRAIN_BATCH = int(
    os.environ.get("TRAIN_BATCH", "1")
)

TRAIN_WORKERS = int(
    os.environ.get("TRAIN_WORKERS", "0")
)

MAX_JSON_BYTES = 15 * 1024 * 1024
MAX_UPLOAD_BYTES = 15 * 1024 * 1024

COUNT_COOLDOWN_SECONDS = 4.0


# ============================================================
# GLOBAL STATE
# ============================================================

MODEL = None
MODEL_ERROR = ""

MODEL_LOCK = threading.Lock()

TRAIN_LOCK = threading.Lock()

TRAINING = False
TRAINING_RUN_ID = None

LAST_COUNT_TIME = 0.0

SERVER_INSTANCE_ID = uuid.uuid4().hex

TRAINING_LOCK_CONN = None

TRAINING_LOCK_KEY = 918273645


CLASS_IDS = {
    0: "BUCKET_LOADED",
    1: "BUCKET_EMPTY",
    2: "PEOPLE",
    3: "EQUIPMENT",
}


# ============================================================
# DIRECTORY SETUP
# ============================================================

def ensure_directories():
    directories = [
        AI_DIR,
        DATASET_DIR,
        TRAIN_IMAGES_DIR,
        VAL_IMAGES_DIR,
        TRAIN_LABELS_DIR,
        VAL_LABELS_DIR,
        MODEL_DIR,
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)


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


def init_db():

    ensure_directories()

    connection = db()
    cursor = connection.cursor()

    try:

        # ----------------------------------------------------
        # BUCKETS
        # ----------------------------------------------------

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS buckets (
                id SERIAL PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT DEFAULT '',
                active BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)

        # ----------------------------------------------------
        # DATASET IMAGES
        # ----------------------------------------------------

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS dataset_images (
                id SERIAL PRIMARY KEY,
                filename TEXT NOT NULL,
                image_data BYTEA NOT NULL,
                mime_type TEXT DEFAULT 'image/jpeg',
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)

        # ----------------------------------------------------
        # ANNOTATIONS
        # ----------------------------------------------------

        cursor.execute("""
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

        # ----------------------------------------------------
        # TRAINING STATE
        # ----------------------------------------------------

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS training_state (
                id INTEGER PRIMARY KEY,

                status TEXT DEFAULT 'idle',

                message TEXT DEFAULT '',

                progress INTEGER DEFAULT 0,

                epoch INTEGER DEFAULT 0,

                epochs INTEGER DEFAULT 20,

                run_id TEXT,

                server_id TEXT,

                started_at TIMESTAMPTZ,

                finished_at TIMESTAMPTZ,

                updated_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)

        # ----------------------------------------------------
        # TRAINED MODEL
        # ----------------------------------------------------

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trained_model (
                id INTEGER PRIMARY KEY,

                model_data BYTEA NOT NULL,

                filename TEXT DEFAULT 'bucket_best.pt',

                created_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)

        # ----------------------------------------------------
        # DAILY COUNTS
        # ----------------------------------------------------

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS daily_counts (
                id SERIAL PRIMARY KEY,

                count_date DATE UNIQUE NOT NULL,

                bucket_count INTEGER DEFAULT 0,

                updated_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)

        # ----------------------------------------------------
        # DETECTION EVENTS
        # ----------------------------------------------------

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS detection_events (
                id SERIAL PRIMARY KEY,

                class_name TEXT NOT NULL,

                confidence DOUBLE PRECISION DEFAULT 0,

                counted BOOLEAN DEFAULT FALSE,

                created_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)

        # ----------------------------------------------------
        # MIGRATIONS
        # ----------------------------------------------------

        cursor.execute("""
            ALTER TABLE dataset_images
            ADD COLUMN IF NOT EXISTS mime_type TEXT
            DEFAULT 'image/jpeg'
        """)

        cursor.execute("""
            ALTER TABLE dataset_images
            ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ
            DEFAULT NOW()
        """)

        cursor.execute("""
            ALTER TABLE training_state
            ADD COLUMN IF NOT EXISTS epoch INTEGER
            DEFAULT 0
        """)

        cursor.execute("""
            ALTER TABLE training_state
            ADD COLUMN IF NOT EXISTS epochs INTEGER
            DEFAULT 20
        """)

        cursor.execute("""
            ALTER TABLE training_state
            ADD COLUMN IF NOT EXISTS run_id TEXT
        """)

        cursor.execute("""
            ALTER TABLE training_state
            ADD COLUMN IF NOT EXISTS server_id TEXT
        """)

        cursor.execute("""
            ALTER TABLE training_state
            ADD COLUMN IF NOT EXISTS started_at TIMESTAMPTZ
        """)

        cursor.execute("""
            ALTER TABLE training_state
            ADD COLUMN IF NOT EXISTS finished_at TIMESTAMPTZ
        """)

        cursor.execute("""
            ALTER TABLE training_state
            ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ
            DEFAULT NOW()
        """)

        cursor.execute("""
            INSERT INTO training_state
            (
                id,
                status,
                message,
                progress,
                epoch,
                epochs
            )
            VALUES
            (
                1,
                'idle',
                'Ready',
                0,
                0,
                %s
            )
            ON CONFLICT(id) DO NOTHING
        """, (TRAIN_EPOCHS,))

        cursor.execute("""
            UPDATE training_state
            SET epochs = %s
            WHERE id = 1
        """, (TRAIN_EPOCHS,))

        connection.commit()

    except Exception:
        connection.rollback()
        raise

    finally:
        cursor.close()
        connection.close()


# ============================================================
# HELPERS
# ============================================================

def esc(value):

    if value is None:
        return ""

    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#039;")
    )


def json_out(handler, data, status=200):

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
        "no-store"
    )

    handler.end_headers()

    handler.wfile.write(body)


def html_out(handler, text, status=200):

    body = text.encode("utf-8")

    handler.send_response(status)

    handler.send_header(
        "Content-Type",
        "text/html; charset=utf-8"
    )

    handler.send_header(
        "Content-Length",
        str(len(body))
    )

    handler.send_header(
        "Cache-Control",
        "no-store"
    )

    handler.end_headers()

    handler.wfile.write(body)


def error_out(handler, message, status=500):

    json_out(
        handler,
        {
            "error": str(message)
        },
        status
    )


# ============================================================
# HTML LAYOUT
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

    nav = ""

    for name, path in items:

        active_class = (
            "active"
            if name == active
            else ""
        )

        nav += (
            '<a class="nav-item '
            + active_class
            + '" href="'
            + path
            + '">'
            + name
            + "</a>"
        )

    page = r"""
<!doctype html>

<html>

<head>

<meta charset="utf-8">

<meta
    name="viewport"
    content="width=device-width,initial-scale=1"
>

<title>
__TITLE__ - NEERIKA BUCKET AI
</title>

<style>

* {
    box-sizing: border-box;
}

body {
    margin: 0;
    font-family: Arial, sans-serif;
    background: #f4f6f8;
    color: #17202a;
}

header {
    background: #111827;
    color: white;
    padding: 16px;
}

.brand {
    font-size: 22px;
    font-weight: 800;
}

.sub {
    font-size: 13px;
    opacity: .75;
    margin-top: 4px;
}

.nav {
    display: flex;
    gap: 7px;
    overflow-x: auto;
    background: #1f2937;
    padding: 8px;
}

.nav-item {
    color: white;
    text-decoration: none;
    padding: 10px 13px;
    border-radius: 8px;
    white-space: nowrap;
}

.nav-item.active,
.nav-item:hover {
    background: #374151;
}

main {
    max-width: 1200px;
    margin: auto;
    padding: 16px;
}

.card,
.stat {
    background: white;
    border-radius: 14px;
    padding: 18px;
    margin-bottom: 16px;
    box-shadow: 0 2px 10px rgba(0,0,0,.07);
}

.grid {
    display: grid;
    grid-template-columns:
        repeat(auto-fit,minmax(210px,1fr));
    gap: 14px;
}

.num {
    font-size: 38px;
    font-weight: 800;
    margin-top: 7px;
}

button {
    border: 0;
    border-radius: 9px;
    padding: 10px 15px;
    background: #111827;
    color: white;
    cursor: pointer;
    margin: 3px;
}

button:disabled {
    opacity: .5;
    cursor: not-allowed;
}

button.secondary {
    background: #6b7280;
}

button.success {
    background: #166534;
}

button.danger {
    background: #991b1b;
}

input,
select {
    width: 100%;
    padding: 10px;
    border: 1px solid #d1d5db;
    border-radius: 8px;
}

label {
    font-weight: 700;
    display: block;
    margin-bottom: 5px;
}

.row {
    margin-bottom: 13px;
}

.status {
    padding: 10px;
    background: #f3f4f6;
    border-radius: 8px;
    margin-top: 10px;
}

table {
    width: 100%;
    border-collapse: collapse;
}

th,
td {
    text-align: left;
    padding: 9px;
    border-bottom: 1px solid #e5e7eb;
}

.footer {
    text-align: center;
    color: #6b7280;
    font-size: 12px;
    padding: 25px;
}

.progress {
    height: 24px;
    background: #e5e7eb;
    border-radius: 12px;
    overflow: hidden;
}

.bar {
    height: 100%;
    background: #111827;
    width: 0%;
    transition: width .4s;
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
    font-size: 12px;
    font-weight: bold;
}

.preview {
    max-width: 100%;
    max-height: 500px;
    border-radius: 10px;
}

.small {
    font-size: 13px;
    color: #6b7280;
}

.training-log {
    background: #111827;
    color: #fff;
    padding: 12px;
    border-radius: 8px;
    font-family: monospace;
    white-space: pre-wrap;
    min-height: 50px;
}

@media(max-width:600px) {

    main {
        padding: 10px;
    }

    .num {
        font-size: 30px;
    }

}

</style>

</head>

<body>

<header>

<div class="brand">
NEERIKA BUCKET AI
</div>

<div class="sub">
Mining Production Bucket Counter
</div>

</header>

<nav class="nav">
__NAV__
</nav>

<main>
__BODY__
</main>

<div class="footer">
Geology &amp; Mining Services
</div>

</body>

</html>
"""

    return (
        page
        .replace("__TITLE__", esc(title))
        .replace("__NAV__", nav)
        .replace("__BODY__", body)
    )


# ============================================================
# MODEL
# ============================================================

def restore_model_from_supabase():

    if os.path.exists(MODEL_OUTPUT):
        return True

    try:

        connection = db()
        cursor = connection.cursor()

        cursor.execute("""
            SELECT model_data
            FROM trained_model
            WHERE id = 1
        """)

        row = cursor.fetchone()

        cursor.close()
        connection.close()

        if row and row["model_data"]:

            os.makedirs(
                os.path.dirname(MODEL_OUTPUT),
                exist_ok=True
            )

            with open(
                MODEL_OUTPUT,
                "wb"
            ) as file:

                file.write(
                    bytes(row["model_data"])
                )

            print(
                "Trained model restored from Supabase."
            )

            return True

    except Exception:

        traceback.print_exc()

    return False


def load_model():

    global MODEL
    global MODEL_ERROR

    with MODEL_LOCK:

        if MODEL is not None:
            return MODEL

        try:

            ensure_directories()

            restore_model_from_supabase()

            model_path = None

            if os.path.exists(MODEL_OUTPUT):

                model_path = MODEL_OUTPUT

            elif os.path.exists(MODEL_BACKUP):

                model_path = MODEL_BACKUP

            if model_path:

                print(
                    "Loading trained model:",
                    model_path
                )

                MODEL = YOLO(model_path)

                MODEL_ERROR = ""

                return MODEL

            print(
                "Loading base YOLO model: yolo11n.pt"
            )

            MODEL = YOLO("yolo11n.pt")

            MODEL_ERROR = (
                "Base model loaded. "
                "Train the bucket model."
            )

            return MODEL

        except Exception as error:

            MODEL_ERROR = str(error)

            traceback.print_exc()

            return None


def save_model_to_supabase(path):

    with open(path, "rb") as file:
        model_bytes = file.read()

    connection = db()
    cursor = connection.cursor()

    try:

        cursor.execute(
            """
            INSERT INTO trained_model
            (
                id,
                model_data,
                filename,
                created_at
            )
            VALUES
            (
                1,
                %s,
                %s,
                NOW()
            )

            ON CONFLICT(id)
            DO UPDATE SET
                model_data = EXCLUDED.model_data,
                filename = EXCLUDED.filename,
                created_at = NOW()
            """,
            (
                psycopg2.Binary(model_bytes),
                "bucket_best.pt"
            )
        )

        connection.commit()

    finally:

        cursor.close()
        connection.close()


# ============================================================
# DETECTION
# ============================================================

def normalize_class(class_id, class_name=""):

    if class_id in CLASS_IDS:
        return CLASS_IDS[class_id]

    name = str(class_name).upper()

    if "EMPTY" in name:
        return "BUCKET_EMPTY"

    if (
        "PEOPLE" in name
        or "PERSON" in name
    ):
        return "PEOPLE"

    if (
        "EQUIPMENT" in name
        or "MACHINE" in name
    ):
        return "EQUIPMENT"

    if "LOADED" in name:
        return "BUCKET_LOADED"

    return "UNKNOWN"


def detect(image_bytes):

    model = load_model()

    if model is None:

        raise RuntimeError(
            "YOLO model haijapatikana: "
            + MODEL_ERROR
        )

    image = Image.open(
        io.BytesIO(image_bytes)
    ).convert("RGB")

    array = np.array(image)

    results = model.predict(
        source=array,
        conf=YOLO_CONFIDENCE,
        imgsz=YOLO_IMAGE_SIZE,
        device="cpu",
        verbose=False,
    )

    detections = []

    if (
        not results
        or results[0].boxes is None
    ):
        return detections

    names = getattr(
        results[0],
        "names",
        {}
    )

    for box in results[0].boxes:

        try:

            xyxy = (
                box.xyxy[0]
                .cpu()
                .numpy()
                .tolist()
            )

            confidence = float(
                box.conf[0]
                .cpu()
                .item()
            )

            class_id = int(
                box.cls[0]
                .cpu()
                .item()
            )

            raw_name = names.get(
                class_id,
                ""
            )

            detections.append({

                "class_id": class_id,

                "class_name": normalize_class(
                    class_id,
                    raw_name
                ),

                "raw_class_name": str(
                    raw_name
                ),

                "confidence": round(
                    confidence,
                    4
                ),

                "x1": round(
                    float(xyxy[0]),
                    2
                ),

                "y1": round(
                    float(xyxy[1]),
                    2
                ),

                "x2": round(
                    float(xyxy[2]),
                    2
                ),

                "y2": round(
                    float(xyxy[3]),
                    2
                ),

            })

        except Exception:

            traceback.print_exc()

    return detections


# ============================================================
# COUNT LOADED BUCKET
# ============================================================

def count_loaded(detections):

    global LAST_COUNT_TIME

    loaded = [
        item
        for item in detections
        if item["class_name"] == "BUCKET_LOADED"
    ]

    if not loaded:

        return {
            "counted": False,
            "reason": "No loaded bucket detected.",
            "count": 0,
        }

    if (
        time.time() - LAST_COUNT_TIME
        < COUNT_COOLDOWN_SECONDS
    ):

        return {
            "counted": False,
            "reason": "Cooldown: possible same bucket.",
            "count": 0,
        }

    best = max(
        loaded,
        key=lambda item: item["confidence"]
    )

    confidence = float(
        best["confidence"]
    )

    connection = db()
    cursor = connection.cursor()

    try:

        cursor.execute(
            """
            SELECT id
            FROM daily_counts
            WHERE count_date = CURRENT_DATE
            LIMIT 1
            """
        )

        existing = cursor.fetchone()

        if existing:

            cursor.execute(
                """
                UPDATE daily_counts
                SET
                    bucket_count =
                        COALESCE(bucket_count,0) + 1,
                    updated_at = NOW()
                WHERE id = %s
                """,
                (existing["id"],)
            )

        else:

            cursor.execute(
                """
                INSERT INTO daily_counts
                (
                    count_date,
                    bucket_count,
                    updated_at
                )
                VALUES
                (
                    CURRENT_DATE,
                    1,
                    NOW()
                )
                """
            )

        cursor.execute(
            """
            INSERT INTO detection_events
            (
                class_name,
                confidence,
                counted
            )
            VALUES
            (
                %s,
                %s,
                TRUE
            )
            """,
            (
                "BUCKET_LOADED",
                confidence
            )
        )

        connection.commit()

    finally:

        cursor.close()
        connection.close()

    LAST_COUNT_TIME = time.time()

    return {

        "counted": True,

        "reason": "Loaded bucket counted.",

        "count": 1,

        "confidence": confidence,

    }


# ============================================================
# HISTORY
# ============================================================

def today_count():

    connection = db()
    cursor = connection.cursor()

    try:

        cursor.execute(
            """
            SELECT bucket_count
            FROM daily_counts
            WHERE count_date = CURRENT_DATE
            LIMIT 1
            """
        )

        row = cursor.fetchone()

        if row:
            return int(
                row["bucket_count"]
            )

        return 0

    finally:

        cursor.close()
        connection.close()


# ============================================================
# DATASET
# ============================================================

def dataset_rows():

    connection = db()
    cursor = connection.cursor()

    try:

        cursor.execute(
            """
            SELECT
                di.id,
                di.filename,
                di.mime_type,
                di.created_at,
                COUNT(a.id) AS annotation_count

            FROM dataset_images di

            LEFT JOIN annotations a
                ON a.image_id = di.id

            GROUP BY
                di.id,
                di.filename,
                di.mime_type,
                di.created_at

            ORDER BY
                di.id DESC
            """
        )

        return cursor.fetchall()

    finally:

        cursor.close()
        connection.close()


def save_image(
    filename,
    mime_type,
    data
):

    connection = db()
    cursor = connection.cursor()

    try:

        cursor.execute(
            """
            INSERT INTO dataset_images
            (
                filename,
                image_data,
                mime_type
            )
            VALUES
            (
                %s,
                %s,
                %s
            )
            RETURNING id
            """,
            (
                filename,
                psycopg2.Binary(data),
                mime_type
            )
        )

        image_id = cursor.fetchone()["id"]

        connection.commit()

        return image_id

    finally:

        cursor.close()
        connection.close()


def save_annotations(
    image_id,
    annotations
):

    connection = db()
    cursor = connection.cursor()

    try:

        cursor.execute(
            """
            DELETE FROM annotations
            WHERE image_id = %s
            """,
            (image_id,)
        )

        for annotation in annotations:

            class_name = str(
                annotation.get(
                    "class_name",
                    "BUCKET_LOADED"
                )
            )

            x_center = float(
                annotation.get(
                    "x_center",
                    0
                )
            )

            y_center = float(
                annotation.get(
                    "y_center",
                    0
                )
            )

            box_width = float(
                annotation.get(
                    "box_width",
                    0
                )
            )

            box_height = float(
                annotation.get(
                    "box_height",
                    0
                )
            )

            cursor.execute(
                """
                INSERT INTO annotations
                (
                    image_id,
                    class_name,
                    x_center,
                    y_center,
                    box_width,
                    box_height
                )
                VALUES
                (
                    %s,
                    %s,
                    %s,
                    %s,
                    %s,
                    %s
                )
                """,
                (
                    image_id,
                    class_name,
                    x_center,
                    y_center,
                    box_width,
                    box_height
                )
            )

        connection.commit()

    finally:

        cursor.close()
        connection.close()


def delete_image(image_id):

    connection = db()
    cursor = connection.cursor()

    try:

        cursor.execute(
            """
            DELETE FROM dataset_images
            WHERE id = %s
            """,
            (image_id,)
        )

        deleted = cursor.rowcount > 0

        connection.commit()

        return deleted

    finally:

        cursor.close()
        connection.close()


def image_bytes(value):

    if value is None:
        return b""

    if isinstance(value, bytes):
        return value

    if isinstance(value, memoryview):
        return value.tobytes()

    if isinstance(value, bytearray):
        return bytes(value)

    try:
        return bytes(value)
    except Exception:
        return b""


# ============================================================
# TRAINING STATE
# ============================================================

def set_training(
    status,
    message,
    progress,
    run_id=None,
    epoch=None,
    finished=False
):

    progress = max(
        0,
        min(
            100,
            int(progress)
        )
    )

    connection = db()
    cursor = connection.cursor()

    try:

        if run_id:

            if finished:

                cursor.execute(
                    """
                    UPDATE training_state

                    SET
                        status = %s,
                        message = %s,
                        progress = %s,
                        epoch = COALESCE(%s,epoch),
                        updated_at = NOW(),
                        finished_at = NOW()

                    WHERE
                        id = 1
                        AND run_id = %s
                    """,
                    (
                        status,
                        message,
                        progress,
                        epoch,
                        str(run_id)
                    )
                )

            else:

                cursor.execute(
                    """
                    UPDATE training_state

                    SET
                        status = %s,
                        message = %s,
                        progress = %s,
                        epoch = COALESCE(%s,epoch),
                        updated_at = NOW()

                    WHERE
                        id = 1
                        AND run_id = %s
                    """,
                    (
                        status,
                        message,
                        progress,
                        epoch,
                        str(run_id)
                    )
                )

        else:

            cursor.execute(
                """
                UPDATE training_state

                SET
                    status = %s,
                    message = %s,
                    progress = %s,
                    epoch = COALESCE(%s,epoch),
                    updated_at = NOW()

                WHERE id = 1
                """,
                (
                    status,
                    message,
                    progress,
                    epoch
                )
            )

        connection.commit()

    finally:

        cursor.close()
        connection.close()


def mark_interrupted_on_startup():

    connection = db()
    cursor = connection.cursor()

    try:

        cursor.execute(
            """
            UPDATE training_state

            SET
                status = 'interrupted',

                message =
                    'Previous training stopped because Render restarted. Start a new training run.',

                progress = 0,

                epoch = 0,

                updated_at = NOW(),

                finished_at = NOW()

            WHERE
                id = 1

                AND status IN
                (
                    'preparing',
                    'training',
                    'saving'
                )
            """
        )

        if cursor.rowcount:

            print(
                "Previous training marked interrupted."
            )

        connection.commit()

    finally:

        cursor.close()
        connection.close()


# ============================================================
# POSTGRES TRAINING LOCK
# ============================================================

def acquire_training_lock():

    global TRAINING_LOCK_CONN

    connection = psycopg2.connect(
        DATABASE_URL,
        cursor_factory=RealDictCursor,
        connect_timeout=15
    )

    connection.autocommit = True

    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT pg_try_advisory_lock(%s)
        AS locked
        """,
        (TRAINING_LOCK_KEY,)
    )

    row = cursor.fetchone()

    cursor.close()

    if not row or not row["locked"]:

        connection.close()

        return None

    TRAINING_LOCK_CONN = connection

    return connection


def release_training_lock(connection=None):

    global TRAINING_LOCK_CONN

    connection = (
        connection
        or TRAINING_LOCK_CONN
    )

    if not connection:
        return

    try:

        cursor = connection.cursor()

        cursor.execute(
            """
            SELECT pg_advisory_unlock(%s)
            """,
            (TRAINING_LOCK_KEY,)
        )

        cursor.close()

    except Exception:
        pass

    try:
        connection.close()
    except Exception:
        pass

    TRAINING_LOCK_CONN = None


# ============================================================
# TRAINING CLAIM
# ============================================================

def claim_training_run():

    global TRAINING
    global TRAINING_RUN_ID

    lock_connection = (
        acquire_training_lock()
    )

    if not lock_connection:
        return None

    run_id = uuid.uuid4().hex

    connection = db()
    cursor = connection.cursor()

    try:

        cursor.execute(
            """
            UPDATE training_state

            SET
                status = 'preparing',

                message =
                    'Training accepted. Preparing dataset...',

                progress = 1,

                epoch = 0,

                epochs = %s,

                run_id = %s,

                server_id = %s,

                started_at = NOW(),

                finished_at = NULL,

                updated_at = NOW()

            WHERE
                id = 1

                AND status NOT IN
                (
                    'preparing',
                    'training',
                    'saving'
                )

            RETURNING id
            """,
            (
                TRAIN_EPOCHS,
                run_id,
                SERVER_INSTANCE_ID
            )
        )

        row = cursor.fetchone()

        if not row:

            connection.rollback()

            release_training_lock(
                lock_connection
            )

            return None

        connection.commit()

        TRAINING = True

        TRAINING_RUN_ID = run_id

        return run_id

    except Exception:

        connection.rollback()

        release_training_lock(
            lock_connection
        )

        raise

    finally:

        cursor.close()
        connection.close()


# ============================================================
# BUILD TRAINING DATASET
# ============================================================

def build_dataset():

    rows = dataset_rows()

    if not rows:

        raise RuntimeError(
            "Hakuna picha kwenye dataset."
        )

    root = tempfile.mkdtemp(
        prefix="neerika_train_"
    )

    train_images = os.path.join(
        root,
        "images",
        "train"
    )

    val_images = os.path.join(
        root,
        "images",
        "val"
    )

    train_labels = os.path.join(
        root,
        "labels",
        "train"
    )

    val_labels = os.path.join(
        root,
        "labels",
        "val"
    )

    os.makedirs(
        train_images,
        exist_ok=True
    )

    os.makedirs(
        val_images,
        exist_ok=True
    )

    os.makedirs(
        train_labels,
        exist_ok=True
    )

    os.makedirs(
        val_labels,
        exist_ok=True
    )

    usable = []

    connection = db()
    cursor = connection.cursor()

    try:

        for row in rows:

            image_id = row["id"]

            cursor.execute(
                """
                SELECT
                    image_data,
                    mime_type,
                    filename

                FROM dataset_images

                WHERE id = %s
                """,
                (image_id,)
            )

            image_row = cursor.fetchone()

            if not image_row:
                continue

            cursor.execute(
                """
                SELECT
                    class_name,
                    x_center,
                    y_center,
                    box_width,
                    box_height

                FROM annotations

                WHERE image_id = %s

                ORDER BY id
                """,
                (image_id,)
            )

            annotations = cursor.fetchall()

            loaded_annotations = [
                annotation
                for annotation in annotations
                if str(
                    annotation["class_name"]
                ).upper()
                == "BUCKET_LOADED"
            ]

            if not loaded_annotations:

                print(
                    "Skipping image",
                    image_id,
                    "because it has no BUCKET_LOADED label."
                )

                continue

            raw = image_bytes(
                image_row["image_data"]
            )

            if not raw:

                continue

            filename = str(
                image_row["filename"]
                or ""
            ).lower()

            mime = str(
                image_row["mime_type"]
                or ""
            ).lower()

            if (
                "png" in mime
                or filename.endswith(".png")
            ):

                extension = ".png"

            elif (
                "webp" in mime
                or filename.endswith(".webp")
            ):

                extension = ".webp"

            else:

                extension = ".jpg"

            usable.append(
                (
                    image_id,
                    raw,
                    extension,
                    loaded_annotations
                )
            )

    finally:

        cursor.close()
        connection.close()

    if not usable:

        shutil.rmtree(
            root,
            ignore_errors=True
        )

        raise RuntimeError(
            "Hakuna picha yenye BUCKET_LOADED annotation."
        )

    # ========================================================
    # TRAIN / VALIDATION SPLIT
    #
    # If only one image exists:
    # use same image for train and validation.
    #
    # This is only a pipeline fallback.
    # For real ML quality, add many different images.
    # ========================================================

    train_items = usable[:]

    if len(usable) >= 2:

        split_index = max(
            1,
            int(len(usable) * 0.8)
        )

        if split_index >= len(usable):

            split_index = len(usable) - 1

        train_items = usable[:split_index]

        val_items = usable[split_index:]

    else:

        train_items = usable

        val_items = usable

    # ========================================================
    # WRITE DATA
    # ========================================================

    def write_item(
        item,
        image_directory,
        label_directory,
        prefix
    ):

        image_id, raw, extension, annotations = item

        filename = (
            prefix
            + "_"
            + str(image_id)
            + extension
        )

        image_path = os.path.join(
            image_directory,
            filename
        )

        label_path = os.path.join(
            label_directory,
            os.path.splitext(filename)[0]
            + ".txt"
        )

        with open(
            image_path,
            "wb"
        ) as file:

            file.write(raw)

        with open(
            label_path,
            "w",
            encoding="utf-8"
        ) as file:

            for annotation in annotations:

                try:

                    x_center = max(
                        0.0,
                        min(
                            1.0,
                            float(
                                annotation["x_center"]
                            )
                        )
                    )

                    y_center = max(
                        0.0,
                        min(
                            1.0,
                            float(
                                annotation["y_center"]
                            )
                        )
                    )

                    box_width = max(
                        0.0,
                        min(
                            1.0,
                            float(
                                annotation["box_width"]
                            )
                        )
                    )

                    box_height = max(
                        0.0,
                        min(
                            1.0,
                            float(
                                annotation["box_height"]
                            )
                        )
                    )

                    file.write(
                        "0 "
                        f"{x_center:.6f} "
                        f"{y_center:.6f} "
                        f"{box_width:.6f} "
                        f"{box_height:.6f}\n"
                    )

                except Exception:

                    traceback.print_exc()

        return image_path, label_path

    for item in train_items:

        write_item(
            item,
            train_images,
            train_labels,
            "train"
        )

    for item in val_items:

        write_item(
            item,
            val_images,
            val_labels,
            "val"
        )

    # ========================================================
    # DATASET YAML
    # ========================================================

    yaml_path = os.path.join(
        root,
        "data.yaml"
    )

    with open(
        yaml_path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "path: "
            + root.replace("\\", "/")
            + "\n"
        )

        file.write(
            "train: images/train\n"
        )

        file.write(
            "val: images/val\n"
        )

        file.write(
            "names:\n"
        )

        file.write(
            "  0: BUCKET_LOADED\n"
        )

    return (
        root,
        yaml_path,
        len(train_items),
        len(val_items)
    )


# ============================================================
# TRAINING CALLBACK
# ============================================================

def make_training_callback(
    run_id
):

    def update_epoch(
        trainer
    ):

        try:

            epoch = int(
                getattr(
                    trainer,
                    "epoch",
                    0
                )
            ) + 1

            total_epochs = int(
                getattr(
                    trainer,
                    "epochs",
                    TRAIN_EPOCHS
                )
                or TRAIN_EPOCHS
            )

            if total_epochs <= 0:
                total_epochs = TRAIN_EPOCHS

            progress = 20 + int(
                (
                    epoch
                    / total_epochs
                ) * 70
            )

            progress = min(
                90,
                max(
                    20,
                    progress
                )
            )

            message = (
                f"Training epoch "
                f"{epoch}/{total_epochs}..."
            )

            print(
                "NEERIKA TRAINING:",
                message,
                progress,
                "%"
            )

            set_training(
                "training",
                message,
                progress,
                run_id,
                epoch
            )

        except Exception:

            traceback.print_exc()

    return update_epoch


# ============================================================
# TRAINING WORKER
# ============================================================

def training_worker(
    run_id,
    lock_connection
):

    global TRAINING
    global TRAINING_RUN_ID
    global MODEL
    global MODEL_ERROR

    root = None

    try:

        print(
            "============================================================"
        )

        print(
            "NEERIKA YOLO TRAINING STARTED"
        )

        print(
            "RUN:",
            run_id
        )

        print(
            "CPU MODE"
        )

        print(
            "IMAGE SIZE:",
            YOLO_IMAGE_SIZE
        )

        print(
            "BATCH:",
            TRAIN_BATCH
        )

        print(
            "WORKERS:",
            TRAIN_WORKERS
        )

        print(
            "EPOCHS:",
            TRAIN_EPOCHS
        )

        print(
            "============================================================"
        )

        # ----------------------------------------------------
        # STEP 1
        # ----------------------------------------------------

        set_training(
            "preparing",
            "Preparing training dataset...",
            5,
            run_id,
            0
        )

        root, yaml_path, train_count, val_count = (
            build_dataset()
        )

        print(
            "TRAIN IMAGES:",
            train_count
        )

        print(
            "VALIDATION IMAGES:",
            val_count
        )

        set_training(
            "preparing",
            (
                f"Dataset ready. "
                f"Training images: {train_count}. "
                f"Validation images: {val_count}."
            ),
            10,
            run_id,
            0
        )

        # ----------------------------------------------------
        # STEP 2 - MODEL
        # ----------------------------------------------------

        set_training(
            "preparing",
            "Loading YOLO base model...",
            12,
            run_id,
            0
        )

        print(
            "Loading YOLO model: yolo11n.pt"
        )

        model = YOLO(
            "yolo11n.pt"
        )

        print(
            "YOLO base model loaded."
        )

        # ----------------------------------------------------
        # CALLBACKS
        # ----------------------------------------------------

        epoch_callback = (
            make_training_callback(
                run_id
            )
        )

        # Current Ultralytics callback
        model.add_callback(
            "on_train_epoch_end",
            epoch_callback
        )

        # Compatibility callback
        model.add_callback(
            "on_fit_epoch_end",
            epoch_callback
        )

        # ----------------------------------------------------
        # STEP 3
        # ----------------------------------------------------

        set_training(
            "training",
            (
                f"Starting YOLO training "
                f"for {TRAIN_EPOCHS} epochs..."
            ),
            15,
            run_id,
            0
        )

        print(
            "STARTING YOLO TRAINING"
        )

        print(
            "Using:",
            "device=cpu",
            "workers=0",
            "batch=1",
            f"imgsz={YOLO_IMAGE_SIZE}"
        )

        # ----------------------------------------------------
        # IMPORTANT:
        # These settings are deliberately lightweight for
        # Render Free CPU.
        # ----------------------------------------------------

        output_project = os.path.join(
            root,
            "runs"
        )

        os.makedirs(
            output_project,
            exist_ok=True
        )

        model.train(

            data=yaml_path,

            epochs=TRAIN_EPOCHS,

            imgsz=YOLO_IMAGE_SIZE,

            batch=TRAIN_BATCH,

            workers=TRAIN_WORKERS,

            device="cpu",

            cache=False,

            project=output_project,

            name="neerika_bucket",

            exist_ok=True,

            pretrained=True,

            verbose=True,

            amp=False,

            plots=False,

            save=True,

            val=True,

            patience=100,

        )

        print(
            "YOLO model.train() returned successfully."
        )

        # ----------------------------------------------------
        # STEP 4 - FIND BEST MODEL
        # ----------------------------------------------------

        set_training(
            "saving",
            "Training finished. Searching for best.pt...",
            92,
            run_id,
            TRAIN_EPOCHS
        )

        possible_paths = [

            os.path.join(
                output_project,
                "neerika_bucket",
                "weights",
                "best.pt"
            ),

            os.path.join(
                output_project,
                "neerika_bucket",
                "weights",
                "last.pt"
            ),

        ]

        best_model = None

        for path in possible_paths:

            if os.path.exists(path):

                best_model = path

                break

        if not best_model:

            raise RuntimeError(
                "Training imekwisha lakini "
                "best.pt haikupatikana."
            )

        print(
            "BEST MODEL:",
            best_model
        )

        # ----------------------------------------------------
        # STEP 5 - SAVE LOCAL MODEL
        # ----------------------------------------------------

        set_training(
            "saving",
            "Saving trained bucket model...",
            94,
            run_id,
            TRAIN_EPOCHS
        )

        os.makedirs(
            MODEL_DIR,
            exist_ok=True
        )

        shutil.copy2(
            best_model,
            MODEL_OUTPUT
        )

        shutil.copy2(
            best_model,
            MODEL_BACKUP
        )

        print(
            "Model saved:",
            MODEL_OUTPUT
        )

        # ----------------------------------------------------
        # STEP 6 - SAVE SUPABASE
        # ----------------------------------------------------

        set_training(
            "saving",
            "Saving trained model to Supabase...",
            97,
            run_id,
            TRAIN_EPOCHS
        )

        save_model_to_supabase(
            MODEL_OUTPUT
        )

        print(
            "Model saved to Supabase."
        )

        # ----------------------------------------------------
        # STEP 7 - LOAD NEW MODEL
        # ----------------------------------------------------

        set_training(
            "saving",
            "Loading newly trained model...",
            98,
            run_id,
            TRAIN_EPOCHS
        )

        with MODEL_LOCK:

            MODEL = YOLO(
                MODEL_OUTPUT
            )

            MODEL_ERROR = ""

        # ----------------------------------------------------
        # COMPLETE
        # ----------------------------------------------------

        set_training(
            "completed",
            (
                "Training completed successfully. "
                f"{train_count} training image(s), "
                f"{val_count} validation image(s), "
                f"{TRAIN_EPOCHS} epochs. "
                "bucket_best.pt saved."
            ),
            100,
            run_id,
            TRAIN_EPOCHS,
            finished=True
        )

        print(
            "============================================================"
        )

        print(
            "NEERIKA TRAINING COMPLETE"
        )

        print(
            "MODEL:",
            MODEL_OUTPUT
        )

        print(
            "============================================================"
        )

    except Exception as error:

        traceback.print_exc()

        error_message = (
            "Training error: "
            + str(error)
        )

        print(
            "NEERIKA TRAINING ERROR:",
            error_message
        )

        try:

            set_training(
                "error",
                error_message,
                0,
                run_id,
                finished=True
            )

        except Exception:

            traceback.print_exc()

    finally:

        if root:

            shutil.rmtree(
                root,
                ignore_errors=True
            )

        TRAINING = False

        TRAINING_RUN_ID = None

        release_training_lock(
            lock_connection
        )


# ============================================================
# START TRAINING
# ============================================================

def start_training_request():

    run_id = claim_training_run()

    if not run_id:

        return None

    lock_connection = (
        TRAINING_LOCK_CONN
    )

    try:

        worker = threading.Thread(
            target=training_worker,
            args=(
                run_id,
                lock_connection
            ),
            daemon=True
        )

        worker.start()

        return run_id

    except Exception:

        TRAINING = False

        release_training_lock(
            lock_connection
        )

        raise


# ============================================================
# DASHBOARD
# ============================================================

def dashboard():

    count = today_count()

    body = f"""

<div class="card">

<h2>
Mining Production Dashboard
</h2>

<p>
NEERIKA BUCKET AI counts loaded ore/material
buckets coming from the mine shaft.
</p>

</div>

<div class="grid">

<div class="stat">

<div class="small">
Today's Loaded Buckets
</div>

<div class="num">
{esc(count)}
</div>

</div>

<div class="stat">

<div class="small">
AI System
</div>

<div class="num">
READY
</div>

</div>

<div class="stat">

<div class="small">
Counting Target
</div>

<div class="num">
LOADED
</div>

</div>

</div>

<div class="card">

<h3>
Not counted
</h3>

<ul>

<li>
Empty buckets
</li>

<li>
People
</li>

<li>
Equipment
</li>

</ul>

</div>

<div class="card">

<a href="/camera">
<button>
Open Camera
</button>
</a>

<a href="/training">
<button class="secondary">
Training
</button>
</a>

</div>

"""

    return layout(
        "Dashboard",
        body,
        "Dashboard"
    )


# ============================================================
# CAMERA
# ============================================================

def camera():

    body = r"""

<div class="card">

<h2>
Bucket Camera
</h2>

<p class="small">
Only BUCKET_LOADED is eligible for counting.
</p>

<video
    id="video"
    autoplay
    playsinline
    style="
        width:100%;
        border-radius:12px;
        background:#000;
    "
></video>

<canvas
    id="canvas"
    style="display:none"
></canvas>

<p>

<button onclick="startCamera()">
Start Camera
</button>

<button
    class="secondary"
    onclick="stopCamera()"
>
Stop
</button>

<button
    class="success"
    onclick="detectNow()"
>
Detect & Count
</button>

</p>

<div
    id="result"
    class="status"
>
Camera not started.
</div>

<div
    id="count"
    class="num"
>
0
</div>

</div>

<script>

let stream = null;

async function startCamera() {

    try {

        stream =
            await navigator.mediaDevices.getUserMedia({

                video: {
                    facingMode: {
                        ideal: "environment"
                    }
                },

                audio: false

            });

        document
            .getElementById("video")
            .srcObject = stream;

        document
            .getElementById("result")
            .innerText =
                "Camera started.";

    }

    catch(error) {

        document
            .getElementById("result")
            .innerText =
                "Camera error: "
                + error.message;

    }

}


function stopCamera() {

    if(stream) {

        stream
            .getTracks()
            .forEach(
                track => track.stop()
            );

    }

    stream = null;

    document
        .getElementById("result")
        .innerText =
            "Camera stopped.";

}


async function detectNow() {

    const video =
        document.getElementById("video");

    const canvas =
        document.getElementById("canvas");

    if(!video.videoWidth) {

        alert(
            "Start camera first."
        );

        return;
    }

    canvas.width =
        video.videoWidth;

    canvas.height =
        video.videoHeight;

    const context =
        canvas.getContext("2d");

    context.drawImage(
        video,
        0,
        0,
        canvas.width,
        canvas.height
    );

    document
        .getElementById("result")
        .innerText =
            "Detecting...";

    try {

        const response =
            await fetch(
                "/api/detect",
                {

                    method: "POST",

                    headers: {
                        "Content-Type":
                            "application/json"
                    },

                    body: JSON.stringify({

                        image:
                            canvas.toDataURL(
                                "image/jpeg",
                                0.85
                            ),

                        count: true

                    })

                }
            );

        const data =
            await response.json();

        if(!response.ok) {

            throw new Error(
                data.error
                || "Detection failed."
            );

        }

        const loaded =
            data.detections
                .filter(
                    x =>
                        x.class_name
                        ===
                        "BUCKET_LOADED"
                )
                .length;

        document
            .getElementById("result")
            .innerText =
                "Detected: "
                + data.detections.length
                + " | Loaded: "
                + loaded
                + " | "
                + (
                    data.count_result
                    ? data.count_result.reason
                    : ""
                );

        refreshCount();

    }

    catch(error) {

        document
            .getElementById("result")
            .innerText =
                "Error: "
                + error.message;

    }

}


async function refreshCount() {

    try {

        const response =
            await fetch(
                "/api/history",
                {
                    cache: "no-store"
                }
            );

        const data =
            await response.json();

        document
            .getElementById("count")
            .innerText =
                data.today_count || 0;

    }

    catch(error) {}

}

refreshCount();

</script>

"""

    return layout(
        "Camera",
        body,
        "Camera"
    )


# ============================================================
# BUCKET REGISTRATION
# ============================================================

def buckets():

    body = r"""

<div class="card">

<h2>
Bucket Registration
</h2>

<div class="row">

<label>
Bucket Name
</label>

<input
    id="bucketName"
    placeholder="NEERIKA Loaded Bucket"
>

</div>

<div class="row">

<label>
Description
</label>

<input
    id="bucketDescription"
    placeholder="Bucket description"
>

</div>

<button
    onclick="addBucket()"
>
Register Bucket
</button>

<div
    id="message"
    class="status"
></div>

</div>


<div class="card">

<h3>
Registered Buckets
</h3>

<div id="bucketList">
Loading...
</div>

</div>


<script>

function escapeHtml(value) {

    return String(value)
        .replaceAll("&","&amp;")
        .replaceAll("<","&lt;")
        .replaceAll(">","&gt;")
        .replaceAll('"',"&quot;")
        .replaceAll("'","&#039;");

}


async function loadBuckets() {

    const response =
        await fetch(
            "/api/buckets"
        );

    const data =
        await response.json();

    if(!data.buckets.length) {

        document
            .getElementById("bucketList")
            .innerHTML =
                "<p>No buckets registered.</p>";

        return;

    }

    let html =
        "<table>";

    html +=
        "<tr>"
        + "<th>Name</th>"
        + "<th>Description</th>"
        + "<th>Status</th>"
        + "<th>Action</th>"
        + "</tr>";

    data.buckets.forEach(
        bucket => {

            html +=
                "<tr>";

            html +=
                "<td>"
                + escapeHtml(
                    bucket.name
                )
                + "</td>";

            html +=
                "<td>"
                + escapeHtml(
                    bucket.description
                    || ""
                )
                + "</td>";

            html +=
                "<td>"
                + (
                    bucket.active
                    ? "ACTIVE"
                    : ""
                )
                + "</td>";

            html +=
                "<td>"
                + (
                    bucket.active
                    ?
                    "<button disabled>Active</button>"
                    :
                    "<button onclick='activateBucket("
                    + bucket.id
                    + ")'>Activate</button>"
                )
                + "</td>";

            html +=
                "</tr>";

        }
    );

    html +=
        "</table>";

    document
        .getElementById("bucketList")
        .innerHTML =
            html;

}


async function addBucket() {

    const name =
        document
            .getElementById(
                "bucketName"
            )
            .value
            .trim();

    const description =
        document
            .getElementById(
                "bucketDescription"
            )
            .value
            .trim();

    if(!name) {

        alert(
            "Enter bucket name."
        );

        return;
    }

    const response =
        await fetch(
            "/api/buckets",
            {

                method: "POST",

                headers: {
                    "Content-Type":
                        "application/json"
                },

                body: JSON.stringify({

                    name: name,

                    description:
                        description

                })

            }
        );

    const data =
        await response.json();

    document
        .getElementById("message")
        .innerText =
            data.message
            || data.error
            || "";

    if(response.ok) {

        document
            .getElementById(
                "bucketName"
            )
            .value = "";

        document
            .getElementById(
                "bucketDescription"
            )
            .value = "";

        loadBuckets();

    }

}


async function activateBucket(id) {

    const response =
        await fetch(
            "/api/buckets/active",
            {

                method: "POST",

                headers: {
                    "Content-Type":
                        "application/json"
                },

                body: JSON.stringify({
                    id: id
                })

            }
        );

    const data =
        await response.json();

    if(!response.ok) {

        alert(
            data.error
            || "Activation failed."
        );

    }

    loadBuckets();

}


loadBuckets();

</script>

"""

    return layout(
        "Buckets",
        body,
        "Buckets"
    )


# ============================================================
# TRAINING PAGE
# ============================================================

def training():

    connection = db()
    cursor = connection.cursor()

    try:

        cursor.execute(
            """
            SELECT
                status,
                message,
                progress,
                epoch,
                epochs,
                updated_at,
                run_id

            FROM training_state

            WHERE id = 1
            """
        )

        state = cursor.fetchone()

    finally:

        cursor.close()
        connection.close()

    if not state:

        state = {
            "status": "idle",
            "message": "Ready",
            "progress": 0,
            "epoch": 0,
            "epochs": TRAIN_EPOCHS,
            "updated_at": None,
            "run_id": None,
        }

    rows = dataset_rows()

    table_rows = ""

    for row in rows:

        table_rows += (
            "<tr>"
            "<td>"
            + str(row["id"])
            + "</td>"
            "<td>"
            + esc(row["filename"])
            + "</td>"
            "<td>"
            + str(row["annotation_count"])
            + "</td>"
            "<td>"
            "<button class='danger' "
            "onclick='deleteImage("
            + str(row["id"])
            + ")'>"
            "Delete"
            "</button>"
            "</td>"
            "</tr>"
        )

    if not table_rows:

        table_rows = (
            "<tr>"
            "<td colspan='4'>"
            "No dataset images."
            "</td>"
            "</tr>"
        )

    body = r"""

<div class="card">

<h2>
YOLO Training Dataset
</h2>

<p class="small">
Upload images and annotate them as BUCKET_LOADED.
</p>

<div class="row">

<label>
Image
</label>

<input
    id="imageFile"
    type="file"
    accept="image/*"
>

</div>

<button
    onclick="uploadImage()"
>
Upload Image
</button>

<div
    id="uploadMessage"
    class="status"
></div>

</div>


<div class="card">

<h3>
Training Status
</h3>

<p>
Status:
<b id="trainingStatus">
__STATUS__
</b>
</p>

<p id="trainingMessage">
__MESSAGE__
</p>

<p>
Epoch:
<b id="trainingEpoch">
__EPOCH__
</b>
/
<b id="trainingEpochs">
__EPOCHS__
</b>
</p>

<div class="progress">

<div
    id="trainingBar"
    class="bar"
    style="width:__PROGRESS__%"
>
__PROGRESS__%
</div>

</div>

<br>

<button
    id="startTrainingButton"
    class="success"
    onclick="startTraining()"
>
Start Training
</button>

<button
    class="secondary"
    onclick="pollTraining()"
>
Refresh
</button>

</div>


<div class="card">

<h3>
Training Settings
</h3>

<p>
CPU: <b>CPU</b>
</p>

<p>
Image Size: <b>320</b>
</p>

<p>
Batch: <b>1</b>
</p>

<p>
Workers: <b>0</b>
</p>

<p>
Epochs: <b>20</b>
</p>

<p class="small">
These settings are optimized for Render Free.
</p>

</div>


<div class="card">

<h3>
Dataset
</h3>

<table>

<tr>

<th>
ID
</th>

<th>
Filename
</th>

<th>
Annotations
</th>

<th>
Action
</th>

</tr>

__ROWS__

</table>

</div>


<script>

let trainingTimer = null;


function updateTrainingUI(data) {

    const status =
        data.status
        || "idle";

    const message =
        data.message
        || "";

    const progress =
        Number(
            data.progress
            || 0
        );

    const epoch =
        Number(
            data.epoch
            || 0
        );

    const epochs =
        Number(
            data.epochs
            || 20
        );

    document
        .getElementById(
            "trainingStatus"
        )
        .innerText =
            status;

    document
        .getElementById(
            "trainingMessage"
        )
        .innerText =
            message;

    document
        .getElementById(
            "trainingEpoch"
        )
        .innerText =
            epoch;

    document
        .getElementById(
            "trainingEpochs"
        )
        .innerText =
            epochs;

    const bar =
        document
            .getElementById(
                "trainingBar"
            );

    bar.style.width =
        progress + "%";

    bar.innerText =
        progress + "%";

    const active =
        status === "preparing"
        || status === "training"
        || status === "saving";

    const button =
        document
            .getElementById(
                "startTrainingButton"
            );

    button.disabled =
        active;

    button.innerText =
        active
        ? "Training Running..."
        : "Start Training";

    if(active) {

        if(trainingTimer) {

            clearTimeout(
                trainingTimer
            );

        }

        trainingTimer =
            setTimeout(
                pollTraining,
                2000
            );

    }

}


async function pollTraining() {

    try {

        const response =
            await fetch(
                "/api/training/status",
                {
                    cache: "no-store"
                }
            );

        const data =
            await response.json();

        updateTrainingUI(
            data
        );

    }

    catch(error) {

        document
            .getElementById(
                "trainingMessage"
            )
            .innerText =
                "Status connection error. Retrying...";

        trainingTimer =
            setTimeout(
                pollTraining,
                3000
            );

    }

}


async function startTraining() {

    const button =
        document
            .getElementById(
                "startTrainingButton"
            );

    button.disabled = true;

    button.innerText =
        "Starting...";

    try {

        const response =
            await fetch(
                "/api/training/start",
                {
                    method: "POST",
                    headers: {
                        "Content-Type":
                            "application/json"
                    }
                }
            );

        const data =
            await response.json();

        if(!response.ok) {

            alert(
                data.error
                || "Training could not be started."
            );

        }

        await pollTraining();

    }

    catch(error) {

        alert(
            "Connection error: "
            + error.message
        );

        await pollTraining();

    }

}


async function uploadImage() {

    const input =
        document
            .getElementById(
                "imageFile"
            );

    const file =
        input.files[0];

    if(!file) {

        alert(
            "Choose an image first."
        );

        return;
    }

    if(
        file.size
        > 15 * 1024 * 1024
    ) {

        alert(
            "Image is larger than 15 MB."
        );

        return;
    }

    const reader =
        new FileReader();

    reader.onload =
        async function() {

            try {

                const response =
                    await fetch(
                        "/api/dataset/upload",
                        {

                            method: "POST",

                            headers: {
                                "Content-Type":
                                    "application/json"
                            },

                            body:
                                JSON.stringify({

                                    filename:
                                        file.name,

                                    mime_type:
                                        file.type,

                                    data:
                                        reader.result

                                })

                        }
                    );

                const data =
                    await response.json();

                document
                    .getElementById(
                        "uploadMessage"
                    )
                    .innerText =
                        data.message
                        || data.error
                        || "";

                if(response.ok) {

                    setTimeout(
                        () =>
                            location.reload(),
                        500
                    );

                }

            }

            catch(error) {

                document
                    .getElementById(
                        "uploadMessage"
                    )
                    .innerText =
                        error.message;

            }

        };

    reader.readAsDataURL(
        file
    );

}


async function deleteImage(id) {

    if(
        !confirm(
            "Delete this image?"
        )
    ) {

        return;

    }

    const response =
        await fetch(
            "/api/dataset/delete",
            {

                method: "POST",

                headers: {
                    "Content-Type":
                        "application/json"
                },

                body: JSON.stringify({
                    id: id
                })

            }
        );

    const data =
        await response.json();

    if(!response.ok) {

        alert(
            data.error
            || "Delete failed."
        );

        return;
    }

    location.reload();

}


pollTraining();

</script>

"""

    body = (
        body
        .replace(
            "__STATUS__",
            esc(state["status"])
        )
        .replace(
            "__MESSAGE__",
            esc(state["message"])
        )
        .replace(
            "__EPOCH__",
            esc(state["epoch"])
        )
        .replace(
            "__EPOCHS__",
            esc(state["epochs"])
        )
        .replace(
            "__PROGRESS__",
            esc(state["progress"])
        )
        .replace(
            "__ROWS__",
            table_rows
        )
    )

    return layout(
        "Training",
        body,
        "Training"
    )


# ============================================================
# HISTORY
# ============================================================

def history():

    body = r"""

<div class="card">

<h2>
Production History
</h2>

<button
    onclick="loadHistory()"
>
Refresh
</button>

<a href="/api/history.csv">

<button class="secondary">
Export CSV
</button>

</a>

</div>


<div
    class="card"
    id="historyTable"
>
Loading...
</div>


<script>

async function loadHistory() {

    try {

        const response =
            await fetch(
                "/api/history",
                {
                    cache: "no-store"
                }
            );

        const data =
            await response.json();

        if(!data.history.length) {

            document
                .getElementById(
                    "historyTable"
                )
                .innerHTML =
                    "<p>No history yet.</p>";

            return;

        }

        let html =
            "<table>";

        html +=
            "<tr>"
            + "<th>Date</th>"
            + "<th>Loaded Buckets</th>"
            + "</tr>";

        data.history.forEach(
            row => {

                html +=
                    "<tr>"
                    + "<td>"
                    + row.count_date
                    + "</td>"
                    + "<td>"
                    + row.bucket_count
                    + "</td>"
                    + "</tr>";

            }
        );

        html +=
            "</table>";

        document
            .getElementById(
                "historyTable"
            )
            .innerHTML =
                html;

    }

    catch(error) {

        document
            .getElementById(
                "historyTable"
            )
            .innerText =
                error.message;

    }

}


loadHistory();

</script>

"""

    return layout(
        "History",
        body,
        "History"
    )


# ============================================================
# SETTINGS
# ============================================================

def settings():

    model_status = (
        "Loaded"
        if MODEL is not None
        else "Not loaded"
    )

    body = f"""

<div class="card">

<h2>
Settings
</h2>

<p>
YOLO confidence:
<b>
{esc(YOLO_CONFIDENCE)}
</b>
</p>

<p>
YOLO image size:
<b>
{esc(YOLO_IMAGE_SIZE)}
</b>
</p>

<p>
Training batch:
<b>
{esc(TRAIN_BATCH)}
</b>
</p>

<p>
Training workers:
<b>
{esc(TRAIN_WORKERS)}
</b>
</p>

<p>
Training epochs:
<b>
{esc(TRAIN_EPOCHS)}
</b>
</p>

<p>
Count cooldown:
<b>
{esc(COUNT_COOLDOWN_SECONDS)}
seconds
</b>
</p>

</div>


<div class="card">

<h3>
Model
</h3>

<p>
Status:
<b>
{esc(model_status)}
</b>
</p>

<p>
Model output:
<b>
ai/models/bucket_best.pt
</b>
</p>

</div>


<div class="card">

<h3>
Database
</h3>

<p>
Supabase / PostgreSQL:
<b>
{
    "Connected"
    if DATABASE_URL
    else
    "DATABASE_URL missing"
}
</b>
</p>

</div>

"""

    return layout(
        "Settings",
        body,
        "Settings"
    )


# ============================================================
# HTTP HANDLER
# ============================================================

class Handler(
    BaseHTTPRequestHandler
):

    def log_message(
        self,
        format_string,
        *args
    ):

        print(
            f"{self.address_string()} - "
            f"{format_string % args}"
        )


    def body_json(self):

        content_length = int(
            self.headers.get(
                "Content-Length",
                "0"
            )
        )

        if (
            content_length
            > MAX_JSON_BYTES
        ):

            raise ValueError(
                "Request too large."
            )

        raw = self.rfile.read(
            content_length
        )

        if not raw:

            return {}

        return json.loads(
            raw.decode("utf-8")
        )


    def do_HEAD(self):

        self.send_response(200)

        self.send_header(
            "Content-Type",
            "text/html; charset=utf-8"
        )

        self.end_headers()


    # ========================================================
    # GET
    # ========================================================

    def do_GET(self):

        try:

            path = urlparse(
                self.path
            ).path

            pages = {

                "/": dashboard,

                "/camera": camera,

                "/buckets": buckets,

                "/training": training,

                "/history": history,

                "/settings": settings,

            }

            if path in pages:

                return html_out(
                    self,
                    pages[path]()
                )

            # ------------------------------------------------
            # HEALTH
            # ------------------------------------------------

            if path == "/health":

                return json_out(
                    self,
                    {
                        "status": "ok",

                        "database_configured":
                            bool(DATABASE_URL),

                        "model_loaded":
                            MODEL is not None,

                        "training":
                            TRAINING,

                        "server_id":
                            SERVER_INSTANCE_ID,

                    }
                )

            # ------------------------------------------------
            # BUCKETS
            # ------------------------------------------------

            if path == "/api/buckets":

                connection = db()
                cursor = connection.cursor()

                try:

                    cursor.execute(
                        """
                        SELECT
                            id,
                            name,
                            description,
                            active,
                            created_at

                        FROM buckets

                        ORDER BY id DESC
                        """
                    )

                    rows =   cursor.fetchall()

                    return json_out(
                        self,
                        {
                            "buckets": rows
                        }
                    )

                finally:

                    cursor.close()
                    connection.close()

            # ------------------------------------------------
            # TRAINING STATUS
            # ------------------------------------------------

            if (
                path
                == "/api/training/status"
            ):

                connection = db()
                cursor = connection.cursor()

                try:

                    cursor.execute(
                        """
                        SELECT
                            status,
                            message,
                            progress,
                            epoch,
                            epochs,
                            updated_at,
                            run_id,
                            server_id,
                            started_at,
                            finished_at

                        FROM training_state

                        WHERE id = 1
                        """
                    )

                    row = cursor.fetchone()

                    if not row:

                        row = {

                            "status":
                                "idle",

                            "message":
                                "Ready",

                            "progress":
                                0,

                            "epoch":
                                0,

                            "epochs":
                                TRAIN_EPOCHS,

                        }

                    return json_out(
                        self,
                        row
                    )

                finally:

                    cursor.close()
                    connection.close()

            # ------------------------------------------------
            # HISTORY
            # ------------------------------------------------

            if path == "/api/history":

                connection = db()
                cursor = connection.cursor()

                try:

                    cursor.execute(
                        """
                        SELECT
                            count_date,
                            bucket_count,
                            updated_at

                        FROM daily_counts

                        ORDER BY
                            count_date DESC

                        LIMIT 100
                        """
                    )

                    rows =   cursor.fetchall()

                    current_count =  today_count()

                    return json_out(
                        self,
                        {
                            "history":
                                rows,

                            "today_count":
                                current_count,

                        }
                    )

                finally:

                    cursor.close()
                    connection.close()

            # ------------------------------------------------
            # CSV
            # ------------------------------------------------

            if (
                path
                == "/api/history.csv"
            ):

                connection = db()
                cursor = connection.cursor()

                try:

                    cursor.execute(
                        """
                        SELECT
                            count_date,
                            bucket_count,
                            updated_at

                        FROM daily_counts

                        ORDER BY
                            count_date DESC
                        """
                    )

                    rows =    cursor.fetchall()

                finally:

                    cursor.close()
                    connection.close()

                output =   io.StringIO()

                writer = csv.writer(
                        output
                    )

                writer.writerow(
                    [
                        "Date",
                        "Loaded Buckets",
                        "Updated At"
                    ]
                )

                for row in rows:

                    writer.writerow(
                        [
                            row["count_date"],
                            row["bucket_count"],
                            row["updated_at"]
                        ]
                    )

                body =    output.getvalue().encode(
                        "utf-8"
                    )

                self.send_response(
                    200
                )

                self.send_header(
                    "Content-Type",
                    "text/csv; charset=utf-8"
                )

                self.send_header(
                    "Content-Length",
                    str(len(body))
                )

                self.send_header(
                    "Content-Disposition",
                    "attachment; "
                    "filename=neerika_history.csv"
                )

                self.end_headers()

                self.wfile.write(
                    body
                )

                return

            # ------------------------------------------------
            # DATASET IMAGE
            # ------------------------------------------------

            if path.startswith(
                "/api/dataset/image/"
            ):

                image_id = int(
                    path.rsplit(
                        "/",
                        1
                    )[1]
                )

                connection = db()
                cursor = connection.cursor()

                try:

                    cursor.execute(
                        """
                        SELECT
                            image_data,
                            mime_type

                        FROM dataset_images

                        WHERE id = %s
                        """,
                        (image_id,)
                    )

                    row =     cursor.fetchone()

                finally:

                    cursor.close()
                    connection.close()

                if not row:

                    return error_out(
                        self,
                        "Image not found.",
                        404
                    )

                body =  image_bytes(
                        row["image_data"]
                    )

                self.send_response(
                    200
                )

                self.send_header(
                    "Content-Type",
                    row["mime_type"]
                    or "image/jpeg"
                )

                self.send_header(
                    "Content-Length",
                    str(len(body))
                )

                self.end_headers()

                self.wfile.write(
                    body
                )

                return

            return error_out(
                self,
                "Not found.",
                404
            )

        except Exception as error:

            traceback.print_exc()

            return error_out(
                self,
                error,
                500
            )


    # ========================================================
    # POST
    # ========================================================

    def do_POST(self):

        try:

            path = urlparse(
                self.path
            ).path

            # ------------------------------------------------
            # DETECTION
            # ------------------------------------------------

            if path == "/api/detect":

                data =   self.body_json()

                image_data = data.get(
                        "image",
                        ""
                    )

                if "," in image_data:

                    image_data = image_data.split(
                            ",",
                            1
                        )[1]

                image_bytes_data = base64.b64decode(
                        image_data
                    )

                detections = detect(
                        image_bytes_data
                    )

                count_result = None

                if data.get(
                    "count",
                    True
                ):

                    count_result =  count_loaded(
                            detections
                        )

                return json_out(
                    self,
                    {
                        "detections":
                            detections,

                        "count_result":
                            count_result,
                    }
                )

            # ------------------------------------------------
            # BUCKET
            # ------------------------------------------------

            if path == "/api/buckets":

                data = self.body_json()

                name =   str(
                        data.get(
                            "name",
                            ""
                        )
                    ).strip()

                description =  str(
                        data.get(
                            "description",
                            ""
                        )
                    ).strip()

                if not name:

                    raise ValueError(
                        "Bucket name is required."
                    )

                connection = db()
                cursor = connection.cursor()

                try:

                    cursor.execute(
                        """
                        INSERT INTO buckets
                        (
                            name,
                            description
                        )
                        VALUES
                        (
                            %s,
                            %s
                        )
                        RETURNING id
                        """,
                        (
                            name,
                            description
                        )
                    )

                    bucket_id =cursor.fetchone()["id"]

                    connection.commit()

                finally:

                    cursor.close()
                    connection.close()

                return json_out(
                    self,
                    {
                        "message":
                            "Bucket registered successfully.",

                        "id":
                            bucket_id
                    }
                )

            # ------------------------------------------------
            # ACTIVE BUCKET
            # ------------------------------------------------

            if (
                path
                == "/api/buckets/active"
            ):

                data = self.body_json()

                bucket_id = int(
                        data["id"]
                    )

                connection = db()
                cursor = connection.cursor()

                try:

                    cursor.execute(
                        """
                        UPDATE buckets
                        SET active = FALSE
                        """
                    )

                    cursor.execute(
                        """
                        UPDATE buckets
                        SET active = TRUE
                        WHERE id = %s
                        """,
                        (bucket_id,)
                    )

                    found =   cursor.rowcount > 0

                    connection.commit()

                finally:

                    cursor.close()
                    connection.close()

                if not found:

                    return error_out(
                        self,
                        "Bucket not found.",
                        404
                    )

                return json_out(
                    self,
                    {
                        "message":
                            "Bucket activated."
                    }
                )

            # ------------------------------------------------
            # DATASET UPLOAD
            # ------------------------------------------------

            if (
                path
                == "/api/dataset/upload"
            ):

                data =  self.body_json()

                encoded =  data.get(
                        "data",
                        ""
                    )

                if "," in encoded:

                    encoded =  encoded.split(
                            ",",
                            1
                        )[1]

                image_data = base64.b64decode(
                        encoded
                    )

                if (
                    len(image_data)
                    > MAX_UPLOAD_BYTES
                ):

                    raise ValueError(
                        "Image too large. Maximum 15 MB."
                    )

                image_id =   save_image(
                        str(
                            data.get(
                                "filename",
                                "image.jpg"
                            )
                        ),

                        str(
                            data.get(
                                "mime_type",
                                "image/jpeg"
                            )
                        ),

                        image_data
                    )

                return json_out(
                    self,
                    {
                        "message":
                            "Image uploaded successfully.",

                        "id":
                            image_id
                    }
                )

            # ------------------------------------------------
            # DATASET DELETE
            # ------------------------------------------------

            if (
                path
                == "/api/dataset/delete"
            ):

                data =self.body_json()

                image_id =  int(
                        data["id"]
                    )

                deleted =  delete_image(
                        image_id
                    )

                if not deleted:

                    return error_out(
                        self,
                        "Image not found.",
                        404
                    )

                return json_out(
                    self,
                    {
                        "message":
                            "Image deleted successfully."
                    }
                )

            # ------------------------------------------------
            # ANNOTATIONS
            # ------------------------------------------------

            if (
                path
                == "/api/dataset/annotations"
            ):

                data = self.body_json()

                save_annotations(
                    int(
                        data["image_id"]
                    ),
                    data.get(
                        "annotations",
                        []
                    )
                )

                return json_out(
                    self,
                    {
                        "message":
                            "Annotations saved successfully."
                    }
                )

            # ------------------------------------------------
            # TRAINING
            # ------------------------------------------------

            if (
                path
                == "/api/training/start"
            ):

                run_id =   start_training_request()

                if not run_id:

                    return error_out(
                        self,
                        "Training is already running. Wait for the current run to finish.",
                        409
                    )

                return json_out(
                    self,
                    {
                        "message":
                            "Training started.",

                        "run_id":
                            run_id
                    },
                    202
                )

            return error_out(
                self,
                "Not found.",
                404
            )

        except json.JSONDecodeError:

            return error_out(
                self,
                "Invalid JSON.",
                400
            )

        except ValueError as error:

            return error_out(
                self,
                error,
                400
            )

        except Exception as error:

            traceback.print_exc()

            return error_out(
                self,
                error,
                500
            )


# ============================================================
# MAIN
# ============================================================

def main():

    ensure_directories()

    print(
        "============================================================"
    )

    print(
        "NEERIKA BUCKET AI"
    )

    print(
        "Mining Production Bucket Counter"
    )

    print(
        "============================================================"
    )

    print(
        "BASE_DIR:",
        BASE_DIR
    )

    print(
        "DATASET_DIR:",
        DATASET_DIR
    )

    print(
        "MODEL_OUTPUT:",
        MODEL_OUTPUT
    )

    print(
        "YOLO_IMAGE_SIZE:",
        YOLO_IMAGE_SIZE
    )

    print(
        "TRAIN_EPOCHS:",
        TRAIN_EPOCHS
    )

    print(
        "TRAIN_BATCH:",
        TRAIN_BATCH
    )

    print(
        "TRAIN_WORKERS:",
        TRAIN_WORKERS
    )

    print(
        "PORT:",
        PORT
    )

    if DATABASE_URL:

        try:

            init_db()

            print(
                "Database initialized successfully."
            )

            mark_interrupted_on_startup()

        except Exception:

            traceback.print_exc()

    else:

        print(
            "WARNING: DATABASE_URL is not configured."
        )

    try:

        load_model()

    except Exception:

        traceback.print_exc()

    server =  ThreadingHTTPServer(
            (
                HOST,
                PORT
            ),
            Handler
        )

    print(
        "NEERIKA BUCKET AI running on port",
        PORT
    )

    server.serve_forever()


if __name__ == "__main__":

    main()
