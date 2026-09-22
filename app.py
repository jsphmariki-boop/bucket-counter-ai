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
import re
from datetime import datetime, date
from urllib.parse import urlparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# ============================================================
# NEERIKA BUCKET AI
# Mining Production Bucket Counter
# Full app.py replacement
# ============================================================

os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/Ultralytics")

import psycopg2
from psycopg2.extras import RealDictCursor

from PIL import Image, ImageDraw, ImageFont

try:
    import numpy as np
except Exception:
    np = None

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None


# ============================================================
# CONFIGURATION
# ============================================================

APP_NAME = "NEERIKA BUCKET AI"
APP_SUBTITLE = "Mining Production Bucket Counter"

PORT = int(os.environ.get("PORT", "10000"))

DATABASE_URL = os.environ.get("DATABASE_URL", "").strip()

APP_TIMEZONE = os.environ.get(
    "APP_TIMEZONE",
    "Africa/Dar_es_Salaam"
)

API_KEY = os.environ.get(
    "NEERIKA_API_KEY",
    ""
).strip()

MODEL_PATH = os.environ.get(
    "YOLO_MODEL_PATH",
    "ai/models/best.pt"
)

BASE_MODEL = os.environ.get(
    "YOLO_BASE_MODEL",
    "yolo11n.pt"
)

UPLOAD_DIR = "bucket_images"
DATASET_DIR = "dataset"
MODEL_DIR = "ai/models"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

CLASS_NAMES = {
    0: "BUCKET_LOADED",
    1: "BUCKET_EMPTY",
    2: "PEOPLE",
    3: "EQUIPMENT",
}


# ============================================================
# GLOBALS
# ============================================================

MODEL = None
MODEL_LOCK = threading.Lock()

TRAINING_LOCK = threading.Lock()
TRAINING_STATUS = {
    "status": "idle",
    "message": "Training has not started.",
    "progress": 0,
    "started_at": None,
    "finished_at": None,
    "error": None,
}

TRACKER_LOCK = threading.Lock()

TRACKS = {}
NEXT_TRACK_ID = 1

COUNT_LINE_Y = 55

SERVER_STARTED = time.time()


# ============================================================
# DATABASE
# ============================================================

def db():
    if not DATABASE_URL:
        raise RuntimeError(
            "DATABASE_URL is not configured."
        )

    return psycopg2.connect(
        DATABASE_URL,
        cursor_factory=RealDictCursor,
        connect_timeout=15,
        sslmode="require"
    )


def execute_sql(sql, params=None, fetch=False):
    conn = None

    try:
        conn = db()

        with conn.cursor() as cur:
            cur.execute(sql, params or ())

            result = None

            if fetch:
                result = cur.fetchall()

        conn.commit()

        return result

    except Exception:
        if conn:
            conn.rollback()
        raise

    finally:
        if conn:
            conn.close()


# ============================================================
# DATE / TIME
# ============================================================

def get_today_date():
    """
    Returns today's date using APP_TIMEZONE.

    Tanzania:
    Africa/Dar_es_Salaam
    """

    conn = None

    try:
        conn = db()

        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    (
                        NOW() AT TIME ZONE %s
                    )::DATE AS today
                """,
                (APP_TIMEZONE,)
            )

            row = cur.fetchone()

            return row["today"]

    finally:
        if conn:
            conn.close()


def now_string():
    conn = None

    try:
        conn = db()

        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    NOW() AT TIME ZONE %s AS local_time
                """,
                (APP_TIMEZONE,)
            )

            row = cur.fetchone()

            return str(row["local_time"])

    except Exception:
        return datetime.now().isoformat()

    finally:
        if conn:
            conn.close()


# ============================================================
# DATABASE INITIALIZATION / MIGRATION
# ============================================================

def init_db():
    conn = None

    try:
        conn = db()

        with conn.cursor() as cur:

            # ------------------------------------------------
            # DAILY COUNTS
            # ------------------------------------------------

            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS daily_counts (
                    id SERIAL PRIMARY KEY,
                    count_date DATE NOT NULL UNIQUE,
                    bucket_count INTEGER NOT NULL DEFAULT 0,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )

            # ------------------------------------------------
            # IMPORTANT MIGRATION
            #
            # Old versions may have created count_date as TEXT.
            # CREATE TABLE IF NOT EXISTS does NOT change an
            # existing column type.
            #
            # Therefore convert TEXT -> DATE here.
            # ------------------------------------------------

            cur.execute(
                """
                DO $$
                BEGIN

                    IF EXISTS (
                        SELECT 1
                        FROM information_schema.columns
                        WHERE table_schema = 'public'
                          AND table_name = 'daily_counts'
                          AND column_name = 'count_date'
                          AND data_type = 'text'
                    ) THEN

                        ALTER TABLE daily_counts
                        ALTER COLUMN count_date TYPE DATE
                        USING NULLIF(TRIM(count_date), '')::DATE;

                    END IF;

                END
                $$;
                """
            )

            # ------------------------------------------------
            # BUCKETS
            # ------------------------------------------------

            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS buckets (
                    id SERIAL PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT DEFAULT '',
                    active BOOLEAN NOT NULL DEFAULT FALSE,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )

            # ------------------------------------------------
            # BUCKET REFERENCES
            # ------------------------------------------------

            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS bucket_reference_images (
                    id SERIAL PRIMARY KEY,
                    bucket_id INTEGER NOT NULL
                        REFERENCES buckets(id)
                        ON DELETE CASCADE,
                    filename TEXT NOT NULL,
                    image_data TEXT NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )

            # ------------------------------------------------
            # DETECTION HISTORY
            # ------------------------------------------------

            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS detection_history (
                    id SERIAL PRIMARY KEY,
                    bucket_id INTEGER,
                    class_name TEXT NOT NULL,
                    confidence REAL DEFAULT 0,
                    source TEXT DEFAULT 'camera',
                    image_name TEXT DEFAULT '',
                    detected_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )

            # ------------------------------------------------
            # DATASET IMAGES
            # ------------------------------------------------

            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS dataset_images (
                    id SERIAL PRIMARY KEY,
                    filename TEXT NOT NULL,
                    image_data TEXT NOT NULL,
                    annotation_data TEXT DEFAULT '',
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )

            # ------------------------------------------------
            # APP SETTINGS
            # ------------------------------------------------

            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS app_settings (
                    setting_key TEXT PRIMARY KEY,
                    setting_value TEXT DEFAULT '',
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )

            # ------------------------------------------------
            # INDEXES
            # ------------------------------------------------

            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_daily_counts_date
                ON daily_counts(count_date)
                """
            )

            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_detection_history_date
                ON detection_history(detected_at)
                """
            )

            cur.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS one_active_bucket
                ON buckets(active)
                WHERE active = TRUE
                """
            )

        conn.commit()

    except Exception:
        if conn:
            conn.rollback()

        raise

    finally:
        if conn:
            conn.close()


# ============================================================
# DAILY COUNT
# ============================================================

def get_today_count():
    conn = None

    try:
        conn = db()

        with conn.cursor() as cur:

            cur.execute(
                """
                SELECT bucket_count
                FROM daily_counts
                WHERE count_date =
                    (
                        NOW() AT TIME ZONE %s
                    )::DATE
                LIMIT 1
                """,
                (APP_TIMEZONE,)
            )

            row = cur.fetchone()

            if not row:
                return 0

            return int(row["bucket_count"] or 0)

    finally:
        if conn:
            conn.close()


def atomic_increment_daily_count():
    """
    Safely increments today's bucket count.

    count_date is explicitly cast to DATE.
    """

    conn = None

    try:
        conn = db()

        with conn.cursor() as cur:

            cur.execute(
                """
                INSERT INTO daily_counts
                    (
                        count_date,
                        bucket_count,
                        updated_at
                    )
                VALUES
                    (
                        (
                            NOW() AT TIME ZONE %s
                        )::DATE,
                        1,
                        NOW()
                    )

                ON CONFLICT (count_date)

                DO UPDATE SET
                    bucket_count =
                        daily_counts.bucket_count + 1,
                    updated_at = NOW()

                RETURNING bucket_count
                """,
                (APP_TIMEZONE,)
            )

            row = cur.fetchone()

        conn.commit()

        return int(row["bucket_count"])

    except Exception:
        if conn:
            conn.rollback()

        raise

    finally:
        if conn:
            conn.close()


# ============================================================
# BUCKET MANAGEMENT
# ============================================================

def get_buckets():
    return execute_sql(
        """
        SELECT
            id,
            name,
            description,
            active,
            created_at
        FROM buckets
        ORDER BY id DESC
        """,
        fetch=True
    )


def get_active_bucket():
    rows = execute_sql(
        """
        SELECT
            id,
            name,
            description,
            active,
            created_at
        FROM buckets
        WHERE active = TRUE
        LIMIT 1
        """,
        fetch=True
    )

    if not rows:
        return None

    return rows[0]


def create_bucket(name, description=""):
    name = str(name or "").strip()

    if not name:
        raise ValueError("Bucket name is required.")

    rows = execute_sql(
        """
        INSERT INTO buckets
            (
                name,
                description,
                active
            )
        VALUES
            (
                %s,
                %s,
                FALSE
            )
        RETURNING id, name, description, active, created_at
        """,
        (name, description or ""),
        fetch=True
    )

    return rows[0]


def activate_bucket(bucket_id):
    conn = None

    try:
        conn = db()

        with conn.cursor() as cur:

            cur.execute(
                """
                UPDATE buckets
                SET active = FALSE
                """
            )

            cur.execute(
                """
                UPDATE buckets
                SET active = TRUE
                WHERE id = %s
                RETURNING id, name, description, active
                """,
                (int(bucket_id),)
            )

            row = cur.fetchone()

            if not row:
                raise ValueError("Bucket not found.")

        conn.commit()

        return row

    except Exception:
        if conn:
            conn.rollback()

        raise

    finally:
        if conn:
            conn.close()


def delete_bucket(bucket_id):
    execute_sql(
        """
        DELETE FROM buckets
        WHERE id = %s
        """,
        (int(bucket_id),)
    )


# ============================================================
# BUCKET REFERENCE IMAGES
# ============================================================

def save_bucket_reference(bucket_id, filename, image_data):
    execute_sql(
        """
        INSERT INTO bucket_reference_images
            (
                bucket_id,
                filename,
                image_data
            )
        VALUES
            (
                %s,
                %s,
                %s
            )
        """,
        (
            int(bucket_id),
            filename,
            image_data
        )
    )


def get_bucket_references(bucket_id):
    return execute_sql(
        """
        SELECT
            id,
            bucket_id,
            filename,
            image_data,
            created_at
        FROM bucket_reference_images
        WHERE bucket_id = %s
        ORDER BY id DESC
        """,
        (int(bucket_id),),
        fetch=True
    )


# ============================================================
# IMAGE UTILITIES
# ============================================================

def data_url_to_bytes(data_url):
    if not data_url:
        raise ValueError("Image data is empty.")

    if "," in data_url:
        data_url = data_url.split(",", 1)[1]

    return base64.b64decode(data_url)


def bytes_to_image(data):
    return Image.open(
        io.BytesIO(data)
    ).convert("RGB")


def image_to_data_url(image):
    buffer = io.BytesIO()

    image.save(
        buffer,
        format="JPEG",
        quality=88
    )

    encoded = base64.b64encode(
        buffer.getvalue()
    ).decode("utf-8")

    return "data:image/jpeg;base64," + encoded


def save_uploaded_image(data, filename):
    safe_name = re.sub(
        r"[^a-zA-Z0-9_.-]",
        "_",
        filename or "image.jpg"
    )

    unique_name = (
        str(uuid.uuid4())
        + "_"
        + safe_name
    )

    path = os.path.join(
        UPLOAD_DIR,
        unique_name
    )

    with open(path, "wb") as f:
        f.write(data)

    return path


# ============================================================
# YOLO MODEL
# ============================================================

def model_exists():
    return os.path.exists(MODEL_PATH)


def load_model_internal():
    global MODEL

    if YOLO is None:
        raise RuntimeError(
            "Ultralytics YOLO is not installed."
        )

    if not os.path.exists(MODEL_PATH):
        return None

    with MODEL_LOCK:

        if MODEL is None:
            MODEL = YOLO(MODEL_PATH)

    return MODEL


def reload_model():
    global MODEL

    with MODEL_LOCK:
        MODEL = None

    return load_model_internal()


def model_status():
    if YOLO is None:
        return {
            "status": "error",
            "message": "Ultralytics is not installed."
        }

    if not model_exists():
        return {
            "status": "not_ready",
            "message": "Custom YOLO model not found."
        }

    try:
        load_model_internal()

        return {
            "status": "ready",
            "message": "AI model loaded."
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }


# ============================================================
# DETECTION
# ============================================================

def detection_from_image(
    image,
    confidence=0.35
):
    model = load_model_internal()

    if model is None:
        raise RuntimeError(
            "AI model is not available. "
            "Train a model first and create "
            "ai/models/best.pt."
        )

    if np is None:
        raise RuntimeError(
            "NumPy is not installed."
        )

    image_np = np.array(image)

    results = model.predict(
        source=image_np,
        conf=float(confidence),
        verbose=False
    )

    detections = []

    for result in results:

        boxes = getattr(
            result,
            "boxes",
            None
        )

        if boxes is None:
            continue

        for box in boxes:

            xyxy = box.xyxy[0].tolist()

            cls = int(
                box.cls[0].item()
            )

            conf = float(
                box.conf[0].item()
            )

            x1, y1, x2, y2 = [
                float(v)
                for v in xyxy
            ]

            detections.append(
                {
                    "class_id": cls,
                    "class_name": CLASS_NAMES.get(
                        cls,
                        f"CLASS_{cls}"
                    ),
                    "confidence": round(
                        conf,
                        4
                    ),
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "cx": (x1 + x2) / 2,
                    "cy": (y1 + y2) / 2,
                }
            )

    return detections


# ============================================================
# SIMPLE TRACKER
# ============================================================

def reset_tracker():
    global TRACKS
    global NEXT_TRACK_ID

    with TRACKER_LOCK:
        TRACKS = {}
        NEXT_TRACK_ID = 1


def distance(a, b):
    dx = a["cx"] - b["cx"]
    dy = a["cy"] - b["cy"]

    return (
        dx * dx + dy * dy
    ) ** 0.5


def update_tracker(detections):
    global TRACKS
    global NEXT_TRACK_ID

    loaded = [
        d for d in detections
        if d["class_id"] == 0
    ]

    results = []

    with TRACKER_LOCK:

        used_tracks = set()

        for det in loaded:

            best_id = None
            best_distance = 999999

            for track_id, track in TRACKS.items():

                if track_id in used_tracks:
                    continue

                d = distance(
                    det,
                    track
                )

                if d < best_distance:
                    best_distance = d
                    best_id = track_id

            if best_id is None or best_distance > 100:

                best_id = NEXT_TRACK_ID
                NEXT_TRACK_ID += 1

                TRACKS[best_id] = {
                    "cx": det["cx"],
                    "cy": det["cy"],
                    "previous_y": det["cy"],
                    "counted": False,
                    "last_seen": time.time(),
                }

            track = TRACKS[best_id]

            track["previous_y"] = track["cy"]

            track["cx"] = det["cx"]
            track["cy"] = det["cy"]

            track["last_seen"] = time.time()

            used_tracks.add(best_id)

            det = dict(det)

            det["track_id"] = best_id

            det["previous_y"] = track["previous_y"]
            det["current_y"] = track["cy"]
            det["counted"] = track["counted"]

            results.append(det)

        # Remove old tracks
        now = time.time()

        old_ids = [
            tid
            for tid, track in TRACKS.items()
            if now - track["last_seen"] > 5
        ]

        for tid in old_ids:
            TRACKS.pop(tid, None)

    return results


# ============================================================
# COUNTING
# ============================================================

def count_loaded_bucket(detection):
    """
    Counts ONLY BUCKET_LOADED.

    Empty buckets, people and equipment
    are ignored.
    """

    if detection.get("class_id") != 0:
        return {
            "counted": False,
            "total": get_today_count()
        }

    track_id = detection.get(
        "track_id"
    )

    if track_id is None:
        return {
            "counted": False,
            "total": get_today_count()
        }

    current_y = float(
        detection.get(
            "current_y",
            0
        )
    )

    previous_y = float(
        detection.get(
            "previous_y",
            current_y
        )
    )

    counted_now = False

    with TRACKER_LOCK:

        track = TRACKS.get(
            track_id
        )

        if track is None:
            return {
                "counted": False,
                "total": get_today_count()
            }

        # Detect crossing of counting line.
        crossed = (
            previous_y < COUNT_LINE_Y
            and current_y >= COUNT_LINE_Y
        )

        if crossed and not track["counted"]:

            track["counted"] = True

            counted_now = True

    if counted_now:

        total = atomic_increment_daily_count()

        try:
            execute_sql(
                """
                INSERT INTO detection_history
                    (
                        bucket_id,
                        class_name,
                        confidence,
                        source,
                        image_name
                    )
                VALUES
                    (
                        %s,
                        %s,
                        %s,
                        %s,
                        %s
                    )
                """,
                (
                    (
                        get_active_bucket() or {}
                    ).get("id"),
                    "BUCKET_LOADED",
                    detection.get(
                        "confidence",
                        0
                    ),
                    "camera",
                    ""
                )
            )

        except Exception:
            pass

        return {
            "counted": True,
            "total": total
        }

    return {
        "counted": False,
        "total": get_today_count()
    }


# ============================================================
# TRAINING
# ============================================================

def set_training_status(
    status,
    message,
    progress=0,
    error=None
):
    TRAINING_STATUS.update(
        {
            "status": status,
            "message": message,
            "progress": progress,
            "error": error,
        }
    )


def get_training_status():
    return dict(TRAINING_STATUS)


def create_dataset_yaml(dataset_path):
    yaml_path = os.path.join(
        dataset_path,
        "data.yaml"
    )

    content = f"""
path: {os.path.abspath(dataset_path)}
train: images
val: images

names:
  0: BUCKET_LOADED
  1: BUCKET_EMPTY
  2: PEOPLE
  3: EQUIPMENT
"""

    with open(
        yaml_path,
        "w",
        encoding="utf-8"
    ) as f:
        f.write(content.strip())

    return yaml_path


def train_model():
    global MODEL

    if YOLO is None:
        raise RuntimeError(
            "Ultralytics YOLO is not installed."
        )

    train_images = os.path.join(
        DATASET_DIR,
        "images"
    )

    labels = os.path.join(
        DATASET_DIR,
        "labels"
    )

    if not os.path.isdir(train_images):
        raise RuntimeError(
            "Training images folder does not exist."
        )

    if not os.path.isdir(labels):
        raise RuntimeError(
            "Training labels folder does not exist."
        )

    image_files = [
        x for x in os.listdir(
            train_images
        )
        if x.lower().endswith(
            (
                ".jpg",
                ".jpeg",
                ".png",
                ".webp"
            )
        )
    ]

    if len(image_files) < 5:
        raise RuntimeError(
            "At least 5 training images are required."
        )

    yaml_path = create_dataset_yaml(
        DATASET_DIR
    )

    epochs = int(
        os.environ.get(
            "YOLO_EPOCHS",
            "20"
        )
    )

    set_training_status(
        "training",
        f"Training YOLO for {epochs} epochs...",
        1
    )

    base_model = YOLO(
        BASE_MODEL
    )

    result = base_model.train(
        data=yaml_path,
        epochs=epochs,
        imgsz=640,
        project=MODEL_DIR,
        name="neerika_bucket",
        exist_ok=True
    )

    best_path = os.path.join(
        MODEL_DIR,
        "neerika_bucket",
        "weights",
        "best.pt"
    )

    if not os.path.exists(best_path):
        raise RuntimeError(
            "Training finished but best.pt was not found."
        )

    final_path = os.path.join(
        MODEL_DIR,
        "best.pt"
    )

    shutil.copy2(
        best_path,
        final_path
    )

    MODEL_PATH = final_path

    with MODEL_LOCK:
        MODEL = None

    load_model_internal()

    set_training_status(
        "completed",
        "Training completed successfully.",
        100
    )

    return {
        "success": True,
        "model": final_path
    }


def training_worker():
    try:

        train_model()

        TRAINING_STATUS[
            "finished_at"
        ] = now_string()

    except Exception as e:

        set_training_status(
            "error",
            str(e),
            0,
            str(e)
        )

        TRAINING_STATUS[
            "finished_at"
        ] = now_string()


def start_training():
    if not TRAINING_LOCK.acquire(
        blocking=False
    ):
        return {
            "success": False,
            "message": "Training is already running."
        }

    TRAINING_STATUS.update(
        {
            "status": "starting",
            "message": "Starting training...",
            "progress": 0,
            "started_at": now_string(),
            "finished_at": None,
            "error": None,
        }
    )

    thread = threading.Thread(
        target=training_worker,
        daemon=True
    )

    thread.start()

    return {
        "success": True,
        "message": "Training started."
    }


# ============================================================
# DATASET
# ============================================================

def save_dataset_image(
    filename,
    image_data
):
    image_bytes = data_url_to_bytes(
        image_data
    )

    safe_name = re.sub(
        r"[^a-zA-Z0-9_.-]",
        "_",
        filename or "image.jpg"
    )

    image_dir = os.path.join(
        DATASET_DIR,
        "images"
    )

    label_dir = os.path.join(
        DATASET_DIR,
        "labels"
    )

    os.makedirs(
        image_dir,
        exist_ok=True
    )

    os.makedirs(
        label_dir,
        exist_ok=True
    )

    unique = str(uuid.uuid4())

    ext = os.path.splitext(
        safe_name
    )[1].lower()

    if ext not in (
        ".jpg",
        ".jpeg",
        ".png"
    ):
        ext = ".jpg"

    final_name = unique + ext

    image_path = os.path.join(
        image_dir,
        final_name
    )

    with open(
        image_path,
        "wb"
    ) as f:
        f.write(image_bytes)

    # Empty label initially.
    label_path = os.path.join(
        label_dir,
        os.path.splitext(
            final_name
        )[0] + ".txt"
    )

    if not os.path.exists(
        label_path
    ):
        open(
            label_path,
            "w"
        ).close()

    execute_sql(
        """
        INSERT INTO dataset_images
            (
                filename,
                image_data,
                annotation_data
            )
        VALUES
            (
                %s,
                %s,
                %s
            )
        """,
        (
            final_name,
            image_data,
            ""
        )
    )

    return final_name


# ============================================================
# API KEY
# ============================================================

def check_api_key(headers):
    """
    If NEERIKA_API_KEY is empty, API authentication is disabled.

    If configured, clients must send:
        X-API-Key: your-key
    """

    if not API_KEY:
        return True

    supplied = headers.get(
        "X-API-Key",
        ""
    ).strip()

    return supplied == API_KEY


# ============================================================
# JSON HELPERS
# ============================================================

def json_response(
    handler,
    data,
    status=200
):
    raw = json.dumps(
        data,
        default=str
    ).encode(
        "utf-8"
    )

    handler.send_response(status)

    handler.send_header(
        "Content-Type",
        "application/json; charset=utf-8"
    )

    handler.send_header(
        "Content-Length",
        str(len(raw))
    )

    handler.send_header(
        "Access-Control-Allow-Origin",
        "*"
    )

    handler.send_header(
        "Access-Control-Allow-Headers",
        "Content-Type, X-API-Key"
    )

    handler.send_header(
        "Access-Control-Allow-Methods",
        "GET, POST, PUT, DELETE, OPTIONS"
    )

    handler.end_headers()

    handler.wfile.write(raw)


def error_response(
    handler,
    message,
    status=400
):
    json_response(
        handler,
        {
            "success": False,
            "error": str(message)
        },
        status
    )


def read_json(handler):
    length = int(
        handler.headers.get(
            "Content-Length",
            "0"
        )
    )

    if length <= 0:
        return {}

    raw = handler.rfile.read(
        length
    )

    if not raw:
        return {}

    return json.loads(
        raw.decode("utf-8")
    )


# ============================================================
# HTTP SERVER
# ============================================================

class Handler(BaseHTTPRequestHandler):

    server_version = "NEERIKA-BUCKET-AI/1.0"

    def log_message(
        self,
        format,
        *args
    ):
        print(
            "%s - %s"
            % (
                self.address_string(),
                format % args
            )
        )

    def do_OPTIONS(self):
        self.send_response(204)

        self.send_header(
            "Access-Control-Allow-Origin",
            "*"
        )

        self.send_header(
            "Access-Control-Allow-Headers",
            "Content-Type, X-API-Key"
        )

        self.send_header(
            "Access-Control-Allow-Methods",
            "GET, POST, PUT, DELETE, OPTIONS"
        )

        self.end_headers()

    def do_GET(self):

        parsed = urlparse(
            self.path
        )

        path = parsed.path

        try:

            # ---------------------------------------------
            # FRONTEND
            # ---------------------------------------------

            if path == "/":

                html = render_html()

                raw = html.encode(
                    "utf-8"
                )

                self.send_response(200)

                self.send_header(
                    "Content-Type",
                    "text/html; charset=utf-8"
                )

                self.send_header(
                    "Content-Length",
                    str(len(raw))
                )

                self.end_headers()

                self.wfile.write(raw)

                return

            # ---------------------------------------------
            # HEALTH
            # ---------------------------------------------

            if path == "/health":

                json_response(
                    self,
                    {
                        "status": "ok",
                        "app": APP_NAME,
                        "database": bool(
                            DATABASE_URL
                        ),
                        "uptime": round(
                            time.time()
                            - SERVER_STARTED,
                            2
                        )
                    }
                )

                return

            # ---------------------------------------------
            # DASHBOARD
            # ---------------------------------------------

            if path == "/api/dashboard":

                active = get_active_bucket()

                json_response(
                    self,
                    {
                        "success": True,
                        "today_count":
                            get_today_count(),
                        "active_bucket":
                            active,
                        "model":
                            model_status(),
                        "time":
                            now_string()
                    }
                )

                return

            # ---------------------------------------------
            # BUCKETS
            # ---------------------------------------------

            if path == "/api/buckets":

                json_response(
                    self,
                    {
                        "success": True,
                        "buckets":
                            get_buckets()
                    }
                )

                return

            # ---------------------------------------------
            # ACTIVE BUCKET
            # ---------------------------------------------

            if path == "/api/active-bucket":

                json_response(
                    self,
                    {
                        "success": True,
                        "bucket":
                            get_active_bucket()
                    }
                )

                return

            # ---------------------------------------------
            # REFERENCES
            # ---------------------------------------------

            if path.startswith(
                "/api/bucket-references/"
            ):

                bucket_id = int(
                    path.rsplit(
                        "/",
                        1
                    )[1]
                )

                refs = get_bucket_references(
                    bucket_id
                )

                json_response(
                    self,
                    {
                        "success": True,
                        "references": refs
                    }
                )

                return

            # ---------------------------------------------
            # TRAINING STATUS
            # ---------------------------------------------

            if path == "/api/training/status":

                json_response(
                    self,
                    {
                        "success": True,
                        **get_training_status()
                    }
                )

                return

            # ---------------------------------------------
            # HISTORY
            # ---------------------------------------------

            if path == "/api/history":

                rows = execute_sql(
                    """
                    SELECT
                        id,
                        count_date,
                        bucket_count,
                        updated_at
                    FROM daily_counts
                    ORDER BY count_date DESC
                    LIMIT 100
                    """,
                    fetch=True
                )

                json_response(
                    self,
                    {
                        "success": True,
                        "history": rows
                    }
                )

                return

            # ---------------------------------------------
            # UNKNOWN
            # ---------------------------------------------

            error_response(
                self,
                "Endpoint not found.",
                404
            )

        except Exception as e:

            traceback.print_exc()

            error_response(
                self,
                str(e),
                500
            )

    def do_POST(self):

        path = urlparse(
            self.path
        ).path

        try:

            # API authentication
            if path.startswith("/api/"):

                if not check_api_key(
                    self.headers
                ):
                    error_response(
                        self,
                        "Invalid API key.",
                        401
                    )
                    return

            # ---------------------------------------------
            # CREATE BUCKET
            # ---------------------------------------------

            if path == "/api/buckets":

                data = read_json(
                    self
                )

                bucket = create_bucket(
                    data.get("name"),
                    data.get(
                        "description",
                        ""
                    )
                )

                json_response(
                    self,
                    {
                        "success": True,
                        "bucket": bucket
                    }
                )

                return

            # ---------------------------------------------
            # ACTIVATE BUCKET
            # ---------------------------------------------

            if path.startswith(
                "/api/buckets/"
            ) and path.endswith(
                "/activate"
            ):

                parts = path.strip(
                    "/"
                ).split("/")

                bucket_id = int(
                    parts[2]
                )

                bucket = activate_bucket(
                    bucket_id
                )

                reset_tracker()

                json_response(
                    self,
                    {
                        "success": True,
                        "bucket": bucket
                    }
                )

                return

            # ---------------------------------------------
            # DELETE BUCKET
            # ---------------------------------------------

            if path.startswith(
                "/api/buckets/"
            ) and path.endswith(
                "/delete"
            ):

                parts = path.strip(
                    "/"
                ).split("/")

                bucket_id = int(
                    parts[2]
                )

                delete_bucket(
                    bucket_id
                )

                json_response(
                    self,
                    {
                        "success": True
                    }
                )

                return

            # ---------------------------------------------
            # BUCKET REFERENCE IMAGE
            # ---------------------------------------------

            if path == "/api/bucket-reference":

                data = read_json(
                    self
                )

                bucket_id = int(
                    data.get(
                        "bucket_id"
                    )
                )

                filename = data.get(
                    "filename",
                    "reference.jpg"
                )

                image_data = data.get(
                    "image_data"
                )

                save_bucket_reference(
                    bucket_id,
                    filename,
                    image_data
                )

                json_response(
                    self,
                    {
                        "success": True,
                        "message":
                            "Reference image saved."
                    }
                )

                return

            # ---------------------------------------------
            # DETECT IMAGE
            # ---------------------------------------------

            if path == "/api/detect":

                data = read_json(
                    self
                )

                image_data = data.get(
                    "image"
                )

                if not image_data:
                    raise ValueError(
                        "Image is required."
                    )

                image_bytes = (
                    data_url_to_bytes(
                        image_data
                    )
                )

                image = bytes_to_image(
                    image_bytes
                )

                confidence = float(
                    data.get(
                        "confidence",
                        0.35
                    )
                )

                detections = (
                    detection_from_image(
                        image,
                        confidence
                    )
                )

                tracked = update_tracker(
                    detections
                )

                counted = False
                total = get_today_count()

                for det in tracked:

                    result = (
                        count_loaded_bucket(
                            det
                        )
                    )

                    if result["counted"]:
                        counted = True

                    total = result[
                        "total"
                    ]

                json_response(
                    self,
                    {
                        "success": True,
                        "detections":
                            tracked,
                        "counted":
                            counted,
                        "total":
                            total,
                        "line_y":
                            COUNT_LINE_Y
                    }
                )

                return

            # ---------------------------------------------
            # RESET TRACKER
            # ---------------------------------------------

            if path == "/api/tracker/reset":

                reset_tracker()

                json_response(
                    self,
                    {
                        "success": True,
                        "message":
                            "Tracker reset."
                    }
                )

                return

            # ---------------------------------------------
            # TRAINING
            # ---------------------------------------------

            if path == "/api/training/start":

                result = start_training()

                json_response(
                    self,
                    result
                )

                return

            # ---------------------------------------------
            # DATASET IMAGE
            # ---------------------------------------------

            if path == "/api/dataset/upload":

                data = read_json(
                    self
                )

                filename = data.get(
                    "filename",
                    "image.jpg"
                )

                image_data = data.get(
                    "image_data"
                )

                if not image_data:
                    raise ValueError(
                        "image_data is required."
                    )

                filename = (
                    save_dataset_image(
                        filename,
                        image_data
                    )
                )

                json_response(
                    self,
                    {
                        "success": True,
                        "filename":
                            filename
                    }
                )

                return

            # ---------------------------------------------
            # UNKNOWN
            # ---------------------------------------------

            error_response(
                self,
                "Endpoint not found.",
                404
            )

        except Exception as e:

            traceback.print_exc()

            error_response(
                self,
                str(e),
                500
            )


# ============================================================
# HTML FRONTEND
# ============================================================

def render_html():

    return r"""
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
    font-family:
        Arial,
        Helvetica,
        sans-serif;
    background: #f3f5f7;
    color: #17202a;
}

header {
    background: #111827;
    color: white;
    padding: 18px;
}

header h1 {
    margin: 0;
    font-size: 22px;
}

header p {
    margin: 5px 0 0;
    opacity: .8;
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
    padding: 15px 18px;
    cursor: pointer;
    font-weight: bold;
}

nav button:hover {
    background: #eef2f7;
}

.page {
    padding: 18px;
    max-width: 1200px;
    margin: auto;
}

.tab {
    display: none;
}

.tab.active {
    display: block;
}

.cards {
    display: grid;
    grid-template-columns:
        repeat(
            auto-fit,
            minmax(200px, 1fr)
        );
    gap: 15px;
}

.card {
    background: white;
    border-radius: 12px;
    padding: 20px;
    box-shadow:
        0 2px 10px
        rgba(0,0,0,.06);
}

.card h3 {
    margin-top: 0;
}

.big-number {
    font-size: 42px;
    font-weight: bold;
}

.status {
    padding: 10px;
    border-radius: 8px;
    background: #eef2f7;
}

button.primary {
    background: #111827;
    color: white;
}

button,
input,
textarea,
select {
    font: inherit;
}

button {
    padding: 10px 14px;
    border: 1px solid #ccd2d9;
    border-radius: 7px;
    cursor: pointer;
}

input,
textarea {
    width: 100%;
    padding: 10px;
    margin: 5px 0 12px;
    border: 1px solid #ccd2d9;
    border-radius: 7px;
}

video {
    width: 100%;
    max-height: 600px;
    background: #000;
    border-radius: 10px;
}

.camera-box {
    position: relative;
    width: 100%;
}

#cameraCanvas {
    position: absolute;
    left: 0;
    top: 0;
    width: 100%;
    height: 100%;
    pointer-events: none;
}

table {
    width: 100%;
    border-collapse: collapse;
    background: white;
}

th,
td {
    padding: 10px;
    border-bottom: 1px solid #ddd;
    text-align: left;
}

.bucket-item {
    background: white;
    padding: 15px;
    margin-bottom: 10px;
    border-radius: 10px;
    border: 1px solid #ddd;
}

.active {
    border: 2px solid #111827;
}

footer {
    text-align: center;
    padding: 30px;
    color: #777;
}

#message {
    margin-top: 10px;
    padding: 10px;
    border-radius: 8px;
    background: #eef2f7;
}

</style>

</head>

<body>

<header>

<h1>NEERIKA BUCKET AI</h1>

<p>
Mining Production Bucket Counter
</p>

</header>

<nav>

<button onclick="showTab('dashboard')">
Dashboard
</button>

<button onclick="showTab('camera')">
Camera
</button>

<button onclick="showTab('buckets')">
Buckets
</button>

<button onclick="showTab('training')">
Training
</button>

<button onclick="showTab('history')">
History
</button>

<button onclick="showTab('settings')">
Settings
</button>

</nav>

<div class="page">

<!-- =====================================================
     DASHBOARD
===================================================== -->

<section
    id="dashboard"
    class="tab active">

<h2>Dashboard</h2>

<div class="cards">

<div class="card">

<h3>
Today's Loaded Buckets
</h3>

<div
    id="todayCount"
    class="big-number">
0
</div>

</div>

<div class="card">

<h3>
Active Bucket
</h3>

<div id="activeBucket">
None
</div>

</div>

<div class="card">

<h3>
AI Model
</h3>

<div
    id="modelStatus"
    class="status">
Checking...
</div>

</div>

<div class="card">

<h3>
System Status
</h3>

<div
    id="systemStatus"
    class="status">
Checking...
</div>

</div>

</div>

</section>


<!-- =====================================================
     CAMERA
===================================================== -->

<section
    id="camera"
    class="tab">

<h2>Camera</h2>

<p>
The AI counts only
<strong>BUCKET_LOADED</strong>
objects crossing the counting line.
</p>

<div class="camera-box">

<video
    id="video"
    autoplay
    playsinline>
</video>

<canvas
    id="cameraCanvas">
</canvas>

</div>

<br>

<button
    class="primary"
    onclick="startCamera()">
Start Camera
</button>

<button
    onclick="stopCamera()">
Stop Camera
</button>

<button
    onclick="captureFrame()">
Detect / Count
</button>

<div id="cameraMessage"></div>

</section>


<!-- =====================================================
     BUCKETS
===================================================== -->

<section
    id="buckets"
    class="tab">

<h2>Bucket Registration</h2>

<div class="card">

<h3>
Register Bucket Type
</h3>

<input
    id="bucketName"
    placeholder="Example: Standard Ore Bucket">

<textarea
    id="bucketDescription"
    placeholder="Description">
</textarea>

<button
    class="primary"
    onclick="createBucket()">
Create Bucket
</button>

</div>

<br>

<div class="card">

<h3>
Reference Image
</h3>

<input
    type="file"
    id="referenceFile"
    accept="image/*">

<br>

<input
    id="referenceBucketId"
    placeholder="Bucket ID">

<br>

<button
    onclick="uploadReference()">
Save Reference Image
</button>

</div>

<br>

<div id="bucketList"></div>

</section>


<!-- =====================================================
     TRAINING
===================================================== -->

<section
    id="training"
    class="tab">

<h2>YOLO Training Dataset</h2>

<div class="card">

<p>
Upload images for training.
</p>

<input
    type="file"
    id="datasetFile"
    accept="image/*">

<br>

<button
    class="primary"
    onclick="uploadDataset()">
Upload Image
</button>

</div>

<br>

<div class="card">

<h3>
Training Status
</h3>

<div id="trainingStatus">
Checking...
</div>

<br>

<button
    class="primary"
    onclick="startTraining()">
Start YOLO Training
</button>

</div>

</section>


<!-- =====================================================
     HISTORY
===================================================== -->

<section
    id="history"
    class="tab">

<h2>Production History</h2>

<div class="card">

<table>

<thead>

<tr>
<th>Date</th>
<th>Loaded Buckets</th>
<th>Updated</th>
</tr>

</thead>

<tbody
    id="historyBody">
</tbody>

</table>

</div>

</section>


<!-- =====================================================
     SETTINGS
===================================================== -->

<section
    id="settings"
    class="tab">

<h2>Settings</h2>

<div class="card">

<h3>
Counting Line
</h3>

<input
    type="number"
    id="lineY"
    value="55">

<button
    onclick="saveLine()">
Save Counting Line
</button>

<p>
Only BUCKET_LOADED objects crossing
the counting line are counted.
</p>

</div>

<br>

<div class="card">

<h3>
System
</h3>

<p>
Application:
NEERIKA BUCKET AI
</p>

<p>
Timezone:
Africa/Dar_es_Salaam
</p>

<p>
Database:
Supabase PostgreSQL
</p>

</div>

</section>


<div id="message"></div>

</div>


<footer>
Geology & Mining Services
</footer>


<script>

let stream = null;
let detectionTimer = null;

let countLineY = 55;


// ======================================================
// TABS
// ======================================================

function showTab(name) {

    document
        .querySelectorAll(".tab")
        .forEach(
            x => x.classList.remove(
                "active"
            )
        );

    const element =
        document.getElementById(name);

    if (element) {
        element.classList.add(
            "active"
        );
    }

    if (name === "dashboard") {
        loadDashboard();
    }

    if (name === "buckets") {
        loadBuckets();
    }

    if (name === "history") {
        loadHistory();
    }

    if (name === "training") {
        loadTrainingStatus();
    }
}


// ======================================================
// MESSAGE
// ======================================================

function message(text) {

    document.getElementById(
        "message"
    ).textContent = text;
}


// ======================================================
// DASHBOARD
// ======================================================

async function loadDashboard() {

    try {

        const response =
            await fetch(
                "/api/dashboard"
            );

        const data =
            await response.json();

        if (!data.success) {
            throw new Error(
                data.error
            );
        }

        document.getElementById(
            "todayCount"
        ).textContent =
            data.today_count;

        const bucket =
            data.active_bucket;

        document.getElementById(
            "activeBucket"
        ).textContent =
            bucket
            ? bucket.name
            : "None";

        const model =
            data.model;

        document.getElementById(
            "modelStatus"
        ).textContent =
            model.message;

        document.getElementById(
            "systemStatus"
        ).textContent =
            "Online";

    } catch (error) {

        document.getElementById(
            "systemStatus"
        ).textContent =
            error.message;
    }
}


// ======================================================
// CAMERA
// ======================================================

async function startCamera() {

    try {

        stream =
            await navigator.mediaDevices
                .getUserMedia({
                    video: {
                        facingMode:
                            "environment"
                    },
                    audio: false
                });

        document.getElementById(
            "video"
        ).srcObject = stream;

        document.getElementById(
            "cameraMessage"
        ).textContent =
            "Camera started.";

    } catch (error) {

        document.getElementById(
            "cameraMessage"
        ).textContent =
            "Camera error: "
            + error.message;
    }
}


function stopCamera() {

    if (stream) {

        stream
            .getTracks()
            .forEach(
                track => track.stop()
            );

        stream = null;
    }

    document.getElementById(
        "cameraMessage"
    ).textContent =
        "Camera stopped.";
}


async function captureFrame() {

    const video =
        document.getElementById(
            "video"
        );

    if (!video.videoWidth) {

        document.getElementById(
            "cameraMessage"
        ).textContent =
            "Start camera first.";

        return;
    }

    const canvas =
        document.createElement(
            "canvas"
        );

    canvas.width =
        video.videoWidth;

    canvas.height =
        video.videoHeight;

    const ctx =
        canvas.getContext(
            "2d"
        );

    ctx.drawImage(
        video,
        0,
        0,
        canvas.width,
        canvas.height
    );

    const image =
        canvas.toDataURL(
            "image/jpeg",
            .85
        );

    try {

        document.getElementById(
            "cameraMessage"
        ).textContent =
            "AI detecting...";

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
                        image: image,
                        confidence: 0.35
                    })
                }
            );

        const data =
            await response.json();

        if (!data.success) {
            throw new Error(
                data.error
            );
        }

        drawDetections(
            data.detections,
            canvas.width,
            canvas.height
        );

        document.getElementById(
            "cameraMessage"
        ).textContent =
            "Detected: "
            + data.detections.length
            + " | Today: "
            + data.total
            + (
                data.counted
                ? " | BUCKET COUNTED"
                : ""
            );

        loadDashboard();

    } catch (error) {

        document.getElementById(
            "cameraMessage"
        ).textContent =
            error.message;
    }
}


function drawDetections(
    detections,
    width,
    height
) {

    const canvas =
        document.getElementById(
            "cameraCanvas"
        );

    canvas.width = width;
    canvas.height = height;

    const ctx =
        canvas.getContext(
            "2d"
        );

    ctx.clearRect(
        0,
        0,
        width,
        height
    );

    const line =
        height *
        (
            countLineY / 100
        );

    ctx.beginPath();

    ctx.moveTo(
        0,
        line
    );

    ctx.lineTo(
        width,
        line
    );

    ctx.strokeStyle =
        "red";

    ctx.lineWidth = 4;

    ctx.stroke();

    detections.forEach(
        d => {

            ctx.strokeStyle =
                d.class_id === 0
                ? "lime"
                : "yellow";

            ctx.lineWidth = 3;

            ctx.strokeRect(
                d.x1,
                d.y1,
                d.x2 - d.x1,
                d.y2 - d.y1
            );

            ctx.fillStyle =
                "white";

            ctx.font =
                "16px Arial";

            ctx.fillText(
                d.class_name
                + " "
                + (
                    d.confidence * 100
                ).toFixed(0)
                + "%",
                d.x1,
                Math.max(
                    18,
                    d.y1 - 5
                )
            );
        }
    );
}


// ======================================================
// BUCKETS
// ======================================================

async function loadBuckets() {

    try {

        const response =
            await fetch(
                "/api/buckets"
            );

        const data =
            await response.json();

        const list =
            document.getElementById(
                "bucketList"
            );

        list.innerHTML = "";

        data.buckets.forEach(
            bucket => {

                const div =
                    document.createElement(
                        "div"
                    );

                div.className =
                    "bucket-item"
                    + (
                        bucket.active
                        ? " active"
                        : ""
                    );

                div.innerHTML = `
                    <h3>
                        ${escapeHtml(
                            bucket.name
                        )}
                    </h3>

                    <p>
                        ID:
                        ${bucket.id}
                    </p>

                    <p>
                        ${escapeHtml(
                            bucket.description || ""
                        )}
                    </p>

                    <p>
                        ${
                            bucket.active
                            ? "ACTIVE"
                            : "Not active"
                        }
                    </p>

                    <button
                        onclick="activateBucket(
                            ${bucket.id}
                        )">
                        Make Active
                    </button>
                `;

                list.appendChild(
                    div
                );
            }
        );

    } catch (error) {

        message(
            error.message
        );
    }
}


async function createBucket() {

    const name =
        document.getElementById(
            "bucketName"
        ).value.trim();

    const description =
        document.getElementById(
            "bucketDescription"
        ).value.trim();

    if (!name) {
        message(
            "Enter bucket name."
        );
        return;
    }

    try {

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
                        name,
                        description
                    })
                }
            );

        const data =
            await response.json();

        if (!data.success) {
            throw new Error(
                data.error
            );
        }

        message(
            "Bucket created successfully."
        );

        document.getElementById(
            "bucketName"
        ).value = "";

        document.getElementById(
            "bucketDescription"
        ).value = "";

        loadBuckets();

    } catch (error) {

        message(
            error.message
        );
    }
}


async function activateBucket(
    id
) {

    try {

        const response =
            await fetch(
                `/api/buckets/${id}/activate`,
                {
                    method: "POST"
                }
            );

        const data =
            await response.json();

        if (!data.success) {
            throw new Error(
                data.error
            );
        }

        message(
            "Active bucket changed."
        );

        resetTracker();

        loadBuckets();
        loadDashboard();

    } catch (error) {

        message(
            error.message
        );
    }
}


async function uploadReference() {

    const file =
        document.getElementById(
            "referenceFile"
        ).files[0];

    const bucketId =
        document.getElementById(
            "referenceBucketId"
        ).value;

    if (!file || !bucketId) {

        message(
            "Choose image and enter bucket ID."
        );

        return;
    }

    const reader =
        new FileReader();

    reader.onload = async function() {

        try {

            const response =
                await fetch(
                    "/api/bucket-reference",
                    {
                        method: "POST",
                        headers: {
                            "Content-Type":
                                "application/json"
                        },
                        body:
                            JSON.stringify({
                                bucket_id:
                                    bucketId,
                                filename:
                                    file.name,
                                image_data:
                                    reader.result
                            })
                    }
                );

            const data =
                await response.json();

            if (!data.success) {
                throw new Error(
                    data.error
                );
            }

            message(
                "Reference image saved."
            );

        } catch (error) {

            message(
                error.message
            );
        }
    };

    reader.readAsDataURL(
        file
    );
}


// ======================================================
// TRAINING
// ======================================================

async function uploadDataset() {

    const file =
        document.getElementById(
            "datasetFile"
        ).files[0];

    if (!file) {

        message(
            "Choose an image first."
        );

        return;
    }

    const reader =
        new FileReader();

    reader.onload = async function() {

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
                                image_data:
                                    reader.result
                            })
                    }
                );

            const data =
                await response.json();

            if (!data.success) {
                throw new Error(
                    data.error
                );
            }

            message(
                "Training image uploaded."
            );

        } catch (error) {

            message(
                error.message
            );
        }
    };

    reader.readAsDataURL(
        file
    );
}


async function startTraining() {

    try {

        const response =
            await fetch(
                "/api/training/start",
                {
                    method: "POST"
                }
            );

        const data =
            await response.json();

        if (!data.success) {
            throw new Error(
                data.message
                || data.error
            );
        }

        message(
            "Training started."
        );

        loadTrainingStatus();

    } catch (error) {

        message(
            error.message
        );
    }
}


async function loadTrainingStatus() {

    try {

        const response =
            await fetch(
                "/api/training/status"
            );

        const data =
            await response.json();

        document.getElementById(
            "trainingStatus"
        ).innerHTML = `
            <strong>Status:</strong>
            ${escapeHtml(
                data.status || ""
            )}
            <br>
            <strong>Message:</strong>
            ${escapeHtml(
                data.message || ""
            )}
            <br>
            <strong>Progress:</strong>
            ${data.progress || 0}%
        `;

    } catch (error) {

        document.getElementById(
            "trainingStatus"
        ).textContent =
            error.message;
    }
}


// ======================================================
// HISTORY
// ======================================================

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
            row => {

                const tr =
                    document.createElement(
                        "tr"
                    );

                tr.innerHTML = `
                    <td>
                        ${escapeHtml(
                            row.count_date
                        )}
                    </td>

                    <td>
                        ${row.bucket_count}
                    </td>

                    <td>
                        ${escapeHtml(
                            row.updated_at
                        )}
                    </td>
                `;

                body.appendChild(
                    tr
                );
            }
        );

    } catch (error) {

        message(
            error.message
        );
    }
}


// ======================================================
// TRACKER
// ======================================================

async function resetTracker() {

    try {

        await fetch(
            "/api/tracker/reset",
            {
                method: "POST"
            }
        );

    } catch (error) {

        console.error(
            error
        );
    }
}


// ======================================================
// SETTINGS
// ======================================================

function saveLine() {

    const value =
        parseFloat(
            document.getElementById(
                "lineY"
            ).value
        );

    if (
        Number.isFinite(value)
        && value >= 0
        && value <= 100
    ) {

        countLineY = value;

        message(
            "Counting line saved."
        );

    } else {

        message(
            "Counting line must be between 0 and 100."
        );
    }
}


// ======================================================
// SECURITY / HTML
// ======================================================

function escapeHtml(value) {

    return String(
        value ?? ""
    )
    .replace(
        /&/g,
        "&amp;"
    )
    .replace(
        /</g,
        "&lt;"
    )
    .replace(
        />/g,
        "&gt;"
    )
    .replace(
        /"/g,
        "&quot;"
    )
    .replace(
        /'/g,
        "&#039;"
    );
}


// ======================================================
// AUTO REFRESH
// ======================================================

setInterval(
    loadDashboard,
    5000
);

setInterval(
    loadTrainingStatus,
    5000
);


// ======================================================
// START
// ======================================================

loadDashboard();

</script>

</body>

</html>
"""


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("NEERIKA BUCKET AI")
    print("Mining Production Bucket Counter")
    print("=" * 60)

    print(
        "Database configured:",
        bool(DATABASE_URL)
    )

    print(
        "Timezone:",
        APP_TIMEZONE
    )

    # Initialize database and perform
    # schema migrations.
    init_db()

    print(
        "Database initialization completed."
    )

    # Try loading custom model.
    if model_exists():

        try:

            load_model_internal()

            print(
                "YOLO model loaded:",
                MODEL_PATH
            )

        except Exception as e:

            print(
                "YOLO model could not be loaded:"
            )

            print(e)

    else:

        print(
            "Custom YOLO model not found:"
        )

        print(
            MODEL_PATH
        )

    server = ThreadingHTTPServer(
        (
            "0.0.0.0",
            PORT
        ),
        Handler
    )

    print(
        f"Server running on port {PORT}"
    )

    server.serve_forever()


if __name__ == "__main__":
    main()
