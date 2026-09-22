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
import hmac
from datetime import datetime
from urllib.parse import urlparse, parse_qs
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/Ultralytics")

import psycopg2
from psycopg2.extras import RealDictCursor
from ultralytics import YOLO
from PIL import Image, ImageOps, UnidentifiedImageError
import numpy as np


# =========================================================
# NEERIKA BUCKET AI
# MINING PRODUCTION BUCKET COUNTER
# FULL REPLACEMENT
# =========================================================


# =========================================================
# SERVER CONFIG
# =========================================================

HOST = "0.0.0.0"
PORT = int(os.environ.get("PORT", "8080"))

DATABASE_URL = (
    os.environ.get("DATABASE_URL")
    or os.environ.get("SUPABASE_DB_URL")
    or os.environ.get("POSTGRES_URL")
)

NEERIKA_API_KEY = os.environ.get(
    "NEERIKA_API_KEY",
    ""
).strip()

APP_TIMEZONE = os.environ.get(
    "APP_TIMEZONE",
    "Africa/Dar_es_Salaam"
)


# =========================================================
# PATHS
# =========================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "best.pt"
)

MODEL_BACKUP_PATH = os.path.join(
    BASE_DIR,
    "ai",
    "models",
    "best.pt"
)


# =========================================================
# YOLO SETTINGS
# =========================================================

YOLO_CONFIDENCE = float(
    os.environ.get(
        "YOLO_CONFIDENCE",
        "0.25"
    )
)

YOLO_IMAGE_SIZE = int(
    os.environ.get(
        "YOLO_IMAGE_SIZE",
        "640"
    )
)

TRAIN_EPOCHS = int(
    os.environ.get(
        "TRAIN_EPOCHS",
        "20"
    )
)

MIN_TRAIN_IMAGES = int(
    os.environ.get(
        "MIN_TRAIN_IMAGES",
        "5"
    )
)


# =========================================================
# SECURITY / UPLOAD LIMITS
# =========================================================

# Raw image maximum.
MAX_UPLOAD_BYTES = 10 * 1024 * 1024

# JSON is larger because base64 increases size.
MAX_JSON_BYTES = 15 * 1024 * 1024

MAX_IMAGE_DIMENSION = 4096
MAX_IMAGE_PIXELS = 20_000_000

MAX_ANNOTATIONS_PER_IMAGE = 100
MAX_REFERENCE_IMAGES_PER_BUCKET = 50


Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS


# =========================================================
# CUSTOM CLASSES
# =========================================================

CLASS_NAMES = {
    0: "BUCKET_LOADED",
    1: "BUCKET_EMPTY",
    2: "PEOPLE",
    3: "EQUIPMENT",
}

ALLOWED_CLASSES = set(
    CLASS_NAMES.values()
)


# =========================================================
# GLOBAL MODEL STATE
# =========================================================

MODEL = None

MODEL_ERROR = ""

# VERY IMPORTANT:
# True only when our custom NEERIKA model is loaded.
MODEL_IS_CUSTOM = False

MODEL_LOCK = threading.Lock()


# =========================================================
# TRACKER STATE
# =========================================================

TRACKED_BUCKET_HITS = {}
TRACKED_BUCKET_COUNTED = {}
TRACKED_BUCKET_LAST_SEEN = {}

TRACK_CONSECUTIVE_HITS = 2

TRACK_EXPIRY_SECONDS = 300

TRACKER_LOCK = threading.Lock()


# =========================================================
# TRAINING
# =========================================================

TRAINING_THREAD = None

TRAINING_LOCK_KEY = 98234751

SERVER_INSTANCE_ID = uuid.uuid4().hex


# =========================================================
# DATABASE
# =========================================================

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


def init_db():

    conn = db()
    cur = conn.cursor()

    try:

        cur.execute("""
            CREATE TABLE IF NOT EXISTS buckets (
                id SERIAL PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT DEFAULT '',
                active BOOLEAN NOT NULL DEFAULT FALSE,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS bucket_references (
                id SERIAL PRIMARY KEY,
                bucket_id INTEGER NOT NULL
                    REFERENCES buckets(id)
                    ON DELETE CASCADE,
                image_data BYTEA NOT NULL,
                mime_type TEXT NOT NULL
                    DEFAULT 'image/jpeg',
                created_at TIMESTAMPTZ NOT NULL
                    DEFAULT NOW()
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS dataset_images (
                id SERIAL PRIMARY KEY,
                filename TEXT NOT NULL,
                image_data BYTEA NOT NULL,
                mime_type TEXT NOT NULL
                    DEFAULT 'image/jpeg',
                created_at TIMESTAMPTZ NOT NULL
                    DEFAULT NOW()
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS annotations (
                id SERIAL PRIMARY KEY,
                image_id INTEGER NOT NULL
                    REFERENCES dataset_images(id)
                    ON DELETE CASCADE,
                class_name TEXT NOT NULL,
                x_center DOUBLE PRECISION NOT NULL,
                y_center DOUBLE PRECISION NOT NULL,
                box_width DOUBLE PRECISION NOT NULL,
                box_height DOUBLE PRECISION NOT NULL,
                created_at TIMESTAMPTZ NOT NULL
                    DEFAULT NOW()
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS training_state (
                id INTEGER PRIMARY KEY,
                status TEXT NOT NULL DEFAULT 'idle',
                message TEXT NOT NULL DEFAULT '',
                progress INTEGER NOT NULL DEFAULT 0,
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                run_id TEXT,
                server_id TEXT,
                started_at TIMESTAMPTZ,
                finished_at TIMESTAMPTZ
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS trained_model (
                id INTEGER PRIMARY KEY,
                model_data BYTEA NOT NULL,
                filename TEXT NOT NULL DEFAULT 'best.pt',
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS daily_counts (
                id SERIAL PRIMARY KEY,
                count_date DATE NOT NULL UNIQUE,
                bucket_count INTEGER NOT NULL DEFAULT 0,
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS detection_events (
                id BIGSERIAL PRIMARY KEY,
                class_name TEXT NOT NULL,
                confidence DOUBLE PRECISION NOT NULL DEFAULT 0,
                counted BOOLEAN NOT NULL DEFAULT FALSE,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
        """)

        # Safe migration for old tables.

        cur.execute("""
            ALTER TABLE buckets
            ADD COLUMN IF NOT EXISTS description TEXT DEFAULT ''
        """)

        cur.execute("""
            ALTER TABLE buckets
            ADD COLUMN IF NOT EXISTS active BOOLEAN
            NOT NULL DEFAULT FALSE
        """)

        cur.execute("""
            ALTER TABLE bucket_references
            ADD COLUMN IF NOT EXISTS mime_type TEXT
            NOT NULL DEFAULT 'image/jpeg'
        """)

        cur.execute("""
            ALTER TABLE dataset_images
            ADD COLUMN IF NOT EXISTS mime_type TEXT
            NOT NULL DEFAULT 'image/jpeg'
        """)

        cur.execute("""
            ALTER TABLE training_state
            ADD COLUMN IF NOT EXISTS run_id TEXT
        """)

        cur.execute("""
            ALTER TABLE training_state
            ADD COLUMN IF NOT EXISTS server_id TEXT
        """)

        cur.execute("""
            ALTER TABLE training_state
            ADD COLUMN IF NOT EXISTS started_at TIMESTAMPTZ
        """)

        cur.execute("""
            ALTER TABLE training_state
            ADD COLUMN IF NOT EXISTS finished_at TIMESTAMPTZ
        """)

        cur.execute("""
            INSERT INTO training_state
                (
                    id,
                    status,
                    message,
                    progress,
                    server_id
                )
            VALUES
                (
                    1,
                    'idle',
                    'Ready',
                    0,
                    %s
                )
            ON CONFLICT (id)
            DO NOTHING
        """, (
            SERVER_INSTANCE_ID,
        ))

        # If old database has more than one active bucket,
        # keep newest active only.

        cur.execute("""
            UPDATE buckets
            SET active = FALSE
            WHERE active = TRUE
              AND id NOT IN (
                    SELECT id
                    FROM buckets
                    WHERE active = TRUE
                    ORDER BY id DESC
                    LIMIT 1
              )
        """)

        cur.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS
            one_active_bucket
            ON buckets (active)
            WHERE active = TRUE
        """)

        cur.execute("""
            CREATE INDEX IF NOT EXISTS
            idx_bucket_references_bucket
            ON bucket_references(bucket_id)
        """)

        cur.execute("""
            CREATE INDEX IF NOT EXISTS
            idx_annotations_image
            ON annotations(image_id)
        """)

        cur.execute("""
            CREATE INDEX IF NOT EXISTS
            idx_detection_events_created
            ON detection_events(created_at)
        """)

        conn.commit()

    except Exception:
        conn.rollback()
        raise

    finally:
        cur.close()
        conn.close()


# =========================================================
# DATE / TIME
# =========================================================

def get_tz_date():

    conn = db()
    cur = conn.cursor()

    try:

        cur.execute(
            """
            SELECT
                (
                    NOW()
                    AT TIME ZONE %s
                )::date AS tdate
            """,
            (
                APP_TIMEZONE,
            )
        )

        row = cur.fetchone()

        if row and row["tdate"]:
            return row["tdate"]

        return datetime.now().date()

    finally:

        cur.close()
        conn.close()


# =========================================================
# DAILY COUNT
# =========================================================

def atomic_increment_daily_count(
    cur,
    tdate
):

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
                %s,
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
        (
            tdate,
        )
    )

    row = cur.fetchone()

    if row:
        return int(
            row["bucket_count"]
        )

    return 1


def get_today_count():

    conn = db()
    cur = conn.cursor()

    try:

        cur.execute(
            """
            SELECT bucket_count
            FROM daily_counts
            WHERE count_date =
                (
                    NOW()
                    AT TIME ZONE %s
                )::date
            LIMIT 1
            """,
            (
                APP_TIMEZONE,
            )
        )

        row = cur.fetchone()

        if row:
            return int(
                row["bucket_count"]
            )

        return 0

    finally:

        cur.close()
        conn.close()


# =========================================================
# MODEL RESTORE
# =========================================================

def restore_model():

    if os.path.exists(MODEL_PATH):
        return True

    if not DATABASE_URL:
        return False

    conn = None
    cur = None

    try:

        conn = db()
        cur = conn.cursor()

        cur.execute("""
            SELECT model_data
            FROM trained_model
            WHERE id = 1
            LIMIT 1
        """)

        row = cur.fetchone()

        if not row:
            return False

        model_bytes = bytes(
            row["model_data"]
        )

        with open(
            MODEL_PATH,
            "wb"
        ) as f:
            f.write(model_bytes)

        return True

    except Exception:

        traceback.print_exc()

        return False

    finally:

        if cur:
            cur.close()

        if conn:
            conn.close()


# =========================================================
# LOAD MODEL
# =========================================================

def load_model_internal():

    global MODEL
    global MODEL_ERROR
    global MODEL_IS_CUSTOM

    if MODEL is not None:
        return MODEL

    MODEL_ERROR = ""
    MODEL_IS_CUSTOM = False

    try:

        # First choice: local trained model.
        if restore_model():

            MODEL = YOLO(
                MODEL_PATH
            )

            MODEL_IS_CUSTOM = True

            return MODEL

        # Second choice: repository model.
        if os.path.exists(
            MODEL_BACKUP_PATH
        ):

            MODEL = YOLO(
                MODEL_BACKUP_PATH
            )

            MODEL_IS_CUSTOM = True

            return MODEL

        # DO NOT load COCO yolo11n for counting.
        MODEL = None

        MODEL_IS_CUSTOM = False

        MODEL_ERROR = (
            "Custom best.pt model is not available. "
            "Train the NEERIKA model first."
        )

        return None

    except Exception as e:

        MODEL = None
        MODEL_IS_CUSTOM = False

        MODEL_ERROR = str(e)

        traceback.print_exc()

        return None


def reload_model_after_training():

    global MODEL
    global MODEL_ERROR
    global MODEL_IS_CUSTOM

    with MODEL_LOCK:

        MODEL = None
        MODEL_ERROR = ""
        MODEL_IS_CUSTOM = False

        return load_model_internal()


# =========================================================
# TRACKER
# =========================================================

def cleanup_tracker():

    now = time.time()

    with TRACKER_LOCK:

        expired = []

        for track_id, last_seen in (
            TRACKED_BUCKET_LAST_SEEN.items()
        ):

            if (
                now - last_seen
                > TRACK_EXPIRY_SECONDS
            ):
                expired.append(
                    track_id
                )

        for track_id in expired:

            TRACKED_BUCKET_LAST_SEEN.pop(
                track_id,
                None
            )

            TRACKED_BUCKET_HITS.pop(
                track_id,
                None
            )

            TRACKED_BUCKET_COUNTED.pop(
                track_id,
                None
            )


def reset_tracker():

    with TRACKER_LOCK:

        TRACKED_BUCKET_HITS.clear()

        TRACKED_BUCKET_COUNTED.clear()

        TRACKED_BUCKET_LAST_SEEN.clear()


# =========================================================
# IMAGE HANDLING
# =========================================================

def decode_base64_image(
    value
):

    if not isinstance(
        value,
        str
    ):

        raise ValueError(
            "Image data is required."
        )

    if value.startswith(
        "data:"
    ) and "," in value:

        value = value.split(
            ",",
            1
        )[1]

    try:

        raw = base64.b64decode(
            value,
            validate=True
        )

    except Exception:

        raise ValueError(
            "Invalid base64 image."
        )

    if not raw:

        raise ValueError(
            "Empty image."
        )

    if len(raw) > MAX_UPLOAD_BYTES:

        raise ValueError(
            "Image is too large. "
            "Maximum is 10 MB."
        )

    return raw


def validate_and_load_image(
    data_bytes
):

    if not data_bytes:

        raise ValueError(
            "Empty image."
        )

    if len(data_bytes) > MAX_UPLOAD_BYTES:

        raise ValueError(
            "Image is too large."
        )

    try:

        with Image.open(
            io.BytesIO(data_bytes)
        ) as im:

            im.verify()

        with Image.open(
            io.BytesIO(data_bytes)
        ) as im:

            im = ImageOps.exif_transpose(
                im
            )

            im = im.convert(
                "RGB"
            )

            if (
                im.width >
                MAX_IMAGE_DIMENSION
                or
                im.height >
                MAX_IMAGE_DIMENSION
            ):

                im.thumbnail(
                    (
                        MAX_IMAGE_DIMENSION,
                        MAX_IMAGE_DIMENSION
                    ),
                    Image.Resampling.LANCZOS
                )

            return im.copy()

    except Image.DecompressionBombError:

        raise ValueError(
            "Image dimensions are unsafe."
        )

    except UnidentifiedImageError:

        raise ValueError(
            "Unsupported or corrupt image."
        )

    except Exception as e:

        raise ValueError(
            f"Invalid image: {e}"
        )


def image_to_jpeg_bytes(
    image,
    quality=90
):

    output = io.BytesIO()

    image.save(
        output,
        format="JPEG",
        quality=quality,
        optimize=True
    )

    data = output.getvalue()

    if len(data) > MAX_UPLOAD_BYTES:

        raise ValueError(
            "Processed image is too large."
        )

    return data


# =========================================================
# ANNOTATION VALIDATION
# =========================================================

def validate_annotation_values(
    class_name,
    xc,
    yc,
    bw,
    bh
):

    if class_name not in ALLOWED_CLASSES:

        raise ValueError(
            f"Invalid class: {class_name}"
        )

    values = [
        xc,
        yc,
        bw,
        bh
    ]

    if not all(
        np.isfinite(v)
        for v in values
    ):

        raise ValueError(
            "Annotation contains invalid numbers."
        )

    if not (
        0 <= xc <= 1
        and
        0 <= yc <= 1
        and
        0 < bw <= 1
        and
        0 < bh <= 1
    ):

        raise ValueError(
            "Annotation values must be normalized "
            "between 0 and 1."
        )

    if xc - bw / 2 < 0:

        raise ValueError(
            "Annotation extends outside image."
        )

    if xc + bw / 2 > 1:

        raise ValueError(
            "Annotation extends outside image."
        )

    if yc - bh / 2 < 0:

        raise ValueError(
            "Annotation extends outside image."
        )

    if yc + bh / 2 > 1:

        raise ValueError(
            "Annotation extends outside image."
        )


# =========================================================
# DATASET BUILD
# =========================================================

def build_dataset(
    tmp_dir
):

    train_images = os.path.join(
        tmp_dir,
        "images",
        "train"
    )

    train_labels = os.path.join(
        tmp_dir,
        "labels",
        "train"
    )

    val_images = os.path.join(
        tmp_dir,
        "images",
        "val"
    )

    val_labels = os.path.join(
        tmp_dir,
        "labels",
        "val"
    )

    os.makedirs(
        train_images,
        exist_ok=True
    )

    os.makedirs(
        train_labels,
        exist_ok=True
    )

    os.makedirs(
        val_images,
        exist_ok=True
    )

    os.makedirs(
        val_labels,
        exist_ok=True
    )

    conn = db()
    cur = conn.cursor()

    try:

        cur.execute("""
            SELECT DISTINCT
                di.id,
                di.filename,
                di.image_data
            FROM dataset_images di
            INNER JOIN annotations a
                ON a.image_id = di.id
            ORDER BY di.id
        """)

        rows = cur.fetchall()

    finally:

        cur.close()
        conn.close()

    if len(rows) < MIN_TRAIN_IMAGES:

        raise ValueError(
            f"At least {MIN_TRAIN_IMAGES} "
            "annotated images are required."
        )

    rows = list(rows)

    random.seed(42)

    random.shuffle(rows)

    val_count = max(
        1,
        int(
            round(
                len(rows) * 0.20
            )
        )
    )

    val_rows = rows[:val_count]

    train_rows = rows[val_count:]

    if not train_rows:

        raise ValueError(
            "Training dataset is empty."
        )

    conn = db()
    cur = conn.cursor()

    try:

        groups = [
            (
                True,
                train_rows,
                train_images,
                train_labels
            ),
            (
                False,
                val_rows,
                val_images,
                val_labels
            )
        ]

        for (
            is_train,
            group,
            image_folder,
            label_folder
        ) in groups:

            for row in group:

                image_id = int(
                    row["id"]
                )

                try:

                    image = validate_and_load_image(
                        bytes(
                            row["image_data"]
                        )
                    )

                except Exception as e:

                    raise ValueError(
                        "Invalid dataset image "
                        f"{row['filename']}: {e}"
                    )

                image_path = os.path.join(
                    image_folder,
                    f"{image_id}.jpg"
                )

                label_path = os.path.join(
                    label_folder,
                    f"{image_id}.txt"
                )

                # Save actual JPEG.
                image.save(
                    image_path,
                    format="JPEG",
                    quality=95
                )

                cur.execute(
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
                    (
                        image_id,
                    )
                )

                annotations = cur.fetchall()

                with open(
                    label_path,
                    "w",
                    encoding="utf-8"
                ) as f:

                    for annotation in annotations:

                        class_name = (
                            annotation[
                                "class_name"
                            ]
                        )

                        if (
                            class_name
                            not in
                            ALLOWED_CLASSES
                        ):

                            raise ValueError(
                                "Invalid class "
                                f"{class_name}."
                            )

                        class_id = None

                        for k, v in (
                            CLASS_NAMES.items()
                        ):

                            if v == class_name:
                                class_id = k
                                break

                        xc = float(
                            annotation[
                                "x_center"
                            ]
                        )

                        yc = float(
                            annotation[
                                "y_center"
                            ]
                        )

                        bw = float(
                            annotation[
                                "box_width"
                            ]
                        )

                        bh = float(
                            annotation[
                                "box_height"
                            ]
                        )

                        validate_annotation_values(
                            class_name,
                            xc,
                            yc,
                            bw,
                            bh
                        )

                        f.write(
                            f"{class_id} "
                            f"{xc:.6f} "
                            f"{yc:.6f} "
                            f"{bw:.6f} "
                            f"{bh:.6f}\n"
                        )

    finally:

        cur.close()
        conn.close()

    yaml_path = os.path.join(
        tmp_dir,
        "dataset.yaml"
    )

    safe_path = (
        tmp_dir
        .replace("\\", "/")
    )

    yaml = f"""path: "{safe_path}"
train: images/train
val: images/val

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

        f.write(yaml)

    return (
        yaml_path,
        len(train_rows),
        len(val_rows)
    )


# =========================================================
# TRAINING STATE
# =========================================================

def set_training(
    status,
    message,
    progress,
    run_id=None,
    started_at=None,
    finished_at=None
):

    conn = db()
    cur = conn.cursor()

    try:

        cur.execute(
            """
            UPDATE training_state
            SET
                status = %s,
                message = %s,
                progress = %s,
                updated_at = NOW(),
                run_id =
                    COALESCE(
                        %s,
                        run_id
                    ),
                server_id = %s,
                started_at =
                    COALESCE(
                        %s,
                        started_at
                    ),
                finished_at = %s
            WHERE id = 1
            """,
            (
                status,
                message,
                int(progress),
                run_id,
                SERVER_INSTANCE_ID,
                started_at,
                finished_at
            )
        )

        conn.commit()

    finally:

        cur.close()
        conn.close()


def get_training_state():

    conn = db()
    cur = conn.cursor()

    try:

        cur.execute("""
            SELECT
                id,
                status,
                message,
                progress,
                updated_at,
                run_id,
                server_id,
                started_at,
                finished_at
            FROM training_state
            WHERE id = 1
            LIMIT 1
        """)

        row = cur.fetchone()

        if not row:

            return {
                "status": "idle",
                "message": "Ready",
                "progress": 0
            }

        return dict(row)

    finally:

        cur.close()
        conn.close()


# =========================================================
# TRAINING ADVISORY LOCK
# =========================================================

def acquire_training_lock(
    cur
):

    cur.execute(
        """
        SELECT
            pg_try_advisory_lock(%s)
            AS locked
        """,
        (
            TRAINING_LOCK_KEY,
        )
    )

    row = cur.fetchone()

    return bool(
        row and row["locked"]
    )


def release_training_lock(
    cur
):

    try:

                cur.execute(
            """
            SELECT
                pg_advisory_unlock(%s)
                AS unlocked
            """,
            (
                TRAINING_LOCK_KEY,
            )
        )

    except Exception:

        traceback.print_exc()


# =========================================================
# TRAINING
# =========================================================

def train_model():

    global TRAINING_THREAD

    run_id = uuid.uuid4().hex

    started_at = datetime.utcnow()

    tmp_dir = None

    conn = None
    cur = None

    try:

        conn = db()
        cur = conn.cursor()

        if not acquire_training_lock(cur):

            return {
                "ok": False,
                "error": (
                    "Another training process "
                    "is already running."
                )
            }

        set_training(
            "training",
            "Preparing training dataset...",
            5,
            run_id=run_id,
            started_at=started_at,
            finished_at=None
        )

        tmp_dir = tempfile.mkdtemp(
            prefix="neerika_training_"
        )

        (
            yaml_path,
            train_count,
            val_count
        ) = build_dataset(
            tmp_dir
        )

        set_training(
            "training",
            (
                f"Dataset ready. "
                f"Training images: {train_count}, "
                f"validation images: {val_count}"
            ),
            10,
            run_id=run_id
        )

        # -------------------------------------------------
        # Find starting model
        # -------------------------------------------------

        base_model = None

        if os.path.exists(
            MODEL_PATH
        ):

            base_model = MODEL_PATH

        elif os.path.exists(
            MODEL_BACKUP_PATH
        ):

            base_model = MODEL_BACKUP_PATH

        else:

            base_model = "yolo11n.pt"

        set_training(
            "training",
            (
                "Loading YOLO model "
                f"{os.path.basename(base_model)}..."
            ),
            15,
            run_id=run_id
        )

        model = YOLO(
            base_model
        )

        # -------------------------------------------------
        # Train
        # -------------------------------------------------

        set_training(
            "training",
            (
                f"Training YOLO for "
                f"{TRAIN_EPOCHS} epochs..."
            ),
            20,
            run_id=run_id
        )

        results = model.train(
            data=yaml_path,
            epochs=TRAIN_EPOCHS,
            imgsz=YOLO_IMAGE_SIZE,
            project=tmp_dir,
            name="neerika_training",
            exist_ok=True,
            verbose=True
        )

        # -------------------------------------------------
        # Find best.pt
        # -------------------------------------------------

        possible_best = [
            os.path.join(
                tmp_dir,
                "neerika_training",
                "weights",
                "best.pt"
            ),
            os.path.join(
                tmp_dir,
                "neerika_training",
                "weights",
                "last.pt"
            )
        ]

        best_path = None

        for path in possible_best:

            if os.path.exists(path):

                best_path = path

                if path.endswith(
                    "best.pt"
                ):

                    break

        if not best_path:

            raise RuntimeError(
                "Training finished but "
                "best.pt was not created."
            )

        set_training(
            "training",
            "Training completed. "
            "Saving trained model...",
            90,
            run_id=run_id
        )

        # -------------------------------------------------
        # Read trained model
        # -------------------------------------------------

        with open(
            best_path,
            "rb"
        ) as f:

            model_bytes = f.read()

        if not model_bytes:

            raise RuntimeError(
                "Trained model file is empty."
            )

        # -------------------------------------------------
        # Save model permanently in Supabase/PostgreSQL
        # -------------------------------------------------

        conn2 = db()
        cur2 = conn2.cursor()

        try:

            cur2.execute(
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
                        'best.pt',
                        NOW()
                    )
                ON CONFLICT (id)
                DO UPDATE SET
                    model_data = EXCLUDED.model_data,
                    filename = EXCLUDED.filename,
                    created_at = NOW()
                """,
                (
                    psycopg2.Binary(
                        model_bytes
                    ),
                )
            )

            conn2.commit()

        except Exception:

            conn2.rollback()

            raise

        finally:

            cur2.close()
            conn2.close()

        # -------------------------------------------------
        # Save local copy if filesystem allows it
        # -------------------------------------------------

        try:

            os.makedirs(
                os.path.dirname(
                    MODEL_PATH
                ),
                exist_ok=True
            )

            shutil.copyfile(
                best_path,
                MODEL_PATH
            )

        except Exception:

            traceback.print_exc()

        # -------------------------------------------------
        # Reload model
        # -------------------------------------------------

        set_training(
            "training",
            "Loading newly trained model...",
            95,
            run_id=run_id
        )

        reload_model_after_training()

        finished_at = datetime.utcnow()

        if MODEL_IS_CUSTOM:

            set_training(
                "completed",
                (
                    "Training completed successfully. "
                    "NEERIKA custom model is ready."
                ),
                100,
                run_id=run_id,
                finished_at=finished_at
            )

        else:

            set_training(
                "error",
                (
                    "Training completed but "
                    "custom model could not be loaded."
                ),
                100,
                run_id=run_id,
                finished_at=finished_at
            )

        return {
            "ok": True,
            "message": (
                "Training completed successfully."
            )
        }

    except Exception as e:

        traceback.print_exc()

        try:

            set_training(
                "error",
                str(e),
                0,
                run_id=run_id,
                finished_at=datetime.utcnow()
            )

        except Exception:

            traceback.print_exc()

        return {
            "ok": False,
            "error": str(e)
        }

    finally:

        try:

            if conn and cur:

                release_training_lock(
                    cur
                )

        except Exception:

            traceback.print_exc()

        if conn:

            try:

                conn.commit()

            except Exception:

                pass

            try:

                cur.close()

            except Exception:

                pass

            try:

                conn.close()

            except Exception:

                pass

        if tmp_dir:

            try:

                shutil.rmtree(
                    tmp_dir,
                    ignore_errors=True
                )

            except Exception:

                pass


def training_worker():

    global TRAINING_THREAD

    try:

        train_model()

    finally:

        TRAINING_THREAD = None


def start_training():

    global TRAINING_THREAD

    if (
        TRAINING_THREAD is not None
        and TRAINING_THREAD.is_alive()
    ):

        return {
            "ok": False,
            "error": (
                "Training is already running."
            )
        }

    try:

        state = get_training_state()

        if state.get(
            "status"
        ) == "training":

            return {
                "ok": False,
                "error": (
                    "Training is already "
                    "running."
                )
            }

    except Exception:

        pass

    TRAINING_THREAD = threading.Thread(
        target=training_worker,
        daemon=True
    )

    TRAINING_THREAD.start()

    return {
        "ok": True,
        "message": (
            "Training started."
        )
    }


# =========================================================
# DETECTION
# =========================================================

def detection_from_image(
    image
):

    global MODEL_ERROR

    with MODEL_LOCK:

        model = load_model_internal()

    if model is None:

        raise RuntimeError(
            MODEL_ERROR
            or
            "Custom YOLO model is not available."
        )

    try:

        results = model.predict(
            source=np.array(image),
            conf=YOLO_CONFIDENCE,
            imgsz=YOLO_IMAGE_SIZE,
            verbose=False
        )

    except Exception as e:

        MODEL_ERROR = str(e)

        raise RuntimeError(
            f"YOLO detection failed: {e}"
        )

    detections = []

    if not results:

        return detections

    result = results[0]

    if result.boxes is None:

        return detections

    names = getattr(
        result,
        "names",
        CLASS_NAMES
    )

    for box in result.boxes:

        try:

            cls_id = int(
                box.cls[0].item()
            )

            confidence = float(
                box.conf[0].item()
            )

            xyxy = box.xyxy[
                0
            ].tolist()

            x1, y1, x2, y2 = (
                float(xyxy[0]),
                float(xyxy[1]),
                float(xyxy[2]),
                float(xyxy[3])
            )

            if isinstance(
                names,
                dict
            ):

                class_name = names.get(
                    cls_id,
                    CLASS_NAMES.get(
                        cls_id,
                        f"class_{cls_id}"
                    )
                )

            else:

                class_name = (
                    names[cls_id]
                    if cls_id < len(names)
                    else CLASS_NAMES.get(
                        cls_id,
                        f"class_{cls_id}"
                    )
                )

            detections.append(
                {
                    "class_id": cls_id,
                    "class_name": class_name,
                    "confidence": confidence,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2
                }
            )

        except Exception:

            traceback.print_exc()

    return detections


# =========================================================
# COUNT TRACK
# =========================================================

def make_detection_track_id(
    detection,
    image_width,
    image_height
):

    x1 = detection["x1"]
    y1 = detection["y1"]
    x2 = detection["x2"]
    y2 = detection["y2"]

    cx = (
        x1 + x2
    ) / 2

    cy = (
        y1 + y2
    ) / 2

    # Convert coordinates to coarse grid.
    gx = int(
        cx / max(
            1,
            image_width
        ) * 100
    )

    gy = int(
        cy / max(
            1,
            image_height
        ) * 100
    )

    gw = int(
        (x2 - x1)
        / max(
            1,
            image_width
        ) * 100
    )

    gh = int(
        (y2 - y1)
        / max(
            1,
            image_height
        ) * 100
    )

    return (
        f"{detection['class_name']}:"
        f"{gx}:{gy}:{gw}:{gh}"
    )


def register_detection_event(
    class_name,
    confidence,
    counted
):

    conn = db()
    cur = conn.cursor()

    try:

        cur.execute(
            """
            INSERT INTO detection_events
                (
                    class_name,
                    confidence,
                    counted,
                    created_at
                )
            VALUES
                (
                    %s,
                    %s,
                    %s,
                    NOW()
                )
            """,
            (
                class_name,
                float(confidence),
                bool(counted)
            )
        )

        conn.commit()

    finally:

        cur.close()
        conn.close()


def count_loaded_bucket(
    detection,
    image_width,
    image_height
):

    if (
        detection["class_name"]
        !=
        "BUCKET_LOADED"
    ):

        return {
            "counted": False,
            "total": get_today_count()
        }

    track_id = make_detection_track_id(
        detection,
        image_width,
        image_height
    )

    now = time.time()

    with TRACKER_LOCK:

        previous_hits = TRACKED_BUCKET_HITS.get(
            track_id,
            0
        )

        hits = (
            previous_hits + 1
        )

        TRACKED_BUCKET_HITS[
            track_id
        ] = hits

        TRACKED_BUCKET_LAST_SEEN[
            track_id
        ] = now

        already_counted = (
            TRACKED_BUCKET_COUNTED.get(
                track_id,
                False
            )
        )

        if already_counted:

            return {
                "counted": False,
                "total": get_today_count()
            }

        if hits < TRACK_CONSECUTIVE_HITS:

            return {
                "counted": False,
                "total": get_today_count()
            }

        TRACKED_BUCKET_COUNTED[
            track_id
        ] = True

    # Atomic DB count.
    conn = db()
    cur = conn.cursor()

    try:

        tdate = get_tz_date()

        total = atomic_increment_daily_count(
            cur,
            tdate
        )

        cur.execute(
            """
            INSERT INTO detection_events
                (
                    class_name,
                    confidence,
                    counted,
                    created_at
                )
            VALUES
                (
                    %s,
                    %s,
                    TRUE,
                    NOW()
                )
            """,
            (
                detection["class_name"],
                float(
                    detection["confidence"]
                )
            )
        )

        conn.commit()

    except Exception:

        conn.rollback()

        with TRACKER_LOCK:

            TRACKED_BUCKET_COUNTED.pop(
                track_id,
                None
            )

        raise

    finally:

        cur.close()
        conn.close()

    return {
        "counted": True,
        "total": total
    }


# =========================================================
# ACTIVE BUCKET
# =========================================================

def get_active_bucket():

    conn = db()
    cur = conn.cursor()

    try:

        cur.execute("""
            SELECT
                id,
                name,
                description,
                active,
                created_at
            FROM buckets
            WHERE active = TRUE
            ORDER BY id DESC
            LIMIT 1
        """)

        row = cur.fetchone()

        if row:

            return dict(row)

        return None

    finally:

        cur.close()
        conn.close()


def get_buckets():

    conn = db()
    cur = conn.cursor()

    try:

        cur.execute("""
            SELECT
                b.id,
                b.name,
                b.description,
                b.active,
                b.created_at,
                COUNT(br.id) AS reference_count
            FROM buckets b
            LEFT JOIN bucket_references br
                ON br.bucket_id = b.id
            GROUP BY
                b.id,
                b.name,
                b.description,
                b.active,
                b.created_at
            ORDER BY b.id DESC
        """)

        return [
            dict(row)
            for row in cur.fetchall()
        ]

    finally:

        cur.close()
        conn.close()


# =========================================================
# JSON HELPERS
# =========================================================

def json_bytes(
    payload
):

    return json.dumps(
        payload,
        ensure_ascii=False,
        default=str
    ).encode(
        "utf-8"
    )


def safe_int(
    value,
    default=0
):

    try:

        return int(value)

    except Exception:

        return default


def safe_float(
    value,
    default=0.0
):

    try:

        return float(value)

    except Exception:

        return default


# =========================================================
# AUTHENTICATION
# =========================================================

def constant_time_equal(
    a,
    b
):

    try:

        return hmac.compare_digest(
            str(a),
            str(b)
        )

    except Exception:

        return False


# =========================================================
# HTTP HANDLER
# =========================================================

class Handler(
    BaseHTTPRequestHandler
):

    server_version = (
        "NEERIKA-BUCKET-AI/1.0"
    )

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

    # -----------------------------------------------------
    # HEADERS
    # -----------------------------------------------------

    def send_json(
        self,
        payload,
        status=200
    ):

        body = json_bytes(
            payload
        )

        self.send_response(
            status
        )

        self.send_header(
            "Content-Type",
            "application/json; charset=utf-8"
        )

        self.send_header(
            "Content-Length",
            str(len(body))
        )

        self.send_header(
            "Cache-Control",
            "no-store"
        )

        self.send_header(
            "X-Content-Type-Options",
            "nosniff"
        )

        self.send_header(
            "X-Frame-Options",
            "SAMEORIGIN"
        )

        self.send_header(
            "Referrer-Policy",
            "same-origin"
        )

        self.send_header(
            "Permissions-Policy",
            "camera=(self), microphone=(self)"
        )

        self.end_headers()

        self.wfile.write(
            body
        )

    def send_text(
        self,
        body,
        content_type="text/plain; charset=utf-8",
        status=200
    ):

        data = body.encode(
            "utf-8"
        )

        self.send_response(
            status
        )

        self.send_header(
            "Content-Type",
            content_type
        )

        self.send_header(
            "Content-Length",
            str(len(data))
        )

        self.send_header(
            "Cache-Control",
            "no-store"
        )

        self.send_header(
            "X-Content-Type-Options",
            "nosniff"
        )

        self.end_headers()

        self.wfile.write(
            data
        )

    # -----------------------------------------------------
    # AUTH
    # -----------------------------------------------------

    def authorized(self):

        # If API key is not configured,
        # keep compatibility with current app.
        if not NEERIKA_API_KEY:

            return True

        supplied = (
            self.headers.get(
                "X-NEERIKA-API-KEY",
                ""
            )
            or ""
        ).strip()

        if constant_time_equal(
            supplied,
            NEERIKA_API_KEY
        ):

            return True

        parsed = urlparse(
            self.path
        )

        query = parse_qs(
            parsed.query
        )

        query_key = (
            query.get(
                "api_key",
                [""]
            )[0]
            or ""
        )

        return constant_time_equal(
            query_key,
            NEERIKA_API_KEY
        )

    # -----------------------------------------------------
    # BODY
    # -----------------------------------------------------

    def read_body(
        self
    ):

        length_header = (
            self.headers.get(
                "Content-Length"
            )
        )

        if not length_header:

            raise ValueError(
                "Content-Length is required."
            )

        try:

            length = int(
                length_header
            )

        except Exception:

            raise ValueError(
                "Invalid Content-Length."
            )

        content_type = (
            self.headers.get(
                "Content-Type",
                ""
            )
        ).lower()

        limit = (
            MAX_JSON_BYTES
            if "application/json"
            in content_type
            else MAX_UPLOAD_BYTES
        )

        if length < 0:

            raise ValueError(
                "Invalid request size."
            )

        if length > limit:

            raise ValueError(
                "Request body is too large."
            )

        return self.rfile.read(
            length
        )

    def read_json(
        self
    ):

        raw = self.read_body()

        if not raw:

            raise ValueError(
                "Empty request body."
            )

        try:

            return json.loads(
                raw.decode(
                    "utf-8"
                )
            )

        except Exception:

            raise ValueError(
                "Invalid JSON."
            )

    # -----------------------------------------------------
    # GET
    # -----------------------------------------------------

    def do_GET(
        self
    ):

        try:

            if not self.authorized():

                self.send_json(
                    {
                        "ok": False,
                        "error": "Unauthorized"
                    },
                    401
                )

                return

            parsed = urlparse(
                self.path
            )

            path = parsed.path

            if path == "/":

                self.send_text(
                    HTML,
                    "text/html; charset=utf-8"
                )

                return

            if path == "/api/health":

                self.send_json(
                    {
                        "ok": True,
                        "app": (
                            "NEERIKA BUCKET AI"
                        ),
                        "database": bool(
                            DATABASE_URL
                        ),
                        "model_loaded": (
                            MODEL is not None
                        ),
                        "custom_model": (
                            MODEL_IS_CUSTOM
                        ),
                        "model_error": (
                            MODEL_ERROR
                        ),
                        "today_count": (
                            get_today_count()
                        )
                    }
                )

                return

            if path == "/api/count":

                self.send_json(
                    {
                        "ok": True,
                        "count": (
                            get_today_count()
                        )
                    }
                )

                return

            if path == "/api/buckets":

                self.send_json(
                    {
                        "ok": True,
                        "buckets": (
                            get_buckets()
                        ),
                        "active": (
                            get_active_bucket()
                        )
                    }
                )

                return

            if path == "/api/training/status":

                self.send_json(
                    {
                        "ok": True,
                        "training": (
                            get_training_state()
                        )
                    }
                )

                return

            if path == "/api/model/status":

                with MODEL_LOCK:

                    load_model_internal()

                self.send_json(
                    {
                        "ok": True,
                        "loaded": (
                            MODEL is not None
                        ),
                        "custom": (
                            MODEL_IS_CUSTOM
                        ),
                        "error": (
                            MODEL_ERROR
                        )
                    }
                )

                return

            if path == "/api/history":

                conn = db()
                cur = conn.cursor()

                try:

                    cur.execute("""
                        SELECT
                            count_date,
                            bucket_count,
                            updated_at
                        FROM daily_counts
                        ORDER BY count_date DESC
                        LIMIT 100
                    """)

                    rows = [
                        dict(row)
                        for row in cur.fetchall()
                    ]

                finally:

                    cur.close()
                    conn.close()

                self.send_json(
                    {
                        "ok": True,
                        "history": rows
                    }
                )

                return

            self.send_json(
                {
                    "ok": False,
                    "error": "Not found"
                },
                404
            )

        except Exception as e:

            traceback.print_exc()

            self.send_json(
                {
                    "ok": False,
                    "error": str(e)
                },
                500
            )

    # -----------------------------------------------------
    # POST
    # -----------------------------------------------------

    def do_POST(
        self
    ):

        try:

            if not self.authorized():

                self.send_json(
                    {
                        "ok": False,
                        "error": "Unauthorized"
                    },
                    401
                )

                return

            parsed = urlparse(
                self.path
            )

            path = parsed.path

            # ---------------------------------------------
            # Detection
            # ---------------------------------------------

            if path == "/api/detect":

                payload = self.read_json()

                image_value = (
                    payload.get(
                        "image"
                    )
                )

                raw = decode_base64_image(
                    image_value
                )

                image = validate_and_load_image(
                    raw
                )

                cleanup_tracker()

                detections = (
                    detection_from_image(
                        image
                    )
                )

                counted = []

                for detection in detections:

                    if (
                        detection[
                            "class_name"
                        ]
                        ==
                        "BUCKET_LOADED"
                    ):

                        result = (
                            count_loaded_bucket(
                                detection,
                                image.width,
                                image.height
                            )
                        )

                        detection[
                            "counted"
                        ] = result[
                            "counted"
                        ]

                        detection[
                            "total"
                        ] = result[
                            "total"
                        ]

                        if result[
                            "counted"
                        ]:

                            counted.append(
                                detection
                            )

                    else:

                        detection[
                            "counted"
                        ] = False

                self.send_json(
                    {
                        "ok": True,
                        "detections": detections,
                        "counted": len(
                            counted
                        ),
                        "total": (
                            get_today_count()
                        ),
                        "model_custom": (
                            MODEL_IS_CUSTOM
                        )
                    }
                )

                return

            # ---------------------------------------------
            # Reset tracker
            # ---------------------------------------------

            if path == "/api/tracker/reset":

                reset_tracker()

                self.send_json(
                    {
                        "ok": True,
                        "message": (
                            "Tracker reset."
                        )
                    }
                )

                return

            # ---------------------------------------------
            # Create bucket
            # ---------------------------------------------

            if path == "/api/buckets":

                payload = self.read_json()

                name = str(
                    payload.get(
                        "name",
                        ""
                    )
                ).strip()

                description = str(
                    payload.get(
                        "description",
                        ""
                    )
                ).strip()

                if not name:

                    raise ValueError(
                        "Bucket name is required."
                    )

                if len(name) > 100:

                    raise ValueError(
                        "Bucket name is too long."
                    )

                if len(description) > 1000:

                    raise ValueError(
                        "Description is too long."
                    )

                conn = db()
                cur = conn.cursor()

                try:

                    cur.execute(
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
                        RETURNING
                            id,
                            name,
                            description,
                            active,
                            created_at
                        """,
                        (
                            name,
                            description
                        )
                    )

                    row = cur.fetchone()

                    conn.commit()

                except Exception:

                    conn.rollback()

                    raise

                finally:

                    cur.close()
                    conn.close()

                self.send_json(
                    {
                        "ok": True,
                        "bucket": dict(row)
                    }
                )

                return

            # ---------------------------------------------
            # Set active bucket
            # ---------------------------------------------

            if path.startswith(
                "/api/buckets/"
            ) and path.endswith(
                "/activate"
            ):

                parts = path.strip(
                    "/"
                ).split("/")

                if len(parts) != 3:

                    raise ValueError(
                        "Invalid bucket ID."
                    )

                bucket_id = safe_int(
                    parts[2],
                    -1
                )

                if bucket_id < 1:

                    raise ValueError(
                        "Invalid bucket ID."
                    )

                conn = db()
                cur = conn.cursor()

                try:

                    # Transaction ensures only one active bucket.

                    cur.execute(
                        """
                        SELECT id
                        FROM buckets
                        WHERE id = %s
                        FOR UPDATE
                        """,
                        (
                            bucket_id,
                        )
                    )

                    if not cur.fetchone():

                        raise ValueError(
                            "Bucket not found."
                        )

                    cur.execute(
                        """
                        UPDATE buckets
                        SET active = FALSE
                        WHERE active = TRUE
                        """
                    )

                    cur.execute(
                        """
                        UPDATE buckets
                        SET active = TRUE
                        WHERE id = %s
                        """,
                        (
                            bucket_id,
                        )
                    )

                    conn.commit()

                except Exception:

                    conn.rollback()

                    raise

                finally:

                    cur.close()
                    conn.close()

                reset_tracker()

                self.send_json(
                    {
                        "ok": True,
                        "active": (
                            get_active_bucket()
                        )
                    }
                )

                return

            # ---------------------------------------------
            # Upload bucket reference image
            # ---------------------------------------------

            if path == "/api/bucket-reference":

                payload = self.read_json()

                bucket_id = safe_int(
                    payload.get(
                        "bucket_id"
                    ),
                    -1
                )

                if bucket_id < 1:

                    raise ValueError(
                        "Valid bucket_id is required."
                    )

                raw = decode_base64_image(
                    payload.get(
                        "image"
                    )
                )

                image = validate_and_load_image(
                    raw
                )

                jpeg = image_to_jpeg_bytes(
                    image
                )

                conn = db()
                cur = conn.cursor()

                try:

                    cur.execute(
                        """
                        SELECT
                            COUNT(*) AS total
                        FROM bucket_references
                        WHERE bucket_id = %s
                        """,
                        (
                            bucket_id,
                        )
                    )

                    total = int(
                        cur.fetchone()[
                            "total"
                        ]
                    )

                    if (
                        total
                        >=
                        MAX_REFERENCE_IMAGES_PER_BUCKET
                    ):

                        raise ValueError(
                            "Maximum reference "
                            "images reached."
                        )

                    cur.execute(
                        """
                        SELECT id
                        FROM buckets
                        WHERE id = %s
                        """,
                        (
                            bucket_id,
                        )
                    )

                    if not cur.fetchone():

                        raise ValueError(
                            "Bucket not found."
                        )

                    cur.execute(
                        """
                        INSERT INTO bucket_references
                            (
                                bucket_id,
                                image_data,
                                mime_type
                            )
                        VALUES
                            (
                                %s,
                                %s,
                                'image/jpeg'
                            )
                        RETURNING id
                        """,
                        (
                            bucket_id,
                            psycopg2.Binary(
                                jpeg
                            )
                        )
                    )

                    reference_id = cur.fetchone()[
                        "id"
                    ]

                    conn.commit()

                except Exception:

                    conn.rollback()

                    raise

                finally:

                    cur.close()
                    conn.close()

                self.send_json(
                    {
                        "ok": True,
                        "reference_id": (
                            reference_id
                        )
                    }
                )

                return

            # ---------------------------------------------
            # Dataset image upload
            # ---------------------------------------------

            if path == "/api/dataset/upload":

                payload = self.read_json()

                filename = str(
                    payload.get(
                        "filename",
                        "image.jpg"
                    )
                ).strip()

                if not filename:

                    filename = "image.jpg"

                filename = os.path.basename(
                    filename
                )

                if len(filename) > 255:

                    filename = filename[
                        :255
                    ]

                raw = decode_base64_image(
                    payload.get(
                        "image"
                    )
                )

                image = validate_and_load_image(
                    raw
                )

                jpeg = image_to_jpeg_bytes(
                    image
                )

                conn = db()
                cur = conn.cursor()

                try:

                    cur.execute(
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
                                'image/jpeg'
                            )
                        RETURNING id
                        """,
                        (
                            filename,
                            psycopg2.Binary(
                                jpeg
                            )
                        )
                    )

                    row = cur.fetchone()

                    conn.commit()

                except Exception:

                    conn.rollback()

                    raise

                finally:

                    cur.close()
                    conn.close()

                self.send_json(
                    {
                        "ok": True,
                        "image_id": row[
                            "id"
                        ],
                        "filename": filename
                    }
                )

                return

            # ---------------------------------------------
            # Save annotations
            # ---------------------------------------------

            if path == "/api/dataset/annotate":

                payload = self.read_json()

                image_id = safe_int(
                    payload.get(
                        "image_id"
                    ),
                    -1
                )

                annotations = payload.get(
                    "annotations",
                    []
                )

                if image_id < 1:

                    raise ValueError(
                        "Valid image_id is required."
                    )

                if not isinstance(
                    annotations,
                    list
                ):

                    raise ValueError(
                        "annotations must be a list."
                    )

                if len(
                    annotations
                ) > MAX_ANNOTATIONS_PER_IMAGE:

                    raise ValueError(
                        "Too many annotations."
                    )

                conn = db()
                cur = conn.cursor()

                try:

                    cur.execute(
                        """
                        SELECT id
                        FROM dataset_images
                        WHERE id = %s
                        """,
                        (
                            image_id,
                        )
                    )

                    if not cur.fetchone():

                        raise ValueError(
                            "Dataset image not found."
                        )

                    # Replace previous annotations.

                    cur.execute(
                        """
                        DELETE FROM annotations
                        WHERE image_id = %s
                        """,
                        (
                            image_id,
                        )
                    )

                    for annotation in annotations:

                        if not isinstance(
                            annotation,
                            dict
                        ):

                            raise ValueError(
                                "Invalid annotation."
                            )

                        class_name = str(
                            annotation.get(
                                "class_name",
                                ""
                            )
                        ).strip()

                        xc = safe_float(
                            annotation.get(
                                "x_center"
                            ),
                            -1
                        )

                        yc = safe_float(
                            annotation.get(
                                "y_center"
                            ),
                            -1
                        )

                        bw = safe_float(
                            annotation.get(
                                "box_width"
                            ),
                            -1
                        )

                        bh = safe_float(
                            annotation.get(
                                "box_height"
                            ),
                            -1
                        )

                        validate_annotation_values(
                            class_name,
                            xc,
                            yc,
                            bw,
                            bh
                        )

                        cur.execute(
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
                                xc,
                                yc,
                                bw,
                                bh
                            )
                        )

                    conn.commit()

                except Exception:

                    conn.rollback()

                    raise

                finally:

                    cur.close()
                    conn.close()

                self.send_json(
                    {
                        "ok": True,
                        "message": (
                            "Annotations saved."
                        ),
                        "count": len(
                            annotations
                        )
                    }
                )

                return

            # ---------------------------------------------
            # Start training
            # ---------------------------------------------

            if path == "/api/training/start":

                result = start_training()

                self.send_json(
                    result,
                    200
                    if result.get("ok")
                    else 409
                )

                return

            # ---------------------------------------------
            # 404
            # ---------------------------------------------

            self.send_json(
                {
                    "ok": False,
                    "error": "Not found"
                },
                404
            )

        except ValueError as e:

            self.send_json(
                {
                    "ok": False,
                    "error": str(e)
                },
                400
            )

        except Exception as e:

            traceback.print_exc()

            self.send_json(
                {
                    "ok": False,
                    "error": str(e)
                },
                500
            )


# =========================================================
# HTML APPLICATION
# =========================================================

HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>

<meta charset="UTF-8">

<meta
    name="viewport"
    content="width=device-width,
             initial-scale=1.0"
>

<title>
NEERIKA BUCKET AI
</title>

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
    background: #f4f6f8;
    color: #17202a;
}

header {
    background: #111827;
    color: white;
    padding: 18px;
    position: sticky;
    top: 0;
    z-index: 10;
}

.header-title {
    font-size: 22px;
    font-weight: 700;
}

.header-subtitle {
    font-size: 13px;
    opacity: .75;
    margin-top: 4px;
}

nav {
    display: flex;
    gap: 8px;
    overflow-x: auto;
    padding: 10px;
    background: white;
    border-bottom: 1px solid #ddd;
    position: sticky;
    top: 76px;
    z-index: 9;
}

nav button {
    border: 0;
    background: #e5e7eb;
    padding: 10px 14px;
    border-radius: 8px;
    cursor: pointer;
    white-space: nowrap;
}

nav button.active {
    background: #111827;
    color: white;
}

main {
    max-width: 1100px;
    margin: auto;
    padding: 16px;
}

.section {
    display: none;
}

.section.active {
    display: block;
}

.card {
    background: white;
    border-radius: 14px;
    padding: 18px;
    margin-bottom: 16px;
    box-shadow:
        0 2px 8px
        rgba(0,0,0,.07);
}

.grid {
    display: grid;
    grid-template-columns:
        repeat(
            auto-fit,
            minmax(220px,1fr)
        );
    gap: 14px;
}

.stat {
    background: #f9fafb;
    border-radius: 12px;
    padding: 16px;
}

.stat-label {
    color: #6b7280;
    font-size: 13px;
}

.stat-value {
    font-size: 30px;
    font-weight: 700;
    margin-top: 5px;
}

button,
input,
textarea,
select {
    font: inherit;
}

.action {
    border: 0;
    border-radius: 9px;
    padding: 11px 15px;
    cursor: pointer;
    background: #111827;
    color: white;
}

.action.secondary {
    background: #e5e7eb;
    color: #111827;
}

.action.success {
    background: #166534;
}

.action.danger {
    background: #991b1b;
}

input,
textarea,
select {
    width: 100%;
    padding: 11px;
    border: 1px solid #d1d5db;
    border-radius: 8px;
    margin-top: 6px;
    margin-bottom: 12px;
}

label {
    font-weight: 600;
    font-size: 14px;
}

video,
canvas,
.preview {
    width: 100%;
    max-height: 600px;
    object-fit: contain;
    background: #111;
    border-radius: 12px;
}

.camera-wrap {
    position: relative;
}

#cameraCanvas {
    position: absolute;
    left: 0;
    top: 0;
    pointer-events: none;
}

.status {
    padding: 10px;
    border-radius: 8px;
    background: #f3f4f6;
    margin-top: 10px;
}

.status.ok {
    background: #dcfce7;
    color: #166534;
}

.status.error {
    background: #fee2e2;
    color: #991b1b;
}

table {
    width: 100%;
    border-collapse: collapse;
}

th,
td {
    border-bottom: 1px solid #e5e7eb;
    padding: 10px;
    text-align: left;
}

.small {
    font-size: 12px;
    color: #6b7280;
}

.badge {
    display: inline-block;
    padding: 4px 8px;
    border-radius: 20px;
    background: #e5e7eb;
    font-size: 12px;
}

.badge.active {
    background: #dcfce7;
    color: #166534;
}

.preview-grid {
    display: grid;
    grid-template-columns:
        repeat(
            auto-fill,
            minmax(130px,1fr)
        );
    gap: 10px;
}

.preview-grid img {
    width: 100%;
    height: 120px;
    object-fit: cover;
    border-radius: 8px;
}

.log {
    white-space: pre-wrap;
    max-height: 250px;
    overflow: auto;
    background: #111827;
    color: #e5e7eb;
    padding: 12px;
    border-radius: 8px;
    font-family: monospace;
    font-size: 12px;
}

footer {
    text-align: center;
    color: #6b7280;
    padding: 30px 10px;
}

@media (
    max-width: 600px
) {

    main {
        padding: 10px;
    }

    .card {
        padding: 14px;
    }

    .stat-value {
        font-size: 25px;
    }

}

</style>

</head>

<body>

<header>

<div class="header-title">
NEERIKA BUCKET AI
</div>

<div class="header-subtitle">
Mining Production Bucket Counter
</div>

</header>

<nav>

<button
    class="nav-btn active"
    onclick="showSection('dashboard',this)"
>
Dashboard
</button>

<button
    class="nav-btn"
    onclick="showSection('camera',this)"
>
Camera
</button>

<button
    class="nav-btn"
    onclick="showSection('buckets',this)"
>
Buckets
</button>

<button
    class="nav-btn"
    onclick="showSection('training',this)"
>
Training
</button>

<button
    class="nav-btn"
    onclick="showSection('history',this)"
>
History
</button>

<button
    class="nav-btn"
    onclick="showSection('settings',this)"
>
Settings
</button>

</nav>

<main>

<!-- =====================================================
     DASHBOARD
===================================================== -->

<section
    id="dashboard"
    class="section active"
>

<div class="card">

<h2>
Dashboard
</h2>

<div class="grid">

<div class="stat">

<div class="stat-label">
Today's Loaded Buckets
</div>

<div
    class="stat-value"
    id="todayCount"
>
0
</div>

</div>

<div class="stat">

<div class="stat-label">
Active Bucket
</div>

<div
    class="stat-value"
    id="activeBucketName"
>
None
</div>

</div>

<div class="stat">

<div class="stat-label">
AI Model
</div>

<div
    class="stat-value"
    id="modelStatus"
>
Checking...
</div>

</div>

</div>

</div>

<div class="card">

<h3>
System Status
</h3>

<div
    id="dashboardStatus"
    class="status"
>
Checking...
</div>

</div>

</section>


<!-- =====================================================
     CAMERA
===================================================== -->

<section
    id="camera"
    class="section"
>

<div class="card">

<h2>
Camera Counter
</h2>

<p class="small">
Only BUCKET_LOADED detections are counted.
Empty buckets, people and equipment are ignored.
</p>

<div class="camera-wrap">

<video
    id="cameraVideo"
    autoplay
    playsinline
>
</video>

<canvas
    id="cameraCanvas"
>
</canvas>

</div>

<div
    class="grid"
    style="margin-top:12px"
>

<button
    class="action"
    onclick="startCamera()"
>
Start Camera
</button>

<button
    class="action secondary"
    onclick="stopCamera()"
>
Stop Camera
</button>

<button
    class="action success"
    onclick="captureAndDetect()"
>
Detect Bucket
</button>

<button
    class="action danger"
    onclick="resetTracker()"
>
Reset Tracker
</button>

</div>

<div
    id="cameraStatus"
    class="status"
>
Camera stopped.
</div>

<div class="stat">

<div class="stat-label">
Count
</div>

<div
    id="cameraCount"
    class="stat-value"
>
0
</div>

</div>

</div>


<div class="card">

<h3>
Upload Image Instead
</h3>

<input
    type="file"
    id="detectFile"
    accept="image/*"
>

<button
    class="action"
    onclick="detectUploadedImage()"
>
Detect Uploaded Image
</button>

<img
    id="detectPreview"
    class="preview"
    style="display:none;margin-top:12px"
>

</div>

</section>


<!-- =====================================================
     BUCKETS
===================================================== -->

<section
    id="buckets"
    class="section"
>

<div class="card">

<h2>
Bucket Registration
</h2>

<label>
Bucket Name
</label>

<input
    id="bucketName"
    placeholder="Example: 800 KG Bucket"
>

<label>
Description
</label>

<textarea
    id="bucketDescription"
    placeholder="Bucket description..."
></textarea>

<button
    class="action"
    onclick="createBucket()"
>
Register Bucket
</button>

</div>

<div class="card">

<h3>
Registered Buckets
</h3>

<div
    id="bucketList"
>
Loading...
</div>

</div>

</section>


<!-- =====================================================
     TRAINING
===================================================== -->

<section
    id="training"
    class="section"
>

<div class="card">

<h2>
YOLO Training Dataset
</h2>

<p class="small">
Upload images and create annotations.
At least 5 annotated images are required.
</p>

<input
    type="file"
    id="datasetFile"
    accept="image/*"
>

<button
    class="action"
    onclick="uploadDatasetImage()"
>
Upload Image
</button>

<div
    id="datasetUploadStatus"
    class="status"
>
Ready.
</div>

</div>

<div class="card">

<h3>
Training Status
</h3>

<div
    id="trainingStatus"
    class="status"
>
Checking...
</div>

<div
    id="trainingProgress"
    style="
        margin-top:10px;
        height:12px;
        background:#e5e7eb;
        border-radius:10px;
        overflow:hidden;
    "
>

<div
    id="trainingProgressBar"
    style="
        width:0%;
        height:100%;
        background:#166534;
    "
></div>

</div>

<button
    class="action success"
    style="margin-top:12px"
    onclick="startTraining()"
>
Start Training
</button>

</div>

</section>


<!-- =====================================================
     HISTORY
===================================================== -->

<section
    id="history"
    class="section"
>

<div class="card">

<h2>
Production History
</h2>

<button
    class="action secondary"
    onclick="loadHistory()"
>
Refresh
</button>

<div
    id="historyTable"
    style="margin-top:12px"
>
Loading...
</div>

</div>

</section>


<!-- =====================================================
     SETTINGS
===================================================== -->

<section
    id="settings"
    class="section"
>

<div class="card">

<h2>
Settings
</h2>

<p class="small">
NEERIKA BUCKET AI
</p>

<div id="settingsStatus">
Loading...
</div>

</div>

</section>

</main>

<footer>
Geology &amp; Mining Services
</footer>


<script>

let cameraStream = null;

let detectBusy = false;

let cameraTimer = null;

let lastDetections = [];


// =====================================================
// API
// =====================================================

async function api(
    url,
    options = {}
) {

    const response =
        await fetch(
            url,
            options
        );

    let data = {};

    try {

        data =
            await response.json();

    } catch (e) {

        throw new Error(
            "Invalid server response."
        );

    }

    if (!response.ok) {

        throw new Error(
            data.error
            ||
            "Request failed."
        );

    }

    return data;
}


// =====================================================
// NAVIGATION
// =====================================================

function showSection(
    id,
    button
) {

    document
        .querySelectorAll(
            ".section"
        )
        .forEach(
            section => {
                section
                    .classList
                    .remove("active");
            }
        );

    document
        .getElementById(
            id
        )
        .classList
        .add("active");

    document
        .querySelectorAll(
            ".nav-btn"
        )
        .forEach(
            btn => {
                btn
                    .classList
                    .remove("active");
            }
        );

    button.classList.add(
        "active"
    );

    if (
        id === "dashboard"
    ) {

        refreshDashboard();

    }

    if (
        id === "buckets"
    ) {

        loadBuckets();

    }

    if (
        id === "history"
    ) {

        loadHistory();

    }

    if (
        id === "training"
    ) {

        loadTrainingStatus();

    }

}


// =====================================================
// DASHBOARD
// =====================================================

async function refreshDashboard() {

    try {

        const health =
            await api(
                "/api/health"
            );

        document
            .getElementById(
                "todayCount"
            )
            .textContent =
            health.today_count;

        document
            .getElementById(
                "modelStatus"
            )
            .textContent =
            health.custom_model
                ? "Ready"
                : "Not Ready";

        const active =
            await api(
                "/api/buckets"
            );

        document
            .getElementById(
                "activeBucketName"
            )
            .textContent =
            active.active
                ? active.active.name
                : "None";

        const status =
            document
                .getElementById(
                    "dashboardStatus"
                );

        if (
            health.ok
            &&
            health.database
            &&
            health.custom_model
        ) {

            status.textContent =
                "System ready.";

            status.className =
                "status ok";

        } else {

            status.textContent =
                health.model_error
                ||
                "System needs attention.";

            status.className =
                "status error";

        }

    } catch (e) {

        document
            .getElementById(
                "dashboardStatus"
            )
            .textContent =
            e.message;

        document
            .getElementById(
                "dashboardStatus"
            )
            .className =
            "status error";

    }

}


// =====================================================
// CAMERA
// =====================================================

async function startCamera() {

    try {

        cameraStream =
            await navigator
                .mediaDevices
                .getUserMedia(
                    {
                        video: {
                            facingMode: {
                                ideal:
                                    "environment"
                            }
                        },
                        audio: false
                    }
                );

        document
            .getElementById(
                "cameraVideo"
            )
            .srcObject =
            cameraStream;

        document
            .getElementById(
                "cameraStatus"
            )
            .textContent =
            "Camera started.";

    } catch (e) {

        document
            .getElementById(
                "cameraStatus"
            )
            .textContent =
            "Camera error: "
            + e.message;

    }

}


function stopCamera() {

    if (
        cameraStream
    ) {

        cameraStream
            .getTracks()
            .forEach(
                track =>
                    track.stop()
            );

        cameraStream = null;

    }

    if (
        cameraTimer
    ) {

        clearInterval(
            cameraTimer
        );

        cameraTimer = null;

    }

    document
        .getElementById(
            "cameraVideo"
        )
        .srcObject =
        null;

    document
        .getElementById(
            "cameraStatus"
        )
        .textContent =
        "Camera stopped.";

}


async function captureFrame() {

    const video =
        document
            .getElementById(
                "cameraVideo"
            );

    if (
        !video.videoWidth
        ||
        !video.videoHeight
    ) {

        throw new Error(
            "Camera is not ready."
        );

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

    return canvas.toDataURL(
        "image/jpeg",
        0.85
    );

}


async function captureAndDetect() {

    if (
        detectBusy
    ) {

        return;

    }

    detectBusy = true;

    try {

        const image =
            await captureFrame();

        const result =
            await api(
                "/api/detect",
                {
                    method: "POST",
                    headers: {
                        "Content-Type":
                            "application/json"
                    },
                    body:
                        JSON.stringify(
                            {
                                image
                            }
                        )
                }
            );

        document
            .getElementById(
                "cameraCount"
            )
            .textContent =
            result.total;

        document
            .getElementById(
                "cameraStatus"
            )
            .textContent =
            result.counted
            + " loaded bucket(s) counted.";

        drawDetections(
            result.detections
        );

    } catch (e) {

        document
            .getElementById(
                "cameraStatus"
            )
            .textContent =
            e.message;

    } finally {

        detectBusy = false;

    }

}


function drawDetections(
    detections
) {

    const video =
        document
            .getElementById(
                "cameraVideo"
            );

    const canvas =
        document
            .getElementById(
                "cameraCanvas"
            );

    if (
        !video.videoWidth
    ) {

        return;

    }

    canvas.width =
        video.clientWidth;

    canvas.height =
        video.clientHeight;

    const ctx =
        canvas.getContext(
            "2d"
        );

    ctx.clearRect(
        0,
        0,
        canvas.width,
        canvas.height
    );

    const scaleX =
        canvas.width /
        video.videoWidth;

    const scaleY =
        canvas.height /
        video.videoHeight;

    detections.forEach(
        d => {

            const x =
                d.x1 * scaleX;

            const y =
                d.y1 * scaleY;

            const w =
                (
                    d.x2 -
                    d.x1
                ) * scaleX;

            const h =
                (
                    d.y2 -
                    d.y1
                ) * scaleY;

            ctx.strokeStyle =
                d.class_name ===
                "BUCKET_LOADED"
                    ? "#22c55e"
                    : "#ef4444";

            ctx.lineWidth = 3;

            ctx.strokeRect(
                x,
                y,
                w,
                h
            );

            ctx.fillStyle =
                "rgba(0,0,0,.65)";

            ctx.fillRect(
                x,
                Math.max(
                    0,
                    y - 24
                ),
                Math.min(
                    220,
                    w + 100
                ),
                24
            );

            ctx.fillStyle =
                "#fff";

            ctx.font =
                "14px Arial";

            ctx.fillText(
                d.class_name
                + " "
                + (
                    d.confidence * 100
                ).toFixed(1)
                + "%",
                x + 5,
                Math.max(
                    16,
                    y - 7
                )
            );

        }
    );

}


async function detectUploadedImage() {

    const file =
        document
            .getElementById(
                "detectFile"
            )
            .files[0];

    if (!file) {

        alert(
            "Choose an image first."
        );

        return;

    }

    if (
        file.size >
        10 * 1024 * 1024
    ) {

        alert(
            "Maximum image size is 10 MB."
        );

        return;

    }

    const reader =
        new FileReader();

    reader.onload =
        async function() {

            const image =
                reader.result;

            document
                .getElementById(
                    "detectPreview"
                )
                .src =
                image;

            document
                .getElementById(
                    "detectPreview"
                )
                .style
                .display =
                "block";

            try {

                const result =
                    await api(
                        "/api/detect",
                        {
                            method: "POST",
                            headers: {
                                "Content-Type":
                                    "application/json"
                            },
                            body:
                                JSON.stringify(
                                    {
                                        image
                                    }
                                )
                        }
                    );

                document
                    .getElementById(
                        "cameraCount"
                    )
                    .textContent =
                    result.total;

                document
                    .getElementById(
                        "cameraStatus"
                    )
                    .textContent =
                    result.counted
                    + " loaded bucket(s) counted.";

            } catch (e) {

                alert(
                    e.message
                );

            }

        };

    reader.readAsDataURL(
        file
    );

}


// =====================================================
// TRACKER RESET
// =====================================================

async function resetTracker() {

    try {

        await api(
            "/api/tracker/reset",
            {
                method: "POST"
            }
        );

        document
            .getElementById(
                "cameraStatus"
            )
            .textContent =
            "Tracker reset.";

    } catch (e) {

        alert(
            e.message
        );

    }

}


// =====================================================
// BUCKETS
// =====================================================

async function createBucket() {

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

    if (!name) {

        alert(
            "Enter bucket name."
        );

        return;

    }

    try {

        await api(
            "/api/buckets",
            {
                method: "POST",
                headers: {
                    "Content-Type":
                        "application/json"
                },
                body:
                    JSON.stringify(
                        {
                            name,
                            description
                        }
                    )
            }
        );

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

        await loadBuckets();

    } catch (e) {

        alert(
            e.message
        );

    }

}


async function loadBuckets() {

    try {

        const data =
            await api(
                "/api/buckets"
            );

        const container =
            document
                .getElementById(
                    "bucketList"
                );

        if (
            !data.buckets.length
        ) {

            container.innerHTML =
                "<p>No buckets registered.</p>";

            return;

        }

        container.innerHTML =
            data.buckets
                .map(
                    bucket => `
                    <div
                        class="card"
                        style="
                            margin-bottom:10px;
                            box-shadow:none;
                            border:1px solid #ddd;
                        "
                    >

                        <strong>
                            ${escapeHtml(
                                bucket.name
                            )}
                        </strong>

                        ${
                            bucket.active
                            ?
                            `
                            <span
                                class="badge active"
                            >
                                ACTIVE
                            </span>
                            `
                            :
                            ""
                        }

                        <div
                            class="small"
                        >
                            ${escapeHtml(
                                bucket.description
                                || ""
                            )}
                        </div>

                        <div
                            class="small"
                            style="margin-top:6px"
                        >
                            References:
                            ${
                                bucket.reference_count
                            }
                        </div>

                        ${
                            bucket.active
                            ?
                            ""
                            :
                            `
                            <button
                                class="action"
                                style="margin-top:8px"
                                onclick="
                                    activateBucket(
                                        ${bucket.id}
                                    )
                                "
                            >
                                Set Active
                            </button>
                            `
                        }

                    </div>
                    `
                )
                .join("");

    } catch (e) {

        document
            .getElementById(
                "bucketList"
            )
            .textContent =
            e.message;

    }

}


async function activateBucket(
    id
) {

    try {

        await api(
            `/api/buckets/${id}/activate`,
            {
                method: "POST"
            }
        );

        await loadBuckets();

        await refreshDashboard();

    } catch (e) {

        alert(
            e.message
        );

    }

}


// =====================================================
// DATASET
// =====================================================

async function uploadDatasetImage() {

    const file =
        document
            .getElementById(
                "datasetFile"
            )
            .files[0];

    if (!file) {

        alert(
            "Choose an image."
        );

        return;

    }

    if (
        file.size >
        10 * 1024 * 1024
    ) {

        alert(
            "Maximum image size is 10 MB."
        );

        return;

    }

    const reader =
        new FileReader();

    reader.onload =
        async function() {

            try {

                const result =
                    await api(
                        "/api/dataset/upload",
                        {
                            method: "POST",
                            headers: {
                                "Content-Type":
                                    "application/json"
                            },
                            body:
                                JSON.stringify(
                                    {
                                        filename:
                                            file.name,
                                        image:
                                            reader.result
                                    }
                                )
                        }
                    );

                document
                    .getElementById(
                        "datasetUploadStatus"
                    )
                    .textContent =
                    "Uploaded image ID: "
                    + result.image_id;

                document
                    .getElementById(
                        "datasetUploadStatus"
                    )
                    .className =
                    "status ok";

            } catch (e) {

                document
                    .getElementById(
                        "datasetUploadStatus"
                    )
                    .textContent =
                    e.message;

                document
                    .getElementById(
                        "datasetUploadStatus"
                    )
                    .className =
                    "status error";

            }

        };

    reader.readAsDataURL(
        file
    );

}


// =====================================================
// TRAINING
// =====================================================

async function loadTrainingStatus() {

    try {

        const data =
            await api(
                "/api/training/status"
            );

        const state =
            data.training;

        const status =
            document
                .getElementById(
                    "trainingStatus"
                );

        status.textContent =
            state.status
            + ": "
            + state.message;

        if (
            state.status ===
            "completed"
        ) {

            status.className =
                "status ok";

        } else if (
            state.status ===
            "error"
        ) {

            status.className =
                "status error";

        } else {

            status.className =
                "status";

        }

        const progress =
            Math.max(
                0,
                Math.min(
                    100,
                    Number(
                        state.progress
                    ) || 0
                )
            );

        document
            .getElementById(
                "trainingProgressBar"
            )
            .style
            .width =
            progress
            + "%";

    } catch (e) {

        document
            .getElementById(
                "trainingStatus"
            )
            .textContent =
            e.message;

    }

}


async function startTraining() {

    if (
        !confirm(
            "Start YOLO training now?"
        )
    ) {

        return;

    }

    try {

        const result =
            await api(
                "/api/training/start",
                {
                    method: "POST"
                }
            );

        document
            .getElementById(
                "trainingStatus"
            )
            .textContent =
            result.message
            || "Training started.";

        pollTraining();

    } catch (e) {

        alert(
            e.message
        );

    }

}


function pollTraining() {

    loadTrainingStatus();

    setTimeout(
        async function loop() {

            try {

                const data =
                    await api(
                        "/api/training/status"
                    );

                const state =
                    data.training;

                document
                    .getElementById(
                        "trainingStatus"
                    )
                    .textContent =
                    state.status
                    + ": "
                    + state.message;

                const progress =
                    Math.max(
                        0,
                        Math.min(
                            100,
                            Number(
                                state.progress
                            ) || 0
                        )
                    );

                document
                    .getElementById(
                        "trainingProgressBar"
                    )
                    .style
                    .width =
                    progress
                    + "%";

                if (
                    state.status ===
                        "training"
                ) {

                    setTimeout(
                        loop,
                        3000
                    );

                } else {

                    refreshDashboard();

                }

            } catch (e) {

                setTimeout(
                    loop,
                    5000
                );

            }

        },
        1000
    );

}


// =====================================================
// HISTORY
// =====================================================

async function loadHistory() {

    try {

        const data =
            await api(
                "/api/history"
            );

        if (
            !data.history.length
        ) {

            document
                .getElementById(
                    "historyTable"
                )
                .innerHTML =
                "<p>No history yet.</p>";

            return;

        }

        document
            .getElementById(
                "historyTable"
            )
            .innerHTML = `
                <table>

                    <thead>

                        <tr>
                            <th>Date</th>
                            <th>Loaded Buckets</th>
                            <th>Updated</th>
                        </tr>

                    </thead>

                    <tbody>

                        ${
                            data.history
                                .map(
                                    row => `
                                    <tr>
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
                                    </tr>
                                    `
                                )
                                .join("")
                        }

                    </tbody>

                </table>
            `;

    } catch (e) {

        document
            .getElementById(
                "historyTable"
            )
            .textContent =
            e.message;

    }

}


// =====================================================
// SETTINGS
// =====================================================

async function loadSettings() {

    try {

        const health =
            await api(
                "/api/health"
            );

        document
            .getElementById(
                "settingsStatus"
            )
            .innerHTML = `

                <p>
                    Database:
                    <strong>
                        ${
                            health.database
                            ? "Connected"
                            : "Not configured"
                        }
                    </strong>
                </p>

                <p>
                    Custom Model:
                    <strong>
                        ${
                            health.custom_model
                            ? "Loaded"
                            : "Not loaded"
                        }
                    </strong>
                </p>

                <p>
                    Today's Count:
                    <strong>
                        ${health.today_count}
                    </strong>
                </p>

                <p class="small">
                    API security is controlled
                    by NEERIKA_API_KEY.
                </p>
            `;

    } catch (e) {

        document
            .getElementById(
                "settingsStatus"
            )
            .textContent =
            e.message;

    }

}


// =====================================================
// HTML ESCAPE
// =====================================================

function escapeHtml(
    value
) {

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


// =====================================================
// STARTUP
// =====================================================

async function startup() {

    await refreshDashboard();

    await loadBuckets();

    await loadHistory();

    await loadTrainingStatus();

    await loadSettings();

}

startup();

</script>

</body>
</html>
"""


# =========================================================
# SERVER START
# =========================================================

def main():

    print(
        "=========================================="
    )

    print(
        "NEERIKA BUCKET AI"
    )

    print(
        "Mining Production Bucket Counter"
    )

    print(
        "=========================================="
    )

    print(
        f"PORT: {PORT}"
    )

    print(
        f"DATABASE: "
        f"{'CONFIGURED' if DATABASE_URL else 'NOT CONFIGURED'}"
    )

    print(
        f"API KEY: "
        f"{'ENABLED' if NEERIKA_API_KEY else 'DISABLED'}"
    )

    print(
        f"TIMEZONE: {APP_TIMEZONE}"
    )

    try:

        init_db()

        print(
            "Database initialized successfully."
        )

    except Exception as e:

        print(
            "DATABASE INITIALIZATION ERROR:"
        )

        traceback.print_exc()

    try:

        with MODEL_LOCK:

            load_model_internal()

        if MODEL_IS_CUSTOM:

            print(
                "Custom YOLO model loaded."
            )

        else:

            print(
                "Custom YOLO model NOT loaded."
            )

            print(
                MODEL_ERROR
            )

    except Exception:

        traceback.print_exc()

    server = ThreadingHTTPServer(
        (
            HOST,
            PORT
        ),
        Handler
    )

    print(
        f"Server running on "
        f"http://{HOST}:{PORT}"
    )

    try:

        server.serve_forever()

    except KeyboardInterrupt:

        print(
            "Server stopping..."
        )

    finally:

        server.server_close()


if __name__ == "__main__":

    main()
