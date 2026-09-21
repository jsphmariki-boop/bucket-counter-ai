import os
import io
import csv
import json
import base64
import traceback
import zipfile
import tempfile
import threading
import subprocess
import sys
from datetime import datetime
from urllib.parse import urlparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import psycopg2
from psycopg2.extras import RealDictCursor

from ultralytics import YOLO
from PIL import Image
import numpy as np


# =========================================================
# NEERIKA BUCKET AI
# FULL REPLACEMENT VERSION
# =========================================================

HOST = "0.0.0.0"
PORT = int(os.environ.get("PORT", "8080"))

DATABASE_URL = (
    os.environ.get("DATABASE_URL")
    or os.environ.get("SUPABASE_DB_URL")
    or os.environ.get("POSTGRES_URL")
)

MAX_JSON_BYTES = 12 * 1024 * 1024
MAX_UPLOAD_BYTES = 15 * 1024 * 1024

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best.pt")
MODEL_BACKUP_PATH = os.path.join(BASE_DIR, "ai", "models", "best.pt")

YOLO_CONFIDENCE = 0.25
YOLO_IMAGE_SIZE = 640

MODEL = None
MODEL_ERROR = None
MODEL_LOCK = threading.Lock()

# The current trained model is intended to have:
# class 0 = loaded_bucket
CLASSES = {
    "BUCKET_LOADED": "count",
    "BUCKET_EMPTY": "no count",
    "PEOPLE": "no count",
    "EQUIPMENT": "no count",
}

CLASS_IDS = {
    "BUCKET_LOADED": 0,
    "BUCKET_EMPTY": 1,
    "PEOPLE": 2,
    "EQUIPMENT": 3,
}


# =========================================================
# DATABASE
# =========================================================

def db():
    if not DATABASE_URL:
        raise RuntimeError(
            "DATABASE_URL is not configured in Render Environment Variables."
        )

    return psycopg2.connect(
        DATABASE_URL,
        connect_timeout=10,
    )


def init_db():
    conn = db()
    try:
        c = conn.cursor()

        c.execute("""
            CREATE TABLE IF NOT EXISTS buckets (
                id BIGSERIAL PRIMARY KEY,
                bucket_code TEXT,
                bucket_name TEXT NOT NULL,
                location TEXT,
                status TEXT DEFAULT 'ACTIVE',
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)

        c.execute("""
            CREATE TABLE IF NOT EXISTS dataset_images (
                id BIGSERIAL PRIMARY KEY,
                filename TEXT NOT NULL,
                image_data BYTEA NOT NULL,
                labeled BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)

        c.execute("""
            ALTER TABLE dataset_images
            ADD COLUMN IF NOT EXISTS labeled BOOLEAN DEFAULT FALSE
        """)

        c.execute("""
            ALTER TABLE dataset_images
            ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ DEFAULT NOW()
        """)

        c.execute("""
            CREATE TABLE IF NOT EXISTS annotations (
                id BIGSERIAL PRIMARY KEY,
                image_id BIGINT NOT NULL
                    REFERENCES dataset_images(id)
                    ON DELETE CASCADE,
                class_name TEXT NOT NULL,
                x_center DOUBLE PRECISION NOT NULL DEFAULT 0,
                y_center DOUBLE PRECISION NOT NULL DEFAULT 0,
                width DOUBLE PRECISION NOT NULL DEFAULT 0,
                height DOUBLE PRECISION NOT NULL DEFAULT 0,
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)

        for statement in [
            "ALTER TABLE annotations ADD COLUMN IF NOT EXISTS image_id BIGINT",
            "ALTER TABLE annotations ADD COLUMN IF NOT EXISTS class_name TEXT",
            "ALTER TABLE annotations ADD COLUMN IF NOT EXISTS x_center DOUBLE PRECISION DEFAULT 0",
            "ALTER TABLE annotations ADD COLUMN IF NOT EXISTS y_center DOUBLE PRECISION DEFAULT 0",
            "ALTER TABLE annotations ADD COLUMN IF NOT EXISTS width DOUBLE PRECISION DEFAULT 0",
            "ALTER TABLE annotations ADD COLUMN IF NOT EXISTS height DOUBLE PRECISION DEFAULT 0",
            "ALTER TABLE annotations ADD COLUMN IF NOT EXISTS box_width DOUBLE PRECISION DEFAULT 0",
            "ALTER TABLE annotations ADD COLUMN IF NOT EXISTS box_height DOUBLE PRECISION DEFAULT 0",
            "ALTER TABLE annotations ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ DEFAULT NOW()",
        ]:
            try:
                c.execute(statement)
            except Exception:
                conn.rollback()
                c = conn.cursor()

        c.execute("""
            CREATE TABLE IF NOT EXISTS training_state (
                id INTEGER PRIMARY KEY,
                status TEXT NOT NULL DEFAULT 'WAITING',
                progress INTEGER NOT NULL DEFAULT 0,
                message TEXT,
                updated_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)

        c.execute("""
            INSERT INTO training_state
                (id, status, progress, message)
            VALUES
                (1, 'WAITING', 0, 'YOLO model ready')
            ON CONFLICT(id) DO NOTHING
        """)

        c.execute("""
            CREATE TABLE IF NOT EXISTS trained_model (
                id INTEGER PRIMARY KEY,
                model_name TEXT,
                model_data BYTEA,
                uploaded_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)

        c.execute("""
            CREATE TABLE IF NOT EXISTS daily_counts (
                id BIGSERIAL PRIMARY KEY,
                count_date DATE NOT NULL DEFAULT CURRENT_DATE,
                loaded_count INTEGER NOT NULL DEFAULT 0,
                empty_count INTEGER NOT NULL DEFAULT 0,
                notes TEXT,
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)

        c.execute("""
            CREATE TABLE IF NOT EXISTS detection_events (
                id BIGSERIAL PRIMARY KEY,
                detected_class TEXT NOT NULL,
                confidence DOUBLE PRECISION,
                counted BOOLEAN DEFAULT FALSE,
                detected_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)

        conn.commit()
        print("Database initialization: OK")

    finally:
        conn.close()


# =========================================================
# MODEL
# =========================================================

def load_model():
    global MODEL
    global MODEL_ERROR

    MODEL = None
    MODEL_ERROR = None

    path = MODEL_PATH

    if not os.path.exists(path) and os.path.exists(MODEL_BACKUP_PATH):
        path = MODEL_BACKUP_PATH

    if not os.path.exists(path):
        MODEL_ERROR = (
            "best.pt was not found. Put best.pt in the same folder as app.py."
        )
        print(MODEL_ERROR)
        return

    try:
        print("==============================================")
        print("Loading NEERIKA BUCKET AI model")
        print("Model:", path)
        print("==============================================")

        with MODEL_LOCK:
            MODEL = YOLO(path)

        print("YOLO model loaded successfully.")

        try:
            print("Model classes:", MODEL.names)
        except Exception:
            pass

    except Exception as e:
        MODEL_ERROR = str(e)
        print("YOLO MODEL ERROR:")
        traceback.print_exc()


def model_ready():
    return MODEL is not None


def model_status():
    if model_ready():
        try:
            names = MODEL.names
        except Exception:
            names = {}

        return {
            "ready": True,
            "file": os.path.basename(MODEL_PATH),
            "classes": names,
            "error": None,
        }

    return {
        "ready": False,
        "file": "best.pt",
        "classes": {},
        "error": MODEL_ERROR,
    }


# =========================================================
# HELPERS
# =========================================================

def esc(v):
    return (
        str(v if v is not None else "")
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


def send_json(h, data, status=200):
    raw = json.dumps(
        data,
        ensure_ascii=False,
        default=str,
    ).encode("utf-8")

    try:
        h.send_response(status)
        h.send_header(
            "Content-Type",
            "application/json; charset=utf-8",
        )
        h.send_header("Content-Length", str(len(raw)))
        h.send_header("Cache-Control", "no-store")
        h.end_headers()
        h.wfile.write(raw)
    except (BrokenPipeError, ConnectionResetError):
        pass


def send_html(h, html, status=200):
    raw = html.encode("utf-8")

    try:
        h.send_response(status)
        h.send_header(
            "Content-Type",
            "text/html; charset=utf-8",
        )
        h.send_header("Content-Length", str(len(raw)))
        h.send_header("Cache-Control", "no-store")
        h.end_headers()
        h.wfile.write(raw)
    except (BrokenPipeError, ConnectionResetError):
        pass


def read_json(h):
    n = int(h.headers.get("Content-Length", "0") or 0)

    if n > MAX_JSON_BYTES:
        raise ValueError("Request is too large.")

    if not n:
        return {}

    raw = h.rfile.read(n)
    return json.loads(raw.decode("utf-8"))


def set_training_state(status, progress, message):
    conn = db()
    try:
        c = conn.cursor()
        c.execute("""
            UPDATE training_state
            SET status=%s,
                progress=%s,
                message=%s,
                updated_at=NOW()
            WHERE id=1
        """, (status, int(progress), message))
        conn.commit()
    finally:
        conn.close()


def get_training_state():
    conn = db()
    try:
        c = conn.cursor(cursor_factory=RealDictCursor)
        c.execute("""
            SELECT status, progress, message, updated_at
            FROM training_state
            WHERE id=1
        """)
        row = c.fetchone()

        if not row:
            return {
                "status": "WAITING",
                "progress": 0,
                "message": "YOLO model ready",
            }

        return dict(row)
    finally:
        conn.close()


# =========================================================
# SUMMARY / COUNTS
# =========================================================

def summary():
    conn = db()
    try:
        c = conn.cursor()

        c.execute("SELECT COUNT(*) FROM dataset_images")
        total = int(c.fetchone()[0])

        c.execute("""
            SELECT COUNT(*)
            FROM dataset_images
            WHERE labeled=TRUE
        """)
        labeled = int(c.fetchone()[0])

        c.execute("SELECT COUNT(*) FROM annotations")
        annotations = int(c.fetchone()[0])

        return {
            "total": total,
            "labeled": labeled,
            "annotations": annotations,
        }
    finally:
        conn.close()


def class_counts():
    conn = db()
    try:
        c = conn.cursor()

        c.execute("""
            SELECT class_name, COUNT(*)
            FROM annotations
            GROUP BY class_name
        """)

        out = {k: 0 for k in CLASSES}

        for k, v in c.fetchall():
            if k in out:
                out[k] = int(v)

        return out
    finally:
        conn.close()


def get_today_count():
    conn = db()
    try:
        c = conn.cursor()

        c.execute("""
            SELECT loaded_count
            FROM daily_counts
            WHERE count_date=CURRENT_DATE
            ORDER BY id DESC
            LIMIT 1
        """)

        row = c.fetchone()

        if row:
            return int(row[0])

        return 0
    finally:
        conn.close()


def add_daily_count(amount=1):
    conn = db()
    try:
        c = conn.cursor()

        c.execute("""
            SELECT id, loaded_count
            FROM daily_counts
            WHERE count_date=CURRENT_DATE
            ORDER BY id DESC
            LIMIT 1
        """)

        row = c.fetchone()

        if row:
            new_count = int(row[1]) + int(amount)

            c.execute("""
                UPDATE daily_counts
                SET loaded_count=%s
                WHERE id=%s
            """, (new_count, row[0]))

        else:
            new_count = int(amount)

            c.execute("""
                INSERT INTO daily_counts
                    (count_date, loaded_count)
                VALUES
                    (CURRENT_DATE, %s)
            """, (new_count,))

        conn.commit()
        return new_count

    finally:
        conn.close()


def save_detection_event(detected_class, confidence, counted):
    conn = db()
    try:
        c = conn.cursor()

        c.execute("""
            INSERT INTO detection_events
                (detected_class, confidence, counted)
            VALUES
                (%s, %s, %s)
        """, (
            detected_class,
            confidence,
            counted,
        ))

        conn.commit()

    finally:
        conn.close()


# =========================================================
# IMAGE
# =========================================================

def decode_image_data(src):
    if not src:
        raise ValueError("Image data is empty.")

    if "," in src:
        src = src.split(",", 1)[1]

    raw = base64.b64decode(src, validate=True)

    if not raw:
        raise ValueError("Could not decode image.")

    if len(raw) > MAX_UPLOAD_BYTES:
        raise ValueError("Image is too large.")

    image = Image.open(io.BytesIO(raw)).convert("RGB")
    return image


# =========================================================
# DETECTION
# =========================================================

def normalize_class(cls_id, model_name):
    name = str(model_name or "").strip().lower()

    if cls_id == 0:
        return "BUCKET_LOADED"

    if name in (
        "loaded_bucket",
        "bucket_loaded",
        "loaded bucket",
        "loaded-bucket",
    ):
        return "BUCKET_LOADED"

    if name in (
        "empty_bucket",
        "bucket_empty",
        "empty bucket",
        "empty-bucket",
    ):
        return "BUCKET_EMPTY"

    if name in ("person", "people", "worker"):
        return "PEOPLE"

    if name in ("equipment", "machine", "vehicle"):
        return "EQUIPMENT"

    upper = str(model_name or "").upper()

    if upper in CLASSES:
        return upper

    return str(model_name or "UNKNOWN")


def detect_image(src):
    if not model_ready():
        raise RuntimeError(
            "YOLO model is not ready. "
            + str(MODEL_ERROR or "")
        )

    image = decode_image_data(src)
    image_np = np.array(image)

    with MODEL_LOCK:
        results = MODEL.predict(
            source=image_np,
            conf=YOLO_CONFIDENCE,
            imgsz=YOLO_IMAGE_SIZE,
            verbose=False,
        )

    detections = []

    if not results:
        return detections

    result = results[0]

    try:
        names = MODEL.names
    except Exception:
        names = {}

    if result.boxes is None:
        return detections

    for box in result.boxes:
        try:
            cls_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())

            xyxy = (
                box.xyxy[0]
                .cpu()
                .numpy()
                .tolist()
            )

            x1, y1, x2, y2 = xyxy

            if isinstance(names, dict):
                model_class = str(
                    names.get(cls_id, cls_id)
                )
            else:
                model_class = str(names[cls_id])

            app_class = normalize_class(
                cls_id,
                model_class,
            )

            detections.append({
                "class_id": cls_id,
                "class_name": app_class,
                "model_class": model_class,
                "confidence": round(confidence, 4),
                "x1": round(float(x1), 2),
                "y1": round(float(y1), 2),
                "x2": round(float(x2), 2),
                "y2": round(float(y2), 2),
            })

        except Exception:
            continue

    return detections


# =========================================================
# DATASET
# =========================================================

def training_images():
    conn = db()
    try:
        c = conn.cursor(cursor_factory=RealDictCursor)

        c.execute("""
            SELECT
                d.id,
                d.filename,
                d.created_at,
                d.labeled,
                COUNT(a.id) AS annotation_count
            FROM dataset_images d
            LEFT JOIN annotations a
                ON a.image_id=d.id
            GROUP BY
                d.id,
                d.filename,
                d.created_at,
                d.labeled
            ORDER BY d.id DESC
        """)

        rows = []

        for r in c.fetchall():
            r = dict(r)

            if r.get("created_at"):
                r["created_at"] = str(r["created_at"])

            r["annotation_count"] = int(
                r.get("annotation_count") or 0
            )

            rows.append(r)

        return rows

    finally:
        conn.close()


def get_image(image_id):
    conn = db()
    try:
        c = conn.cursor(
            cursor_factory=RealDictCursor
        )

        c.execute("""
            SELECT id, filename, image_data, labeled, created_at
            FROM dataset_images
            WHERE id=%s
        """, (image_id,))

        row = c.fetchone()

        if not row:
            return None

        return dict(row)

    finally:
        conn.close()


def upload_dataset_image(filename, image_data):
    image = decode_image_data(image_data)

    output = io.BytesIO()
    image.save(output, format="JPEG", quality=90)
    raw = output.getvalue()

    conn = db()
    try:
        c = conn.cursor()

        c.execute("""
            INSERT INTO dataset_images
                (filename, image_data, labeled)
            VALUES
                (%s, %s, FALSE)
            RETURNING id
        """, (
            filename or "image.jpg",
            psycopg2.Binary(raw),
        ))

        image_id = int(c.fetchone()[0])
        conn.commit()

        return image_id

    finally:
        conn.close()


def save_annotations(image_id, annotations):
    conn = db()
    try:
        c = conn.cursor()

        c.execute("""
            DELETE FROM annotations
            WHERE image_id=%s
        """, (image_id,))

        valid = 0

        for a in annotations:
            class_name = str(
                a.get("class_name", "BUCKET_LOADED")
            )

            x = float(a.get("x_center", 0))
            y = float(a.get("y_center", 0))
            w = float(a.get("width", 0))
            h = float(a.get("height", 0))

            x = max(0, min(1, x))
            y = max(0, min(1, y))
            w = max(0, min(1, w))
            h = max(0, min(1, h))

            c.execute("""
                INSERT INTO annotations
                    (
                        image_id,
                        class_name,
                        x_center,
                        y_center,
                        width,
                        height,
                        box_width,
                        box_height
                    )
                VALUES
                    (%s,%s,%s,%s,%s,%s,%s,%s)
            """, (
                image_id,
                class_name,
                x,
                y,
                w,
                h,
                w,
                h,
            ))

            valid += 1

        c.execute("""
            UPDATE dataset_images
            SET labeled=%s
            WHERE id=%s
        """, (
            valid > 0,
            image_id,
        ))

        conn.commit()

        return valid

    finally:
        conn.close()


def delete_dataset_image(image_id):
    conn = db()
    try:
        c = conn.cursor()

        c.execute("""
            DELETE FROM dataset_images
            WHERE id=%s
        """, (image_id,))

        conn.commit()

        return c.rowcount > 0

    finally:
        conn.close()


# =========================================================
# DATASET EXPORT
# =========================================================

def build_dataset_zip():
    conn = db()

    try:
        c = conn.cursor(
            cursor_factory=RealDictCursor
        )

        c.execute("""
            SELECT id, filename, image_data
            FROM dataset_images
            WHERE labeled=TRUE
            ORDER BY id
        """)

        images = [dict(x) for x in c.fetchall()]

        if not images:
            raise ValueError(
                "No labeled images available."
            )

        c.execute("""
            SELECT
                image_id,
                class_name,
                x_center,
                y_center,
                width,
                height
            FROM annotations
            ORDER BY image_id, id
        """)

        annotations = {}

        for r in c.fetchall():
            r = dict(r)
            annotations.setdefault(
                int(r["image_id"]),
                [],
            ).append(r)

        output = io.BytesIO()

        with zipfile.ZipFile(
            output,
            "w",
            zipfile.ZIP_DEFLATED,
        ) as z:

            yaml = """path: .
train: images
val: images

names:
  0: loaded_bucket
"""

            z.writestr(
                "data.yaml",
                yaml,
            )

            for item in images:
                image_id = int(item["id"])
                original = str(
                    item["filename"] or
                    f"image_{image_id}.jpg"
                )

                safe = (
                    os.path.basename(original)
                    .replace(" ", "_")
                )

                if not safe.lower().endswith(
                    (".jpg", ".jpeg", ".png")
                ):
                    safe += ".jpg"

                image_path = f"images/{safe}"

                z.writestr(
                    image_path,
                    bytes(item["image_data"]),
                )

                lines = []

                for a in annotations.get(
                    image_id,
                    [],
                ):
                    # Current training model uses one class:
                    # 0 = loaded_bucket
                    class_id = 0

                    lines.append(
                        "%d %.6f %.6f %.6f %.6f"
                        % (
                            class_id,
                            float(a["x_center"]),
                            float(a["y_center"]),
                            float(a["width"]),
                            float(a["height"]),
                        )
                    )

                label_name = os.path.splitext(
                    safe
                )[0] + ".txt"

                z.writestr(
                    f"labels/{label_name}",
                    "\n".join(lines),
                )

        output.seek(0)
        return output.getvalue()

    finally:
        conn.close()


# =========================================================
# TRAINING
# =========================================================

TRAIN_THREAD = None
TRAIN_LOCK = threading.Lock()


def training_worker():
    global MODEL
    global MODEL_ERROR

    try:
        set_training_state(
            "PREPARING",
            5,
            "Preparing training dataset...",
        )

        conn = db()

        try:
            c = conn.cursor(
                cursor_factory=RealDictCursor
            )

            c.execute("""
                SELECT id, filename, image_data
                FROM dataset_images
                WHERE labeled=TRUE
                ORDER BY id
            """)

            images = [
                dict(x)
                for x in c.fetchall()
            ]

            c.execute("""
                SELECT
                    image_id,
                    class_name,
                    x_center,
                    y_center,
                    width,
                    height
                FROM annotations
                ORDER BY image_id, id
            """)

            annotations = {}

            for r in c.fetchall():
                r = dict(r)
                annotations.setdefault(
                    int(r["image_id"]),
                    [],
                ).append(r)

        finally:
            conn.close()

        if len(images) < 2:
            raise RuntimeError(
                "At least 2 labeled images are required before training."
            )

        temp_dir = tempfile.mkdtemp(
            prefix="neerika_train_"
        )

        images_dir = os.path.join(
            temp_dir,
            "images",
        )

        labels_dir = os.path.join(
            temp_dir,
            "labels",
        )

        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)

        for index, item in enumerate(images):
            image_id = int(item["id"])

            filename = os.path.basename(
                str(item["filename"])
            )

            if not filename.lower().endswith(
                (".jpg", ".jpeg", ".png")
            ):
                filename += ".jpg"

            image_path = os.path.join(
                images_dir,
                filename,
            )

            with open(
                image_path,
                "wb",
            ) as f:
                f.write(
                    bytes(item["image_data"])
                )

            label_name = os.path.splitext(
                filename
            )[0] + ".txt"

            label_path = os.path.join(
                labels_dir,
                label_name,
            )

            with open(
                label_path,
                "w",
                encoding="utf-8",
            ) as f:

                for a in annotations.get(
                    image_id,
                    [],
                ):
                    f.write(
                        "0 %.6f %.6f %.6f %.6f\n"
                        % (
                            float(a["x_center"]),
                            float(a["y_center"]),
                            float(a["width"]),
                            float(a["height"]),
                        )
                    )

            progress = 5 + int(
                (index + 1) / len(images) * 20
            )

            set_training_state(
                "PREPARING",
                progress,
                f"Preparing image {index + 1}/{len(images)}",
            )

        data_yaml = os.path.join(
            temp_dir,
            "data.yaml",
        )

        with open(
            data_yaml,
            "w",
            encoding="utf-8",
        ) as f:
            f.write(
                "path: %s\n"
                "train: images\n"
                "val: images\n\n"
                "names:\n"
                "  0: loaded_bucket\n"
                % temp_dir.replace("\\", "/")
            )

        set_training_state(
            "TRAINING",
            30,
            "YOLO training started...",
        )

        base_model = (
            MODEL_PATH
            if os.path.exists(MODEL_PATH)
            else "yolo11n.pt"
        )

        trainer = YOLO(base_model)

        results = trainer.train(
            data=data_yaml,
            epochs=20,
            imgsz=640,
            project=temp_dir,
            name="neerika_bucket",
            exist_ok=True,
            verbose=True,
        )

        best_candidate = os.path.join(
            temp_dir,
            "neerika_bucket",
            "weights",
            "best.pt",
        )

        if not os.path.exists(best_candidate):
            raise RuntimeError(
                "Training finished but best.pt was not produced."
            )

        set_training_state(
            "SAVING",
            90,
            "Saving trained model...",
        )

        with open(
            best_candidate,
            "rb",
        ) as f:
            model_bytes = f.read()

        # Save local best.pt for Render instance.
        with open(
            MODEL_PATH,
            "wb",
        ) as f:
            f.write(model_bytes)

        # Save a copy to PostgreSQL/Supabase.
        conn = db()

        try:
            c = conn.cursor()

            c.execute("""
                INSERT INTO trained_model
                    (id, model_name, model_data)
                VALUES
                    (1, %s, %s)
                ON CONFLICT(id)
                DO UPDATE SET
                    model_name=EXCLUDED.model_name,
                    model_data=EXCLUDED.model_data,
                    uploaded_at=NOW()
            """, (
                "best.pt",
                psycopg2.Binary(model_bytes),
            ))

            conn.commit()

        finally:
            conn.close()

        # Reload model.
        MODEL = None
        MODEL_ERROR = None

        try:
            MODEL = YOLO(MODEL_PATH)
        except Exception as e:
            MODEL_ERROR = str(e)
            raise

        set_training_state(
            "COMPLETED",
            100,
            "Training completed successfully. New best.pt is ready.",
        )

    except Exception as e:
        traceback.print_exc()

        try:
            set_training_state(
                "ERROR",
                0,
                str(e),
            )
        except Exception:
            pass


def start_training():
    global TRAIN_THREAD

    with TRAIN_LOCK:
        if TRAIN_THREAD and TRAIN_THREAD.is_alive():
            return False

        TRAIN_THREAD = threading.Thread(
            target=training_worker,
            daemon=True,
        )

        TRAIN_THREAD.start()
        return True


# =========================================================
# DAILY HISTORY
# =========================================================

def history():
    conn = db()
    try:
        c = conn.cursor(
            cursor_factory=RealDictCursor
        )

        c.execute("""
            SELECT
                count_date,
                loaded_count,
                empty_count,
                notes
            FROM daily_counts
            ORDER BY count_date DESC, id DESC
            LIMIT 90
        """)

        rows = []

        for r in c.fetchall():
            rows.append(dict(r))

        return rows

    finally:
        conn.close()


def detection_history():
    conn = db()
    try:
        c = conn.cursor(
            cursor_factory=RealDictCursor
        )

        c.execute("""
            SELECT
                id,
                detected_class,
                confidence,
                counted,
                detected_at
            FROM detection_events
            ORDER BY id DESC
            LIMIT 100
        """)

        return [
            dict(x)
            for x in c.fetchall()
        ]

    finally:
        conn.close()


# =========================================================
# HTML LAYOUT
# =========================================================

def layout(title, body, active):
    links = [
        ("Dashboard", "/"),
        ("Camera", "/camera"),
        ("Buckets", "/buckets"),
        ("AI Training", "/training"),
        ("History", "/history"),
        ("Settings", "/settings"),
    ]

    nav = ""

    for name, url in links:
        cls = "active" if name == active else ""

        nav += (
            '<a class="%s" href="%s">%s</a>'
            % (cls, url, esc(name))
        )

    return """
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport"
      content="width=device-width,initial-scale=1">

<title>%s - NEERIKA BUCKET AI</title>

<style>
*{box-sizing:border-box}

body{
    margin:0;
    font-family:Arial,sans-serif;
    background:#f4f6f8;
    color:#17202a
}

header{
    background:#111827;
    color:white;
    padding:16px;
    position:sticky;
    top:0;
    z-index:5
}

.brand{
    font-size:21px;
    font-weight:800
}

.sub{
    font-size:12px;
    color:#cbd5e1;
    margin-top:4px
}

nav{
    display:flex;
    gap:6px;
    overflow:auto;
    margin-top:12px
}

nav a{
    color:#dbeafe;
    text-decoration:none;
    padding:9px 11px;
    border-radius:9px;
    white-space:nowrap
}

nav a.active,
nav a:hover{
    background:#2563eb;
    color:white
}

main{
    max-width:1100px;
    margin:auto;
    padding:18px
}

.card{
    background:white;
    border-radius:14px;
    padding:18px;
    margin-bottom:16px;
    box-shadow:0 2px 9px #00000012
}

h1{
    margin:0 0 15px;
    font-size:25px
}

h2{
    margin:0 0 12px;
    font-size:19px
}

.grid{
    display:grid;
    grid-template-columns:repeat(auto-fit,minmax(180px,1fr));
    gap:12px
}

.stat{
    padding:15px;
    border:1px solid #e2e8f0;
    border-radius:12px;
    background:#f8fafc
}

.stat b{
    display:block;
    font-size:28px;
    margin-top:5px
}

button{
    border:0;
    border-radius:9px;
    padding:11px 15px;
    background:#2563eb;
    color:white;
    font-weight:700;
    cursor:pointer
}

button:hover{
    opacity:.9
}

button.danger{
    background:#dc2626
}

button.gray{
    background:#64748b
}

input,select,textarea{
    width:100%;
    padding:11px;
    border:1px solid #cbd5e1;
    border-radius:9px;
    margin:5px 0 12px;
    background:white
}

label{
    font-weight:700;
    font-size:13px
}

table{
    width:100%;
    border-collapse:collapse
}

th,td{
    border-bottom:1px solid #e2e8f0;
    padding:10px;
    text-align:left;
    font-size:14px
}

th{
    background:#f8fafc
}

.badge{
    display:inline-block;
    padding:5px 8px;
    border-radius:999px;
    background:#dbeafe;
    color:#1d4ed8;
    font-size:12px;
    font-weight:700
}

.ok{
    color:#15803d;
    font-weight:700
}

.err{
    color:#dc2626;
    font-weight:700
}

.video-wrap{
    background:#000;
    border-radius:14px;
    overflow:hidden;
    position:relative
}

video{
    width:100%;
    display:block;
    max-height:65vh;
    object-fit:contain
}

canvas{
    width:100%;
    display:block
}

.camera-controls{
    display:flex;
    flex-wrap:wrap;
    gap:8px;
    margin-top:12px
}

.result{
    padding:14px;
    border-radius:12px;
    background:#f8fafc;
    margin-top:12px
}

.big{
    font-size:42px;
    font-weight:800
}

.small{
    font-size:12px;
    color:#64748b
}

.progress{
    height:18px;
    background:#e2e8f0;
    border-radius:999px;
    overflow:hidden
}

.progress > div{
    height:100%;
    background:#2563eb;
    width:0%;
    transition:width .3s
}

.thumb-grid{
    display:grid;
    grid-template-columns:repeat(auto-fill,minmax(150px,1fr));
    gap:12px
}

.thumb{
    border:1px solid #e2e8f0;
    border-radius:12px;
    overflow:hidden;
    background:#fff
}

.thumb img{
    width:100%;
    aspect-ratio:4/3;
    object-fit:cover;
    display:block
}

.thumb-body{
    padding:9px
}

footer{
    text-align:center;
    padding:25px;
    color:#64748b;
    font-size:12px
}

@media(max-width:600px){
    main{padding:10px}
    .card{padding:13px}
    h1{font-size:21px}
}
</style>
</head>

<body>

<header>
    <div class="brand">NEERIKA BUCKET AI</div>
    <div class="sub">Mining Production Bucket Counter</div>
    <nav>%s</nav>
</header>

<main>
%s
</main>

<footer>
    Geology &amp; Mining Services
</footer>

</body>
</html>
""" % (
        esc(title),
        nav,
        body,
    )


# =========================================================
# DASHBOARD
# =========================================================

def dashboard_page():
    try:
        s = summary()
        counts = class_counts()
        today = get_today_count()
        model = model_status()
        train = get_training_state()

        model_html = (
            '<span class="ok">READY</span>'
            if model["ready"]
            else '<span class="err">NOT READY</span>'
        )

        body = """
<h1>Dashboard</h1>

<div class="grid">

<div class="stat">
    Today's Loaded Buckets
    <b id="today">%s</b>
</div>

<div class="stat">
    Dataset Images
    <b>%s</b>
</div>

<div class="stat">
    Labeled Images
    <b>%s</b>
</div>

<div class="stat">
    Annotations
    <b>%s</b>
</div>

</div>

<div class="card">
<h2>AI Model</h2>
<p>Status: %s</p>
<p>Model: <b>best.pt</b></p>
<p>Confidence: <b>%s</b></p>
<p>Image size: <b>%s</b></p>
%s
</div>

<div class="card">
<h2>Training</h2>
<p>Status: <b>%s</b></p>
<p>%s</p>
<div class="progress">
<div style="width:%s%%"></div>
</div>
<p class="small">%s%%</p>
</div>

<div class="card">
<h2>Detection Classes</h2>
<table>
<tr><th>Class</th><th>Action</th><th>Annotations</th></tr>
<tr>
<td>Loaded Bucket</td>
<td><span class="badge">COUNT</span></td>
<td>%s</td>
</tr>
<tr>
<td>Empty Bucket</td>
<td>NO COUNT</td>
<td>%s</td>
</tr>
<tr>
<td>People</td>
<td>NO COUNT</td>
<td>%s</td>
</tr>
<tr>
<td>Equipment</td>
<td>NO COUNT</td>
<td>%s</td>
</tr>
</table>
</div>

<div class="card">
<h2>How it works</h2>
<ol>
<li>Open Camera.</li>
<li>Allow camera access.</li>
<li>Point the camera at the bucket path.</li>
<li>Run AI detection.</li>
<li>Only a detected <b>loaded_bucket</b> is eligible for counting.</li>
<li>Empty buckets, people and equipment are not counted.</li>
</ol>
</div>
""" % (
            today,
            s["total"],
            s["labeled"],
            s["annotations"],
            model_html,
            YOLO_CONFIDENCE,
            YOLO_IMAGE_SIZE,
            (
                '<p class="err">%s</p>'
                % esc(model["error"])
                if model["error"]
                else ""
            ),
            esc(train.get("status", "")),
            esc(train.get("message", "")),
            int(train.get("progress") or 0),
            int(train.get("progress") or 0),
            counts.get("BUCKET_LOADED", 0),
            counts.get("BUCKET_EMPTY", 0),
            counts.get("PEOPLE", 0),
            counts.get("EQUIPMENT", 0),
        )

        return layout(
            "Dashboard",
            body,
            "Dashboard",
        )

    except Exception as e:
        return layout(
            "Dashboard Error",
            '<div class="card"><h1>Error</h1><p class="err">%s</p></div>'
            % esc(str(e)),
            "Dashboard",
        )


# =========================================================
# CAMERA PAGE
# =========================================================

def camera_page():
    body = """
<h1>Camera</h1>

<div class="card">

<div class="video-wrap">
    <video id="video"
           autoplay
           playsinline
           muted></video>
</div>

<div class="camera-controls">
    <button onclick="startCamera()">Start Camera</button>
    <button onclick="stopCamera()" class="gray">Stop</button>
    <button onclick="detectFrame()">Detect Bucket</button>
    <button onclick="autoToggle()" id="autoBtn">Auto: OFF</button>
</div>

<div class="result">
    <div class="small">Today's count</div>
    <div class="big" id="count">...</div>
</div>

<div class="result" id="status">
    Camera is not started.
</div>

<div class="result" id="detections">
    No detection yet.
</div>

</div>

<div class="card">
<h2>Counting rule</h2>
<p>
Only <b>BUCKET_LOADED</b> is counted.
Empty buckets, people and equipment are not counted.
</p>
<p class="small">
The current trained model is configured for class 0 =
loaded_bucket.
</p>
</div>

<script>
let stream = null;
let auto = false;
let busy = false;
let lastCountTime = 0;

const video = document.getElementById("video");

async function startCamera(){
    try{
        stream = await navigator.mediaDevices.getUserMedia({
            video:{
                facingMode:{ideal:"environment"},
                width:{ideal:1280},
                height:{ideal:720}
            },
            audio:false
        });

        video.srcObject = stream;

        document.getElementById("status").innerHTML =
            '<span class="ok">Camera started.</span>';

    }catch(e){
        document.getElementById("status").innerHTML =
            '<span class="err">Camera error: '+e.message+'</span>';
    }
}

function stopCamera(){
    if(stream){
        stream.getTracks().forEach(t=>t.stop());
        stream=null;
    }

    video.srcObject=null;
    auto=false;
    document.getElementById("autoBtn").innerText="Auto: OFF";
}

function autoToggle(){
    auto=!auto;

    document.getElementById("autoBtn").innerText =
        auto ? "Auto: ON" : "Auto: OFF";

    if(auto){
        autoLoop();
    }
}

async function autoLoop(){
    if(!auto) return;

    await detectFrame();

    setTimeout(autoLoop, 1800);
}

async function detectFrame(){
    if(busy) return;

    if(!video.videoWidth){
        document.getElementById("status").innerHTML =
            '<span class="err">Start the camera first.</span>';
        return;
    }

    busy=true;

    try{
        const canvas=document.createElement("canvas");

        canvas.width=video.videoWidth;
        canvas.height=video.videoHeight;

        const ctx=canvas.getContext("2d");
        ctx.drawImage(video,0,0);

        const image=canvas.toDataURL("image/jpeg",0.85);

        const response=await fetch("/api/detect",{
            method:"POST",
            headers:{
                "Content-Type":"application/json"
            },
            body:JSON.stringify({
                image:image
            })
        });

        const data=await response.json();

        if(!response.ok){
            throw new Error(data.error || "Detection failed");
        }

        document.getElementById("count").innerText =
            data.today_count;

        renderDetections(data);

    }catch(e){
        document.getElementById("status").innerHTML =
            '<span class="err">'+escapeHtml(e.message)+'</span>';
    }finally{
        busy=false;
    }
}

function renderDetections(data){
    const box=document.getElementById("detections");

    if(!data.detections || !data.detections.length){
        box.innerHTML="No bucket detected.";
        return;
    }

    let html="";

    data.detections.forEach(d=>{
        html +=
            "<div><b>"+
            escapeHtml(d.class_name)+
            "</b> — confidence "+
            Number(d.confidence).toFixed(2)+
            " — "+
            (d.counted ? "COUNTED" : "NOT COUNTED")+
            "</div>";
    });

    box.innerHTML=html;

    document.getElementById("status").innerHTML =
        data.counted
        ? '<span class="ok">Loaded bucket counted.</span>'
        : 'Detection completed. No new bucket counted.';
}

function escapeHtml(s){
    return String(s)
      .replaceAll("&","&amp;")
      .replaceAll("<","&lt;")
      .replaceAll(">","&gt;")
      .replaceAll('"',"&quot;")
      .replaceAll("'","&#039;");
}

async function loadCount(){
    try{
        const r=await fetch("/api/today");
        const d=await r.json();
        document.getElementById("count").innerText=d.count;
    }catch(e){}
}

loadCount();
</script>
"""

    return layout(
        "Camera",
        body,
        "Camera",
    )


# =========================================================
# TRAINING PAGE
# =========================================================

def training_page():
    try:
        s = summary()
        train = get_training_state()

        images_json = json.dumps(
            training_images(),
            default=str,
        )

        body = """
<h1>AI Training</h1>

<div class="card">
<h2>Dataset Summary</h2>

<div class="grid">

<div class="stat">
Images
<b>%s</b>
</div>

<div class="stat">
Labeled
<b>%s</b>
</div>

<div class="stat">
Annotations
<b>%s</b>
</div>

</div>
</div>

<div class="card">

<h2>1. Upload Training Images</h2>

<input id="imageFiles"
       type="file"
       accept="image/*"
       multiple>

<button onclick="uploadImages()">
Upload Images
</button>

<div id="uploadStatus"
     class="result">
Ready.
</div>

</div>

<div class="card">

<h2>2. Label Images</h2>

<p>
Open an image below and create a box around the loaded bucket.
For the current model use class:
<b>BUCKET_LOADED</b>.
</p>

<div id="images"
     class="thumb-grid">
Loading...
</div>

</div>

<div class="card">

<h2>3. Train Model</h2>

<p>
Training uses the labeled images stored in Supabase/PostgreSQL.
The current training configuration uses one class:
<b>loaded_bucket</b>.
</p>

<button onclick="startTraining()">
Start YOLO Training
</button>

<div class="result">
Status:
<b id="trainStatus">%s</b>

<p id="trainMessage">%s</p>

<div class="progress">
<div id="trainBar"
     style="width:%s%%"></div>
</div>

<p id="trainProgress">%s%%</p>
</div>

</div>

<script>
let dataset=%s;

function renderImages(){
    const box=document.getElementById("images");

    if(!dataset.length){
        box.innerHTML="<p>No images uploaded.</p>";
        return;
    }

    let html="";

    dataset.forEach(x=>{
        html += `
        <div class="thumb">
            <img src="/api/image/${x.id}">
            <div class="thumb-body">
                <b>${escapeHtml(x.filename)}</b>
                <div class="small">
                    ID: ${x.id}<br>
                    Annotations: ${x.annotation_count}<br>
                    Labeled: ${x.labeled ? "YES" : "NO"}
                </div>
                <br>
                <button onclick="labelImage(${x.id})">
                    Label
                </button>
                <button class="danger"
                        onclick="deleteImage(${x.id})">
                    Delete
                </button>
            </div>
        </div>`;
    });

    box.innerHTML=html;
}

async function uploadImages(){
    const files=document.getElementById("imageFiles").files;

    if(!files.length){
        alert("Choose image files first.");
        return;
    }

    const status=document.getElementById("uploadStatus");
    status.innerText="Uploading...";

    let success=0;

    for(let i=0;i<files.length;i++){
        try{
            const file=files[i];

            const reader=new FileReader();

            const data=await new Promise((resolve,reject)=>{
                reader.onload=()=>resolve(reader.result);
                reader.onerror=reject;
                reader.readAsDataURL(file);
            });

            const r=await fetch("/api/dataset/upload",{
                method:"POST",
                headers:{
                    "Content-Type":"application/json"
                },
                body:JSON.stringify({
                    filename:file.name,
                    image:data
                })
            });

            const d=await r.json();

            if(!r.ok){
                throw new Error(d.error || "Upload failed");
            }

            success++;

            status.innerText =
                "Uploaded "+success+"/"+files.length;

        }catch(e){
            status.innerText =
                "Upload error: "+e.message;
        }
    }

    await reloadImages();
    status.innerText =
        "Finished. Uploaded "+success+" image(s).";
}

async function reloadImages(){
    const r=await fetch("/api/dataset");
    const d=await r.json();

    dataset=d.images || [];
    renderImages();
}

async function deleteImage(id){
    if(!confirm("Delete this image and its annotations?")){
        return;
    }

    const r=await fetch(
        "/api/dataset/delete/"+id,
        {method:"POST"}
    );

    const d=await r.json();

    if(!r.ok){
        alert(d.error || "Delete failed");
        return;
    }

    await reloadImages();
}

async function labelImage(id){
    const response=await fetch("/api/dataset/"+id);
    const d=await response.json();

    if(!response.ok){
        alert(d.error || "Could not load image");
        return;
    }

    const imageUrl="/api/image/"+id;

    const x=prompt(
        "X center (%) from left, e.g. 50"
    );

    if(x===null) return;

    const y=prompt(
        "Y center (%) from top, e.g. 50"
    );

    if(y===null) return;

    const w=prompt(
        "Box width (%) e.g. 40"
    );

    if(w===null) return;

    const h=prompt(
        "Box height (%) e.g. 50"
    );

    if(h===null) return;

    const values=[
        Number(x)/100,
        Number(y)/100,
        Number(w)/100,
        Number(h)/100
    ];

    if(values.some(v=>isNaN(v) || v<0 || v>1)){
        alert("Values must be between 0 and 100.");
        return;
    }

    const r=await fetch("/api/annotations",{
        method:"POST",
        headers:{
            "Content-Type":"application/json"
        },
        body:JSON.stringify({
            image_id:id,
            annotations:[{
                class_name:"BUCKET_LOADED",
                x_center:values[0],
                y_center:values[1],
                width:values[2],
                height:values[3]
            }]
        })
    });

    const result=await r.json();

    if(!r.ok){
        alert(result.error || "Annotation failed");
        return;
    }

    alert("Annotation saved.");
    await reloadImages();
}

async function startTraining(){
    if(!confirm(
        "Start YOLO training now? It may take time on Render."
    )){
        return;
    }

    const r=await fetch("/api/train",{
        method:"POST"
    });

    const d=await r.json();

    if(!r.ok){
        alert(d.error || "Could not start training");
        return;
    }

    updateTraining();
}

async function updateTraining(){
    try{
        const r=await fetch("/api/training/status");
        const d=await r.json();

        document.getElementById("trainStatus").innerText =
            d.status || "";

        document.getElementById("trainMessage").innerText =
            d.message || "";

        const p=Number(d.progress || 0);

        document.getElementById("trainBar").style.width =
            p+"%";

        document.getElementById("trainProgress").innerText =
            p+"%";
    }catch(e){}
}

function escapeHtml(s){
    return String(s)
      .replaceAll("&","&amp;")
      .replaceAll("<","&lt;")
      .replaceAll(">","&gt;")
      .replaceAll('"',"&quot;")
      .replaceAll("'","&#039;");
}

renderImages();
updateTraining();

setInterval(updateTraining,3000);
</script>
""" % (
            s["total"],
            s["labeled"],
            s["annotations"],
            esc(train.get("status", "")),
            esc(train.get("message", "")),
            int(train.get("progress") or 0),
            int(train.get("progress") or 0),
            images_json,
        )

        return layout(
            "AI Training",
            body,
            "AI Training",
        )

    except Exception as e:
        return layout(
            "Training Error",
            '<div class="card"><h1>Error</h1><p class="err">%s</p></div>'
            % esc(str(e)),
            "AI Training",
        )


# =========================================================
# BUCKETS PAGE
# =========================================================

def buckets_page():
    conn = db()

    try:
        c = conn.cursor(
            cursor_factory=RealDictCursor
        )

        c.execute("""
            SELECT
                id,
                bucket_code,
                bucket_name,
                location,
                status,
                created_at
            FROM buckets
            ORDER BY id DESC
        """)

        rows = [
            dict(x)
            for x in c.fetchall()
        ]

    finally:
        conn.close()

    table = ""

    for r in rows:
        table += """
<tr>
<td>%s</td>
<td>%s</td>
<td>%s</td>
<td>%s</td>
<td>%s</td>
</tr>
""" % (
            esc(r["id"]),
            esc(r["bucket_code"]),
            esc(r["bucket_name"]),
            esc(r["location"]),
            esc(r["status"]),
        )

    if not table:
        table = """
<tr>
<td colspan="5">No buckets registered.</td>
</tr>
"""

    body = """
<h1>Buckets</h1>

<div class="card">

<h2>Register Bucket Type</h2>

<form method="post" action="/buckets/add">

<label>Bucket Code</label>
<input name="bucket_code"
       placeholder="e.g. BKT-001">

<label>Bucket Name</label>
<input name="bucket_name"
       placeholder="Loaded Ore Bucket"
       required>

<label>Location</label>
<input name="location"
       placeholder="Shaft / Level">

<button type="submit">
Save Bucket
</button>

</form>

</div>

<div class="card">

<h2>Registered Buckets</h2>

<table>
<tr>
<th>ID</th>
<th>Code</th>
<th>Name</th>
<th>Location</th>
<th>Status</th>
</tr>

%s

</table>

</div>
""" % table

    return layout(
        "Buckets",
        body,
        "Buckets",
    )


# =========================================================
# HISTORY PAGE
# =========================================================

def history_page():
    rows = history()
    events = detection_history()

    table = ""

    for r in rows:
        table += """
<tr>
<td>%s</td>
<td><b>%s</b></td>
<td>%s</td>
<td>%s</td>
</tr>
""" % (
            esc(r["count_date"]),
            esc(r["loaded_count"]),
            esc(r["empty_count"]),
            esc(r["notes"] or ""),
        )

    if not table:
        table = """
<tr>
<td colspan="4">No history yet.</td>
</tr>
"""

    event_table = ""

    for r in events:
        event_table += """
<tr>
<td>%s</td>
<td>%s</td>
<td>%s</td>
<td>%s</td>
</tr>
""" % (
            esc(r["detected_at"]),
            esc(r["detected_class"]),
            (
                "%.2f"
                % float(r["confidence"] or 0)
            ),
            "YES" if r["counted"] else "NO",
        )

    if not event_table:
        event_table = """
<tr>
<td colspan="4">No detection events.</td>
</tr>
"""

    body = """
<h1>History</h1>

<div class="card">

<h2>Daily Production</h2>

<table>
<tr>
<th>Date</th>
<th>Loaded Buckets</th>
<th>Empty Buckets</th>
<th>Notes</th>
</tr>

%s

</table>

</div>

<div class="card">

<h2>Recent AI Detection Events</h2>

<table>
<tr>
<th>Time</th>
<th>Class</th>
<th>Confidence</th>
<th>Counted</th>
</tr>

%s

</table>

</div>

<div class="card">
<a href="/api/history.csv">
<button>Download CSV</button>
</a>
</div>
""" % (
        table,
        event_table,
    )

    return layout(
        "History",
        body,
        "History",
    )


# =========================================================
# SETTINGS
# =========================================================

def settings_page():
    model = model_status()

    body = """
<h1>Settings</h1>

<div class="card">
<h2>AI Model</h2>

<p>
Status:
%s
</p>

<p>
Model file:
<b>best.pt</b>
</p>

<p>
Confidence:
<b>%s</b>
</p>

<p>
Image size:
<b>%s</b>
</p>

<p>
Current application class:
<b>BUCKET_LOADED</b>
</p>

<p class="small">
Class 0 is treated as loaded_bucket and is eligible for counting.
</p>

</div>

<div class="card">
<h2>Database</h2>

<p>
The application stores dataset images, annotations,
training state, trained model, daily counts and detection
events in PostgreSQL/Supabase.
</p>
</div>

<div class="card">
<h2>System</h2>

<p>
Host: <b>%s</b>
</p>

<p>
Port: <b>%s</b>
</p>

<p>
Python: <b>%s</b>
</p>
</div>
""" % (
        '<span class="ok">READY</span>'
        if model["ready"]
        else '<span class="err">NOT READY</span>',
        YOLO_CONFIDENCE,
        YOLO_IMAGE_SIZE,
        HOST,
        PORT,
        esc(sys.version.split()[0]),
    )

    return layout(
        "Settings",
        body,
        "Settings",
    )


# =========================================================
# HTTP HANDLER
# =========================================================

class Handler(BaseHTTPRequestHandler):

    def log_message(self, fmt, *args):
        print(
            "%s - %s"
            % (
                self.address_string(),
                fmt % args,
            )
        )

    def do_GET(self):
        try:
            parsed = urlparse(self.path)
            path = parsed.path

            if path == "/":
                send_html(
                    self,
                    dashboard_page(),
                )
                return

            if path == "/camera":
                send_html(
                    self,
                    camera_page(),
                )
                return

            if path == "/training":
                send_html(
                    self,
                    training_page(),
                )
                return

            if path == "/buckets":
                send_html(
                    self,
                    buckets_page(),
                )
                return

            if path == "/history":
                send_html(
                    self,
                    history_page(),
                )
                return

            if path == "/settings":
                send_html(
                    self,
                    settings_page(),
                )
                return

            if path == "/health":
                send_json(
                    self,
                    {
                        "status": "ok",
                        "model": model_status(),
                    },
                )
                return

            if path == "/api/model":
                send_json(
                    self,
                    model_status(),
                )
                return

            if path == "/api/today":
                send_json(
                    self,
                    {
                        "count": get_today_count(),
                    },
                )
                return

            if path == "/api/summary":
                send_json(
                    self,
                    {
                        **summary(),
                        "classes": class_counts(),
                        "today": get_today_count(),
                    },
                )
                return

            if path == "/api/dataset":
                send_json(
                    self,
                    {
                        "images": training_images(),
                        "summary": summary(),
                    },
                )
                return

            if path.startswith("/api/dataset/"):
                image_id = path.rsplit("/", 1)[1]

                try:
                    image_id = int(image_id)
                except ValueError:
                    send_json(
                        self,
                        {"error": "Invalid image ID"},
                        400,
                    )
                    return

                row = get_image(image_id)

                if not row:
                    send_json(
                        self,
                        {"error": "Image not found"},
                        404,
                    )
                    return

                conn = db()

                try:
                    c = conn.cursor(
                        cursor_factory=RealDictCursor
                    )

                    c.execute("""
                        SELECT
                            id,
                            class_name,
                            x_center,
                            y_center,
                            width,
                            height
                        FROM annotations
                        WHERE image_id=%s
                        ORDER BY id
                    """, (image_id,))

                    annotations = [
                        dict(x)
                        for x in c.fetchall()
                    ]

                finally:
                    conn.close()

                send_json(
                    self,
                    {
                        "id": row["id"],
                        "filename": row["filename"],
                        "labeled": row["labeled"],
                        "annotations": annotations,
                    },
                )
                return

            if path.startswith("/api/image/"):
                image_id = path.rsplit("/", 1)[1]

                try:
                    image_id = int(image_id)
                except ValueError:
                    send_json(
                        self,
                        {"error": "Invalid image ID"},
                        400,
                    )
                    return

                row = get_image(image_id)

                if not row:
                    send_json(
                        self,
                        {"error": "Image not found"},
                        404,
                    )
                    return

                raw = bytes(row["image_data"])

                self.send_response(200)
                self.send_header(
                    "Content-Type",
                    "image/jpeg",
                )
                self.send_header(
                    "Content-Length",
                    str(len(raw)),
                )
                self.send_header(
                    "Cache-Control",
                    "no-store",
                )
                self.end_headers()
                self.wfile.write(raw)
                return

            if path == "/api/training/status":
                send_json(
                    self,
                    get_training_state(),
                )
                return

            if path == "/api/history.csv":
                rows = history()

                output = io.StringIO()

                writer = csv.writer(output)

                writer.writerow([
                    "date",
                    "loaded_buckets",
                    "empty_buckets",
                    "notes",
                ])

                for r in rows:
                    writer.writerow([
                        r["count_date"],
                        r["loaded_count"],
                        r["empty_count"],
                        r["notes"] or "",
                    ])

                raw = output.getvalue().encode(
                    "utf-8-sig"
                )

                self.send_response(200)
                self.send_header(
                    "Content-Type",
                    "text/csv; charset=utf-8",
                )
                self.send_header(
                    "Content-Disposition",
                    'attachment; filename="neerika_history.csv"',
                )
                self.send_header(
                    "Content-Length",
                    str(len(raw)),
                )
                self.end_headers()
                self.wfile.write(raw)
                return

            send_json(
                self,
                {"error": "Not found"},
                404,
            )

        except Exception as e:
            traceback.print_exc()

            send_json(
                self,
                {
                    "error": str(e),
                },
                500,
            )

    def do_POST(self):
        try:
            parsed = urlparse(self.path)
            path = parsed.path

            # -------------------------------------------------
            # DETECTION
            # -------------------------------------------------

            if path == "/api/detect":
                data = read_json(self)

                detections = detect_image(
                    data.get("image")
                )

                counted = False

                # Important:
                # This application counts only a loaded bucket.
                # To avoid counting multiple detections in one
                # frame as multiple buckets, the frame contributes
                # one count at most.
                loaded = [
                    d for d in detections
                    if d["class_name"] == "BUCKET_LOADED"
                    and float(d["confidence"]) >= YOLO_CONFIDENCE
                ]

                if loaded:
                    # One camera frame = at most one new bucket.
                    # The frontend requests a frame approximately
                    # every 1.8 seconds, so this is a simple
                    # frame-based counter.
                    now = datetime.now().timestamp()

                    # Server-side global cooldown.
                    # This prevents accidental duplicate counting
                    # from rapid repeated requests.
                    global LAST_COUNT_TIME

                    if now - LAST_COUNT_TIME >= 4.0:
                        add_daily_count(1)
                        LAST_COUNT_TIME = now
                        counted = True

                        save_detection_event(
                            "BUCKET_LOADED",
                            max(
                                float(d["confidence"])
                                for d in loaded
                            ),
                            True,
                        )

                    else:
                        save_detection_event(
                            "BUCKET_LOADED",
                            max(
                                float(d["confidence"])
                                for d in loaded
                            ),
                            False,
                        )

                else:
                    if detections:
                        best = max(
                            detections,
                            key=lambda x: float(
                                x["confidence"]
                            ),
                        )

                        save_detection_event(
                            best["class_name"],
                            float(best["confidence"]),
                            False,
                        )

                send_json(
                    self,
                    {
                        "ok": True,
                        "counted": counted,
                        "today_count": get_today_count(),
                        "detections": detections,
                    },
                )
                return

            # -------------------------------------------------
            # DATASET UPLOAD
            # -------------------------------------------------

            if path == "/api/dataset/upload":
                data = read_json(self)

                filename = (
                    data.get("filename")
                    or "image.jpg"
                )

                image = data.get("image")

                image_id = upload_dataset_image(
                    filename,
                    image,
                )

                send_json(
                    self,
                    {
                        "ok": True,
                        "id": image_id,
                        "message": "Image uploaded successfully.",
                    },
                )
                return

            # -------------------------------------------------
            # SAVE ANNOTATIONS
            # -------------------------------------------------

            if path == "/api/annotations":
                data = read_json(self)

                image_id = int(
                    data.get("image_id")
                )

                annotations = data.get(
                    "annotations",
                    [],
                )

                saved = save_annotations(
                    image_id,
                    annotations,
                )

                send_json(
                    self,
                    {
                        "ok": True,
                        "saved": saved,
                    },
                )
                return

            # -------------------------------------------------
            # DELETE DATASET IMAGE
            # -------------------------------------------------

            if path.startswith(
                "/api/dataset/delete/"
            ):
                image_id = path.rsplit("/", 1)[1]

                image_id = int(image_id)

                ok = delete_dataset_image(
                    image_id
                )

                send_json(
                    self,
                    {
                        "ok": ok,
                    },
                )
                return

            # -------------------------------------------------
            # START TRAINING
            # -------------------------------------------------

            if path == "/api/train":
                started = start_training()

                if not started:
                    send_json(
                        self,
                        {
                            "error":
                                "Training is already running."
                        },
                        409,
                    )
                    return

                send_json(
                    self,
                    {
                        "ok": True,
                        "message":
                            "Training started in background.",
                    },
                )
                return

            # -------------------------------------------------
            # ADD BUCKET
            # -------------------------------------------------

            if path == "/buckets/add":
                length = int(
                    self.headers.get(
                        "Content-Length",
                        "0",
                    )
                    or 0
                )

                raw = self.rfile.read(length)

                from urllib.parse import parse_qs

                form = parse_qs(
                    raw.decode("utf-8")
                )

                bucket_code = (
                    form.get(
                        "bucket_code",
                        [""],
                    )[0]
                )

                bucket_name = (
                    form.get(
                        "bucket_name",
                        [""],
                    )[0]
                )

                location = (
                    form.get(
                        "location",
                        [""],
                    )[0]
                )

                if not bucket_name.strip():
                    raise ValueError(
                        "Bucket name is required."
                    )

                conn = db()

                try:
                    c = conn.cursor()

                    c.execute("""
                        INSERT INTO buckets
                            (
                                bucket_code,
                                bucket_name,
                                location
                            )
                        VALUES
                            (%s,%s,%s)
                    """, (
                        bucket_code,
                        bucket_name,
                        location,
                    ))

                    conn.commit()

                finally:
                    conn.close()

                self.send_response(303)
                self.send_header(
                    "Location",
                    "/buckets",
                )
                self.end_headers()
                return

            send_json(
                self,
                {"error": "Not found"},
                404,
            )

        except Exception as e:
            traceback.print_exc()

            send_json(
                self,
                {
                    "error": str(e),
                },
                500,
            )


# =========================================================
# SERVER
# =========================================================

LAST_COUNT_TIME = 0.0


def main():
    print("")
    print("=================================================")
    print("NEERIKA BUCKET AI")
    print("=================================================")
    print("Host:", HOST)
    print("Port:", PORT)
    print("Database configured:", bool(DATABASE_URL))
    print("Model path:", MODEL_PATH)

    if not DATABASE_URL:
        print(
            "WARNING: DATABASE_URL is missing."
        )

    try:
        init_db()
    except Exception:
        print("DATABASE INITIALIZATION ERROR")
        traceback.print_exc()
        raise

    load_model()

    server = ThreadingHTTPServer(
        (HOST, PORT),
        Handler,
    )

    print(
        "Server running on port",
        PORT,
    )

    server.serve_forever()


if __name__ == "__main__":
    main()
