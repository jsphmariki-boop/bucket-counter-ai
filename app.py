import os
import io
import csv
import json
import base64
import traceback
import tempfile
import threading
import shutil
import time
from datetime import datetime
from urllib.parse import urlparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import psycopg2
from psycopg2.extras import RealDictCursor

from ultralytics import YOLO
from PIL import Image
import numpy as np


# ============================================================
# NEERIKA BUCKET AI
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
    "best.pt"
)

YOLO_CONFIDENCE = float(
    os.environ.get("YOLO_CONFIDENCE", "0.25")
)

YOLO_IMAGE_SIZE = int(
    os.environ.get("YOLO_IMAGE_SIZE", "640")
)

MAX_JSON_BYTES = 12 * 1024 * 1024
MAX_UPLOAD_BYTES = 15 * 1024 * 1024

MODEL = None
MODEL_ERROR = ""

MODEL_LOCK = threading.Lock()
TRAIN_LOCK = threading.Lock()

TRAINING = False
TRAINING_MESSAGE = ""
TRAINING_PROGRESS = 0

LAST_COUNT_TIME = 0.0
COUNT_COOLDOWN_SECONDS = 4.0


# ============================================================
# CLASSES
# ============================================================

CLASSES = {
    "BUCKET_LOADED": "count",
    "BUCKET_EMPTY": "no count",
    "PEOPLE": "no count",
    "EQUIPMENT": "no count",
}

CLASS_IDS = {
    0: "BUCKET_LOADED",
    1: "BUCKET_EMPTY",
    2: "PEOPLE",
    3: "EQUIPMENT",
}


# ============================================================
# BASIC HELPERS
# ============================================================

def esc(value):
    value = "" if value is None else str(value)

    return (
        value
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


def now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def today_str():
    return datetime.now().strftime("%Y-%m-%d")


def json_bytes(obj):
    return json.dumps(
        obj,
        ensure_ascii=False
    ).encode("utf-8")


def send_json(handler, obj, status=200):

    data = json_bytes(obj)

    handler.send_response(status)

    handler.send_header(
        "Content-Type",
        "application/json; charset=utf-8"
    )

    handler.send_header(
        "Content-Length",
        str(len(data))
    )

    handler.send_header(
        "Cache-Control",
        "no-store"
    )

    handler.end_headers()

    handler.wfile.write(data)


def send_html(handler, html, status=200):

    data = html.encode("utf-8")

    handler.send_response(status)

    handler.send_header(
        "Content-Type",
        "text/html; charset=utf-8"
    )

    handler.send_header(
        "Content-Length",
        str(len(data))
    )

    handler.send_header(
        "Cache-Control",
        "no-store"
    )

    handler.end_headers()

    handler.wfile.write(data)


def send_bytes(
    handler,
    data,
    content_type,
    filename=None,
    status=200
):

    handler.send_response(status)

    handler.send_header(
        "Content-Type",
        content_type
    )

    handler.send_header(
        "Content-Length",
        str(len(data))
    )

    if filename:

        handler.send_header(
            "Content-Disposition",
            'attachment; filename="%s"'
            % filename.replace('"', "")
        )

    handler.end_headers()

    handler.wfile.write(data)


def parse_body(handler):

    length = int(
        handler.headers.get(
            "Content-Length",
            "0"
        )
    )

    if length > MAX_JSON_BYTES:
        raise ValueError("Request too large")

    raw = handler.rfile.read(length)

    if not raw:
        return {}

    return json.loads(
        raw.decode("utf-8")
    )


# ============================================================
# DATABASE
# ============================================================

def db():

    if not DATABASE_URL:

        raise RuntimeError(
            "DATABASE_URL is missing. "
            "Add your Supabase/PostgreSQL connection string "
            "to Render Environment Variables."
        )

    return psycopg2.connect(
        DATABASE_URL,
        connect_timeout=15,
        sslmode="require"
    )


def db_init():

    conn = db()

    try:

        cur = conn.cursor()

        cur.execute("""
            CREATE TABLE IF NOT EXISTS buckets (
                id SERIAL PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT DEFAULT '',
                active BOOLEAN DEFAULT FALSE,
                reference_images JSONB DEFAULT '[]'::jsonb,
                created_at TIMESTAMP DEFAULT NOW()
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS dataset_images (
                id SERIAL PRIMARY KEY,
                filename TEXT NOT NULL,
                image_data BYTEA NOT NULL,
                labeled BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMP DEFAULT NOW()
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
                created_at TIMESTAMP DEFAULT NOW()
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS training_state (
                id INTEGER PRIMARY KEY,
                status TEXT DEFAULT 'idle',
                progress INTEGER DEFAULT 0,
                message TEXT DEFAULT '',
                updated_at TIMESTAMP DEFAULT NOW()
            )
        """)

        cur.execute("""
            INSERT INTO training_state
            (id, status, progress, message)
            VALUES
            (1, 'idle', 0, '')
            ON CONFLICT (id) DO NOTHING
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS trained_model (
                id INTEGER PRIMARY KEY,
                model_data BYTEA,
                filename TEXT DEFAULT 'best.pt',
                created_at TIMESTAMP DEFAULT NOW()
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS daily_counts (
                count_date DATE PRIMARY KEY,
                bucket_count INTEGER DEFAULT 0,
                updated_at TIMESTAMP DEFAULT NOW()
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS detection_events (
                id BIGSERIAL PRIMARY KEY,
                detected_class TEXT NOT NULL,
                confidence DOUBLE PRECISION DEFAULT 0,
                counted BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMP DEFAULT NOW()
            )
        """)

        conn.commit()

    finally:

        conn.close()


# ============================================================
# MODEL RESTORE
# ============================================================

def restore_model_from_db():

    if os.path.exists(MODEL_PATH):
        return False

    try:

        conn = db()

        try:

            cur = conn.cursor()

            cur.execute("""
                SELECT model_data
                FROM trained_model
                WHERE id=1
            """)

            row = cur.fetchone()

        finally:

            conn.close()

        if row and row[0]:

            os.makedirs(
                os.path.dirname(MODEL_PATH),
                exist_ok=True
            )

            with open(
                MODEL_PATH,
                "wb"
            ) as f:

                f.write(bytes(row[0]))

            print(
                "Restored trained model from database."
            )

            return True

    except Exception:

        return False

    return False


def save_model_to_db(path):

    with open(path, "rb") as f:

        data = f.read()

    conn = db()

    try:

        cur = conn.cursor()

        cur.execute("""
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
        """, (
            psycopg2.Binary(data),
        ))

        conn.commit()

    finally:

        conn.close()


# ============================================================
# LOAD YOLO
# ============================================================

def load_model():

    global MODEL
    global MODEL_ERROR

    with MODEL_LOCK:

        if MODEL is not None:
            return MODEL

        try:

            restore_model_from_db()

            candidate = None

            if os.path.exists(MODEL_PATH):

                candidate = MODEL_PATH

            elif os.path.exists(
                MODEL_BACKUP_PATH
            ):

                candidate = MODEL_BACKUP_PATH

            if candidate:

                print(
                    "Loading model:",
                    candidate
                )

                MODEL = YOLO(candidate)

                MODEL_ERROR = ""

                return MODEL

            print(
                "best.pt not found."
            )

            print(
                "Loading fallback yolo11n.pt..."
            )

            MODEL = YOLO(
                os.environ.get(
                    "BASE_MODEL",
                    "yolo11n.pt"
                )
            )

            MODEL_ERROR = ""

            return MODEL

        except Exception as e:

            MODEL = None

            MODEL_ERROR = str(e)

            print(
                traceback.format_exc()
            )

            return None


# ============================================================
# IMAGE / AI
# ============================================================

def normalize_class(
    cls_id,
    model_name=""
):

    try:

        cls_id = int(cls_id)

    except Exception:

        return "BUCKET_LOADED"

    if cls_id in CLASS_IDS:

        return CLASS_IDS[cls_id]

    if cls_id == 0:

        return "BUCKET_LOADED"

    return "EQUIPMENT"


def decode_image_data(data):

    if not data:

        raise ValueError(
            "No image data"
        )

    if data.startswith("data:"):

        if "," not in data:

            raise ValueError(
                "Invalid data URL"
            )

        data = data.split(
            ",",
            1
        )[1]

    raw = base64.b64decode(data)

    if len(raw) > MAX_UPLOAD_BYTES:

        raise ValueError(
            "Image is too large"
        )

    return Image.open(
        io.BytesIO(raw)
    ).convert("RGB")


def detect_image(data):

    model = load_model()

    if model is None:

        raise RuntimeError(
            MODEL_ERROR
            or
            "YOLO model is not available"
        )

    image = decode_image_data(data)

    with MODEL_LOCK:

        results = model.predict(
            source=np.array(image),
            conf=YOLO_CONFIDENCE,
            imgsz=YOLO_IMAGE_SIZE,
            verbose=False
        )

    detections = []

    for result in results:

        names = (
            getattr(
                result,
                "names",
                {}
            )
            or
            {}
        )

        if result.boxes is None:
            continue

        for box in result.boxes:

            xyxy = (
                box.xyxy[0]
                .cpu()
                .numpy()
                .tolist()
            )

            conf = float(
                box.conf[0]
                .cpu()
                .item()
            )

            cls_id = int(
                box.cls[0]
                .cpu()
                .item()
            )

            model_class = names.get(
                cls_id,
                str(cls_id)
            )

            normalized = normalize_class(
                cls_id,
                model_class
            )

            detections.append({

                "class": normalized,

                "model_class":
                    model_class,

                "confidence":
                    round(
                        conf,
                        4
                    ),

                "x1":
                    round(
                        float(xyxy[0]),
                        2
                    ),

                "y1":
                    round(
                        float(xyxy[1]),
                        2
                    ),

                "x2":
                    round(
                        float(xyxy[2]),
                        2
                    ),

                "y2":
                    round(
                        float(xyxy[3]),
                        2
                    )
            })

    return detections


# ============================================================
# COUNTING
# ============================================================

def add_daily_count(
    amount=1
):

    conn = db()

    try:

        cur = conn.cursor()

        cur.execute("""
            INSERT INTO daily_counts
            (
                count_date,
                bucket_count,
                updated_at
            )
            VALUES
            (
                %s,
                %s,
                NOW()
            )

            ON CONFLICT (count_date)

            DO UPDATE SET

                bucket_count =
                    daily_counts.bucket_count
                    +
                    EXCLUDED.bucket_count,

                updated_at =
                    NOW()
        """, (
            today_str(),
            amount
        ))

        conn.commit()

    finally:

        conn.close()


def get_today_count():

    conn = db()

    try:

        cur = conn.cursor()

        cur.execute("""
            SELECT
                COALESCE(
                    bucket_count,
                    0
                )

            FROM daily_counts

            WHERE count_date=%s
        """, (
            today_str(),
        ))

        row = cur.fetchone()

        if row:
            return int(row[0])

        return 0

    finally:

        conn.close()


def save_detection_event(
    class_name,
    confidence,
    counted
):

    try:

        conn = db()

        try:

            cur = conn.cursor()

            cur.execute("""
                INSERT INTO detection_events
                (
                    detected_class,
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
            """, (
                class_name,
                confidence,
                counted
            ))

            conn.commit()

        finally:

            conn.close()

    except Exception:

        pass


def process_detection_count(
    detections
):

    global LAST_COUNT_TIME

    loaded = [

        d

        for d in detections

        if d.get("class")
        == "BUCKET_LOADED"

        and float(
            d.get(
                "confidence",
                0
            )
        ) >= YOLO_CONFIDENCE
    ]

    counted = False

    if loaded:

        best = max(
            loaded,
            key=lambda x:
                x["confidence"]
        )

        current = time.time()

        if (
            current
            -
            LAST_COUNT_TIME
            >= COUNT_COOLDOWN_SECONDS
        ):

            add_daily_count(1)

            LAST_COUNT_TIME = current

            counted = True

            save_detection_event(
                "BUCKET_LOADED",
                best["confidence"],
                True
            )

        else:

            save_detection_event(
                "BUCKET_LOADED",
                best["confidence"],
                False
            )

    for d in detections:

        if (
            d["class"]
            !=
            "BUCKET_LOADED"
        ):

            save_detection_event(
                d["class"],
                d["confidence"],
                False
            )

    return counted


# ============================================================
# DATASET
# ============================================================

def training_images():

    conn = db()

    try:

        cur = conn.cursor(
            cursor_factory=RealDictCursor
        )

        cur.execute("""
            SELECT
                d.id,
                d.filename,
                d.labeled,
                d.created_at,
                COUNT(a.id)
                AS annotation_count

            FROM dataset_images d

            LEFT JOIN annotations a
                ON a.image_id=d.id

            GROUP BY
                d.id

            ORDER BY
                d.id DESC
        """)

        return [
            dict(x)
            for x in cur.fetchall()
        ]

    finally:

        conn.close()


def get_dataset_image(
    image_id
):

    conn = db()

    try:

        cur = conn.cursor()

        cur.execute("""
            SELECT
                id,
                filename,
                image_data,
                labeled

            FROM dataset_images

            WHERE id=%s
        """, (
            image_id,
        ))

        return cur.fetchone()

    finally:

        conn.close()


def upload_dataset_image(
    filename,
    raw
):

    if len(raw) > MAX_UPLOAD_BYTES:

        raise ValueError(
            "Image is too large"
        )

    Image.open(
        io.BytesIO(raw)
    ).verify()

    conn = db()

    try:

        cur = conn.cursor()

        cur.execute("""
            INSERT INTO dataset_images
            (
                filename,
                image_data,
                labeled
            )

            VALUES
            (
                %s,
                %s,
                FALSE
            )

            RETURNING id
        """, (
            filename,
            psycopg2.Binary(raw)
        ))

        image_id = cur.fetchone()[0]

        conn.commit()

        return image_id

    finally:

        conn.close()


def save_annotations(
    image_id,
    annotations
):

    conn = db()

    try:

        cur = conn.cursor()

        cur.execute(
            """
            DELETE FROM annotations
            WHERE image_id=%s
            """,
            (
                image_id,
            )
        )

        valid = 0

        for a in annotations:

            class_name = str(
                a.get(
                    "class_name",
                    "BUCKET_LOADED"
                )
            )

            if class_name not in CLASSES:

                class_name = (
                    "BUCKET_LOADED"
                )

            x = max(
                0.0,
                min(
                    1.0,
                    float(
                        a.get(
                            "x_center",
                            0
                        )
                    )
                )
            )

            y = max(
                0.0,
                min(
                    1.0,
                    float(
                        a.get(
                            "y_center",
                            0
                        )
                    )
                )
            )

            w = max(
                0.001,
                min(
                    1.0,
                    float(
                        a.get(
                            "box_width",
                            0.1
                        )
                    )
                )
            )

            h = max(
                0.001,
                min(
                    1.0,
                    float(
                        a.get(
                            "box_height",
                            0.1
                        )
                    )
                )
            )

            cur.execute("""
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
            """, (
                image_id,
                class_name,
                x,
                y,
                w,
                h
            ))

            valid += 1

        cur.execute("""
            UPDATE dataset_images

            SET labeled=%s

            WHERE id=%s
        """, (
            valid > 0,
            image_id
        ))

        conn.commit()

        return valid

    finally:

        conn.close()


def delete_dataset_image(
    image_id
):

    conn = db()

    try:

        cur = conn.cursor()

        cur.execute(
            """
            DELETE FROM dataset_images
            WHERE id=%s
            """,
            (
                image_id,
            )
        )

        conn.commit()

    finally:

        conn.close()


# ============================================================
# TRAINING DATASET
# ============================================================

def build_training_dataset():

    conn = db()

    temp_dir = tempfile.mkdtemp(
        prefix="neerika_dataset_"
    )

    try:

        images_dir = os.path.join(
            temp_dir,
            "images"
        )

        labels_dir = os.path.join(
            temp_dir,
            "labels"
        )

        os.makedirs(
            images_dir,
            exist_ok=True
        )

        os.makedirs(
            labels_dir,
            exist_ok=True
        )

        cur = conn.cursor()

        cur.execute("""
            SELECT
                id,
                filename,
                image_data

            FROM dataset_images

            WHERE labeled=TRUE

            ORDER BY id
        """)

        images = cur.fetchall()

        if len(images) < 2:

            raise ValueError(
                "At least 2 labeled images are required. "
                "For better training use many different bucket images."
            )

        for (
            image_id,
            filename,
            image_data
        ) in images:

            safe = os.path.basename(
                filename
                or
                (
                    "image_%s.jpg"
                    % image_id
                )
            )

            root, ext = os.path.splitext(
                safe
            )

            if not ext:
                ext = ".jpg"

            image_name = (
                "%s_%s%s"
                % (
                    image_id,
                    root,
                    ext
                )
            )

            image_path = os.path.join(
                images_dir,
                image_name
            )

            with open(
                image_path,
                "wb"
            ) as f:

                f.write(
                    bytes(image_data)
                )

            cur.execute("""
                SELECT
                    class_name,
                    x_center,
                    y_center,
                    box_width,
                    box_height

                FROM annotations

                WHERE image_id=%s

                ORDER BY id
            """, (
                image_id,
            ))

            labels = []

            for (
                class_name,
                x,
                y,
                w,
                h
            ) in cur.fetchall():

                if (
                    class_name
                    !=
                    "BUCKET_LOADED"
                ):
                    continue

                labels.append(
                    "0 %.6f %.6f %.6f %.6f"
                    % (
                        float(x),
                        float(y),
                        float(w),
                        float(h)
                    )
                )

            label_path = os.path.join(
                labels_dir,
                os.path.splitext(
                    image_name
                )[0]
                +
                ".txt"
            )

            with open(
                label_path,
                "w",
                encoding="utf-8"
            ) as f:

                f.write(
                    "\n".join(labels)
                )

        yaml_path = os.path.join(
            temp_dir,
            "data.yaml"
        )

        yaml_content = (
            "path: %s\n"
            "train: images\n"
            "val: images\n"
            "names:\n"
            "  0: loaded_bucket\n"
            %
            temp_dir.replace(
                "\\",
                "/"
            )
        )

        with open(
            yaml_path,
            "w",
            encoding="utf-8"
        ) as f:

            f.write(
                yaml_content
            )

        return temp_dir

    except Exception:

        shutil.rmtree(
            temp_dir,
            ignore_errors=True
        )

        raise

    finally:

        conn.close()


# ============================================================
# TRAINING STATE
# ============================================================

def set_training_state(
    status,
    progress,
    message
):

    global TRAINING_MESSAGE
    global TRAINING_PROGRESS

    TRAINING_MESSAGE = message
    TRAINING_PROGRESS = int(
        progress
    )

    try:

        conn = db()

        try:

            cur = conn.cursor()

            cur.execute("""
                UPDATE training_state

                SET
                    status=%s,
                    progress=%s,
                    message=%s,
                    updated_at=NOW()

                WHERE id=1
            """, (
                status,
                int(progress),
                message
            ))

            conn.commit()

        finally:

            conn.close()

    except Exception as e:

        print(
            "Training state error:",
            e
        )


def training_worker():

    global TRAINING
    global MODEL
    global MODEL_ERROR

    with TRAIN_LOCK:

        TRAINING = True

        try:

            set_training_state(
                "preparing",
                5,
                "Preparing training dataset..."
            )

            dataset_dir = (
                build_training_dataset()
            )

            try:

                set_training_state(
                    "training",
                    10,
                    "YOLO training has started..."
                )

                base_model = os.environ.get(
                    "TRAIN_BASE_MODEL",
                    "yolo11n.pt"
                )

                model = YOLO(
                    base_model
                )

                try:

                    def on_epoch_end(
                        trainer
                    ):

                        try:

                            epoch = (
                                int(
                                    trainer.epoch
                                )
                                +
                                1
                            )

                            total = int(
                                getattr(
                                    trainer,
                                    "epochs",
                                    20
                                )
                            )

                            progress = (
                                10
                                +
                                int(
                                    (
                                        epoch
                                        /
                                        max(
                                            total,
                                            1
                                        )
                                    )
                                    *
                                    75
                                )
                            )

                            set_training_state(
                                "training",
                                min(
                                    progress,
                                    85
                                ),
                                "Training epoch %d/%d..."
                                % (
                                    epoch,
                                    total
                                )
                            )

                        except Exception:

                            pass

                    model.add_callback(
                        "on_train_epoch_end",
                        on_epoch_end
                    )

                except Exception:

                    pass

                results = model.train(
                    data=os.path.join(
                        dataset_dir,
                        "data.yaml"
                    ),

                    epochs=int(
                        os.environ.get(
                            "TRAIN_EPOCHS",
                            "20"
                        )
                    ),

                    imgsz=YOLO_IMAGE_SIZE,

                    project=os.path.join(
                        dataset_dir,
                        "runs"
                    ),

                    name="neerika_bucket",

                    exist_ok=True,

                    verbose=False
                )

                set_training_state(
                    "saving",
                    90,
                    "Saving trained model..."
                )

                save_dir = getattr(
                    results,
                    "save_dir",
                    None
                )

                candidate = None

                if save_dir:

                    possible = os.path.join(
                        save_dir,
                        "weights",
                        "best.pt"
                    )

                    if os.path.exists(
                        possible
                    ):

                        candidate = possible

                if not candidate:

                    for root, dirs, files in os.walk(
                        dataset_dir
                    ):

                        if "best.pt" in files:

                            candidate = os.path.join(
                                root,
                                "best.pt"
                            )

                            break

                if not candidate:

                    raise RuntimeError(
                        "Training finished but best.pt was not found."
                    )

                os.makedirs(
                    os.path.dirname(
                        MODEL_PATH
                    ),
                    exist_ok=True
                )

                shutil.copy2(
                    candidate,
                    MODEL_PATH
                )

                save_model_to_db(
                    MODEL_PATH
                )

                with MODEL_LOCK:

                    MODEL = YOLO(
                        MODEL_PATH
                    )

                    MODEL_ERROR = ""

                set_training_state(
                    "completed",
                    100,
                    "Training completed successfully. Model saved."
                )

            finally:

                shutil.rmtree(
                    dataset_dir,
                    ignore_errors=True
                )

        except Exception as e:

            print(
                "TRAINING ERROR:"
            )

            print(
                traceback.format_exc()
            )

            set_training_state(
                "failed",
                0,
                str(e)
            )

        finally:

            TRAINING = False


def start_training():

    if TRAINING:

        return (
            False,
            "Training is already running."
        )

    thread = threading.Thread(
        target=training_worker,
        daemon=True
    )

    thread.start()

    return (
        True,
        "Training started."
    )


def get_training_state():

    try:

        conn = db()

        try:

            cur = conn.cursor(
                cursor_factory=RealDictCursor
            )

            cur.execute("""
                SELECT
                    status,
                    progress,
                    message,
                    updated_at

                FROM training_state

                WHERE id=1
            """)

            row = cur.fetchone()

            if row:

                return dict(row)

        finally:

            conn.close()

    except Exception:

        pass

    return {
        "status": "idle",
        "progress": 0,
        "message": ""
    }


# ============================================================
# BUCKET REGISTRATION
# ============================================================

def get_buckets():

    conn = db()

    try:

        cur = conn.cursor(
            cursor_factory=RealDictCursor
        )

        cur.execute("""
            SELECT
                id,
                name,
                description,
                active,
                reference_images,
                created_at

            FROM buckets

            ORDER BY id DESC
        """)

        return [
            dict(x)
            for x in cur.fetchall()
        ]

    finally:

        conn.close()


def add_bucket(
    name,
    description,
    active,
    reference_images
):

    conn = db()

    try:

        cur = conn.cursor()

        if active:

            cur.execute(
                """
                UPDATE buckets
                SET active=FALSE
                """
            )

        cur.execute("""
            INSERT INTO buckets
            (
                name,
                description,
                active,
                reference_images
            )

            VALUES
            (
                %s,
                %s,
                %s,
                %s::jsonb
            )
        """, (
            name,
            description,
            bool(active),
            json.dumps(
                reference_images
                or []
            )
        ))

        conn.commit()

    finally:

        conn.close()


def set_active_bucket(
    bucket_id
):

    conn = db()

    try:

        cur = conn.cursor()

        cur.execute(
            """
            UPDATE buckets
            SET active=FALSE
            """
        )

        cur.execute(
            """
            UPDATE buckets
            SET active=TRUE
            WHERE id=%s
            """,
            (
                bucket_id,
            )
        )

        conn.commit()

    finally:

        conn.close()


# ============================================================
# HTML LAYOUT
# ============================================================

def layout(
    title,
    body,
    active="Dashboard"
):

    nav_items = [
        (
            "Dashboard",
            "/"
        ),
        (
            "Camera",
            "/camera"
        ),
        (
            "Buckets",
            "/buckets"
        ),
        (
            "Training",
            "/training"
        ),
        (
            "History",
            "/history"
        ),
        (
            "Settings",
            "/settings"
        )
    ]

    nav = ""

    for name, href in nav_items:

        cls = (
            "nav-active"
            if name == active
            else ""
        )

        nav += (
            '<a class="nav-item %s" '
            'href="%s">%s</a>'
            %
            (
                cls,
                href,
                esc(name)
            )
        )

    html = """<!doctype html>

<html>

<head>

<meta charset="utf-8">

<meta name="viewport"
      content="width=device-width, initial-scale=1">

<title>__TITLE__ - NEERIKA BUCKET AI</title>

<style>

*{
    box-sizing:border-box;
}

body{
    margin:0;
    font-family:Arial,Helvetica,sans-serif;
    background:#f4f6f8;
    color:#17202a;
}

.topbar{
    background:#111827;
    color:white;
    padding:15px 20px;
    display:flex;
    align-items:center;
    justify-content:space-between;
    gap:15px;
    position:sticky;
    top:0;
    z-index:10;
}

.brand{
    font-size:20px;
    font-weight:800;
}

.brand small{
    display:block;
    font-size:11px;
    opacity:.7;
    font-weight:400;
}

.nav{
    display:flex;
    gap:6px;
    flex-wrap:wrap;
}

.nav-item{
    color:#dbeafe;
    text-decoration:none;
    padding:8px 10px;
    border-radius:7px;
    font-size:13px;
}

.nav-item:hover,
.nav-active{
    background:#2563eb;
    color:white;
}

.container{
    max-width:1200px;
    margin:0 auto;
    padding:20px;
}

.card{
    background:white;
    border-radius:12px;
    padding:18px;
    margin-bottom:18px;
    box-shadow:0 2px 10px rgba(0,0,0,.07);
}

.grid{
    display:grid;
    grid-template-columns:
        repeat(auto-fit,minmax(220px,1fr));
    gap:15px;
}

.stat{
    background:white;
    border-radius:12px;
    padding:20px;
    box-shadow:0 2px 10px rgba(0,0,0,.07);
}

.stat-label{
    color:#64748b;
    font-size:13px;
}

.stat-value{
    font-size:35px;
    font-weight:800;
    margin-top:7px;
}

button,
input,
textarea,
select{
    font:inherit;
}

button{
    border:0;
    border-radius:8px;
    padding:10px 14px;
    cursor:pointer;
    background:#2563eb;
    color:white;
    font-weight:700;
}

button.secondary{
    background:#64748b;
}

button.danger{
    background:#dc2626;
}

button.success{
    background:#15803d;
}

button:disabled{
    opacity:.5;
    cursor:not-allowed;
}

input,
textarea,
select{
    width:100%;
    border:1px solid #cbd5e1;
    border-radius:8px;
    padding:10px;
    background:white;
}

label{
    display:block;
    font-size:13px;
    font-weight:700;
    margin:10px 0 6px;
}

video{
    width:100%;
    max-height:65vh;
    background:#000;
    border-radius:12px;
}

.table-wrap{
    overflow-x:auto;
}

table{
    width:100%;
    border-collapse:collapse;
}

th,
td{
    padding:10px;
    border-bottom:1px solid #e5e7eb;
    text-align:left;
    font-size:13px;
}

.badge{
    display:inline-block;
    padding:5px 8px;
    border-radius:999px;
    background:#e5e7eb;
    font-size:11px;
    font-weight:700;
}

.badge.green{
    background:#dcfce7;
    color:#166534;
}

.progress{
    width:100%;
    height:22px;
    background:#e5e7eb;
    border-radius:999px;
    overflow:hidden;
}

.progress > div{
    height:100%;
    width:0%;
    background:#2563eb;
    transition:width .3s;
}

.msg{
    padding:10px;
    border-radius:8px;
    margin:10px 0;
    background:#eff6ff;
    color:#1e40af;
}

.error{
    background:#fef2f2;
    color:#991b1b;
}

.success-msg{
    background:#f0fdf4;
    color:#166534;
}

.small{
    font-size:12px;
    color:#64748b;
}

.footer{
    text-align:center;
    color:#64748b;
    font-size:12px;
    padding:25px;
}

.row{
    display:flex;
    gap:10px;
    align-items:center;
    flex-wrap:wrap;
}

.camera-wrap{
    position:relative;
}

.overlay{
    position:absolute;
    left:10px;
    top:10px;
    background:rgba(0,0,0,.7);
    color:white;
    padding:10px;
    border-radius:8px;
    font-size:13px;
}

.dataset-grid{
    display:grid;
    grid-template-columns:
        repeat(auto-fill,minmax(210px,1fr));
    gap:15px;
}

.dataset-item{
    border:1px solid #e5e7eb;
    border-radius:10px;
    padding:10px;
    background:#fff;
}

@media(max-width:700px){

    .topbar{
        align-items:flex-start;
        flex-direction:column;
    }

    .container{
        padding:12px;
    }

    .nav{
        width:100%;
    }

}

</style>

</head>

<body>

<div class="topbar">

<div class="brand">

NEERIKA BUCKET AI

<small>
Mining Production Bucket Counter
</small>

</div>

<div class="nav">

__NAV__

</div>

</div>

<div class="container">

__BODY__

</div>

<div class="footer">

Geology &amp; Mining Services

</div>

</body>

</html>
"""

    return (
        html
        .replace(
            "__TITLE__",
            esc(title)
        )
        .replace(
            "__NAV__",
            nav
        )
        .replace(
            "__BODY__",
            body
        )
    )


# ============================================================
# DASHBOARD
# ============================================================

def dashboard_page():

    count = get_today_count()

    state = get_training_state()

    model_ok = (
        load_model()
        is not None
    )

    body = """
<h1>Dashboard</h1>

<div class="grid">

<div class="stat">

<div class="stat-label">
Today's loaded buckets
</div>

<div class="stat-value">
%s
</div>

</div>

<div class="stat">

<div class="stat-label">
AI model
</div>

<div class="stat-value"
     style="font-size:22px">

%s

</div>

</div>

<div class="stat">

<div class="stat-label">
Training status
</div>

<div class="stat-value"
     style="font-size:22px">

%s

</div>

</div>

</div>

<div class="card">

<h2>NEERIKA BUCKET AI</h2>

<p>
The system identifies loaded ore/material buckets.
Empty buckets, people and equipment should not be counted.
</p>

<div class="row">

<a href="/camera">
<button>
Open Camera
</button>
</a>

<a href="/training">
<button class="secondary">
AI Training
</button>
</a>

<a href="/history">
<button class="secondary">
View History
</button>
</a>

</div>

</div>

<div class="card">

<h3>System</h3>

<p class="small">
Database:
%s
</p>

<p class="small">
Model:
%s
</p>

</div>

""" % (

        count,

        (
            "READY"
            if model_ok
            else "ERROR"
        ),

        esc(
            state.get(
                "status",
                "idle"
            )
        ),

        (
            "Connected"
            if DATABASE_URL
            else "NOT CONFIGURED"
        ),

        (
            "Available"
            if os.path.exists(
                MODEL_PATH
            )
            else
            "Fallback / DB restore"
        )
    )

    if MODEL_ERROR:

        body += """
<div class="card">

<div class="msg error">

%s

</div>

</div>
""" % esc(
            MODEL_ERROR
        )

    return layout(
        "Dashboard",
        body,
        "Dashboard"
    )


# ============================================================
# CAMERA
# ============================================================

def camera_page():

    body = r"""
<h1>Camera</h1>

<div class="card">

<div class="row">

<button id="startBtn"
        onclick="startCamera()">

Start Camera

</button>

<button class="secondary"
        onclick="stopCamera()">

Stop

</button>

</div>

<p id="status"
   class="msg">

Camera is stopped.

</p>

<div class="camera-wrap">

<video id="video"
       autoplay
       playsinline
       muted>
</video>

<div class="overlay">

Count:
<span id="count">
0
</span>

</div>

</div>

</div>

<div class="card">

<h3>
Last AI result
</h3>

<div id="result">
No detection yet.
</div>

</div>

<script>

let stream = null;

let timer = null;

let busy = false;


async function startCamera(){

    try{

        stream =
            await navigator.mediaDevices
            .getUserMedia({

                video:{
                    facingMode:"environment",
                    width:{ideal:1280},
                    height:{ideal:720}
                },

                audio:false
            });

        document.getElementById(
            "video"
        ).srcObject = stream;

        document.getElementById(
            "status"
        ).textContent =
            "Camera running. Point it at loaded buckets.";

        if(!timer){

            timer =
                setInterval(
                    captureFrame,
                    1800
                );
        }

    }catch(e){

        document.getElementById(
            "status"
        ).textContent =
            "Camera error: "
            +
            e.message;

        document.getElementById(
            "status"
        ).className =
            "msg error";
    }
}


function stopCamera(){

    if(timer){

        clearInterval(
            timer
        );

        timer = null;
    }

    if(stream){

        stream
            .getTracks()
            .forEach(
                t => t.stop()
            );

        stream = null;
    }

    document.getElementById(
        "video"
    ).srcObject = null;

    document.getElementById(
        "status"
    ).textContent =
        "Camera is stopped.";
}


async function captureFrame(){

    if(
        busy
        ||
        !stream
    ){
        return;
    }

    const video =
        document.getElementById(
            "video"
        );

    if(
        video.videoWidth < 10
        ||
        video.videoHeight < 10
    ){
        return;
    }

    busy = true;

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

    try{

        const image =
            canvas.toDataURL(
                "image/jpeg",
                0.85
            );

        const response =
            await fetch(
                "/api/detect",
                {
                    method:"POST",

                    headers:{
                        "Content-Type":
                            "application/json"
                    },

                    body:
                        JSON.stringify({
                            image:image
                        })
                }
            );

        const data =
            await response.json();

        if(!response.ok){

            throw new Error(
                data.error
                ||
                "Detection failed"
            );
        }

        document.getElementById(
            "count"
        ).textContent =
            data.total_count;

        let html = "";

        if(
            data.detections.length
            ===
            0
        ){

            html =
                "Nothing detected.";

        }else{

            data.detections
                .forEach(
                    d => {

                        html +=
                            "<div><b>"
                            +
                            d.class
                            +
                            "</b> - confidence "
                            +
                            Math.round(
                                d.confidence
                                *
                                100
                            )
                            +
                            "%</div>";
                    }
                );
        }

        if(data.counted){

            html +=
                "<div class='msg success-msg'>"
                +
                "Loaded bucket counted."
                +
                "</div>";
        }

        document.getElementById(
            "result"
        ).innerHTML =
            html;

    }catch(e){

        document.getElementById(
            "result"
        ).innerHTML =
            "<div class='msg error'>"
            +
            e.message
            +
            "</div>";

    }finally{

        busy = false;
    }
}

</script>
"""

    return layout(
        "Camera",
        body,
        "Camera"
    )


# ============================================================
# BUCKETS
# ============================================================

def buckets_page():

    buckets = get_buckets()

    rows = ""

    for b in buckets:

        refs = (
            b.get(
                "reference_images"
            )
            or []
        )

        if isinstance(
            refs,
            str
        ):

            try:

                refs = json.loads(
                    refs
                )

            except Exception:

                refs = []

        status = (
            '<span class="badge green">'
            'ACTIVE'
            '</span>'
            if b.get("active")
            else
            '<span class="badge">'
            'INACTIVE'
            '</span>'
        )

        button_text = (
            "Active"
            if b.get("active")
            else
            "Set Active"
        )

        rows += """
<div class="dataset-item">

<h3>
%s
</h3>

<p class="small">
%s
</p>

<p>
%s
</p>

<p class="small">
Reference photos:
%d
</p>

<button onclick="activateBucket(%d)">
%s
</button>

</div>
""" % (

            esc(
                b.get(
                    "name"
                )
            ),

            esc(
                b.get(
                    "description"
                )
            ),

            status,

            len(refs),

            int(
                b.get(
                    "id"
                )
            ),

            button_text
        )

    if not rows:

        rows = """
<div class="msg">

No bucket type has been registered yet.

</div>
"""

    body = """
<h1>Bucket Registration</h1>

<div class="card">

<h3>
Add bucket type
</h3>

<label>
Bucket name
</label>

<input
id="name"
placeholder="Example: Neerika Kamnara Bucket">

<label>
Description
</label>

<textarea
id="description"
placeholder="Describe the bucket type">
</textarea>

<label>
Reference photo URLs
</label>

<textarea
id="refs"
placeholder="One image URL per line">
</textarea>

<label>

<input
id="active"
type="checkbox"
style="width:auto">

Make this active bucket type

</label>

<button onclick="addBucket()">
Save Bucket Type
</button>

<div id="msg"></div>

</div>

<div class="card">

<h3>
Registered bucket types
</h3>

<div class="dataset-grid">

%s

</div>

</div>

<script>

async function addBucket(){

    const name =
        document
        .getElementById(
            "name"
        )
        .value
        .trim();

    if(!name){

        alert(
            "Enter bucket name"
        );

        return;
    }

    const refs =
        document
        .getElementById(
            "refs"
        )
        .value
        .split("\\n")
        .map(
            x => x.trim()
        )
        .filter(Boolean);

    const response =
        await fetch(
            "/api/buckets",
            {
                method:"POST",

                headers:{
                    "Content-Type":
                        "application/json"
                },

                body:
                    JSON.stringify({

                        name:name,

                        description:
                            document
                            .getElementById(
                                "description"
                            )
                            .value,

                        active:
                            document
                            .getElementById(
                                "active"
                            )
                            .checked,

                        reference_images:
                            refs
                    })
            }
        );

    const data =
        await response.json();

    if(!response.ok){

        alert(
            data.error
            ||
            "Save failed"
        );

        return;
    }

    location.reload();
}


async function activateBucket(id){

    const response =
        await fetch(
            "/api/buckets/active",
            {
                method:"POST",

                headers:{
                    "Content-Type":
                        "application/json"
                },

                body:
                    JSON.stringify({
                        id:id
                    })
            }
        );

    const data =
        await response.json();

    if(!response.ok){

        alert(
            data.error
            ||
            "Could not activate bucket"
        );

        return;
    }

    location.reload();
}

</script>

""" % rows

    return layout(
        "Buckets",
        body,
        "Buckets"
    )


# ============================================================
# TRAINING PAGE
# ============================================================

def training_page():

    images = training_images()

    state = get_training_state()

    items = ""

    for item in images:

        items += """
<div class="dataset-item">

<b>
%s
</b>

<p class="small">

ID:
%s

<br>

Annotations:
%s

</p>

<div class="row">

<a href="/api/dataset/image/%s">

<button class="secondary">
View
</button>

</a>

<button
class="danger"
onclick="deleteImage(%s)">

Delete

</button>

</div>

</div>
""" % (

            esc(
                item.get(
                    "filename"
                )
            ),

            esc(
                item.get(
                    "id"
                )
            ),

            esc(
                item.get(
                    "annotation_count"
                )
            ),

            esc(
                item.get(
                    "id"
                )
            ),

            esc(
                item.get(
                    "id"
                )
            )
        )

    if not items:

        items = """
<div class="msg">

No training images uploaded yet.

</div>
"""

    progress = int(
        state.get(
            "progress",
            0
        )
        or
        0
    )

    status_text = (
        "%s - %s"
        %
        (
            state.get(
                "status",
                "idle"
            ),

            state.get(
                "message",
                ""
            )
        )
    )

    body = """
<h1>
AI Training
</h1>

<div class="card">

<h3>
Upload training image
</h3>

<input
id="imageFile"
type="file"
accept="image/*">

<br>
<br>

<button onclick="uploadImage()">

Upload Image

</button>

<div id="uploadMsg"></div>

</div>


<div class="card">

<h3>
Training
</h3>

<p>

Upload images and annotate them
before starting YOLO training.

The model uses one class:

<b>
loaded_bucket
</b>

</p>

<div class="progress">

<div
id="trainBar"
style="width:%s%%">
</div>

</div>

<p id="trainStatus">
%s
</p>

<button
id="trainBtn"
onclick="startTraining()">

Start Training

</button>

</div>


<div class="card">

<h3>
Dataset
</h3>

<div class="dataset-grid">

%s

</div>

</div>


<script>

async function uploadImage(){

    const file =
        document
        .getElementById(
            "imageFile"
        )
        .files[0];

    if(!file){

        alert(
            "Choose an image first."
        );

        return;
    }

    if(
        file.size
        >
        15 * 1024 * 1024
    ){

        alert(
            "Image is too large. Maximum is 15 MB."
        );

        return;
    }

    const reader =
        new FileReader();

    reader.onload =
        async function(){

            const response =
                await fetch(
                    "/api/dataset/upload",
                    {
                        method:"POST",

                        headers:{
                            "Content-Type":
                                "application/json"
                        },

                        body:
                            JSON.stringify({

                                filename:
                                    file.name,

                                image:
                                    reader.result
                            })
                    }
                );

            const data =
                await response.json();

            if(!response.ok){

                document
                .getElementById(
                    "uploadMsg"
                )
                .innerHTML =
                    "<div class='msg error'>"
                    +
                    (
                        data.error
                        ||
                        "Upload failed"
                    )
                    +
                    "</div>";

                return;
            }

            document
            .getElementById(
                "uploadMsg"
            )
            .innerHTML =
                "<div class='msg success-msg'>"
                +
                "Image uploaded."
                +
                "</div>";

            setTimeout(
                () => location.reload(),
                700
            );
        };

    reader.readAsDataURL(
        file
    );
}


async function deleteImage(id){

    if(
        !confirm(
            "Delete this image?"
        )
    ){

        return;
    }

    const response =
        await fetch(
            "/api/dataset/delete",
            {
                method:"POST",

                headers:{
                    "Content-Type":
                        "application/json"
                },

                body:
                    JSON.stringify({
                        id:id
                    })
            }
        );

    const data =
        await response.json();

    if(!response.ok){

        alert(
            data.error
            ||
            "Delete failed"
        );

        return;
    }

    location.reload();
}


async function startTraining(){

    const button =
        document.getElementById(
            "trainBtn"
        );

    button.disabled = true;

    const response =
        await fetch(
            "/api/training/start",
            {
                method:"POST"
            }
        );

    const data =
        await response.json();

    if(!response.ok){

        alert(
            data.error
            ||
            "Training could not start"
        );

        button.disabled = false;

        return;
    }

    pollTraining();
}


async function pollTraining(){

    const response =
        await fetch(
            "/api/training/status"
        );

    const data =
        await response.json();

    const p =
        Number(
            data.progress
            ||
            0
        );

    document
    .getElementById(
        "trainBar"
    )
    .style
    .width =
        p
        +
        "%";

    document
    .getElementById(
        "trainStatus"
    )
    .textContent =
        data.status
        +
        " - "
        +
        data.message;

    if(
        data.status
        ===
        "training"
        ||
        data.status
        ===
        "preparing"
        ||
        data.status
        ===
        "saving"
    ){

        setTimeout(
            pollTraining,
            2000
        );

    }else{

        document
        .getElementById(
            "trainBtn"
        )
        .disabled = false;
    }
}


pollTraining();

</script>

""" % (

        progress,

        esc(
            status_text
        ),

        items
    )

    return layout(
        "Training",
        body,
        "Training"
    )


# ============================================================
# HISTORY
# ============================================================

def get_history():

    conn = db()

    try:

        cur = conn.cursor(
            cursor_factory=RealDictCursor
        )

        cur.execute("""
            SELECT
                count_date::text AS date,
                bucket_count

            FROM daily_counts

            ORDER BY
                count_date DESC

            LIMIT 100
        """)

        return [
            dict(x)
            for x in cur.fetchall()
        ]

    finally:

        conn.close()


def history_page():

    history = get_history()

    rows = ""

    for item in history:

        rows += """
<tr>

<td>
%s
</td>

<td>
<b>
%s
</b>
</td>

</tr>
""" % (

            esc(
                item.get(
                    "date"
                )
            ),

            esc(
                item.get(
                    "bucket_count"
                )
            )
        )

    if not rows:

        rows = """
<tr>

<td colspan="2">

No count history yet.

</td>

</tr>
"""

    body = """
<h1>
History
</h1>

<div class="card">

<div class="row">

<button onclick="location.reload()">
Refresh
</button>

<a href="/api/history.csv">

<button class="secondary">
Export CSV
</button>

</a>

</div>

</div>


<div class="card table-wrap">

<table>

<thead>

<tr>

<th>
Date
</th>

<th>
Loaded Buckets
</th>

</tr>

</thead>

<tbody>

%s

</tbody>

</table>

</div>

""" % rows

    return layout(
        "History",
        body,
        "History"
    )


# ============================================================
# SETTINGS
# ============================================================

def settings_page():

    body = """
<h1>
Settings
</h1>

<div class="card">

<h3>
AI Configuration
</h3>

<p>
Confidence:
<b>
%s
</b>
</p>

<p>
Image size:
<b>
%s
</b>
</p>

<p>
Count cooldown:
<b>
%s seconds
</b>
</p>

</div>


<div class="card">

<h3>
System
</h3>

<p class="small">

Database:
%s

</p>

<p class="small">

Model file:
%s

</p>

<p class="small">

Server time:
%s

</p>

</div>

""" % (

        YOLO_CONFIDENCE,

        YOLO_IMAGE_SIZE,

        COUNT_COOLDOWN_SECONDS,

        (
            "Connected"
            if DATABASE_URL
            else
            "NOT CONFIGURED"
        ),

        (
            "Available"
            if os.path.exists(
                MODEL_PATH
            )
            else
            "Fallback / DB restore"
        ),

        now_str()
    )

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


    def do_GET(self):

        try:

            path = urlparse(
                self.path
            ).path


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


            if path == "/health":

                send_json(
                    self,
                    {
                        "status":"ok",
                        "time":
                            now_str(),
                        "database_configured":
                            bool(
                                DATABASE_URL
                            ),
                        "model_file":
                            os.path.exists(
                                MODEL_PATH
                            )
                    }
                )

                return


            if path == "/api/training/status":

                send_json(
                    self,
                    get_training_state()
                )

                return


            if path == "/api/history":

                send_json(
                    self,
                    {
                        "history":
                            get_history()
                    }
                )

                return


            if path == "/api/history.csv":

                history = get_history()

                output = io.StringIO()

                writer = csv.writer(
                    output
                )

                writer.writerow(
                    [
                        "date",
                        "loaded_buckets"
                    ]
                )

                for item in history:

                    writer.writerow(
                        [
                            item.get(
                                "date"
                            ),

                            item.get(
                                "bucket_count"
                            )
                        ]
                    )

                send_bytes(
                    self,
                    output
                    .getvalue()
                    .encode("utf-8"),

                    "text/csv; charset=utf-8",

                    "neerika_bucket_history.csv"
                )

                return


            if path.startswith(
                "/api/dataset/image/"
            ):

                image_id = int(
                    path.rsplit(
                        "/",
                        1
                    )[1]
                )

                row = get_dataset_image(
                    image_id
                )

                if not row:

                    send_json(
                        self,
                        {
                            "error":
                                "Image not found"
                        },
                        404
                    )

                    return

                send_bytes(
                    self,
                    bytes(row[2]),
                    "image/jpeg"
                )

                return


            send_json(
                self,
                {
                    "error":
                        "Not found"
                },
                404
            )

        except Exception as e:

            print(
                traceback.format_exc()
            )

            send_json(
                self,
                {
                    "error":
                        str(e)
                },
                500
            )


    def do_POST(self):

        try:

            path = urlparse(
                self.path
            ).path

            body = parse_body(
                self
            )


            # --------------------------------------------
            # AI DETECTION
            # --------------------------------------------

            if path == "/api/detect":

                image = body.get(
                    "image"
                )

                if not image:

                    send_json(
                        self,
                        {
                            "error":
                                "No image supplied"
                        },
                        400
                    )

                    return

                detections = detect_image(
                    image
                )

                counted = (
                    process_detection_count(
                        detections
                    )
                )

                total = get_today_count()

                send_json(
                    self,
                    {
                        "detections":
                            detections,

                        "counted":
                            counted,

                        "total_count":
                            total
                    }
                )

                return


            # --------------------------------------------
            # DATASET UPLOAD
            # --------------------------------------------

            if path == "/api/dataset/upload":

                filename = (
                    body.get(
                        "filename"
                    )
                    or
                    "image.jpg"
                )

                image = body.get(
                    "image"
                )

                if not image:

                    send_json(
                        self,
                        {
                            "error":
                                "No image supplied"
                        },
                        400
                    )

                    return

                if image.startswith(
                    "data:"
                ):

                    image = image.split(
                        ",",
                        1
                    )[1]

                raw = base64.b64decode(
                    image
                )

                image_id = (
                    upload_dataset_image(
                        filename,
                        raw
                    )
                )

                send_json(
                    self,
                    {
                        "ok":True,
                        "id":
                            image_id
                    }
                )

                return


            # --------------------------------------------
            # DATASET DELETE
            # --------------------------------------------

            if path == "/api/dataset/delete":

                image_id = int(
                    body.get(
                        "id"
                    )
                )

                delete_dataset_image(
                    image_id
                )

                send_json(
                    self,
                    {
                        "ok":True
                    }
                )

                return


            # --------------------------------------------
            # ANNOTATIONS
            # --------------------------------------------

            if path == "/api/dataset/annotations":

                image_id = int(
                    body.get(
                        "image_id"
                    )
                )

                annotations = (
                    body.get(
                        "annotations"
                    )
                    or
                    []
                )

                saved = save_annotations(
                    image_id,
                    annotations
                )

                send_json(
                    self,
                    {
                        "ok":True,
                        "annotations_saved":
                            saved
                    }
                )

                return


            # --------------------------------------------
            # TRAINING
            # --------------------------------------------

            if path == "/api/training/start":

                ok, message = (
                    start_training()
                )

                if not ok:

                    send_json(
                        self,
                        {
                            "error":
                                message
                        },
                        409
                    )

                    return

                send_json(
                    self,
                    {
                        "ok":True,
                        "message":
                            message
                    }
                )

                return


            # --------------------------------------------
            # BUCKET REGISTRATION
            # --------------------------------------------

            if path == "/api/buckets":

                name = str(
                    body.get(
                        "name",
                        ""
                    )
                ).strip()

                if not name:

                    send_json(
                        self,
                        {
                            "error":
                                "Bucket name is required"
                        },
                        400
                    )

                    return

                add_bucket(

                    name,

                    str(
                        body.get(
                            "description",
                            ""
                        )
                    ),

                    bool(
                        body.get(
                            "active",
                            False
                        )
                    ),

                    body.get(
                        "reference_images"
                    )
                    or
                    []
                )

                send_json(
                    self,
                    {
                        "ok":True
                    }
                )

                return


            # --------------------------------------------
            # ACTIVE BUCKET
            # --------------------------------------------

            if path == "/api/buckets/active":

                bucket_id = int(
                    body.get(
                        "id"
                    )
                )

                set_active_bucket(
                    bucket_id
                )

                send_json(
                    self,
                    {
                        "ok":True
                    }
                )

                return


            send_json(
                self,
                {
                    "error":
                        "Not found"
                },
                404
            )

        except Exception as e:

            print(
                traceback.format_exc()
            )

            send_json(
                self,
                {
                    "error":
                        str(e)
                },
                500
            )


# ============================================================
# START SERVER
# ============================================================

def startup():

    print(
        "=" * 60
    )

    print(
        "NEERIKA BUCKET AI"
    )

    print(
        "=" * 60
    )

    print(
        "PORT:",
        PORT
    )

    print(
        "DATABASE CONFIGURED:",
        bool(DATABASE_URL)
    )

    if not DATABASE_URL:

        print(
            "WARNING: DATABASE_URL is not configured."
        )

    try:

        db_init()

        print(
            "Database initialization: OK"
        )

    except Exception:

        print(
            "Database initialization failed:"
        )

        print(
            traceback.format_exc()
        )

    try:

        restore_model_from_db()

    except Exception:

        pass

    print(
        "Starting HTTP server..."
    )


if __name__ == "__main__":

    startup()

    server = ThreadingHTTPServer(
        (
            HOST,
            PORT
        ),
        Handler
    )

    print(
        "Server running on %s:%s"
        %
        (
            HOST,
            PORT
        )
    )

    try:

        server.serve_forever()

    except KeyboardInterrupt:

        pass

    finally:

        server.server_close()
