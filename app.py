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
# Full replacement app.py
# ============================================================

os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/Ultralytics")

import psycopg2
from psycopg2.extras import RealDictCursor

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except Exception:
    YOLO_AVAILABLE = False


# ============================================================
# CONFIGURATION
# ============================================================

PORT = int(os.environ.get("PORT", "8080"))

DATABASE_URL = os.environ.get("DATABASE_URL", "").strip()

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

REFERENCE_DIR = os.path.join(BASE_DIR, "bucket_images")

MODEL_OUTPUT = os.path.join(MODELS_DIR, "bucket_best.pt")

DATASET_YAML = os.path.join(AI_DIR, "bucket_dataset.yaml")

EPOCHS = int(os.environ.get("YOLO_EPOCHS", "20"))

IMG_SIZE = int(os.environ.get("YOLO_IMG_SIZE", "640"))

MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "15"))


# ============================================================
# CREATE DIRECTORIES
# ============================================================

for folder in [
    AI_DIR,
    DATASET_DIR,
    IMAGES_DIR,
    LABELS_DIR,
    TRAIN_IMAGES_DIR,
    VAL_IMAGES_DIR,
    TRAIN_LABELS_DIR,
    VAL_LABELS_DIR,
    MODELS_DIR,
    REFERENCE_DIR,
]:
    os.makedirs(folder, exist_ok=True)


# ============================================================
# TRAINING STATE
# ============================================================

TRAINING_STATE = {
    "status": "idle",
    "message": "Ready",
    "progress": 0,
    "epoch": 0,
    "epochs": EPOCHS,
    "error": "",
    "model": "",
    "started_at": None,
    "finished_at": None,
}

TRAINING_LOCK = threading.Lock()


# ============================================================
# DATABASE
# ============================================================

def get_db():
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL environment variable is missing.")

    return psycopg2.connect(
        DATABASE_URL,
        sslmode="require",
        connect_timeout=15
    )


def init_db():
    conn = None

    try:
        conn = get_db()

        with conn.cursor() as cur:

            cur.execute("""
                CREATE TABLE IF NOT EXISTS bucket_counts (
                    id BIGSERIAL PRIMARY KEY,
                    bucket_count INTEGER NOT NULL DEFAULT 0,
                    shift_name TEXT,
                    recorded_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)

        conn.commit()

    except Exception:
        print("DATABASE INIT ERROR")
        traceback.print_exc()

    finally:
        if conn:
            conn.close()


# ============================================================
# DATABASE HELPERS
# ============================================================

def save_bucket_count(count, shift_name=""):
    conn = None

    try:
        conn = get_db()

        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO bucket_counts
                (bucket_count, shift_name)
                VALUES (%s, %s)
                RETURNING id
                """,
                (int(count), shift_name)
            )

            row = cur.fetchone()

        conn.commit()

        return {
            "success": True,
            "id": row[0] if row else None
        }

    except Exception as e:
        if conn:
            conn.rollback()

        return {
            "success": False,
            "error": str(e)
        }

    finally:
        if conn:
            conn.close()


def get_bucket_history():
    conn = None

    try:
        conn = get_db()

        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT
                    id,
                    bucket_count,
                    shift_name,
                    recorded_at
                FROM bucket_counts
                ORDER BY recorded_at DESC
                LIMIT 500
            """)

            rows = cur.fetchall()

        result = []

        for row in rows:
            item = dict(row)

            if item.get("recorded_at"):
                item["recorded_at"] = item["recorded_at"].isoformat()

            result.append(item)

        return result

    except Exception:
        traceback.print_exc()
        return []

    finally:
        if conn:
            conn.close()


# ============================================================
# FILE HELPERS
# ============================================================

ALLOWED_IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp"
}


def safe_filename(name):
    name = os.path.basename(name or "")

    allowed = (
        "abcdefghijklmnopqrstuvwxyz"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "0123456789"
        "._-"
    )

    cleaned = "".join(
        c if c in allowed else "_"
        for c in name
    )

    if not cleaned:
        cleaned = "image.jpg"

    return cleaned


def unique_filename(original):
    original = safe_filename(original)

    root, ext = os.path.splitext(original)

    if not ext:
        ext = ".jpg"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    random_part = uuid.uuid4().hex[:8]

    return f"{root}_{timestamp}_{random_part}{ext.lower()}"


def is_image_file(filename):
    ext = os.path.splitext(filename)[1].lower()
    return ext in ALLOWED_IMAGE_EXTENSIONS


# ============================================================
# MULTIPART PARSER
# ============================================================

def parse_multipart(handler):
    content_type = handler.headers.get("Content-Type", "")

    if "multipart/form-data" not in content_type:
        raise ValueError("Expected multipart/form-data.")

    content_length = int(handler.headers.get("Content-Length", "0"))

    if content_length <= 0:
        raise ValueError("Empty upload.")

    max_bytes = MAX_UPLOAD_MB * 1024 * 1024

    if content_length > max_bytes:
        raise ValueError(
            f"File too large. Maximum is {MAX_UPLOAD_MB} MB."
        )

    body = handler.rfile.read(content_length)

    boundary_marker = "boundary="

    if boundary_marker not in content_type:
        raise ValueError("Multipart boundary missing.")

    boundary = content_type.split(
        boundary_marker,
        1
    )[1].strip()

    if boundary.startswith('"') and boundary.endswith('"'):
        boundary = boundary[1:-1]

    boundary_bytes = (
        b"--" +
        boundary.encode("utf-8")
    )

    parts = body.split(boundary_bytes)

    fields = {}

    files = []

    for part in parts:

        part = part.strip(b"\r\n-")

        if not part:
            continue

        header_end = part.find(b"\r\n\r\n")

        if header_end == -1:
            continue

        header_bytes = part[:header_end]
        data = part[header_end + 4:]

        headers = header_bytes.decode(
            "utf-8",
            errors="ignore"
        )

        disposition = ""

        for line in headers.split("\r\n"):
            if line.lower().startswith(
                "content-disposition:"
            ):
                disposition = line

        field_name = None
        filename = None

        for token in disposition.split(";"):

            token = token.strip()

            if token.startswith("name="):
                field_name = token.split(
                    "=",
                    1
                )[1].strip('"')

            elif token.startswith("filename="):
                filename = token.split(
                    "=",
                    1
                )[1].strip('"')

        if not field_name:
            continue

        if filename:
            files.append({
                "field": field_name,
                "filename": filename,
                "data": data
            })

        else:
            fields[field_name] = data.decode(
                "utf-8",
                errors="ignore"
            )

    return fields, files


# ============================================================
# DATASET HELPERS
# ============================================================

def split_dirs(split):
    split = "val" if split == "val" else "train"

    if split == "val":
        return VAL_IMAGES_DIR, VAL_LABELS_DIR

    return TRAIN_IMAGES_DIR, TRAIN_LABELS_DIR


def write_dataset_yaml():

    content = f"""path: {DATASET_DIR.replace(os.sep, "/")}
train: images/train
val: images/val

names:
  0: BUCKET_LOADED
"""

    with open(
        DATASET_YAML,
        "w",
        encoding="utf-8"
    ) as f:
        f.write(content)


def image_files_in(directory):
    result = []

    if not os.path.isdir(directory):
        return result

    for name in sorted(os.listdir(directory)):

        path = os.path.join(directory, name)

        if not os.path.isfile(path):
            continue

        if is_image_file(name):
            result.append(name)

    return result


def label_path_for(split, filename):
    image_dir, label_dir = split_dirs(split)

    root = os.path.splitext(
        safe_filename(filename)
    )[0]

    return os.path.join(
        label_dir,
        root + ".txt"
    )


def image_path_for(split, filename):
    image_dir, label_dir = split_dirs(split)

    return os.path.join(
        image_dir,
        safe_filename(filename)
    )


def image_has_label(split, filename):

    path = label_path_for(
        split,
        filename
    )

    if not os.path.exists(path):
        return False

    try:
        with open(
            path,
            "r",
            encoding="utf-8"
        ) as f:

            content = f.read().strip()

        return bool(content)

    except Exception:
        return False


def dataset_stats():

    train_images = image_files_in(
        TRAIN_IMAGES_DIR
    )

    val_images = image_files_in(
        VAL_IMAGES_DIR
    )

    train_labels = 0
    val_labels = 0

    for filename in train_images:
        if image_has_label("train", filename):
            train_labels += 1

    for filename in val_images:
        if image_has_label("val", filename):
            val_labels += 1

    return {
        "training_images": len(train_images),
        "validation_images": len(val_images),
        "training_labels": train_labels,
        "validation_labels": val_labels,
        "unlabeled_training": (
            len(train_images) - train_labels
        ),
        "unlabeled_validation": (
            len(val_images) - val_labels
        )
    }


# ============================================================
# YOLO LABEL HELPERS
# ============================================================

def save_yolo_labels(
    split,
    filename,
    boxes
):

    split = "val" if split == "val" else "train"

    label_path = label_path_for(
        split,
        filename
    )

    valid_lines = []

    for box in boxes:

        try:
            x = float(box["x"])
            y = float(box["y"])
            w = float(box["w"])
            h = float(box["h"])

            if w <= 0 or h <= 0:
                continue

            # YOLO normalized coordinates
            valid_lines.append(
                f"0 {x:.6f} {y:.6f} {w:.6f} {h:.6f}"
            )

        except Exception:
            continue

    if not valid_lines:
        raise ValueError(
            "No valid annotation boxes."
        )

    with open(
        label_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "\n".join(valid_lines) +
            "\n"
        )

    return {
        "filename": os.path.basename(
            label_path
        ),
        "boxes": len(valid_lines)
    }


def read_yolo_labels(
    split,
    filename
):

    path = label_path_for(
        split,
        filename
    )

    if not os.path.exists(path):
        return []

    result = []

    try:

        with open(
            path,
            "r",
            encoding="utf-8"
        ) as f:

            lines = f.readlines()

        for line in lines:

            parts = line.strip().split()

            if len(parts) != 5:
                continue

            try:

                cls = int(parts[0])

                x = float(parts[1])
                y = float(parts[2])
                w = float(parts[3])
                h = float(parts[4])

                result.append({
                    "class_id": cls,
                    "x": x,
                    "y": y,
                    "w": w,
                    "h": h
                })

            except Exception:
                continue

    except Exception:
        traceback.print_exc()

    return result


# ============================================================
# VALIDATION DATA PREPARATION
# ============================================================

def prepare_validation_data():

    val_images = image_files_in(
        VAL_IMAGES_DIR
    )

    val_labeled = [
        name
        for name in val_images
        if image_has_label(
            "val",
            name
        )
    ]

    if val_labeled:
        return {
            "created": False,
            "filename": val_labeled[0]
        }

    train_images = image_files_in(
        TRAIN_IMAGES_DIR
    )

    train_labeled = [
        name
        for name in train_images
        if image_has_label(
            "train",
            name
        )
    ]

    if not train_labeled:
        raise RuntimeError(
            "No labeled training images found."
        )

    # If validation is empty, copy one labeled
    # training image to validation as a pipeline
    # fallback.
    source_name = train_labeled[0]

    source_image = os.path.join(
        TRAIN_IMAGES_DIR,
        source_name
    )

    source_label = os.path.join(
        TRAIN_LABELS_DIR,
        os.path.splitext(
            source_name
        )[0] + ".txt"
    )

    val_name = (
        "validation_" +
        source_name
    )

    destination_image = os.path.join(
        VAL_IMAGES_DIR,
        val_name
    )

    destination_label = os.path.join(
        VAL_LABELS_DIR,
        os.path.splitext(
            val_name
        )[0] + ".txt"
    )

    shutil.copy2(
        source_image,
        destination_image
    )

    shutil.copy2(
        source_label,
        destination_label
    )

    return {
        "created": True,
        "filename": val_name
    }


# ============================================================
# TRAINING CALLBACK
# ============================================================

def update_training_state(
    status=None,
    message=None,
    progress=None,
    epoch=None,
    error=None
):

    with TRAINING_LOCK:

        if status is not None:
            TRAINING_STATE["status"] = status

        if message is not None:
            TRAINING_STATE["message"] = message

        if progress is not None:
            TRAINING_STATE["progress"] = max(
                0,
                min(
                    100,
                    int(progress)
                )
            )

        if epoch is not None:
            TRAINING_STATE["epoch"] = epoch

        if error is not None:
            TRAINING_STATE["error"] = error


def reset_training_state():

    with TRAINING_LOCK:

        TRAINING_STATE.update({
            "status": "idle",
            "message": "Ready",
            "progress": 0,
            "epoch": 0,
            "epochs": EPOCHS,
            "error": "",
            "model": "",
            "started_at": None,
            "finished_at": None,
        })


def training_callback(trainer):

    try:

        epoch = int(
            getattr(
                trainer,
                "epoch",
                0
            )
        )

        total_epochs = int(
            getattr(
                trainer,
                "epochs",
                EPOCHS
            )
        )

        # epoch is zero based
        completed = epoch + 1

        progress = int(
            completed /
            max(1, total_epochs)
            * 100
        )

        update_training_state(
            status="training",
            message=(
                f"Training YOLO: "
                f"Epoch {completed}/{total_epochs}"
            ),
            progress=progress,
            epoch=completed
        )

        print(
            f"YOLO TRAINING: "
            f"Epoch {completed}/{total_epochs} "
            f"({progress}%)"
        )

    except Exception:

        traceback.print_exc()


# ============================================================
# TRAIN YOLO
# ============================================================

def train_yolo_worker():

    try:

        update_training_state(
            status="preparing",
            message="Preparing dataset...",
            progress=1,
            epoch=0
        )

        write_dataset_yaml()

        stats = dataset_stats()

        if stats["training_images"] == 0:
            raise RuntimeError(
                "Training images folder is empty."
            )

        if stats["training_labels"] == 0:
            raise RuntimeError(
                "No labeled training images found. "
                "Annotate at least one image first."
            )

        update_training_state(
            status="preparing",
            message=(
                f"Dataset ready: "
                f"{stats['training_labels']} "
                f"labeled training image(s)."
            ),
            progress=5
        )

        # Prepare validation
        prepare_validation_data()

        write_dataset_yaml()

        update_training_state(
            status="preparing",
            message="Validation dataset ready.",
            progress=8
        )

        if not YOLO_AVAILABLE:
            raise RuntimeError(
                "Ultralytics YOLO is not installed. "
                "Add ultralytics to requirements.txt."
            )

        update_training_state(
            status="loading_model",
            message="Loading YOLO model...",
            progress=10
        )

        # Prefer previously trained model.
        # Otherwise use YOLO11 nano.
        if os.path.exists(MODEL_OUTPUT):
            model_source = MODEL_OUTPUT
        else:
            model_source = "yolo11n.pt"

        print(
            "Loading YOLO model:",
            model_source
        )

        model = YOLO(model_source)

        # Register callback
        try:
            model.add_callback(
                "on_fit_epoch_end",
                training_callback
            )
        except Exception:
            try:
                model.add_callback(
                    "on_train_epoch_end",
                    training_callback
                )
            except Exception:
                traceback.print_exc()

        update_training_state(
            status="training",
            message=(
                f"Training YOLO for "
                f"{EPOCHS} epochs..."
            ),
            progress=12,
            epoch=0
        )

        print("STARTING YOLO TRAINING")

        results = model.train(
            data=DATASET_YAML,
            epochs=EPOCHS,
            imgsz=IMG_SIZE,
            project=AI_DIR,
            name="bucket_training",
            exist_ok=True,
            verbose=True
        )

        update_training_state(
            status="saving",
            message="Training finished. Saving best model...",
            progress=95,
            epoch=EPOCHS
        )

        # Locate best.pt
        possible_best = []

        try:
            save_dir = getattr(
                results,
                "save_dir",
                None
            )

            if save_dir:
                possible_best.append(
                    os.path.join(
                        str(save_dir),
                        "weights",
                        "best.pt"
                    )
                )
        except Exception:
            pass

        possible_best.extend([
            os.path.join(
                AI_DIR,
                "bucket_training",
                "weights",
                "best.pt"
            ),
            os.path.join(
                AI_DIR,
                "runs",
                "detect",
                "bucket_training",
                "weights",
                "best.pt"
            ),
            os.path.join(
                AI_DIR,
                "runs",
                "train",
                "bucket_training",
                "weights",
                "best.pt"
            )
        ])

        best_path = None

        for candidate in possible_best:

            if candidate and os.path.exists(
                candidate
            ):
                best_path = candidate
                break

        if best_path:

            os.makedirs(
                MODELS_DIR,
                exist_ok=True
            )

            shutil.copy2(
                best_path,
                MODEL_OUTPUT
            )

            update_training_state(
                status="completed",
                message=(
                    "Training completed successfully. "
                    "Best model saved."
                ),
                progress=100,
                epoch=EPOCHS
            )

            TRAINING_STATE["model"] = MODEL_OUTPUT

            print(
                "BEST MODEL SAVED:",
                MODEL_OUTPUT
            )

        else:

            raise RuntimeError(
                "Training completed but best.pt "
                "was not found."
            )

    except Exception as e:

        error_text = str(e)

        print("YOLO TRAINING ERROR")
        traceback.print_exc()

        update_training_state(
            status="error",
            message="Training failed.",
            progress=0,
            error=error_text
        )

    finally:

        with TRAINING_LOCK:
            TRAINING_STATE["finished_at"] = (
                datetime.utcnow().isoformat()
            )


def start_training():

    with TRAINING_LOCK:

        if TRAINING_STATE["status"] in {
            "preparing",
            "loading_model",
            "training",
            "saving"
        }:

            return {
                "success": False,
                "message": "Training is already running."
            }

        TRAINING_STATE.update({
            "status": "preparing",
            "message": "Starting YOLO training...",
            "progress": 1,
            "epoch": 0,
            "epochs": EPOCHS,
            "error": "",
            "model": "",
            "started_at": datetime.utcnow().isoformat(),
            "finished_at": None,
        })

    thread = threading.Thread(
        target=train_yolo_worker,
        daemon=True
    )

    thread.start()

    return {
        "success": True,
        "message": "Training started."
    }


# ============================================================
# MODEL DETECTION
# ============================================================

DETECTION_MODEL = None
DETECTION_MODEL_MTIME = None


def load_detection_model():

    global DETECTION_MODEL
    global DETECTION_MODEL_MTIME

    if not YOLO_AVAILABLE:
        return None

    if not os.path.exists(
        MODEL_OUTPUT
    ):
        return None

    mtime = os.path.getmtime(
        MODEL_OUTPUT
    )

    if (
        DETECTION_MODEL is None
        or DETECTION_MODEL_MTIME != mtime
    ):

        try:

            DETECTION_MODEL = YOLO(
                MODEL_OUTPUT
            )

            DETECTION_MODEL_MTIME = mtime

        except Exception:

            traceback.print_exc()
            DETECTION_MODEL = None

    return DETECTION_MODEL


# ============================================================
# HTTP RESPONSE HELPERS
# ============================================================

def json_bytes(data):

    return json.dumps(
        data,
        ensure_ascii=False
    ).encode("utf-8")


def send_json(handler, data, status=200):

    body = json_bytes(data)

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


def send_html(handler, html):

    body = html.encode("utf-8")

    handler.send_response(200)

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


def send_file(handler, path):

    if not os.path.exists(path):
        handler.send_error(404)
        return

    ext = os.path.splitext(
        path
    )[1].lower()

    content_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
        ".txt": "text/plain",
        ".yaml": "text/plain",
    }

    content_type = content_types.get(
        ext,
        "application/octet-stream"
    )

    with open(
        path,
        "rb"
    ) as f:
        body = f.read()

    handler.send_response(200)

    handler.send_header(
        "Content-Type",
        content_type
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


# ============================================================
# HTML
# ============================================================

HTML_PAGE = r"""
<!DOCTYPE html>
<html lang="en">

<head>

<meta charset="UTF-8">

<meta
    name="viewport"
    content="width=device-width, initial-scale=1.0"
>

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
    background: #f4f6f8;
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
    color: #cbd5e1;
}

nav {
    display: flex;
    gap: 6px;
    overflow-x: auto;
    background: #1f2937;
    padding: 8px;
}

nav button {
    border: 0;
    background: #374151;
    color: white;
    padding: 10px 14px;
    border-radius: 7px;
    cursor: pointer;
}

nav button.active {
    background: #16a34a;
}

main {
    max-width: 1100px;
    margin: auto;
    padding: 15px;
}

.card {
    background: white;
    border-radius: 12px;
    padding: 16px;
    margin-bottom: 15px;
    box-shadow:
        0 2px 8px rgba(0,0,0,.08);
}

h2 {
    margin-top: 0;
}

.hidden {
    display: none !important;
}

input,
select,
button {
    font-size: 16px;
}

input,
select {
    width: 100%;
    padding: 11px;
    margin: 6px 0 10px;
    border: 1px solid #cbd5e1;
    border-radius: 7px;
}

.btn {
    border: 0;
    border-radius: 7px;
    padding: 11px 15px;
    cursor: pointer;
    background: #2563eb;
    color: white;
    margin: 3px;
}

.btn.green {
    background: #16a34a;
}

.btn.red {
    background: #dc2626;
}

.btn.gray {
    background: #64748b;
}

.stats {
    display: grid;
    grid-template-columns:
        repeat(auto-fit, minmax(150px, 1fr));
    gap: 10px;
}

.stat {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 15px;
}

.stat strong {
    display: block;
    font-size: 26px;
    margin-top: 5px;
}

.dataset-grid {
    display: grid;
    grid-template-columns:
        repeat(auto-fit, minmax(250px, 1fr));
    gap: 12px;
}

.image-card {
    border: 1px solid #dbe2ea;
    border-radius: 10px;
    padding: 10px;
    background: white;
}

.image-card img {
    width: 100%;
    max-height: 220px;
    object-fit: contain;
    background: #111827;
    border-radius: 7px;
}

.filename {
    font-size: 13px;
    word-break: break-all;
    margin: 7px 0;
}

.badge {
    display: inline-block;
    padding: 5px 8px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: bold;
}

.badge.green {
    background: #dcfce7;
    color: #166534;
}

.badge.orange {
    background: #ffedd5;
    color: #9a3412;
}

#annotationCanvas {
    width: 100%;
    max-width: 900px;
    display: block;
    margin: auto;
    background: #111;
    border-radius: 8px;
    touch-action: none;
    cursor: crosshair;
}

.annotation-info {
    padding: 10px;
    background: #f1f5f9;
    border-radius: 8px;
    margin-bottom: 10px;
}

.progress-box {
    margin-top: 10px;
}

.progress-track {
    width: 100%;
    height: 25px;
    background: #e5e7eb;
    border-radius: 20px;
    overflow: hidden;
}

.progress-bar {
    width: 0%;
    height: 100%;
    background: #16a34a;
    transition: width .3s ease;
    text-align: center;
    color: white;
    font-weight: bold;
    line-height: 25px;
}

.status {
    padding: 12px;
    border-radius: 8px;
    background: #f1f5f9;
    margin-top: 10px;
    word-break: break-word;
}

.status.training {
    background: #dbeafe;
    color: #1e3a8a;
}

.status.success {
    background: #dcfce7;
    color: #166534;
}

.status.error {
    background: #fee2e2;
    color: #991b1b;
}

.empty {
    color: #64748b;
    padding: 15px;
}

table {
    width: 100%;
    border-collapse: collapse;
}

th,
td {
    border-bottom: 1px solid #e5e7eb;
    padding: 9px;
    text-align: left;
}

.small {
    color: #64748b;
    font-size: 13px;
}

</style>

</head>

<body>

<header>

<h1>NEERIKA BUCKET AI</h1>

<p>Mining Production Bucket Counter</p>

</header>

<nav>

<button
    class="nav-btn active"
    onclick="showTab('dashboard', this)"
>
Dashboard
</button>

<button
    class="nav-btn"
    onclick="showTab('camera', this)"
>
Camera
</button>

<button
    class="nav-btn"
    onclick="showTab('buckets', this)"
>
Buckets
</button>

<button
    class="nav-btn"
    onclick="showTab('training', this)"
>
Training
</button>

<button
    class="nav-btn"
    onclick="showTab('history', this)"
>
History
</button>

<button
    class="nav-btn"
    onclick="showTab('settings', this)"
>
Settings
</button>

</nav>

<main>

<!-- ======================================================
     DASHBOARD
======================================================= -->

<section id="dashboard">

<div class="card">

<h2>Dashboard</h2>

<div class="stats">

<div class="stat">
Training Images
<strong id="dashTrainImages">0</strong>
</div>

<div class="stat">
Validation Images
<strong id="dashValImages">0</strong>
</div>

<div class="stat">
Training Labels
<strong id="dashTrainLabels">0</strong>
</div>

<div class="stat">
Buckets Counted
<strong id="dashBuckets">0</strong>
</div>

</div>

</div>

</section>


<!-- ======================================================
     CAMERA
======================================================= -->

<section
    id="camera"
    class="hidden"
>

<div class="card">

<h2>Camera</h2>

<p>
Camera detection will use the trained
BUCKET_LOADED model after training.
</p>

<button
    class="btn green"
    onclick="checkModel()"
>
Check AI Model
</button>

<div
    id="cameraStatus"
    class="status"
>
Model not checked.
</div>

</div>

</section>


<!-- ======================================================
     BUCKETS
======================================================= -->

<section
    id="buckets"
    class="hidden"
>

<div class="card">

<h2>Bucket Reference Photos</h2>

<p class="small">
Upload reference photos of the bucket type.
</p>

<form
    id="referenceForm"
>
<input
    type="file"
    id="referenceFile"
    accept="image/*"
    required
>

<button
    class="btn green"
    type="submit"
>
Upload Reference
</button>

</form>

<div id="referenceList"></div>

</div>

</section>


<!-- ======================================================
     TRAINING
======================================================= -->

<section
    id="training"
    class="hidden"
>

<div class="card">

<h2>YOLO Training Dataset</h2>

<p>
<b>Hatua 1:</b>
Upload picha ya bucket.
</p>

<p>
<b>Hatua 2:</b>
Chagua picha hapa chini na bonyeza Annotate.
</p>

<p>
<b>Hatua 3:</b>
Tumia kidole au mouse kuchora rectangle
kuizunguka loaded bucket.
</p>

<p>
<b>Muhimu:</b>
Ukiachia kidole, rectangle
<b>HAITAONDOKA.</b>
Itabaki mpaka uifute au u-save labels.
</p>

<hr>

<label>
<b>Dataset Split</b>
</label>

<select id="trainingSplit">

<option value="train">
Training
</option>

<option value="val">
Validation
</option>

</select>

<form id="datasetUploadForm">

<input
    type="file"
    id="datasetFile"
    accept="image/*"
    required
>

<button
    class="btn green"
    type="submit"
>
Upload Image
</button>

</form>

<div
    id="uploadMessage"
    class="status"
>
Ready.
</div>

</div>


<div class="card">

<h2>Dataset Information</h2>

<div class="stats">

<div class="stat">
Training Images
<strong id="trainingImages">0</strong>
</div>

<div class="stat">
Validation Images
<strong id="validationImages">0</strong>
</div>

<div class="stat">
Training Labels
<strong id="trainingLabels">0</strong>
</div>

<div class="stat">
Validation Labels
<strong id="validationLabels">0</strong>
</div>

<div class="stat">
Unlabeled Training
<strong id="unlabeledTraining">0</strong>
</div>

<div class="stat">
Unlabeled Validation
<strong id="unlabeledValidation">0</strong>
</div>

</div>

</div>


<div class="card">

<h2>Training Images</h2>

<div
    id="trainImagesList"
    class="dataset-grid"
>
Loading...
</div>

</div>


<div class="card">

<h2>Validation Images</h2>

<div
    id="valImagesList"
    class="dataset-grid"
>
Loading...
</div>

</div>


<div
    id="annotationCard"
    class="card hidden"
>

<h2>Draw Bucket Annotation</h2>

<div class="annotation-info">

Draw a rectangle around each loaded
ore/material bucket.

<br><br>

You can draw multiple boxes.

<br>

Use your finger on a phone or mouse
on a computer.

<br>

The line will remain after releasing
your finger.

</div>

<div
    id="annotationImageInfo"
    class="small"
>
No image selected.
</div>

<br>

<canvas
    id="annotationCanvas"
></canvas>

<br>

<div>

<b>
Boxes:
<span id="boxCount">0</span>
</b>

</div>

<br>

<button
    class="btn gray"
    onclick="undoLastBox()"
>
Undo Last
</button>

<button
    class="btn red"
    onclick="clearBoxes()"
>
Clear All
</button>

<button
    class="btn green"
    onclick="saveAnnotations()"
>
Save YOLO Labels
</button>

<div
    id="annotationMessage"
    class="status"
>
Draw boxes around loaded buckets.
</div>

</div>


<div class="card">

<h2>Training Status</h2>

<div
    id="trainingStatus"
    class="status"
>
Status: idle | Ready
</div>

<div class="progress-box">

<div class="progress-track">

<div
    id="trainingProgress"
    class="progress-bar"
>
0%
</div>

</div>

</div>

<p id="epochText">
Epoch: 0 / 20
</p>

<button
    id="startTrainingButton"
    class="btn green"
    onclick="startTraining()"
>
Start YOLO Training
</button>

</div>

</section>


<!-- ======================================================
     HISTORY
======================================================= -->

<section
    id="history"
    class="hidden"
>

<div class="card">

<h2>Bucket History</h2>

<button
    class="btn"
    onclick="loadHistory()"
>
Refresh
</button>

<div id="historyTable">
Loading...
</div>

</div>

</section>


<!-- ======================================================
     SETTINGS
======================================================= -->

<section
    id="settings"
    class="hidden"
>

<div class="card">

<h2>Settings</h2>

<p>
Model:
<b>BUCKET_LOADED</b>
</p>

<p>
Epochs:
<b id="settingsEpochs">
20
</b>
</p>

<p>
Image size:
<b>
640
</b>
</p>

<p>
Purpose:
Count loaded ore/material buckets
coming from the shaft.
</p>

</div>

</section>

</main>


<script>

let annotation = {

    image: null,

    filename: "",

    split: "train",

    boxes: [],

    drawing: false,

    startX: 0,

    startY: 0,

    currentX: 0,

    currentY: 0

};


let annotationCanvas =
    document.getElementById(
        "annotationCanvas"
    );

let ctx =
    annotationCanvas.getContext(
        "2d"
    );


let trainingPoller = null;


function showTab(
    tab,
    button
) {

    document
        .querySelectorAll(
            "main > section"
        )
        .forEach(
            section => {
                section.classList.add(
                    "hidden"
                );
            }
        );

    document
        .getElementById(tab)
        .classList.remove(
            "hidden"
        );

    document
        .querySelectorAll(
            ".nav-btn"
        )
        .forEach(
            btn => {
                btn.classList.remove(
                    "active"
                );
            }
        );

    if (button) {
        button.classList.add(
            "active"
        );
    }

    if (tab === "training") {
        loadDataset();
        loadDatasetImages();
        loadTrainingStatus();
    }

    if (tab === "history") {
        loadHistory();
    }

    if (tab === "buckets") {
        loadReferences();
    }
}


async function api(
    url,
    options = {}
) {

    const response =
        await fetch(
            url,
            options
        );

    const text =
        await response.text();

    let data;

    try {
        data = JSON.parse(text);
    }

    catch {
        throw new Error(
            text ||
            "Server returned invalid response."
        );
    }

    if (!response.ok) {
        throw new Error(
            data.error ||
            data.message ||
            "Request failed."
        );
    }

    return data;
}


/* =========================================================
   DATASET UPLOAD
========================================================= */

document
    .getElementById(
        "datasetUploadForm"
    )
    .addEventListener(
        "submit",
        async function(e) {

            e.preventDefault();

            const file =
                document.getElementById(
                    "datasetFile"
                ).files[0];

            const split =
                document.getElementById(
                    "trainingSplit"
                ).value;

            if (!file) {
                return;
            }

            const form =
                new FormData();

            form.append(
                "file",
                file
            );

            form.append(
                "split",
                split
            );

            const msg =
                document.getElementById(
                    "uploadMessage"
                );

            msg.className =
                "status";

            msg.textContent =
                "Uploading image...";

            try {

                const data =
                    await api(
                        "/api/dataset/upload",
                        {
                            method: "POST",
                            body: form
                        }
                    );

                msg.className =
                    "status success";

                msg.textContent =
                    "Image uploaded. Now draw the bucket box.";

                // Important:
                // use the ACTUAL split returned
                // by server
                document.getElementById(
                    "trainingSplit"
                ).value =
                    data.split;

                await loadDataset();

                await loadDatasetImages();

                openAnnotation(
                    data.filename,
                    data.split
                );

            }

            catch(error) {

                msg.className =
                    "status error";

                msg.textContent =
                    error.message;
            }
        }
    );


/* =========================================================
   DATASET LIST
========================================================= */

async function loadDataset() {

    try {

        const data =
            await api(
                "/api/dataset/stats"
            );

        document.getElementById(
            "trainingImages"
        ).textContent =
            data.training_images;

        document.getElementById(
            "validationImages"
        ).textContent =
            data.validation_images;

        document.getElementById(
            "trainingLabels"
        ).textContent =
            data.training_labels;

        document.getElementById(
            "validationLabels"
        ).textContent =
            data.validation_labels;

        document.getElementById(
            "unlabeledTraining"
        ).textContent =
            data.unlabeled_training;

        document.getElementById(
            "unlabeledValidation"
        ).textContent =
            data.unlabeled_validation;

        document.getElementById(
            "dashTrainImages"
        ).textContent =
            data.training_images;

        document.getElementById(
            "dashValImages"
        ).textContent =
            data.validation_images;

        document.getElementById(
            "dashTrainLabels"
        ).textContent =
            data.training_labels;

    }

    catch(error) {

        console.error(
            error
        );
    }
}


async function loadDatasetImages() {

    await loadDatasetImagesForSplit(
        "train"
    );

    await loadDatasetImagesForSplit(
        "val"
    );
}


async function loadDatasetImagesForSplit(
    split
) {

    const container =
        document.getElementById(
            split === "train"
                ? "trainImagesList"
                : "valImagesList"
        );

    container.innerHTML =
        "Loading...";

    try {

        const data =
            await api(
                "/api/dataset/images?split=" +
                encodeURIComponent(
                    split
                )
            );

        if (
            !data.images ||
            data.images.length === 0
        ) {

            container.innerHTML =
                '<div class="empty">' +
                'No images uploaded yet.' +
                '</div>';

            return;
        }

        container.innerHTML =
            "";

        data.images.forEach(
            image => {

                const card =
                    document.createElement(
                        "div"
                    );

                card.className =
                    "image-card";

                const badgeClass =
                    image.labeled
                        ? "green"
                        : "orange";

                const badgeText =
                    image.labeled
                        ? "LABELED"
                        : "NEEDS ANNOTATION";

                card.innerHTML = `

                    <img
                        src="/api/dataset/image?split=${encodeURIComponent(split)}&filename=${encodeURIComponent(image.filename)}"
                        alt=""
                    >

                    <div class="filename">
                        ${escapeHtml(image.filename)}
                    </div>

                    <span class="badge ${badgeClass}">
                        ${badgeText}
                    </span>

                    <br>

                    <button
                        class="btn"
                        onclick="openAnnotation('${escapeJs(image.filename)}','${split}')"
                    >
                        Annotate
                    </button>

                `;

                container.appendChild(
                    card
                );
            }
        );

    }

    catch(error) {

        container.innerHTML =
            '<div class="empty">' +
            escapeHtml(
                error.message
            ) +
            '</div>';
    }
}


function escapeHtml(
    value
) {

    return String(value)
        .replaceAll(
            "&",
            "&amp;"
        )
        .replaceAll(
            "<",
            "&lt;"
        )
        .replaceAll(
            ">",
            "&gt;"
        )
        .replaceAll(
            '"',
            "&quot;"
        )
        .replaceAll(
            "'",
            "&#039;"
        );
}


function escapeJs(
    value
) {

    return String(value)
        .replaceAll(
            "\\",
            "\\\\"
        )
        .replaceAll(
            "'",
            "\\'"
        );
}


/* =========================================================
   ANNOTATION
========================================================= */

function openAnnotation(
    filename,
    split
) {

    annotation.filename =
        filename;

    annotation.split =
        split;

    annotation.boxes =
        [];

    annotation.drawing =
        false;

    document.getElementById(
        "annotationCard"
    ).classList.remove(
        "hidden"
    );

    document.getElementById(
        "annotationImageInfo"
    ).textContent =
        "Image: " +
        filename +
        " | Split: " +
        split;

    document.getElementById(
        "boxCount"
    ).textContent =
        "0";

    const image =
        new Image();

    image.onload =
        async function() {

            annotation.image =
                image;

            resizeCanvas();

            // Load existing labels
            try {

                const data =
                    await api(
                        "/api/dataset/labels?split=" +
                        encodeURIComponent(
                            split
                        ) +
                        "&filename=" +
                        encodeURIComponent(
                            filename
                        )
                    );

                if (
                    data.boxes &&
                    data.boxes.length
                ) {

                    annotation.boxes =
                        data.boxes.map(
                            box => {

                                return {
                                    x:
                                        box.x,
                                    y:
                                        box.y,
                                    w:
                                        box.w,
                                    h:
                                        box.h
                                };

                            }
                        );

                    document.getElementById(
                        "boxCount"
                    ).textContent =
                        annotation.boxes.length;
                }

            }

            catch(error) {

                console.error(
                    error
                );
            }

            drawCanvas();
        };

    image.onerror =
        function() {

            document.getElementById(
                "annotationMessage"
            ).textContent =
                "Failed to load image.";

        };

    image.src =
        "/api/dataset/image?split=" +
        encodeURIComponent(
            split
        ) +
        "&filename=" +
        encodeURIComponent(
            filename
        );

    document
        .getElementById(
            "annotationCard"
        )
        .scrollIntoView({
            behavior: "smooth",
            block: "start"
        });
}


function resizeCanvas() {

    if (!annotation.image) {
        return;
    }

    const maxWidth =
        Math.min(
            900,
            window.innerWidth - 35
        );

    const ratio =
        annotation.image.width /
        annotation.image.height;

    annotationCanvas.width =
        Math.max(
            300,
            Math.floor(maxWidth)
        );

    annotationCanvas.height =
        Math.floor(
            annotationCanvas.width /
            ratio
        );

    drawCanvas();
}


function canvasPoint(
    event
) {

    const rect =
        annotationCanvas.getBoundingClientRect();

    const scaleX =
        annotationCanvas.width /
        rect.width;

    const scaleY =
        annotationCanvas.height /
        rect.height;

    return {
        x:
            (event.clientX - rect.left) *
            scaleX,

        y:
            (event.clientY - rect.top) *
            scaleY
    };
}


function drawCanvas() {

    if (!annotation.image) {
        return;
    }

    ctx.clearRect(
        0,
        0,
        annotationCanvas.width,
        annotationCanvas.height
    );

    ctx.drawImage(
        annotation.image,
        0,
        0,
        annotationCanvas.width,
        annotationCanvas.height
    );

    ctx.lineWidth =
        Math.max(
            2,
            annotationCanvas.width / 300
        );

    ctx.font =
        "bold 16px Arial";

    annotation.boxes.forEach(
        (box, index) => {

            const x =
                box.x *
                annotationCanvas.width;

            const y =
                box.y *
                annotationCanvas.height;

            const w =
                box.w *
                annotationCanvas.width;

            const h =
                box.h *
                annotationCanvas.height;

            ctx.strokeStyle =
                "#00ff00";

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
                95,
                24
            );

            ctx.fillStyle =
                "#ffffff";

            ctx.fillText(
                "Bucket " +
                (index + 1),
                x + 5,
                Math.max(
                    17,
                    y - 7
                )
            );

        }
    );

    // Current drawing box
    if (
        annotation.drawing
    ) {

        const x =
            Math.min(
                annotation.startX,
                annotation.currentX
            );

        const y =
            Math.min(
                annotation.startY,
                annotation.currentY
            );

        const w =
            Math.abs(
                annotation.currentX -
                annotation.startX
            );

        const h =
            Math.abs(
                annotation.currentY -
                annotation.startY
            );

        ctx.strokeStyle =
            "#ff0000";

        ctx.lineWidth = 3;

        ctx.strokeRect(
            x,
            y,
            w,
            h
        );
    }
}


/* =========================================================
   POINTER EVENTS
========================================================= */

annotationCanvas.addEventListener(
    "pointerdown",
    function(event) {

        if (!annotation.image) {
            return;
        }

        event.preventDefault();

        try {
            annotationCanvas.setPointerCapture(
                event.pointerId
            );
        }

        catch(error) {}

        const p =
            canvasPoint(event);

        annotation.drawing =
            true;

        annotation.startX =
            p.x;

        annotation.startY =
            p.y;

        annotation.currentX =
            p.x;

        annotation.currentY =
            p.y;

        drawCanvas();
    }
);


annotationCanvas.addEventListener(
    "pointermove",
    function(event) {

        if (
            !annotation.drawing
        ) {
            return;
        }

        event.preventDefault();

        const p =
            canvasPoint(event);

        annotation.currentX =
            p.x;

        annotation.currentY =
            p.y;

        drawCanvas();
    }
);


function finishPointer(
    event
) {

    if (
        !annotation.drawing
    ) {
        return;
    }

    event.preventDefault();

    const p =
        canvasPoint(event);

    annotation.currentX =
        p.x;

    annotation.currentY =
        p.y;

    let x =
        Math.min(
            annotation.startX,
            annotation.currentX
        );

    let y =
        Math.min(
            annotation.startY,
            annotation.currentY
        );

    let w =
        Math.abs(
            annotation.currentX -
            annotation.startX
        );

    let h =
        Math.abs(
            annotation.currentY -
            annotation.startY
        );

    annotation.drawing =
        false;

    if (
        w >= 10 &&
        h >= 10
    ) {

        const canvasWidth =
            annotationCanvas.width;

        const canvasHeight =
            annotationCanvas.height;

        // Clamp
        x =
            Math.max(
                0,
                Math.min(
                    x,
                    canvasWidth
                )
            );

        y =
            Math.max(
                0,
                Math.min(
                    y,
                    canvasHeight
                )
            );

        w =
            Math.min(
                w,
                canvasWidth - x
            );

        h =
            Math.min(
                h,
                canvasHeight - y
            );

        annotation.boxes.push({
            x:
                x / canvasWidth,

            y:
                y / canvasHeight,

            w:
                w / canvasWidth,

            h:
                h / canvasHeight
        });

        document.getElementById(
            "boxCount"
        ).textContent =
            annotation.boxes.length;

        document.getElementById(
            "annotationMessage"
        ).textContent =
            "Box added. You can draw another bucket.";
    }

    drawCanvas();

    try {
        annotationCanvas.releasePointerCapture(
            event.pointerId
        );
    }

    catch(error) {}
}


annotationCanvas.addEventListener(
    "pointerup",
    finishPointer
);

annotationCanvas.addEventListener(
    "pointercancel",
    finishPointer
);


window.addEventListener(
    "resize",
    function() {
        resizeCanvas();
    }
);


function undoLastBox() {

    if (
        annotation.boxes.length
    ) {

        annotation.boxes.pop();

        document.getElementById(
            "boxCount"
        ).textContent =
            annotation.boxes.length;

        drawCanvas();
    }
}


function clearBoxes() {

    annotation.boxes =
        [];

    document.getElementById(
        "boxCount"
    ).textContent =
        "0";

    drawCanvas();
}


async function saveAnnotations() {

    if (
        !annotation.filename
    ) {

        alert(
            "Choose an image first."
        );

        return;
    }

    if (
        annotation.boxes.length === 0
    ) {

        alert(
            "Draw at least one bucket box."
        );

        return;
    }

    const msg =
        document.getElementById(
            "annotationMessage"
        );

    msg.className =
        "status";

    msg.textContent =
        "Saving YOLO labels...";

    try {

        const data =
            await api(
                "/api/dataset/labels",
                {
                    method: "POST",

                    headers: {
                        "Content-Type":
                            "application/json"
                    },

                    body:
                        JSON.stringify({
                            filename:
                                annotation.filename,

                            split:
                                annotation.split,

                            boxes:
                                annotation.boxes
                        })
                }
            );

        msg.className =
            "status success";

        msg.textContent =
            "Saved successfully: " +
            data.filename +
            " | " +
            data.boxes +
            " bucket(s).";

        await loadDataset();

        await loadDatasetImages();

    }

    catch(error) {

        msg.className =
            "status error";

        msg.textContent =
            error.message;
    }
}


/* =========================================================
   TRAINING STATUS
========================================================= */

async function loadTrainingStatus() {

    try {

        const data =
            await api(
                "/api/training/status"
            );

        renderTrainingStatus(
            data
        );

        if (
            [
                "preparing",
                "loading_model",
                "training",
                "saving"
            ].includes(
                data.status
            )
        ) {

            startTrainingPolling();

        }
        else {

            stopTrainingPolling();

        }

    }

    catch(error) {

        console.error(
            error
        );
    }
}


function renderTrainingStatus(
    data
) {

    const status =
        document.getElementById(
            "trainingStatus"
        );

    const progress =
        document.getElementById(
            "trainingProgress"
        );

    const epochText =
        document.getElementById(
            "epochText"
        );

    const button =
        document.getElementById(
            "startTrainingButton"
        );

    let className =
        "status";

    if (
        data.status === "training" ||
        data.status === "preparing" ||
        data.status === "loading_model" ||
        data.status === "saving"
    ) {

        className =
            "status training";

    }

    else if (
        data.status === "completed"
    ) {

        className =
            "status success";

    }

    else if (
        data.status === "error"
    ) {

        className =
            "status error";
    }

    status.className =
        className;

    status.textContent =
        "Status: " +
        data.status +
        " | " +
        data.message;

    if (
        data.error
    ) {

        status.textContent +=
            " Error: " +
            data.error;
    }

    const pct =
        Number(
            data.progress || 0
        );

    progress.style.width =
        pct + "%";

    progress.textContent =
        pct + "%";

    epochText.textContent =
        "Epoch: " +
        (data.epoch || 0) +
        " / " +
        (data.epochs || 20);

    if (
        [
            "preparing",
            "loading_model",
            "training",
            "saving"
        ].includes(
            data.status
        )
    ) {

        button.disabled =
            true;

        button.textContent =
            "Training in progress...";

    }

    else {

        button.disabled =
            false;

        button.textContent =
            "Start YOLO Training";
    }
}


function startTrainingPolling() {

    if (trainingPoller) {
        return;
    }

    trainingPoller =
        setInterval(
            loadTrainingStatus,
            1500
        );
}


function stopTrainingPolling() {

    if (trainingPoller) {

        clearInterval(
            trainingPoller
        );

        trainingPoller =
            null;
    }
}


async function startTraining() {

    try {

        const data =
            await api(
                "/api/training/start",
                {
                    method: "POST"
                }
            );

        renderTrainingStatus({
            status: "preparing",
            message:
                data.message ||
                "Starting YOLO training...",
            progress: 1,
            epoch: 0,
            epochs: 20,
            error: ""
        });

        startTrainingPolling();

        // Immediately refresh
        // so user sees actual server state
        setTimeout(
            loadTrainingStatus,
            500
        );

    }

    catch(error) {

        renderTrainingStatus({
            status: "error",
            message: "Training failed to start.",
            progress: 0,
            epoch: 0,
            epochs: 20,
            error:
                error.message
        });
    }
}


/* =========================================================
   REFERENCES
========================================================= */

document
    .getElementById(
        "referenceForm"
    )
    .addEventListener(
        "submit",
        async function(e) {

            e.preventDefault();

            const file =
                document.getElementById(
                    "referenceFile"
                ).files[0];

            if (!file) {
                return;
            }

            const form =
                new FormData();

            form.append(
                "file",
                file
            );

            try {

                const data =
                    await api(
                        "/api/reference/upload",
                        {
                            method: "POST",
                            body: form
                        }
                    );

                alert(
                    data.message ||
                    "Reference uploaded."
                );

                loadReferences();

            }

            catch(error) {

                alert(
                    error.message
                );
            }
        }
    );


async function loadReferences() {

    const container =
        document.getElementById(
            "referenceList"
        );

    try {

        const data =
            await api(
                "/api/reference/list"
            );

        if (
            !data.images ||
            data.images.length === 0
        ) {

            container.innerHTML =
                '<div class="empty">' +
                'No reference photos yet.' +
                '</div>';

            return;
        }

        container.innerHTML =
            "";

        data.images.forEach(
            filename => {

                const img =
                    document.createElement(
                        "img"
                    );

                img.src =
                    "/api/reference/image?filename=" +
                    encodeURIComponent(
                        filename
                    );

                img.style.width =
                    "160px";

                img.style.height =
                    "120px";

                img.style.objectFit =
                    "cover";

                img.style.margin =
                    "5px";

                img.style.borderRadius =
                    "8px";

                container.appendChild(
                    img
                );
            }
        );

    }

    catch(error) {

        container.innerHTML =
            escapeHtml(
                error.message
            );
    }
}


/* =========================================================
   HISTORY
========================================================= */

async function loadHistory() {

    const container =
        document.getElementById(
            "historyTable"
        );

    try {

        const data =
            await api(
                "/api/history"
            );

        if (
            !data.history ||
            data.history.length === 0
        ) {

            container.innerHTML =
                '<div class="empty">' +
                'No bucket history yet.' +
                '</div>';

            return;
        }

        let html = `

            <table>

                <thead>

                    <tr>

                        <th>
                            ID
                        </th>

                        <th>
                            Buckets
                        </th>

                        <th>
                            Shift
                        </th>

                        <th>
                            Date
                        </th>

                    </tr>

                </thead>

                <tbody>

        `;

        data.history.forEach(
            row => {

                html += `

                    <tr>

                        <td>
                            ${row.id}
                        </td>

                        <td>
                            ${row.bucket_count}
                        </td>

                        <td>
                            ${escapeHtml(
                                row.shift_name ||
                                ""
                            )}
                        </td>

                        <td>
                            ${escapeHtml(
                                row.recorded_at ||
                                ""
                            )}
                        </td>

                    </tr>

                `;
            }
        );

        html += `
                </tbody>
            </table>
        `;

        container.innerHTML =
            html;

    }

    catch(error) {

        container.innerHTML =
            '<div class="status error">' +
            escapeHtml(
                error.message
            ) +
            '</div>';
    }
}


/* =========================================================
   MODEL CHECK
========================================================= */

async function checkModel() {

    const box =
        document.getElementById(
            "cameraStatus"
        );

    try {

        const data =
            await api(
                "/api/model/status"
            );

        if (data.ready) {

            box.className =
                "status success";

            box.textContent =
                "AI model ready: " +
                data.model;

        }

        else {

            box.className =
                "status";

            box.textContent =
                "AI model is not ready. " +
                "Train the model first.";
        }

    }

    catch(error) {

        box.className =
            "status error";

        box.textContent =
            error.message;
    }
}


/* =========================================================
   INITIAL LOAD
========================================================= */

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


    def do_GET(self):

        try:

            parsed =    urlparse(
                    self.path
                )

            path =   parsed.path

            query =   parse_qs(
                    parsed.query
                )

            # --------------------------------------------
            # Main page
            # --------------------------------------------

            if path == "/":
                send_html(
                    self,
                    HTML_PAGE
                )
                return


            # --------------------------------------------
            # Dataset stats
            # --------------------------------------------

            if path == "/api/dataset/stats":

                send_json(
                    self,
                    dataset_stats()
                )

                return


            # --------------------------------------------
            # Dataset image list
            # --------------------------------------------

            if path == "/api/dataset/images":

                split =   query.get(
                        "split",
                        ["train"]
                    )[0]

                split =    "val" if split == "val" else "train"

                image_dir, label_dir =  split_dirs(split)

                images = []

                for filename in image_files_in(
                    image_dir
                ):

                    images.append({
                        "filename": filename,
                        "split": split,
                        "labeled":
                            image_has_label(
                                split,
                                filename
                            )
                    })

                send_json(
                    self,
                    {
                        "images": images
                    }
                )

                return


            # --------------------------------------------
            # Dataset image
            # --------------------------------------------

            if path == "/api/dataset/image":

                split =   query.get(
                        "split",
                        ["train"]
                    )[0]

                filename =    query.get(
                        "filename",
                        [""]
                    )[0]

                split =    "val" if split == "val" else "train"

                image_path =   image_path_for(
                        split,
                        filename
                    )

                send_file(
                    self,
                    image_path
                )

                return


            # --------------------------------------------
            # Dataset labels
            # --------------------------------------------

            if path == "/api/dataset/labels":

                split =  query.get(
                        "split",
                        ["train"]
                    )[0]

                filename =  query.get(
                        "filename",
                        [""]
                    )[0]

                boxes =  read_yolo_labels(
                        split,
                        filename
                    )

                send_json(
                    self,
                    {
                        "filename": filename,
                        "split": split,
                        "boxes": boxes
                    }
                )

                return


            # --------------------------------------------
            # Training status
            # --------------------------------------------

            if path == "/api/training/status":

                with TRAINING_LOCK:

                    state =   dict(
                            TRAINING_STATE
                        )

                send_json(
                    self,
                    state
                )

                return


            # --------------------------------------------
            # Reference list
            # --------------------------------------------

            if path == "/api/reference/list":

                images = []

                if os.path.isdir(
                    REFERENCE_DIR
                ):

                    for filename in sorted(
                        os.listdir(
                            REFERENCE_DIR
                        )
                    ):

                        if is_image_file(
                            filename
                        ):

                            images.append(
                                filename
                            )

                send_json(
                    self,
                    {
                        "images": images
                    }
                )

                return


            # --------------------------------------------
            # Reference image
            # --------------------------------------------

            if path == "/api/reference/image":

                filename =   query.get(
                        "filename",
                        [""]
                    )[0]

                path =   os.path.join(
                        REFERENCE_DIR,
                        safe_filename(
                            filename
                        )
                    )

                send_file(
                    self,
                    path
                )

                return


            # --------------------------------------------
            # History
            # --------------------------------------------

            if path == "/api/history":

                send_json(
                    self,
                    {
                        "history":
                            get_bucket_history()
                    }
                )

                return


            # --------------------------------------------
            # Model status
            # --------------------------------------------

            if path == "/api/model/status":

                ready =  os.path.exists(
                        MODEL_OUTPUT
                    )

                send_json(
                    self,
                    {
                        "ready": ready,
                        "model":
                            MODEL_OUTPUT
                        if ready
                        else ""
                    }
                )

                return


            self.send_error(
                404,
                "Not Found"
            )

        except Exception as e:

            traceback.print_exc()

            try:

                send_json(
                    self,
                    {
                        "error": str(e)
                    },
                    500
                )

            except Exception:
                pass


    def do_POST(self):

        try:

            parsed =  urlparse(
                    self.path
                )

            path =  parsed.path


            # =================================================
            # DATASET UPLOAD
            # =================================================

            if path == "/api/dataset/upload":

                fields, files =  parse_multipart(
                        self
                    )

                if not files:
                    raise ValueError(
                        "No image file uploaded."
                    )

                file_info = files[0]

                original_filename = file_info["filename"]

                if not is_image_file(
                    original_filename
                ):

                    raise ValueError(
                        "Only JPG, JPEG, PNG and WEBP images are allowed."
                    )

                split =   fields.get(
                        "split",
                        "train"
                    ).strip().lower()

                if split not in {
                    "train",
                    "val"
                }:

                    split = "train"

                image_dir, label_dir =  split_dirs(
                        split
                    )

                filename =  unique_filename(
                        original_filename
                    )

                destination =   os.path.join(
                        image_dir,
                        filename
                    )

                with open(
                    destination,
                    "wb"
                ) as f:

                    f.write(
                        file_info["data"]
                    )

                send_json(
                    self,
                    {
                        "success": True,
                        "filename": filename,
                        "split": split,
                        "url":
                            "/api/dataset/image?split=" +
                            split +
                            "&filename=" +
                            filename
                    }
                )

                return


            # =================================================
            # SAVE LABELS
            # =================================================

            if path == "/api/dataset/labels":

                content_length = int(
                        self.headers.get(
                            "Content-Length",
                            "0"
                        )
                    )

                body =  self.rfile.read(
                        content_length
                    )

                data = json.loads(
                        body.decode(
                            "utf-8"
                        )
                    )

                filename =    data.get(
                        "filename",
                        ""
                    )

                split = data.get(
                        "split",
                        "train"
                    )

                boxes = data.get(
                        "boxes",
                        []
                    )

                if not filename:
                    raise ValueError(
                        "Filename missing."
                    )

                if not boxes:
                    raise ValueError(
                        "No annotation boxes."
                    )

                result =   save_yolo_labels(
                        split,
                        filename,
                        boxes
                    )

                send_json(
                    self,
                    {
                        "success": True,
                        **result
                    }
                )

                return


            # =================================================
            # START TRAINING
            # =================================================

            if path == "/api/training/start":

                result =  start_training()

                send_json(
                    self,
                    result,
                    200
                    if result["success"]
                    else 409
                )

                return


            # =================================================
            # REFERENCE UPLOAD
            # =================================================

            if path == "/api/reference/upload":

                fields, files = parse_multipart(
                        self
                    )

                if not files:
                    raise ValueError(
                        "No reference image uploaded."
                    )

                file_info =files[0]

                original =  file_info["filename"]

                if not is_image_file(
                    original
                ):

                    raise ValueError(
                        "Invalid image format."
                    )

                filename =  unique_filename(
                        original
                    )

                destination = os.path.join(
                        REFERENCE_DIR,
                        filename
                    )

                with open(
                    destination,
                    "wb"
                ) as f:

                    f.write(
                        file_info["data"]
                    )

                send_json(
                    self,
                    {
                        "success": True,
                        "filename": filename,
                        "message":
                            "Reference photo uploaded successfully."
                    }
                )

                return


            # =================================================
            # SAVE BUCKET COUNT
            # =================================================

            if path == "/api/count":

                content_length =  int(
                        self.headers.get(
                            "Content-Length",
                            "0"
                        )
                    )

                body =   self.rfile.read(
                        content_length
                    )

                data = json.loads(
                        body.decode(
                            "utf-8"
                        )
                    )

                count =  int(
                        data.get(
                            "count",
                            0
                        )
                    )

                shift =   str(
                        data.get(
                            "shift",
                            ""
                        )
                    )

                result =  save_bucket_count(
                        count,
                        shift
                    )

                send_json(
                    self,
                    result,
                    200
                    if result["success"]
                    else 500
                )

                return


            self.send_error(
                404,
                "Not Found"
            )

        except Exception as e:

            traceback.print_exc()

            send_json(
                self,
                {
                    "success": False,
                    "error": str(e)
                },
                500
            )


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("NEERIKA BUCKET AI")
    print("Mining Production Bucket Counter")
    print("=" * 60)

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
        "YOLO_AVAILABLE:",
        YOLO_AVAILABLE
    )

    print(
        "EPOCHS:",
        EPOCHS
    )

    print(
        "PORT:",
        PORT
    )

    init_db()

    write_dataset_yaml()

    server =  ThreadingHTTPServer(
            ("0.0.0.0", PORT),
            Handler
        )

    print(
        f"NEERIKA BUCKET AI running on port {PORT}"
    )

    server.serve_forever()
