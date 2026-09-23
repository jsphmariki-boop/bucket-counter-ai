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
from urllib.parse import urlparse, parse_qs
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/Ultralytics")

import psycopg2
from psycopg2.extras import RealDictCursor

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None


# ============================================================
# CONFIGURATION
# ============================================================

PORT = int(os.environ.get("PORT", "10000"))
DATABASE_URL = os.environ.get("DATABASE_URL", "").strip()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

AI_DIR = os.path.join(BASE_DIR, "ai")
DATASET_DIR = os.path.join(AI_DIR, "dataset")

TRAIN_IMAGES_DIR = os.path.join(DATASET_DIR, "images", "train")
VAL_IMAGES_DIR = os.path.join(DATASET_DIR, "images", "val")

TRAIN_LABELS_DIR = os.path.join(DATASET_DIR, "labels", "train")
VAL_LABELS_DIR = os.path.join(DATASET_DIR, "labels", "val")

RUNS_DIR = os.path.join(AI_DIR, "runs")
MODEL_DIR = os.path.join(AI_DIR, "models")

MODEL_PATH = os.path.join(MODEL_DIR, "bucket_best.pt")
DATASET_YAML = os.path.join(DATASET_DIR, "dataset.yaml")

REFERENCE_DIR = os.path.join(BASE_DIR, "bucket_images")

MAX_UPLOAD_SIZE = 25 * 1024 * 1024


# ============================================================
# CREATE DIRECTORIES
# ============================================================

for folder in [
    AI_DIR,
    DATASET_DIR,
    TRAIN_IMAGES_DIR,
    VAL_IMAGES_DIR,
    TRAIN_LABELS_DIR,
    VAL_LABELS_DIR,
    RUNS_DIR,
    MODEL_DIR,
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
    "epochs": 0,
    "started_at": None,
    "finished_at": None,
    "error": None,
}

TRAIN_LOCK = threading.Lock()


# ============================================================
# DATABASE
# ============================================================

def get_db():
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL is not configured.")

    return psycopg2.connect(
        DATABASE_URL,
        cursor_factory=RealDictCursor,
        connect_timeout=15
    )


def init_database():
    if not DATABASE_URL:
        print("WARNING: DATABASE_URL is not configured.")
        return

    conn = None

    try:
        conn = get_db()

        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS bucket_counts (
                    id BIGSERIAL PRIMARY KEY,
                    bucket_count INTEGER NOT NULL DEFAULT 0,
                    count_date TEXT,
                    shift TEXT,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)

        conn.commit()
        print("Database initialized.")

    except Exception:
        print("DATABASE INITIALIZATION ERROR")
        traceback.print_exc()

    finally:
        if conn:
            conn.close()


def save_bucket_count(count_value, shift="A"):
    conn = None

    try:
        conn = get_db()

        today = datetime.now().strftime("%Y-%m-%d")

        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO bucket_counts
                (bucket_count, count_date, shift)
                VALUES (%s, %s, %s)
            """, (
                int(count_value),
                today,
                shift
            ))

        conn.commit()

        return True

    except Exception:
        traceback.print_exc()
        return False

    finally:
        if conn:
            conn.close()


def get_history(limit=200):
    conn = None

    try:
        conn = get_db()

        with conn.cursor() as cur:
            cur.execute("""
                SELECT
                    id,
                    bucket_count,
                    count_date,
                    shift,
                    created_at
                FROM bucket_counts
                ORDER BY id DESC
                LIMIT %s
            """, (limit,))

            return cur.fetchall()

    except Exception:
        traceback.print_exc()
        return []

    finally:
        if conn:
            conn.close()


def get_dashboard_data():
    conn = None

    try:
        conn = get_db()

        with conn.cursor() as cur:
            cur.execute("""
                SELECT
                    COALESCE(SUM(bucket_count), 0) AS total
                FROM bucket_counts
            """)

            total_row = cur.fetchone()

            cur.execute("""
                SELECT
                    COALESCE(SUM(bucket_count), 0) AS today
                FROM bucket_counts
                WHERE count_date = %s
            """, (
                datetime.now().strftime("%Y-%m-%d"),
            ))

            today_row = cur.fetchone()

        return {
            "total": int(total_row["total"] or 0),
            "today": int(today_row["today"] or 0),
        }

    except Exception:
        traceback.print_exc()

        return {
            "total": 0,
            "today": 0,
        }

    finally:
        if conn:
            conn.close()


# ============================================================
# FILE HELPERS
# ============================================================

def safe_filename(filename):
    filename = os.path.basename(filename or "")

    allowed = (
        "abcdefghijklmnopqrstuvwxyz"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "0123456789"
        "._-"
    )

    cleaned = "".join(
        c if c in allowed else "_"
        for c in filename
    )

    if not cleaned:
        cleaned = "file"

    return cleaned


def unique_filename(filename):
    filename = safe_filename(filename)

    name, ext = os.path.splitext(filename)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = uuid.uuid4().hex[:8]

    return f"{name}_{timestamp}_{unique_id}{ext.lower()}"


def is_image(filename):
    return filename.lower().endswith(
        (".jpg", ".jpeg", ".png", ".webp")
    )


def content_type_for(filename):
    ext = os.path.splitext(filename.lower())[1]

    mapping = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
        ".gif": "image/gif",
        ".css": "text/css",
        ".js": "application/javascript",
        ".json": "application/json",
        ".txt": "text/plain",
        ".csv": "text/csv",
    }

    return mapping.get(ext, "application/octet-stream")


# ============================================================
# DATASET HELPERS
# ============================================================

def get_split_dirs(split):
    if split == "val":
        return VAL_IMAGES_DIR, VAL_LABELS_DIR

    return TRAIN_IMAGES_DIR, TRAIN_LABELS_DIR


def get_dataset_yaml():
    content = f"""path: {DATASET_DIR}
train: images/train
val: images/val

names:
  0: BUCKET_LOADED
"""

    with open(DATASET_YAML, "w", encoding="utf-8") as f:
        f.write(content)

    return DATASET_YAML


def dataset_stats():
    result = {
        "training_images": 0,
        "validation_images": 0,
        "training_labels": 0,
        "validation_labels": 0,
        "training_unlabeled": 0,
        "validation_unlabeled": 0,
    }

    train_images = [
        f for f in os.listdir(TRAIN_IMAGES_DIR)
        if is_image(f)
    ]

    val_images = [
        f for f in os.listdir(VAL_IMAGES_DIR)
        if is_image(f)
    ]

    train_labels = [
        f for f in os.listdir(TRAIN_LABELS_DIR)
        if f.lower().endswith(".txt")
    ]

    val_labels = [
        f for f in os.listdir(VAL_LABELS_DIR)
        if f.lower().endswith(".txt")
    ]

    result["training_images"] = len(train_images)
    result["validation_images"] = len(val_images)
    result["training_labels"] = len(train_labels)
    result["validation_labels"] = len(val_labels)

    train_unlabeled = 0

    for image in train_images:
        base = os.path.splitext(image)[0]
        label = base + ".txt"

        if not os.path.exists(
            os.path.join(TRAIN_LABELS_DIR, label)
        ):
            train_unlabeled += 1

    val_unlabeled = 0

    for image in val_images:
        base = os.path.splitext(image)[0]
        label = base + ".txt"

        if not os.path.exists(
            os.path.join(VAL_LABELS_DIR, label)
        ):
            val_unlabeled += 1

    result["training_unlabeled"] = train_unlabeled
    result["validation_unlabeled"] = val_unlabeled

    return result


def dataset_image_list(split="train"):
    images_dir, labels_dir = get_split_dirs(split)

    result = []

    if not os.path.exists(images_dir):
        return result

    for filename in sorted(os.listdir(images_dir)):
        if not is_image(filename):
            continue

        label_filename = (
            os.path.splitext(filename)[0] + ".txt"
        )

        label_exists = os.path.exists(
            os.path.join(labels_dir, label_filename)
        )

        result.append({
            "filename": filename,
            "split": split,
            "label_exists": label_exists,
            "url": f"/dataset/{split}/{filename}",
        })

    return result


# ============================================================
# MULTIPART UPLOAD PARSER
# ============================================================

def parse_multipart(handler):
    content_type = handler.headers.get("Content-Type", "")

    if "multipart/form-data" not in content_type:
        raise ValueError("Expected multipart/form-data")

    if "boundary=" not in content_type:
        raise ValueError("Multipart boundary missing")

    boundary = content_type.split("boundary=", 1)[1]

    boundary = boundary.strip()

    if boundary.startswith('"') and boundary.endswith('"'):
        boundary = boundary[1:-1]

    boundary_bytes = ("--" + boundary).encode()

    content_length = int(
        handler.headers.get("Content-Length", "0")
    )

    if content_length > MAX_UPLOAD_SIZE:
        raise ValueError(
            "File is too large. Maximum upload size is 25 MB."
        )

    body = handler.rfile.read(content_length)

    parts = body.split(boundary_bytes)

    files = []

    for part in parts:
        if not part or part in (b"--\r\n", b"--"):
            continue

        part = part.strip(b"\r\n")

        if b"\r\n\r\n" not in part:
            continue

        header_block, data = part.split(
            b"\r\n\r\n",
            1
        )

        if data.endswith(b"\r\n"):
            data = data[:-2]

        headers = header_block.decode(
            "utf-8",
            errors="ignore"
        )

        disposition = ""

        for line in headers.split("\r\n"):
            if line.lower().startswith("content-disposition:"):
                disposition = line
                break

        if not disposition:
            continue

        filename = None
        field_name = None

        if 'name="' in disposition:
            field_name = disposition.split(
                'name="',
                1
            )[1].split('"', 1)[0]

        if 'filename="' in disposition:
            filename = disposition.split(
                'filename="',
                1
            )[1].split('"', 1)[0]

        if filename:
            files.append({
                "field": field_name,
                "filename": filename,
                "data": data,
            })

    return files


# ============================================================
# SAVE YOLO LABELS
# ============================================================

def save_yolo_labels(image_filename, boxes, split="train"):
    split = "val" if split == "val" else "train"

    image_filename = safe_filename(image_filename)

    images_dir, labels_dir = get_split_dirs(split)

    image_path = os.path.join(
        images_dir,
        image_filename
    )

    if not os.path.isfile(image_path):
        raise ValueError(
            "Dataset image was not found."
        )

    if not isinstance(boxes, list):
        raise ValueError("boxes must be a list")

    if len(boxes) == 0:
        raise ValueError(
            "At least one bucket box is required."
        )

    if len(boxes) > 100:
        raise ValueError(
            "Maximum 100 boxes per image."
        )

    lines = []

    for box in boxes:

        try:
            x_center = float(box["x_center"])
            y_center = float(box["y_center"])
            width = float(box["width"])
            height = float(box["height"])
        except Exception:
            raise ValueError(
                "Invalid bounding box values."
            )

        if not (
            0 <= x_center <= 1
            and
            0 <= y_center <= 1
            and
            0 < width <= 1
            and
            0 < height <= 1
        ):
            raise ValueError(
                "Bounding box coordinates must be between 0 and 1."
            )

        lines.append(
            "0 "
            f"{x_center:.6f} "
            f"{y_center:.6f} "
            f"{width:.6f} "
            f"{height:.6f}"
        )

    label_filename = (
        os.path.splitext(image_filename)[0]
        + ".txt"
    )

    label_path = os.path.join(
        labels_dir,
        label_filename
    )

    with open(
        label_path,
        "w",
        encoding="utf-8"
    ) as f:
        f.write(
            "\n".join(lines) + "\n"
        )

    return label_filename


# ============================================================
# VALIDATION DATA
# ============================================================

def prepare_validation_data():
    train_images = [
        f for f in os.listdir(TRAIN_IMAGES_DIR)
        if is_image(f)
    ]

    val_images = [
        f for f in os.listdir(VAL_IMAGES_DIR)
        if is_image(f)
    ]

    if val_images:
        return

    labeled_train = []

    for image in train_images:
        label = os.path.splitext(image)[0] + ".txt"

        if os.path.exists(
            os.path.join(TRAIN_LABELS_DIR, label)
        ):
            labeled_train.append(image)

    if not labeled_train:
        raise RuntimeError(
            "No labeled training images found. "
            "Draw a bucket box and save the YOLO label first."
        )

    # For a small test dataset, copy one labeled image
    # to validation when validation is empty.
    source_image = labeled_train[-1]

    source_label = (
        os.path.splitext(source_image)[0] + ".txt"
    )

    shutil.copy2(
        os.path.join(TRAIN_IMAGES_DIR, source_image),
        os.path.join(VAL_IMAGES_DIR, source_image)
    )

    shutil.copy2(
        os.path.join(TRAIN_LABELS_DIR, source_label),
        os.path.join(VAL_LABELS_DIR, source_label)
    )


# ============================================================
# MODEL
# ============================================================

def trained_model_exists():
    return os.path.isfile(MODEL_PATH)


def load_detection_model():
    if YOLO is None:
        raise RuntimeError(
            "Ultralytics is not installed."
        )

    if not os.path.isfile(MODEL_PATH):
        raise RuntimeError(
            "Trained bucket model does not exist yet. "
            "Upload and annotate training images, then train the model."
        )

    return YOLO(MODEL_PATH)


# ============================================================
# TRAINING
# ============================================================

def training_callback(trainer):
    try:
        epoch = int(
            getattr(trainer, "epoch", 0)
        )

        total_epochs = int(
            getattr(
                trainer,
                "epochs",
                TRAINING_STATE["epochs"] or 1
            )
        )

        TRAINING_STATE["epoch"] = epoch + 1
        TRAINING_STATE["epochs"] = total_epochs

        if total_epochs > 0:
            TRAINING_STATE["progress"] = int(
                ((epoch + 1) / total_epochs) * 100
            )

        TRAINING_STATE["message"] = (
            f"Training YOLO... "
            f"Epoch {epoch + 1}/{total_epochs}"
        )

    except Exception:
        pass


def train_model():
    try:
        TRAINING_STATE["status"] = "training"
        TRAINING_STATE["message"] = "Preparing dataset..."
        TRAINING_STATE["progress"] = 0
        TRAINING_STATE["error"] = None
        TRAINING_STATE["started_at"] = datetime.now().isoformat()
        TRAINING_STATE["finished_at"] = None

        stats = dataset_stats()

        if stats["training_images"] == 0:
            raise RuntimeError(
                "Training images folder is empty."
            )

        if stats["training_labels"] == 0:
            raise RuntimeError(
                "No training labels found. "
                "Open an image in Training and draw a box around the loaded bucket."
            )

        if stats["training_unlabeled"] > 0:
            raise RuntimeError(
                f"{stats['training_unlabeled']} training image(s) "
                "do not have labels. Annotate every training image first."
            )

        prepare_validation_data()

        get_dataset_yaml()

        stats = dataset_stats()

        TRAINING_STATE["epochs"] = 20
        TRAINING_STATE["message"] = (
            "Loading YOLO model..."
        )

        if YOLO is None:
            raise RuntimeError(
                "Ultralytics package is not available."
            )

        if os.path.isfile(MODEL_PATH):
            model = YOLO(MODEL_PATH)
        else:
            # Ultralytics can download the base model
            # when internet access is available.
            model = YOLO("yolo11n.pt")

        model.add_callback(
            "on_train_epoch_end",
            training_callback
        )

        TRAINING_STATE["message"] = (
            "Starting YOLO training..."
        )

        run_name = (
            "bucket_training_"
            + datetime.now().strftime("%Y%m%d_%H%M%S")
        )

        model.train(
            data=get_dataset_yaml(),
            epochs=20,
            imgsz=640,
            batch=4,
            workers=0,
            project=RUNS_DIR,
            name=run_name,
            exist_ok=True,
            pretrained=True,
            verbose=True,
        )

        best_path = os.path.join(
            RUNS_DIR,
            run_name,
            "weights",
            "best.pt"
        )

        if not os.path.isfile(best_path):
            raise RuntimeError(
                "Training finished but best.pt was not found."
            )

        shutil.copy2(
            best_path,
            MODEL_PATH
        )

        TRAINING_STATE["status"] = "completed"
        TRAINING_STATE["progress"] = 100
        TRAINING_STATE["message"] = (
            "Training completed successfully. "
            "bucket_best.pt is ready."
        )
        TRAINING_STATE["finished_at"] = (
            datetime.now().isoformat()
        )

    except Exception as e:

        traceback.print_exc()

        TRAINING_STATE["status"] = "error"
        TRAINING_STATE["message"] = str(e)
        TRAINING_STATE["error"] = str(e)
        TRAINING_STATE["finished_at"] = (
            datetime.now().isoformat()
        )

    finally:
        if TRAIN_LOCK.locked():
            TRAIN_LOCK.release()


def start_training():

    if not TRAIN_LOCK.acquire(blocking=False):
        return {
            "ok": False,
            "message": "Training is already running."
        }

    thread = threading.Thread(
        target=train_model,
        daemon=True
    )

    thread.start()

    return {
        "ok": True,
        "message": "Training started."
    }


# ============================================================
# DETECTION
# ============================================================

def detect_image(image_bytes):
    if YOLO is None:
        raise RuntimeError(
            "Ultralytics is not installed."
        )

    model = load_detection_model()

    temp_name = (
        f"/tmp/bucket_detect_{uuid.uuid4().hex}.jpg"
    )

    try:

        with open(temp_name, "wb") as f:
            f.write(image_bytes)

        results = model.predict(
            source=temp_name,
            conf=0.35,
            imgsz=640,
            verbose=False
        )

        detections = []

        for result in results:

            boxes = getattr(result, "boxes", None)

            if boxes is None:
                continue

            for box in boxes:

                try:
                    cls_id = int(
                        box.cls[0].item()
                    )

                    confidence = float(
                        box.conf[0].item()
                    )

                    xyxy = (
                        box.xyxy[0]
                        .cpu()
                        .numpy()
                        .tolist()
                    )

                    x1, y1, x2, y2 = xyxy

                    detections.append({
                        "class_id": cls_id,
                        "class_name": "BUCKET_LOADED",
                        "confidence": round(
                            confidence,
                            4
                        ),
                        "x1": round(x1, 2),
                        "y1": round(y1, 2),
                        "x2": round(x2, 2),
                        "y2": round(y2, 2),
                    })

                except Exception:
                    continue

        return {
            "count": len(detections),
            "detections": detections,
        }

    finally:

        try:
            if os.path.exists(temp_name):
                os.remove(temp_name)
        except Exception:
            pass


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
    content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no"
>

<title>NEERIKA BUCKET AI</title>

<style>

* {
    box-sizing: border-box;
}

body {
    margin: 0;
    font-family: Arial, Helvetica, sans-serif;
    background: #f3f4f6;
    color: #111827;
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
    margin: 6px 0 0;
    color: #d1d5db;
}

nav {
    display: flex;
    overflow-x: auto;
    gap: 6px;
    padding: 10px;
    background: white;
    border-bottom: 1px solid #ddd;
}

nav button {
    border: 0;
    background: #e5e7eb;
    padding: 10px 14px;
    border-radius: 8px;
    font-weight: bold;
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

.tab {
    display: none;
}

.tab.active {
    display: block;
}

.card {
    background: white;
    border-radius: 12px;
    padding: 16px;
    margin-bottom: 16px;
    box-shadow: 0 2px 8px rgba(0,0,0,.08);
}

.card h2 {
    margin-top: 0;
}

.grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 12px;
}

.stat {
    background: #111827;
    color: white;
    padding: 18px;
    border-radius: 12px;
}

.stat strong {
    display: block;
    font-size: 30px;
    margin-top: 5px;
}

button,
input,
select {
    font: inherit;
}

button {
    cursor: pointer;
}

.primary {
    background: #111827;
    color: white;
    border: 0;
    padding: 11px 16px;
    border-radius: 8px;
    font-weight: bold;
}

.success {
    background: #15803d;
    color: white;
    border: 0;
    padding: 11px 16px;
    border-radius: 8px;
    font-weight: bold;
}

.danger {
    background: #b91c1c;
    color: white;
    border: 0;
    padding: 8px 12px;
    border-radius: 7px;
}

input[type=file],
select {
    width: 100%;
    padding: 10px;
    border: 1px solid #ccc;
    border-radius: 8px;
    margin: 6px 0 12px;
}

.message {
    padding: 12px;
    border-radius: 8px;
    margin-top: 10px;
    background: #f3f4f6;
}

.ok {
    background: #dcfce7;
    color: #166534;
}

.error {
    background: #fee2e2;
    color: #991b1b;
}

.warning {
    background: #fef3c7;
    color: #92400e;
}


/* ========================================================
   ANNOTATION AREA
   ======================================================== */

.annotation-stage {
    position: relative;
    width: 100%;
    max-width: 950px;
    margin: 15px auto;
    background: #000;
    overflow: hidden;
    border-radius: 10px;

    /*
       VERY IMPORTANT:
       Prevents the phone browser from scrolling
       while the user is drawing with a finger.
    */
    touch-action: none;
    user-select: none;
    -webkit-user-select: none;
}

.annotation-stage img {
    display: block;
    width: 100%;
    height: auto;
    max-width: 100%;

    user-select: none;
    -webkit-user-select: none;
    -webkit-user-drag: none;
}

.annotation-stage canvas {
    position: absolute;
    left: 0;
    top: 0;

    width: 100%;
    height: 100%;

    touch-action: none;
    cursor: crosshair;

    user-select: none;
    -webkit-user-select: none;
}

.annotation-help {
    background: #eef2ff;
    border-left: 4px solid #4f46e5;
    padding: 12px;
    border-radius: 8px;
    margin: 10px 0;
}

.annotation-buttons {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    margin-top: 10px;
}

.box-list {
    margin-top: 12px;
}

.box-item {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 10px;
    padding: 9px;
    background: #f3f4f6;
    border-radius: 7px;
    margin-bottom: 6px;
}

.dataset-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
    gap: 12px;
}

.dataset-item {
    border: 1px solid #ddd;
    border-radius: 10px;
    overflow: hidden;
    background: white;
}

.dataset-item img {
    width: 100%;
    height: 140px;
    object-fit: cover;
    display: block;
}

.dataset-item-body {
    padding: 10px;
}

.dataset-name {
    font-size: 12px;
    word-break: break-all;
    margin-bottom: 7px;
}

.badge {
    display: inline-block;
    padding: 4px 7px;
    border-radius: 6px;
    font-size: 11px;
    font-weight: bold;
    margin-bottom: 8px;
}

.badge.ok {
    background: #dcfce7;
    color: #166534;
}

.badge.warning {
    background: #fef3c7;
    color: #92400e;
}

table {
    width: 100%;
    border-collapse: collapse;
}

th,
td {
    padding: 9px;
    border-bottom: 1px solid #ddd;
    text-align: left;
}

video {
    width: 100%;
    max-width: 700px;
    background: black;
    border-radius: 10px;
}

.camera-buttons {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    margin-top: 10px;
}

.small {
    font-size: 13px;
    color: #6b7280;
}

.hidden {
    display: none !important;
}

footer {
    text-align: center;
    padding: 30px;
    color: #777;
}

</style>

</head>

<body>

<header>

<h1>NEERIKA BUCKET AI</h1>

<p>Mining Production Bucket Counter</p>

</header>


<nav>

<button class="nav-btn active" data-tab="dashboard">
Dashboard
</button>

<button class="nav-btn" data-tab="camera">
Camera
</button>

<button class="nav-btn" data-tab="buckets">
Buckets
</button>

<button class="nav-btn" data-tab="training">
Training
</button>

<button class="nav-btn" data-tab="history">
History
</button>

<button class="nav-btn" data-tab="settings">
Settings
</button>

</nav>


<main>


<!-- =====================================================
     DASHBOARD
====================================================== -->

<section id="dashboard" class="tab active">

<div class="card">

<h2>Dashboard</h2>

<div class="grid">

<div class="stat">
Today
<strong id="todayCount">0</strong>
</div>

<div class="stat">
Total Buckets
<strong id="totalCount">0</strong>
</div>

<div class="stat">
Model
<strong id="modelStatus">Not trained</strong>
</div>

</div>

</div>

</section>


<!-- =====================================================
     CAMERA
====================================================== -->

<section id="camera" class="tab">

<div class="card">

<h2>Camera Bucket Counter</h2>

<p>
Camera detects only the trained
<strong>BUCKET_LOADED</strong> class.
</p>

<video
    id="cameraVideo"
    autoplay
    playsinline
>
</video>

<div class="camera-buttons">

<button
    class="primary"
    onclick="startCamera()"
>
Start Camera
</button>

<button
    class="danger"
    onclick="stopCamera()"
>
Stop Camera
</button>

<button
    class="success"
    onclick="captureAndDetect()"
>
Detect Buckets
</button>

</div>

<div
    id="cameraResult"
    class="message"
>
Camera ready.
</div>

</div>

</section>


<!-- =====================================================
     BUCKETS
====================================================== -->

<section id="buckets" class="tab">

<div class="card">

<h2>Bucket Reference Photos</h2>

<p class="small">
Upload reference photos of the bucket type used at the mine.
</p>

<input
    type="file"
    id="referenceFile"
    accept="image/*"
>

<button
    class="primary"
    onclick="uploadReference()"
>
Upload Reference Photo
</button>

<div
    id="referenceMessage"
    class="message"
>
Ready.
</div>

</div>

<div
    id="referenceList"
    class="card"
>

<h3>Reference Photos</h3>

<div id="references"></div>

</div>

</section>


<!-- =====================================================
     TRAINING
====================================================== -->

<section id="training" class="tab">

<div class="card">

<h2>YOLO Training Dataset</h2>

<div class="annotation-help">

<strong>Hatua 1:</strong>
Upload picha ya bucket.

<br>

<strong>Hatua 2:</strong>
Chagua picha hapa chini na bonyeza
<strong>Annotate</strong>.

<br>

<strong>Hatua 3:</strong>
Tumia kidole au mouse kuchora rectangle kuzunguka
<strong>loaded bucket</strong>.

<br>

<strong>Muhimu:</strong>
Ukiachia kidole, rectangle
<strong>HAITAONDOKA</strong>.
Itabaki mpaka uifute au u-save labels.

</div>


<label>
Dataset Split
</label>

<select id="trainingSplit">

<option value="train">
Training
</option>

<option value="val">
Validation
</option>

</select>


<label>
Choose Image
</label>

<input
    type="file"
    id="trainingFile"
    accept="image/*"
>

<button
    class="primary"
    onclick="uploadTrainingImage()"
>
Upload Image
</button>


<div
    id="trainingUploadMessage"
    class="message"
>
Ready.
</div>

</div>


<!-- =====================================================
     DATASET INFORMATION
====================================================== -->

<div class="card">

<h2>Dataset Information</h2>

<div
    id="datasetStats"
>
Loading...
</div>

</div>


<!-- =====================================================
     IMAGE LIST
====================================================== -->

<div class="card">

<h2>Uploaded Images</h2>

<p class="small">
Choose an image and annotate the loaded bucket.
</p>

<div
    id="datasetImages"
    class="dataset-grid"
>
Loading...
</div>

</div>


<!-- =====================================================
     ANNOTATOR
====================================================== -->

<div
    id="annotationCard"
    class="card hidden"
>

<h2>Draw Bucket Annotation</h2>

<div class="annotation-help">

Draw a rectangle around each
<strong>loaded ore/material bucket</strong>.

<br>

You can draw multiple boxes.

<br>

Use your finger on a phone or mouse on a computer.

<br>

<strong>The line will remain after releasing your finger.</strong>

</div>


<div
    id="annotationFileName"
    class="small"
>
No image selected.
</div>


<div
    id="annotationStage"
    class="annotation-stage"
>

<img
    id="annotationImage"
    draggable="false"
    alt="Training image"
>

<canvas
    id="annotationCanvas"
>
</canvas>

</div>


<div
    id="annotationInfo"
    class="message"
>
Boxes: 0
</div>


<div
    class="annotation-buttons"
>

<button
    class="primary"
    onclick="undoLastBox()"
>
Undo Last
</button>

<button
    class="danger"
    onclick="clearAnnotations()"
>
Clear All
</button>

<button
    class="success"
    onclick="saveAnnotations()"
>
Save YOLO Labels
</button>

</div>


<div
    id="boxList"
    class="box-list"
>
</div>


<div
    id="annotationMessage"
    class="message"
>
Draw a rectangle around the bucket.
</div>

</div>


<!-- =====================================================
     TRAINING STATUS
====================================================== -->

<div class="card">

<h2>Training Status</h2>

<div id="trainingStatus">
Status: idle
</div>

<div
    id="trainingProgress"
    class="message"
>
Progress: 0%
</div>

<br>

<button
    id="startTrainingButton"
    class="success"
    onclick="startTraining()"
>
Start YOLO Training
</button>

</div>

</section>


<!-- =====================================================
     HISTORY
====================================================== -->

<section id="history" class="tab">

<div class="card">

<h2>Bucket Count History</h2>

<button
    class="primary"
    onclick="loadHistory()"
>
Refresh
</button>

<div
    style="overflow-x:auto;margin-top:15px;"
>

<table>

<thead>

<tr>
<th>ID</th>
<th>Count</th>
<th>Date</th>
<th>Shift</th>
<th>Created</th>
</tr>

</thead>

<tbody id="historyBody">

</tbody>

</table>

</div>

</div>

</section>


<!-- =====================================================
     SETTINGS
====================================================== -->

<section id="settings" class="tab">

<div class="card">

<h2>Settings</h2>

<p>
Application:
<strong>NEERIKA BUCKET AI</strong>
</p>

<p>
Class:
<strong>BUCKET_LOADED</strong>
</p>

<p>
Class ID:
<strong>0</strong>
</p>

<p>
Database:
<strong>Supabase PostgreSQL</strong>
</p>

<p>
Training:
<strong>YOLO</strong>
</p>

</div>

</section>


</main>


<footer>
Geology &amp; Mining Services
</footer>


<script>

/* ========================================================
   GLOBAL STATE
======================================================== */

let cameraStream = null;

let annotation = {
    filename: null,
    split: "train",
    boxes: [],
    drawing: false,
    startX: 0,
    startY: 0,
    current: null
};


/* ========================================================
   NAVIGATION
======================================================== */

document.querySelectorAll(".nav-btn").forEach(function(button) {

    button.addEventListener("click", function() {

        document.querySelectorAll(".nav-btn")
            .forEach(function(btn) {
                btn.classList.remove("active");
            });

        document.querySelectorAll(".tab")
            .forEach(function(tab) {
                tab.classList.remove("active");
            });

        button.classList.add("active");

        const tab = document.getElementById(
            button.dataset.tab
        );

        if (tab) {
            tab.classList.add("active");
        }

        if (button.dataset.tab === "training") {
            loadTrainingImages();
            loadDatasetStats();
            loadTrainingStatus();
        }

        if (button.dataset.tab === "history") {
            loadHistory();
        }

        if (button.dataset.tab === "buckets") {
            loadReferences();
        }

    });

});


/* ========================================================
   BASIC API
======================================================== */

async function apiJSON(url, options) {

    const response = await fetch(
        url,
        options || {}
    );

    const text = await response.text();

    let data;

    try {
        data = JSON.parse(text);
    } catch (e) {
        throw new Error(
            text || "Invalid server response."
        );
    }

    if (!response.ok || data.ok === false) {
        throw new Error(
            data.message || "Request failed."
        );
    }

    return data;
}


/* ========================================================
   DASHBOARD
======================================================== */

async function loadDashboard() {

    try {

        const data = await apiJSON(
            "/api/dashboard"
        );

        document.getElementById(
            "todayCount"
        ).textContent = data.today;

        document.getElementById(
            "totalCount"
        ).textContent = data.total;

        document.getElementById(
            "modelStatus"
        ).textContent = data.model
            ? "Ready"
            : "Not trained";

    } catch (error) {

        console.error(error);

    }
}


/* ========================================================
   CAMERA
======================================================== */

async function startCamera() {

    const video = document.getElementById(
        "cameraVideo"
    );

    const result = document.getElementById(
        "cameraResult"
    );

    try {

        cameraStream =
            await navigator.mediaDevices.getUserMedia({
                video: {
                    facingMode: {
                        ideal: "environment"
                    }
                },
                audio: false
            });

        video.srcObject = cameraStream;

        result.className = "message ok";
        result.textContent =
            "Camera started.";

    } catch (error) {

        result.className = "message error";
        result.textContent =
            "Camera error: " + error.message;
    }
}


function stopCamera() {

    if (cameraStream) {

        cameraStream
            .getTracks()
            .forEach(function(track) {
                track.stop();
            });

        cameraStream = null;
    }

    document.getElementById(
        "cameraVideo"
    ).srcObject = null;

    document.getElementById(
        "cameraResult"
    ).textContent = "Camera stopped.";
}


async function captureAndDetect() {

    const video = document.getElementById(
        "cameraVideo"
    );

    const result = document.getElementById(
        "cameraResult"
    );

    if (!video.videoWidth) {

        result.className = "message error";
        result.textContent =
            "Start camera first.";

        return;
    }

    const canvas =
        document.createElement("canvas");

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    const ctx = canvas.getContext("2d");

    ctx.drawImage(
        video,
        0,
        0,
        canvas.width,
        canvas.height
    );

    canvas.toBlob(async function(blob) {

        try {

            result.className = "message";
            result.textContent =
                "Detecting...";

            const form =
                new FormData();

            form.append(
                "image",
                blob,
                "camera.jpg"
            );

            const data =
                await apiJSON(
                    "/api/detect",
                    {
                        method: "POST",
                        body: form
                    }
                );

            result.className = "message ok";

            result.textContent =
                "Detected loaded buckets: "
                + data.count;

            if (data.count > 0) {

                await apiJSON(
                    "/api/count",
                    {
                        method: "POST",
                        headers: {
                            "Content-Type":
                                "application/json"
                        },
                        body: JSON.stringify({
                            count: data.count,
                            shift: "A"
                        })
                    }
                );

                loadDashboard();
            }

        } catch (error) {

            result.className = "message error";

            result.textContent =
                error.message;
        }

    }, "image/jpeg", 0.9);
}


/* ========================================================
   REFERENCE IMAGES
======================================================== */

async function uploadReference() {

    const input =
        document.getElementById(
            "referenceFile"
        );

    const message =
        document.getElementById(
            "referenceMessage"
        );

    if (!input.files.length) {

        message.className =
            "message error";

        message.textContent =
            "Choose an image first.";

        return;
    }

    const form =
        new FormData();

    form.append(
        "image",
        input.files[0]
    );

    try {

        const data =
            await apiJSON(
                "/api/reference/upload",
                {
                    method: "POST",
                    body: form
                }
            );

        message.className =
            "message ok";

        message.textContent =
            data.message;

        input.value = "";

        loadReferences();

    } catch (error) {

        message.className =
            "message error";

        message.textContent =
            error.message;
    }
}


async function loadReferences() {

    const container =
        document.getElementById(
            "references"
        );

    try {

        const data =
            await apiJSON(
                "/api/references"
            );

        if (!data.references.length) {

            container.innerHTML =
                "<p>No reference photos yet.</p>";

            return;
        }

        container.innerHTML =
            data.references
                .map(function(file) {

                    return `
                    <div style="margin-bottom:15px;">
                        <img
                            src="${file.url}"
                            style="
                                width:180px;
                                max-width:100%;
                                border-radius:8px;
                            "
                        >
                        <div class="small">
                            ${file.filename}
                        </div>
                    </div>
                    `;

                })
                .join("");

    } catch (error) {

        container.textContent =
            error.message;
    }
}


/* ========================================================
   DATASET UPLOAD
======================================================== */

document.getElementById(
    "trainingSplit"
).addEventListener(
    "change",
    function() {

        annotation.split = this.value;

        loadTrainingImages();
    }
);


async function uploadTrainingImage() {

    const input =
        document.getElementById(
            "trainingFile"
        );

    const split =
        document.getElementById(
            "trainingSplit"
        ).value;

    const message =
        document.getElementById(
            "trainingUploadMessage"
        );

    if (!input.files.length) {

        message.className =
            "message error";

        message.textContent =
            "Choose an image first.";

        return;
    }

    const form =
        new FormData();

    form.append(
        "image",
        input.files[0]
    );

    form.append(
        "split",
        split
    );

    try {

        message.className =
            "message";

        message.textContent =
            "Uploading image...";

        const data =
            await apiJSON(
                "/api/dataset/upload",
                {
                    method: "POST",
                    body: form
                }
            );

        message.className =
            "message ok";

        message.textContent =
            "Image uploaded. Now draw the bucket box.";

        input.value = "";

        await loadDatasetStats();

        await loadTrainingImages();

        openAnnotation({
            filename: data.filename,
            split: data.split,
            label_exists: false,
            url: data.url
        });

    } catch (error) {

        message.className =
            "message error";

        message.textContent =
            error.message;
    }
}


/* ========================================================
   DATASET LIST
======================================================== */

async function loadTrainingImages() {

    const split =
        document.getElementById(
            "trainingSplit"
        ).value;

    annotation.split = split;

    const container =
        document.getElementById(
            "datasetImages"
        );

    container.innerHTML =
        "Loading...";

    try {

        const data =
            await apiJSON(
                "/api/dataset/list?split="
                + encodeURIComponent(split)
            );

        if (!data.images.length) {

            container.innerHTML =
                "<p>No images uploaded yet.</p>";

            return;
        }

        container.innerHTML = "";

        data.images.forEach(function(item) {

            const card =
                document.createElement("div");

            card.className =
                "dataset-item";

            const img =
                document.createElement("img");

            img.src =
                item.url
                + "?v="
                + Date.now();

            img.alt =
                item.filename;

            const body =
                document.createElement("div");

            body.className =
                "dataset-item-body";

            const name =
                document.createElement("div");

            name.className =
                "dataset-name";

            name.textContent =
                item.filename;

            const badge =
                document.createElement("span");

            badge.className =
                item.label_exists
                    ? "badge ok"
                    : "badge warning";

            badge.textContent =
                item.label_exists
                    ? "LABELED"
                    : "NEEDS ANNOTATION";

            const button =
                document.createElement("button");

            button.className =
                "primary";

            button.textContent =
                "Annotate";

            button.style.width =
                "100%";

            button.onclick =
                function() {
                    openAnnotation(item);
                };

            body.appendChild(name);
            body.appendChild(badge);
            body.appendChild(button);

            card.appendChild(img);
            card.appendChild(body);

            container.appendChild(card);

        });

    } catch (error) {

        container.innerHTML =
            '<div class="message error">'
            + error.message
            + "</div>";
    }
}


/* ========================================================
   OPEN ANNOTATOR
======================================================== */

function openAnnotation(item) {

    const card =
        document.getElementById(
            "annotationCard"
        );

    const image =
        document.getElementById(
            "annotationImage"
        );

    const canvas =
        document.getElementById(
            "annotationCanvas"
        );

    const fileName =
        document.getElementById(
            "annotationFileName"
        );

    annotation = {
        filename: item.filename,
        split: item.split,
        boxes: [],
        drawing: false,
        startX: 0,
        startY: 0,
        current: null
    };

    fileName.textContent =
        "Image: "
        + item.filename
        + " | Split: "
        + item.split;

    card.classList.remove("hidden");

    /*
       Cache-buster ensures the browser loads
       the latest uploaded image.
    */

    image.onload = function() {

        canvas.width =
            image.naturalWidth;

        canvas.height =
            image.naturalHeight;

        redrawAnnotations();

        updateAnnotationInfo();
    };

    image.src =
        item.url
        + "?v="
        + Date.now();

    card.scrollIntoView({
        behavior: "smooth",
        block: "start"
    });

    document.getElementById(
        "annotationMessage"
    ).className = "message";

    document.getElementById(
        "annotationMessage"
    ).textContent =
        "Now draw a rectangle around the loaded bucket.";
}


/* ========================================================
   CANVAS COORDINATES
======================================================== */

function getCanvasPoint(event) {

    const canvas =
        document.getElementById(
            "annotationCanvas"
        );

    const rect =
        canvas.getBoundingClientRect();

    const scaleX =
        canvas.width / rect.width;

    const scaleY =
        canvas.height / rect.height;

    return {
        x:
            (event.clientX - rect.left)
            * scaleX,

        y:
            (event.clientY - rect.top)
            * scaleY
    };
}


/* ========================================================
   POINTER DOWN
======================================================== */

document.getElementById(
    "annotationCanvas"
).addEventListener(
    "pointerdown",
    function(event) {

        event.preventDefault();

        if (!annotation.filename) {
            return;
        }

        const point =
            getCanvasPoint(event);

        annotation.drawing = true;

        annotation.startX =
            point.x;

        annotation.startY =
            point.y;

        annotation.current = {
            x1: point.x,
            y1: point.y,
            x2: point.x,
            y2: point.y
        };

        /*
           Pointer capture is VERY IMPORTANT
           on mobile. It keeps the drawing alive
           while the finger moves.
        */

        try {
            this.setPointerCapture(
                event.pointerId
            );
        } catch (e) {
        }

        redrawAnnotations();

    },
    { passive: false }
);


/* ========================================================
   POINTER MOVE
======================================================== */

document.getElementById(
    "annotationCanvas"
).addEventListener(
    "pointermove",
    function(event) {

        event.preventDefault();

        if (!annotation.drawing) {
            return;
        }

        const point =
            getCanvasPoint(event);

        annotation.current = {
            x1: annotation.startX,
            y1: annotation.startY,
            x2: point.x,
            y2: point.y
        };

        redrawAnnotations();

    },
    { passive: false }
);


/* ========================================================
   POINTER UP
======================================================== */

document.getElementById(
    "annotationCanvas"
).addEventListener(
    "pointerup",
    function(event) {

        event.preventDefault();

        if (!annotation.drawing) {
            return;
        }

        const point =
            getCanvasPoint(event);

        let x1 =
            Math.min(
                annotation.startX,
                point.x
            );

        let y1 =
            Math.min(
                annotation.startY,
                point.y
            );

        let x2 =
            Math.max(
                annotation.startX,
                point.x
            );

        let y2 =
            Math.max(
                annotation.startY,
                point.y
            );

        const width =
            x2 - x1;

        const height =
            y2 - y1;

        annotation.drawing = false;

        annotation.current = null;

        /*
           Ignore accidental tiny touches.
        */

        if (
            width >= 8 &&
            height >= 8
        ) {

            /*
               THIS IS THE IMPORTANT PART:
               The completed box is pushed into
               annotation.boxes.

               Therefore it stays after finger release.
            */

            annotation.boxes.push({
                x1: x1,
                y1: y1,
                x2: x2,
                y2: y2
            });

        }

        try {
            this.releasePointerCapture(
                event.pointerId
            );
        } catch (e) {
        }

        redrawAnnotations();

        updateAnnotationInfo();

    },
    { passive: false }
);


/* ========================================================
   POINTER CANCEL
======================================================== */

document.getElementById(
    "annotationCanvas"
).addEventListener(
    "pointercancel",
    function(event) {

        annotation.drawing = false;
        annotation.current = null;

        redrawAnnotations();

    },
    { passive: false }
);


/* ========================================================
   DRAW ALL BOXES
======================================================== */

function redrawAnnotations() {

    const canvas =
        document.getElementById(
            "annotationCanvas"
        );

    const ctx =
        canvas.getContext("2d");

    if (!canvas.width || !canvas.height) {
        return;
    }

    ctx.clearRect(
        0,
        0,
        canvas.width,
        canvas.height
    );

    const lineWidth =
        Math.max(
            3,
            canvas.width / 500
        );

    const fontSize =
        Math.max(
            16,
            canvas.width / 45
        );

    /*
       Draw saved boxes first.
    */

    annotation.boxes.forEach(
        function(box, index) {

            drawBox(
                ctx,
                box,
                index + 1,
                lineWidth,
                fontSize
            );

        }
    );

    /*
       Draw current box while finger
       is still moving.
    */

    if (annotation.current) {

        drawBox(
            ctx,
            annotation.current,
            annotation.boxes.length + 1,
            lineWidth,
            fontSize,
            true
        );

    }
}


/* ========================================================
   DRAW SINGLE BOX
======================================================== */

function drawBox(
    ctx,
    box,
    number,
    lineWidth,
    fontSize,
    temporary
) {

    const x =
        Math.min(
            box.x1,
            box.x2
        );

    const y =
        Math.min(
            box.y1,
            box.y2
        );

    const width =
        Math.abs(
            box.x2 - box.x1
        );

    const height =
        Math.abs(
            box.y2 - box.y1
        );

    ctx.save();

    ctx.lineWidth =
        lineWidth;

    ctx.strokeStyle =
        temporary
            ? "#f59e0b"
            : "#22c55e";

    ctx.fillStyle =
        temporary
            ? "rgba(245,158,11,0.15)"
            : "rgba(34,197,94,0.15)";

    ctx.fillRect(
        x,
        y,
        width,
        height
    );

    ctx.strokeRect(
        x,
        y,
        width,
        height
    );

    ctx.font =
        "bold "
        + fontSize
        + "px Arial";

    ctx.fillStyle =
        temporary
            ? "#f59e0b"
            : "#22c55e";

    ctx.fillText(
        "BUCKET "
        + number,
        x + 5,
        Math.max(
            fontSize + 4,
            y + fontSize
        )
    );

    ctx.restore();
}


/* ========================================================
   ANNOTATION INFO
======================================================== */

function updateAnnotationInfo() {

    document.getElementById(
        "annotationInfo"
    ).textContent =
        "Boxes: "
        + annotation.boxes.length;
}


/* ========================================================
   UNDO
======================================================== */

function undoLastBox() {

    if (!annotation.boxes.length) {
        return;
    }

    annotation.boxes.pop();

    redrawAnnotations();

    updateAnnotationInfo();
}


/* ========================================================
   CLEAR
======================================================== */

function clearAnnotations() {

    annotation.boxes = [];

    annotation.current = null;

    annotation.drawing = false;

    redrawAnnotations();

    updateAnnotationInfo();

    document.getElementById(
        "annotationMessage"
    ).textContent =
        "All boxes cleared. Draw again.";
}


/* ========================================================
   SAVE YOLO LABELS
======================================================== */

async function saveAnnotations() {

    const message =
        document.getElementById(
            "annotationMessage"
        );

    if (!annotation.filename) {

        message.className =
            "message error";

        message.textContent =
            "No image selected.";

        return;
    }

    if (!annotation.boxes.length) {

        message.className =
            "message error";

        message.textContent =
            "Draw at least one bucket box first.";

        return;
    }

    const canvas =
        document.getElementById(
            "annotationCanvas"
        );

    const boxes =
        annotation.boxes.map(
            function(box) {

                const x1 =
                    Math.min(
                        box.x1,
                        box.x2
                    );

                const y1 =
                    Math.min(
                        box.y1,
                        box.y2
                    );

                const x2 =
                    Math.max(
                        box.x1,
                        box.x2
                    );

                const y2 =
                    Math.max(
                        box.y1,
                        box.y2
                    );

                return {
                    x_center:
                        ((x1 + x2) / 2)
                        / canvas.width,

                    y_center:
                        ((y1 + y2) / 2)
                        / canvas.height,

                    width:
                        (x2 - x1)
                        / canvas.width,

                    height:
                        (y2 - y1)
                        / canvas.height
                };

            }
        );

    try {

        message.className =
            "message";

        message.textContent =
            "Saving YOLO labels...";

        const data =
            await apiJSON(
                "/api/dataset/labels",
                {
                    method: "POST",
                    headers: {
                        "Content-Type":
                            "application/json"
                    },
                    body: JSON.stringify({
                        filename:
                            annotation.filename,

                        split:
                            annotation.split,

                        boxes:
                            boxes
                    })
                }
            );

        message.className =
            "message ok";

        message.textContent =
            "Saved successfully: "
            + data.label_filename
            + " | "
            + annotation.boxes.length
            + " bucket(s).";

        await loadDatasetStats();

        await loadTrainingImages();

    } catch (error) {

        message.className =
            "message error";

        message.textContent =
            error.message;
    }
}


/* ========================================================
   DATASET STATS
======================================================== */

async function loadDatasetStats() {

    const container =
        document.getElementById(
            "datasetStats"
        );

    try {

        const data =
            await apiJSON(
                "/api/dataset/stats"
            );

        const s = data.stats;

        container.innerHTML = `
            <div class="grid">

                <div class="stat">
                    Training Images
                    <strong>
                        ${s.training_images}
                    </strong>
                </div>

                <div class="stat">
                    Validation Images
                    <strong>
                        ${s.validation_images}
                    </strong>
                </div>

                <div class="stat">
                    Training Labels
                    <strong>
                        ${s.training_labels}
                    </strong>
                </div>

                <div class="stat">
                    Validation Labels
                    <strong>
                        ${s.validation_labels}
                    </strong>
                </div>

            </div>

            <div class="message warning">
                Unlabeled Training Images:
                <strong>
                    ${s.training_unlabeled}
                </strong>
                <br>
                Unlabeled Validation Images:
                <strong>
                    ${s.validation_unlabeled}
                </strong>
            </div>
        `;

    } catch (error) {

        container.innerHTML =
            '<div class="message error">'
            + error.message
            + "</div>";
    }
}


/* ========================================================
   TRAINING
======================================================== */

async function startTraining() {

    const button =
        document.getElementById(
            "startTrainingButton"
        );

    try {

        button.disabled = true;

        const data =
            await apiJSON(
                "/api/training/start",
                {
                    method: "POST"
                }
            );

        alert(data.message);

        loadTrainingStatus();

    } catch (error) {

        alert(error.message);

        button.disabled = false;
    }
}


async function loadTrainingStatus() {

    const status =
        document.getElementById(
            "trainingStatus"
        );

    const progress =
        document.getElementById(
            "trainingProgress"
        );

    const button =
        document.getElementById(
            "startTrainingButton"
        );

    try {

        const data =
            await apiJSON(
                "/api/training/status"
            );

        const s =
            data.training;

        status.textContent =
            "Status: "
            + s.status
            + " | "
            + s.message;

        progress.textContent =
            "Progress: "
            + s.progress
            + "%";

        if (s.status === "training") {

            button.disabled = true;

            setTimeout(
                loadTrainingStatus,
                2500
            );

        } else {

            button.disabled = false;
        }

    } catch (error) {

        status.textContent =
            error.message;

        button.disabled = false;
    }
}


/* ========================================================
   HISTORY
======================================================== */

async function loadHistory() {

    const body =
        document.getElementById(
            "historyBody"
        );

    body.innerHTML =
        "<tr><td colspan='5'>Loading...</td></tr>";

    try {

        const data =
            await apiJSON(
                "/api/history"
            );

        if (!data.history.length) {

            body.innerHTML =
                "<tr><td colspan='5'>No records.</td></tr>";

            return;
        }

        body.innerHTML = "";

        data.history.forEach(
            function(row) {

                const tr =
                    document.createElement(
                        "tr"
                    );

                const values = [
                    row.id,
                    row.bucket_count,
                    row.count_date || "",
                    row.shift || "",
                    row.created_at || ""
                ];

                values.forEach(
                    function(value) {

                        const td =
                            document.createElement(
                                "td"
                            );

                        td.textContent =
                            value;

                        tr.appendChild(td);

                    }
                );

                body.appendChild(tr);

            }
        );

    } catch (error) {

        body.innerHTML =
            "<tr><td colspan='5'>"
            + error.message
            + "</td></tr>";
    }
}


/* ========================================================
   INITIAL LOAD
======================================================== */

loadDashboard();

loadDatasetStats();

loadTrainingStatus();

loadTrainingImages();

</script>

</body>

</html>
"""


# ============================================================
# HTTP HANDLER
# ============================================================

class Handler(BaseHTTPRequestHandler):

    def log_message(self, format, *args):
        print(
            "%s - %s"
            % (
                self.address_string(),
                format % args
            )
        )


    # ========================================================
    # RESPONSE HELPERS
    # ========================================================

    def send_json(
        self,
        data,
        status=200
    ):

        body = json.dumps(
            data,
            default=str
        ).encode("utf-8")

        self.send_response(status)

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

        self.end_headers()

        self.wfile.write(body)


    def send_text(
        self,
        text,
        status=200,
        content_type="text/plain; charset=utf-8"
    ):

        body = text.encode("utf-8")

        self.send_response(status)

        self.send_header(
            "Content-Type",
            content_type
        )

        self.send_header(
            "Content-Length",
            str(len(body))
        )

        self.end_headers()

        self.wfile.write(body)


    def send_file(
        self,
        path,
        content_type=None
    ):

        if not os.path.isfile(path):

            self.send_error(
                404,
                "File not found"
            )

            return

        size = os.path.getsize(path)

        if content_type is None:
            content_type =  content_type_for(path)

        self.send_response(200)

        self.send_header(
            "Content-Type",
            content_type
        )

        self.send_header(
            "Content-Length",
            str(size)
        )

        self.send_header(
            "Cache-Control",
            "no-cache"
        )

        self.end_headers()

        with open(path, "rb") as f:

            while True:

                chunk = f.read(1024 * 64)

                if not chunk:
                    break

                self.wfile.write(chunk)


    def read_json(self):

        length = int(
            self.headers.get(
                "Content-Length",
                "0"
            )
        )

        if length > 5 * 1024 * 1024:
            raise ValueError(
                "Request too large."
            )

        body = self.rfile.read(length)

        if not body:
            return {}

        return json.loads(
            body.decode("utf-8")
        )


    # ========================================================
    # GET
    # ========================================================

    def do_GET(self):

        try:

            parsed =   urlparse(self.path)

            path =parsed.path

            # -----------------------------
            # MAIN APP
            # -----------------------------

            if path == "/":

                self.send_text(
                    HTML_PAGE,
                    200,
                    "text/html; charset=utf-8"
                )

                return


            # -----------------------------
            # DASHBOARD
            # -----------------------------

            if path == "/api/dashboard":

                dashboard =  get_dashboard_data()

                self.send_json({
                    "ok": True,
                    **dashboard,
                    "model":
                        trained_model_exists()
                })

                return


            # -----------------------------
            # HISTORY
            # -----------------------------

            if path == "/api/history":

                history =  get_history()

                self.send_json({
                    "ok": True,
                    "history": history
                })

                return


            # -----------------------------
            # DATASET STATS
            # -----------------------------

            if path == "/api/dataset/stats":

                self.send_json({
                    "ok": True,
                    "stats":
                        dataset_stats()
                })

                return


            # -----------------------------
            # DATASET IMAGE LIST
            # -----------------------------

            if path == "/api/dataset/list":

                query =  parse_qs(
                        parsed.query
                    )

                split =  query.get(
                        "split",
                        ["train"]
                    )[0]

                if split not in (
                    "train",
                    "val"
                ):
                    split = "train"

                self.send_json({
                    "ok": True,
                    "images":
                        dataset_image_list(split)
                })

                return


            # -----------------------------
            # DATASET IMAGE
            # -----------------------------

            if path.startswith("/dataset/"):

                parts = path.split("/")

                if len(parts) < 4:

                    self.send_error(
                        404
                    )

                    return

                split =  parts[2]

                filename =  safe_filename(
                        parts[3]
                    )

                if split == "train":

                    image_path =  os.path.join(
                            TRAIN_IMAGES_DIR,
                            filename
                        )

                elif split == "val":

                    image_path =   os.path.join(
                            VAL_IMAGES_DIR,
                            filename
                        )

                else:

                    self.send_error(
                        404
                    )

                    return

                self.send_file(
                    image_path
                )

                return


            # -----------------------------
            # REFERENCES
            # -----------------------------

            if path == "/api/references":

                refs = []

                for filename in sorted(
                    os.listdir(
                        REFERENCE_DIR
                    )
                ):

                    if not is_image(filename):
                        continue

                    refs.append({
                        "filename": filename,
                        "url":
                            "/references/"
                            + filename
                    })

                self.send_json({
                    "ok": True,
                    "references": refs
                })

                return


            # -----------------------------
            # REFERENCE FILE
            # -----------------------------

            if path.startswith(
                "/references/"
            ):

                filename =safe_filename(
                        path.split(
                            "/references/",
                            1
                        )[1]
                    )

                file_path =   os.path.join(
                        REFERENCE_DIR,
                        filename
                    )

                self.send_file(
                    file_path
                )

                return


            # -----------------------------
            # TRAINING STATUS
            # -----------------------------

            if path == "/api/training/status":

                self.send_json({
                    "ok": True,
                    "training":
                        TRAINING_STATE
                })

                return


            # -----------------------------
            # HEALTH
            # -----------------------------

            if path == "/health":

                self.send_json({
                    "ok": True,
                    "service":
                        "NEERIKA BUCKET AI"
                })

                return


            self.send_error(
                404,
                "Not found"
            )

        except Exception as e:

            traceback.print_exc()

            try:

                self.send_json(
                    {
                        "ok": False,
                        "message": str(e)
                    },
                    500
                )

            except Exception:
                pass


    # ========================================================
    # POST
    # ========================================================

    def do_POST(self):

        try:

            parsed =urlparse(self.path)

            path =  parsed.path


            # =================================================
            # SAVE COUNT
            # =================================================

            if path == "/api/count":

                data = self.read_json()

                count =   int(
                        data.get(
                            "count",
                            0
                        )
                    )

                shift =str(
                        data.get(
                            "shift",
                            "A"
                        )
                    )

                if count < 0:
                    raise ValueError(
                        "Count cannot be negative."
                    )

                if count > 10000:
                    raise ValueError(
                        "Count is too large."
                    )

                saved =  save_bucket_count(
                        count,
                        shift
                    )

                if not saved:

                    raise RuntimeError(
                        "Failed to save count to database."
                    )

                self.send_json({
                    "ok": True,
                    "message":
                        "Bucket count saved.",
                    "count":
                        count
                })

                return


            # =================================================
            # REFERENCE UPLOAD
            # =================================================

            if path == "/api/reference/upload":

                files = parse_multipart(self)

                image_file = None

                for item in files:

                    if item["field"] == "image":
                        image_file = item
                        break

                if image_file is None:
                    raise ValueError(
                        "No image uploaded."
                    )

                filename =  unique_filename(
                        image_file["filename"]
                    )

                if not is_image(filename):
                    raise ValueError(
                        "Only image files are allowed."
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
                        image_file["data"]
                    )

                self.send_json({
                    "ok": True,
                    "message":
                        "Reference photo uploaded.",
                    "filename":
                        filename
                })

                return


            # =================================================
            # DATASET IMAGE UPLOAD
            # =================================================

            if path == "/api/dataset/upload":

                files =   parse_multipart(self)

                image_file = None
                split = "train"

                for item in files:

                    if item["field"] == "image":
                        image_file = item

                # Read split from normal multipart field
                # if browser sent it as a non-file part.
                #
                # The simple multipart parser above focuses
                # on files, so infer train by default.
                #
                # Frontend sends split through query too
                # in the fallback below.

                query =   parse_qs(
                        parsed.query
                    )

                if query.get("split"):
                    split =query["split"][0]

                # Because browser sends split as multipart
                # field, inspect the raw request is not
                # available anymore here. The frontend
                # therefore also gets the split from a
                # custom header.

                header_split =  self.headers.get(
                        "X-Dataset-Split",
                        ""
                    )

                if header_split in (
                    "train",
                    "val"
                ):
                    split = header_split

                if split not in (
                    "train",
                    "val"
                ):
                    split = "train"

                if image_file is None:
                    raise ValueError(
                        "No image uploaded."
                    )

                filename =unique_filename(
                        image_file["filename"]
                    )

                if not is_image(filename):
                    raise ValueError(
                        "Only JPG, JPEG, PNG and WEBP images are allowed."
                    )

                images_dir, labels_dir = get_split_dirs(split)

                destination = os.path.join(
                        images_dir,
                        filename
                    )

                with open(
                    destination,
                    "wb"
                ) as f:

                    f.write(
                        image_file["data"]
                    )

                self.send_json({
                    "ok": True,
                    "message":
                        "Training image uploaded.",
                    "filename":
                        filename,
                    "split":
                        split,
                    "url":
                        "/dataset/"
                        + split
                        + "/"
                        + filename
                })

                return


            # =================================================
            # SAVE MULTIPLE YOLO LABELS
            # =================================================

            if path == "/api/dataset/labels":

                data = self.read_json()

                filename = data.get(
                        "filename"
                    )

                split =  data.get(
                        "split",
                        "train"
                    )

                boxes =  data.get(
                        "boxes"
                    )

                if not filename:
                    raise ValueError(
                        "filename is required."
                    )

                label_filename =  save_yolo_labels(
                        filename,
                        boxes,
                        split
                    )

                self.send_json({
                    "ok": True,
                    "message":
                        "YOLO labels saved successfully.",
                    "label_filename":
                        label_filename,
                    "box_count":
                        len(boxes)
                })

                return


            # =================================================
            # LEGACY SINGLE LABEL API
            # =================================================

            if path == "/api/dataset/label":

                data = self.read_json()

                filename =  data.get(
                        "filename"
                    )

                split = data.get(
                        "split",
                        "train"
                    )

                boxes = [{
                    "x_center":
                        data.get("x_center"),
                    "y_center":
                        data.get("y_center"),
                    "width":
                        data.get("width"),
                    "height":
                        data.get("height"),
                }]

                label_filename =  save_yolo_labels(
                        filename,
                        boxes,
                        split
                    )

                self.send_json({
                    "ok": True,
                    "message":
                        "YOLO label saved.",
                    "label_filename":
                        label_filename
                })

                return


            # =================================================
            # START TRAINING
            # =================================================

            if path == "/api/training/start":

                result =  start_training()

                self.send_json(
                    result,
                    200 if result["ok"]
                    else 409
                )

                return


            # =================================================
            # DETECT
            # =================================================

            if path == "/api/detect":

                files =  parse_multipart(self)

                image_file = None

                for item in files:

                    if item["field"] == "image":
                        image_file = item
                        break

                if image_file is None:
                    raise ValueError(
                        "No image uploaded."
                    )

                result =   detect_image(
                        image_file["data"]
                    )

                self.send_json({
                    "ok": True,
                    **result
                })

                return


            self.send_error(
                404,
                "Not found"
            )

        except Exception as e:

            traceback.print_exc()

            try:

                self.send_json(
                    {
                        "ok": False,
                        "message": str(e)
                    },
                    500
                )

            except Exception:
                pass


# ============================================================
# FIX DATASET UPLOAD SPLIT HEADER
# ============================================================
#
# The frontend needs to tell the server whether the image
# belongs to train or val. Patch fetch by intercepting the
# upload function's request header through the following
# small replacement.
#
# Instead of modifying the large HTML above manually, the
# backend also accepts X-Dataset-Split.
#
# The browser function above sends the multipart form but
# does not yet add the header. We therefore modify the HTML
# string before serving it.


HTML_PAGE = HTML_PAGE.replace(
    'method: "POST",\n                    body: form',
    'method: "POST",\n                    headers: {\n                        "X-Dataset-Split": split\n                    },\n                    body: form',
    1
)


# ============================================================
# START SERVER
# ============================================================

def main():

    print("=" * 60)
    print("NEERIKA BUCKET AI")
    print("Mining Production Bucket Counter")
    print("=" * 60)

    print(
        "PORT:",
        PORT
    )

    print(
        "DATABASE:",
        "configured"
        if DATABASE_URL
        else "NOT CONFIGURED"
    )

    print(
        "MODEL:",
        "READY"
        if trained_model_exists()
        else "NOT TRAINED"
    )

    init_database()

    server =     ThreadingHTTPServer(
            ("0.0.0.0", PORT),
            Handler
        )

    print(
        f"Server running on port {PORT}"
    )

    server.serve_forever()


if __name__ == "__main__":
    main()
