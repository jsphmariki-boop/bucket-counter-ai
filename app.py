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
from datetime import datetime, timezone
from urllib.parse import urlparse, parse_qs
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# ============================================================
# NEERIKA BUCKET AI
# Mining Production Bucket Counter
# Full replacement app.py
# ============================================================

os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/Ultralytics")

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
    PSYCOPG2_AVAILABLE = True
except Exception:
    psycopg2 = None
    RealDictCursor = None
    PSYCOPG2_AVAILABLE = False

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except Exception:
    YOLO = None
    YOLO_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except Exception:
    Image = None
    PIL_AVAILABLE = False


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

AI_DIR = os.path.join(BASE_DIR, "ai")
DATASET_DIR = os.path.join(AI_DIR, "dataset")

IMAGES_DIR = os.path.join(DATASET_DIR, "images")
LABELS_DIR = os.path.join(DATASET_DIR, "labels")

TRAIN_IMAGES = os.path.join(IMAGES_DIR, "train")
VAL_IMAGES = os.path.join(IMAGES_DIR, "val")

TRAIN_LABELS = os.path.join(LABELS_DIR, "train")
VAL_LABELS = os.path.join(LABELS_DIR, "val")

MODEL_DIR = os.path.join(AI_DIR, "models")
MODEL_OUTPUT = os.path.join(MODEL_DIR, "bucket_best.pt")

DATASET_YAML = os.path.join(AI_DIR, "bucket_dataset.yaml")

UPLOAD_DIR = os.path.join(AI_DIR, "uploads")

for folder in [
    AI_DIR,
    DATASET_DIR,
    IMAGES_DIR,
    LABELS_DIR,
    TRAIN_IMAGES,
    VAL_IMAGES,
    TRAIN_LABELS,
    VAL_LABELS,
    MODEL_DIR,
    UPLOAD_DIR,
]:
    os.makedirs(folder, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

PORT = int(os.environ.get("PORT", "10000"))

EPOCHS = 20
IMAGE_SIZE = 320
BATCH_SIZE = 1
WORKERS = 0
DEVICE = "cpu"

CLASS_ID = 0
CLASS_NAME = "BUCKET_LOADED"

DATABASE_URL = os.environ.get("DATABASE_URL", "").strip()

MODEL_DOWNLOAD = "yolo11n.pt"


# ============================================================
# TRAINING STATE
# ============================================================

training_lock = threading.Lock()

TRAINING_STATE = {
    "status": "idle",
    "message": "Ready",
    "progress": 0,
    "epoch": 0,
    "epochs": EPOCHS,
    "error": "",
    "model": "",
    "started_at": "",
    "finished_at": "",
    "pid": os.getpid(),
}


# ============================================================
# DATABASE
# ============================================================

def db_connection():
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL haijawekwa kwenye Render Environment Variables.")

    return psycopg2.connect(
        DATABASE_URL,
        connect_timeout=15,
        sslmode="require"
    )


def db_execute(sql, params=None, fetch=False, fetchone=False):
    conn = None

    try:
        conn = db_connection()

        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, params or ())

            result = None

            if fetchone:
                result = cur.fetchone()

            elif fetch:
                result = cur.fetchall()

            conn.commit()
            return result

    finally:
        if conn:
            conn.close()


def init_database():
    if not PSYCOPG2_AVAILABLE:
        print("WARNING: psycopg2 haipo.")
        return

    if not DATABASE_URL:
        print("WARNING: DATABASE_URL haijawekwa.")
        return

    try:
        conn = db_connection()

        with conn.cursor() as cur:

            cur.execute("""
                CREATE TABLE IF NOT EXISTS bucket_counts (
                    id BIGSERIAL PRIMARY KEY,
                    bucket_type TEXT,
                    count INTEGER DEFAULT 1,
                    shift TEXT,
                    operator_name TEXT,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)

            cur.execute("""
                CREATE TABLE IF NOT EXISTS training_dataset (
                    id BIGSERIAL PRIMARY KEY,
                    filename TEXT NOT NULL,
                    split TEXT DEFAULT 'train',
                    image_path TEXT,
                    label_path TEXT,
                    annotations INTEGER DEFAULT 0,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)

        conn.commit()
        conn.close()

        print("Database initialized.")

    except Exception as e:
        print("Database initialization warning:", e)


# ============================================================
# UTILS
# ============================================================

def now_iso():
    return datetime.now(timezone.utc).isoformat()


def safe_filename(filename):
    filename = os.path.basename(filename or "")

    filename = re.sub(
        r"[^A-Za-z0-9._-]+",
        "_",
        filename
    )

    if not filename:
        filename = "image.jpg"

    return filename


def json_response(handler, data, status=200):
    raw = json.dumps(
        data,
        ensure_ascii=False
    ).encode("utf-8")

    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(raw)))
    handler.send_header("Cache-Control", "no-store")
    handler.end_headers()

    handler.wfile.write(raw)


def html_response(handler, html, status=200):
    raw = html.encode("utf-8")

    handler.send_response(status)
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


def file_response(handler, path):
    if not os.path.exists(path):
        handler.send_error(404)
        return

    ext = os.path.splitext(path)[1].lower()

    content_type = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
        ".txt": "text/plain",
        ".csv": "text/csv",
        ".pt": "application/octet-stream",
    }.get(
        ext,
        "application/octet-stream"
    )

    size = os.path.getsize(path)

    handler.send_response(200)
    handler.send_header(
        "Content-Type",
        content_type
    )
    handler.send_header(
        "Content-Length",
        str(size)
    )
    handler.end_headers()

    with open(path, "rb") as f:
        shutil.copyfileobj(f, handler.wfile)


# ============================================================
# DATASET HELPERS
# ============================================================

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp"
}


def is_image(filename):
    return os.path.splitext(filename)[1].lower() in IMAGE_EXTENSIONS


def image_path_for_split(split, filename):
    filename = safe_filename(filename)

    if split == "val":
        return os.path.join(
            VAL_IMAGES,
            filename
        )

    return os.path.join(
        TRAIN_IMAGES,
        filename
    )


def label_path_for_split(split, filename):
    filename = safe_filename(filename)

    base = os.path.splitext(filename)[0] + ".txt"

    if split == "val":
        return os.path.join(
            VAL_LABELS,
            base
        )

    return os.path.join(
        TRAIN_LABELS,
        base
    )


def get_annotation_count(split, filename):
    path = label_path_for_split(
        split,
        filename
    )

    if not os.path.exists(path):
        return 0

    try:
        with open(
            path,
            "r",
            encoding="utf-8"
        ) as f:
            lines = [
                x.strip()
                for x in f
                if x.strip()
            ]

        valid = 0

        for line in lines:
            parts = line.split()

            if len(parts) >= 5:
                try:
                    cls = int(float(parts[0]))

                    if cls == CLASS_ID:
                        valid += 1

                except Exception:
                    pass

        return valid

    except Exception:
        return 0


def has_valid_bucket_label(split, filename):
    return get_annotation_count(
        split,
        filename
    ) > 0


def find_training_images_with_labels():
    result = []

    if not os.path.exists(TRAIN_IMAGES):
        return result

    for filename in sorted(os.listdir(TRAIN_IMAGES)):

        if not is_image(filename):
            continue

        if has_valid_bucket_label(
            "train",
            filename
        ):
            result.append(filename)

    return result


def find_validation_images_with_labels():
    result = []

    if not os.path.exists(VAL_IMAGES):
        return result

    for filename in sorted(os.listdir(VAL_IMAGES)):

        if not is_image(filename):
            continue

        if has_valid_bucket_label(
            "val",
            filename
        ):
            result.append(filename)

    return result


# ============================================================
# IMPORTANT:
# REBUILD LABELS FROM DATABASE IF NECESSARY
# ============================================================

def ensure_label_file_from_database(
    filename,
    split="train"
):
    """
    The browser saves annotation data through /api/dataset/labels.

    This function is a safety check.

    If the physical TXT file exists and contains class 0,
    nothing is changed.

    It does NOT invent annotations.
    """

    path = label_path_for_split(
        split,
        filename
    )

    if not os.path.exists(path):
        return False

    return has_valid_bucket_label(
        split,
        filename
    )


def validate_training_dataset():
    """
    Strict dataset validation.

    Returns:
        {
            ok,
            train_images,
            val_images,
            train_count,
            val_count,
            message
        }
    """

    train_images = find_training_images_with_labels()

    if len(train_images) == 0:

        return {
            "ok": False,
            "train_images": [],
            "val_images": [],
            "train_count": 0,
            "val_count": 0,
            "message": (
                "Hakuna picha yenye BUCKET_LOADED annotation. "
                "Hakikisha ume-save YOLO labels baada ya kuchora box."
            )
        }

    val_images = find_validation_images_with_labels()

    return {
        "ok": True,
        "train_images": train_images,
        "val_images": val_images,
        "train_count": len(train_images),
        "val_count": len(val_images),
        "message": "Dataset iko tayari."
    }


# ============================================================
# DATASET YAML
# ============================================================

def write_dataset_yaml():

    os.makedirs(AI_DIR, exist_ok=True)

    yaml_text = f"""path: {DATASET_DIR}
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
        f.write(yaml_text)

    return DATASET_YAML


# ============================================================
# VALIDATION FALLBACK
# ============================================================

def create_validation_fallback():

    """
    Render Free test mode:

    If user has training images but no validation images,
    copy one labeled training image to validation.

    This is ONLY a pipeline fallback.
    For serious model training, upload separate validation images.
    """

    val_existing = find_validation_images_with_labels()

    if val_existing:
        return val_existing

    train_existing = find_training_images_with_labels()

    if not train_existing:
        return []

    source_name = train_existing[0]

    source_image = os.path.join(
        TRAIN_IMAGES,
        source_name
    )

    source_label = os.path.join(
        TRAIN_LABELS,
        os.path.splitext(source_name)[0] + ".txt"
    )

    val_image = os.path.join(
        VAL_IMAGES,
        source_name
    )

    val_label = os.path.join(
        VAL_LABELS,
        os.path.splitext(source_name)[0] + ".txt"
    )

    try:

        shutil.copy2(
            source_image,
            val_image
        )

        shutil.copy2(
            source_label,
            val_label
        )

        print(
            "Validation fallback created:",
            source_name
        )

        return [source_name]

    except Exception as e:

        print(
            "Validation fallback failed:",
            e
        )

        return []


# ============================================================
# YOLO CALLBACK
# ============================================================

def update_training_epoch(epoch_number, total_epochs):

    try:

        epoch_number = int(epoch_number)
        total_epochs = int(total_epochs)

        TRAINING_STATE["epoch"] = epoch_number

        # 10% reserved for preparation.
        # Remaining 90% belongs to epochs.
        progress = 10 + int(
            (epoch_number / max(total_epochs, 1))
            * 90
        )

        progress = max(
            10,
            min(
                progress,
                99
            )
        )

        TRAINING_STATE["progress"] = progress

        TRAINING_STATE["message"] = (
            f"Training YOLO... "
            f"Epoch {epoch_number}/{total_epochs}"
        )

    except Exception:
        pass


def training_callback(trainer):

    try:

        epoch = int(
            getattr(
                trainer,
                "epoch",
                0
            )
        )

        # Ultralytics epoch is zero based.
        current = epoch + 1

        total = int(
            getattr(
                trainer,
                "epochs",
                EPOCHS
            )
        )

        update_training_epoch(
            current,
            total
        )

        print(
            f"NEERIKA TRAINING EPOCH "
            f"{current}/{total}"
        )

    except Exception as e:

        print(
            "Training callback warning:",
            e
        )


# ============================================================
# TRAINING WORKER
# ============================================================

def train_yolo_worker():

    try:

        TRAINING_STATE["status"] = "preparing"
        TRAINING_STATE["progress"] = 1
        TRAINING_STATE["epoch"] = 0
        TRAINING_STATE["error"] = ""
        TRAINING_STATE["started_at"] = now_iso()
        TRAINING_STATE["finished_at"] = ""
        TRAINING_STATE["model"] = ""

        print("=" * 60)
        print("NEERIKA YOLO TRAINING START")
        print("=" * 60)

        if not YOLO_AVAILABLE:
            raise RuntimeError(
                "Ultralytics haija-install."
            )

        TRAINING_STATE["message"] = (
            "Checking BUCKET_LOADED annotations..."
        )

        TRAINING_STATE["progress"] = 3

        dataset = validate_training_dataset()

        print(
            "TRAINING DATASET:",
            json.dumps(
                dataset,
                ensure_ascii=False
            )
        )

        if not dataset["ok"]:

            raise RuntimeError(
                dataset["message"]
            )

        TRAINING_STATE["message"] = (
            f"Found {dataset['train_count']} "
            f"labeled training image(s)."
        )

        TRAINING_STATE["progress"] = 5

        # ----------------------------------------------------
        # Validation fallback
        # ----------------------------------------------------

        val_images = create_validation_fallback()

        if not val_images:
            raise RuntimeError(
                "Hakuna validation image yenye "
                "BUCKET_LOADED annotation."
            )

        TRAINING_STATE["message"] = (
            f"Validation ready: "
            f"{len(val_images)} image(s)."
        )

        TRAINING_STATE["progress"] = 8

        # ----------------------------------------------------
        # Dataset YAML
        # ----------------------------------------------------

        write_dataset_yaml()

        print(
            "Dataset YAML:",
            DATASET_YAML
        )

        TRAINING_STATE["progress"] = 10

        # ----------------------------------------------------
        # Load model
        # ----------------------------------------------------

        TRAINING_STATE["status"] = "loading_model"

        TRAINING_STATE["message"] = (
            "Loading YOLO model..."
        )

        print(
            "Loading YOLO model:",
            MODEL_DOWNLOAD
        )

        model = YOLO(
            MODEL_DOWNLOAD
        )

        TRAINING_STATE["progress"] = 12

        # ----------------------------------------------------
        # Register callback
        # ----------------------------------------------------

        try:
            model.add_callback(
                "on_train_epoch_end",
                training_callback
            )

            print(
                "Registered callback: "
                "on_train_epoch_end"
            )

        except Exception as e:

            print(
                "Callback registration warning:",
                e
            )

        # ----------------------------------------------------
        # Training directory
        # ----------------------------------------------------

        run_dir = os.path.join(
            AI_DIR,
            "bucket_training"
        )

        os.makedirs(
            run_dir,
            exist_ok=True
        )

        # ----------------------------------------------------
        # Start training
        # ----------------------------------------------------

        TRAINING_STATE["status"] = "training"

        TRAINING_STATE["message"] = (
            f"Training YOLO for {EPOCHS} epochs..."
        )

        TRAINING_STATE["progress"] = 12

        print("=" * 60)
        print("STARTING YOLO TRAINING")
        print("DEVICE:", DEVICE)
        print("IMAGE SIZE:", IMAGE_SIZE)
        print("BATCH:", BATCH_SIZE)
        print("WORKERS:", WORKERS)
        print("EPOCHS:", EPOCHS)
        print("TRAIN IMAGES:", dataset["train_count"])
        print("VAL IMAGES:", len(val_images))
        print("=" * 60)

        results = model.train(

            data=DATASET_YAML,

            epochs=EPOCHS,

            imgsz=IMAGE_SIZE,

            batch=BATCH_SIZE,

            workers=WORKERS,

            device=DEVICE,

            project=AI_DIR,

            name="bucket_training",

            exist_ok=True,

            pretrained=True,

            verbose=True,

            cache=False,

            amp=False,

            plots=False,

            save=True,

            val=True,

            patience=100
        )

        # ----------------------------------------------------
        # Training finished
        # ----------------------------------------------------

        print(
            "YOLO TRAINING FINISHED"
        )

        TRAINING_STATE["progress"] = 95

        TRAINING_STATE["message"] = (
            "Training finished. Searching for best.pt..."
        )

        # ----------------------------------------------------
        # Find best.pt
        # ----------------------------------------------------

        candidates = [

            os.path.join(
                AI_DIR,
                "bucket_training",
                "weights",
                "best.pt"
            ),

            os.path.join(
                AI_DIR,
                "bucket_training",
                "weights",
                "last.pt"
            ),

            os.path.join(
                run_dir,
                "weights",
                "best.pt"
            ),

            os.path.join(
                run_dir,
                "weights",
                "last.pt"
            ),
        ]

        best_source = None

        for candidate in candidates:

            if os.path.exists(candidate):

                best_source = candidate

                print(
                    "Found model:",
                    candidate
                )

                break

        if not best_source:

            raise RuntimeError(
                "Training imekamilika lakini "
                "best.pt/last.pt haikupatikana."
            )

        os.makedirs(
            MODEL_DIR,
            exist_ok=True
        )

        shutil.copy2(
            best_source,
            MODEL_OUTPUT
        )

        print(
            "Model copied to:",
            MODEL_OUTPUT
        )

        TRAINING_STATE["status"] = "completed"

        TRAINING_STATE["progress"] = 100

        TRAINING_STATE["epoch"] = EPOCHS

        TRAINING_STATE["message"] = (
            "Training completed successfully."
        )

        TRAINING_STATE["model"] = MODEL_OUTPUT

        TRAINING_STATE["finished_at"] = now_iso()

        print("=" * 60)
        print("NEERIKA TRAINING COMPLETED")
        print("=" * 60)

    except Exception as e:

        error_text = (
            str(e)
            or traceback.format_exc()
        )

        print("=" * 60)
        print("YOLO TRAINING ERROR")
        print(error_text)
        print("=" * 60)

        traceback.print_exc()

        TRAINING_STATE["status"] = "error"

        TRAINING_STATE["progress"] = 0

        TRAINING_STATE["epoch"] = 0

        TRAINING_STATE["message"] = (
            "Training error: " + str(e)
        )

        TRAINING_STATE["error"] = traceback.format_exc()

        TRAINING_STATE["finished_at"] = now_iso()

    finally:

        training_lock.release()


# ============================================================
# START TRAINING
# ============================================================

def start_training():

    acquired = training_lock.acquire(
        blocking=False
    )

    if not acquired:

        return False, (
            "Training tayari inaendelea."
        )

    try:

        if TRAINING_STATE["status"] in [
            "preparing",
            "loading_model",
            "training"
        ]:

            training_lock.release()

            return False, (
                "Training tayari inaendelea."
            )

        # Reset immediately.
        TRAINING_STATE["status"] = "starting"
        TRAINING_STATE["message"] = (
            "Starting training..."
        )
        TRAINING_STATE["progress"] = 1
        TRAINING_STATE["epoch"] = 0
        TRAINING_STATE["error"] = ""

        thread = threading.Thread(
            target=train_yolo_worker,
            daemon=True
        )

        thread.start()

        return True, (
            "Training started."
        )

    except Exception:

        try:
            training_lock.release()
        except Exception:
            pass

        raise


# ============================================================
# DATASET STATS
# ============================================================

def dataset_stats():

    train_images = [
        f
        for f in os.listdir(TRAIN_IMAGES)
        if is_image(f)
    ]

    val_images = [
        f
        for f in os.listdir(VAL_IMAGES)
        if is_image(f)
    ]

    train_labeled = sum(
        1
        for f in train_images
        if has_valid_bucket_label(
            "train",
            f
        )
    )

    val_labeled = sum(
        1
        for f in val_images
        if has_valid_bucket_label(
            "val",
            f
        )
    )

    return {
        "training_images": len(train_images),
        "validation_images": len(val_images),
        "training_labels": train_labeled,
        "validation_labels": val_labeled,
        "unlabeled_training": (
            len(train_images) - train_labeled
        ),
        "unlabeled_validation": (
            len(val_images) - val_labeled
        )
    }


# ============================================================
# DATASET IMAGE LIST
# ============================================================

def dataset_images(split):

    directory = (
        VAL_IMAGES
        if split == "val"
        else TRAIN_IMAGES
    )

    results = []

    if not os.path.exists(directory):
        return results

    for filename in sorted(
        os.listdir(directory),
        reverse=True
    ):

        if not is_image(filename):
            continue

        results.append({
            "filename": filename,
            "split": split,
            "annotations": get_annotation_count(
                split,
                filename
            )
        })

    return results


# ============================================================
# MULTIPART PARSER
# ============================================================

def parse_multipart(handler):

    content_type = handler.headers.get(
        "Content-Type",
        ""
    )

    content_length = int(
        handler.headers.get(
            "Content-Length",
            "0"
        )
    )

    body = handler.rfile.read(
        content_length
    )

    fields = {}
    files = {}

    if "boundary=" not in content_type:
        return fields, files

    boundary = content_type.split(
        "boundary=",
        1
    )[1]

    boundary = boundary.strip()

    if boundary.startswith('"') and boundary.endswith('"'):
        boundary = boundary[1:-1]

    marker = (
        b"--" +
        boundary.encode()
    )

    parts = body.split(marker)

    for part in parts:

        if not part:
            continue

        if part in [
            b"--",
            b"--\r\n"
        ]:
            continue

        if part.startswith(b"\r\n"):
            part = part[2:]

        if part.endswith(b"\r\n"):
            part = part[:-2]

        header_end = part.find(
            b"\r\n\r\n"
        )

        if header_end == -1:
            continue

        raw_headers = part[
            :header_end
        ]

        content = part[
            header_end + 4:
        ]

        header_text = raw_headers.decode(
            "utf-8",
            errors="ignore"
        )

        disposition = ""

        for line in header_text.split(
            "\r\n"
        ):

            if line.lower().startswith(
                "content-disposition:"
            ):
                disposition = line

        name_match = re.search(
            r'name="([^"]+)"',
            disposition
        )

        if not name_match:
            continue

        field_name = name_match.group(1)

        filename_match = re.search(
            r'filename="([^"]*)"',
            disposition
        )

        if filename_match:

            filename = filename_match.group(1)

            files[field_name] = {
                "filename": filename,
                "content": content
            }

        else:

            fields[field_name] = content.decode(
                "utf-8",
                errors="replace"
            )

    return fields, files


# ============================================================
# HTML
# ============================================================

HTML = r"""
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

*{
box-sizing:border-box;
}

body{
margin:0;
font-family:Arial,Helvetica,sans-serif;
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
font-size:22px;
}

header p{
margin:5px 0 0;
opacity:.8;
}

nav{
display:flex;
gap:5px;
overflow:auto;
background:#1f2937;
padding:8px;
}

nav button{
border:0;
background:#374151;
color:white;
padding:11px 15px;
border-radius:7px;
cursor:pointer;
white-space:nowrap;
}

nav button.active{
background:#f59e0b;
color:#111;
}

main{
max-width:1100px;
margin:auto;
padding:15px;
}

.tab{
display:none;
}

.tab.active{
display:block;
}

.card{
background:white;
border-radius:12px;
padding:16px;
margin-bottom:15px;
box-shadow:0 2px 8px rgba(0,0,0,.08);
}

button{
padding:10px 14px;
border:0;
border-radius:7px;
background:#2563eb;
color:white;
cursor:pointer;
}

button.danger{
background:#dc2626;
}

button.success{
background:#16a34a;
}

button:disabled{
opacity:.5;
cursor:not-allowed;
}

input,select{
padding:10px;
border:1px solid #d1d5db;
border-radius:7px;
width:100%;
margin:5px 0 10px;
}

.progress{
height:22px;
background:#e5e7eb;
border-radius:20px;
overflow:hidden;
margin:10px 0;
}

.progress-bar{
height:100%;
width:0%;
background:#16a34a;
color:white;
text-align:center;
font-size:13px;
line-height:22px;
transition:width .3s;
}

.status{
padding:12px;
border-radius:8px;
background:#f3f4f6;
margin:10px 0;
}

table{
width:100%;
border-collapse:collapse;
}

th,td{
padding:9px;
border-bottom:1px solid #ddd;
text-align:left;
}

.small{
font-size:13px;
color:#6b7280;
}

.badge{
display:inline-block;
padding:4px 8px;
border-radius:20px;
font-size:12px;
background:#dcfce7;
color:#166534;
}

.error{
background:#fee2e2;
color:#991b1b;
}

.successbox{
background:#dcfce7;
color:#166534;
}

footer{
text-align:center;
padding:20px;
color:#777;
}

#preview{
max-width:100%;
margin-top:10px;
border-radius:10px;
}

</style>
</head>

<body>

<header>
<h1>NEERIKA BUCKET AI</h1>
<p>Mining Production Bucket Counter</p>
</header>

<nav>
<button onclick="showTab('dashboard')" id="nav-dashboard">Dashboard</button>
<button onclick="showTab('camera')" id="nav-camera">Camera</button>
<button onclick="showTab('buckets')" id="nav-buckets">Buckets</button>
<button onclick="showTab('training')" id="nav-training">Training</button>
<button onclick="showTab('history')" id="nav-history">History</button>
<button onclick="showTab('settings')" id="nav-settings">Settings</button>
</nav>

<main>

<section id="dashboard" class="tab">

<div class="card">

<h2>Dashboard</h2>

<p>
NEERIKA BUCKET AI counts loaded ore/material buckets.
</p>

<div id="dashboardStats">
Loading...
</div>

</div>

</section>


<section id="camera" class="tab">

<div class="card">

<h2>Camera</h2>

<p>
Camera counting interface.
</p>

<video
id="cameraVideo"
autoplay
playsinline
style="width:100%;max-width:700px;background:#111;border-radius:10px;"
></video>

<br><br>

<button onclick="startCamera()">
Start Camera
</button>

<button onclick="stopCamera()">
Stop Camera
</button>

</div>

</section>


<section id="buckets" class="tab">

<div class="card">

<h2>Bucket Types</h2>

<p>
Bucket reference management.
</p>

<p class="small">
Active bucket type: BUCKET_LOADED
</p>

</div>

</section>


<section id="training" class="tab">

<div class="card">

<h2>YOLO Training Dataset</h2>

<p>
Upload images and annotate them as
<strong>BUCKET_LOADED</strong>.
</p>

<form
id="uploadForm"
enctype="multipart/form-data"
>

<input
type="file"
name="image"
id="imageInput"
accept="image/*"
required
>

<select name="split" id="split">
<option value="train">Training</option>
<option value="val">Validation</option>
</select>

<button type="submit">
Upload Image
</button>

</form>

<div id="uploadMessage"></div>

</div>


<div class="card">

<h2>Training Status</h2>

<div
id="trainingStatus"
class="status"
>
Loading...
</div>

<div class="progress">
<div
id="trainingProgress"
class="progress-bar"
>
0%
</div>
</div>

<p id="epochText">
Epoch: 0 / 20
</p>

<button
id="startTraining"
class="success"
onclick="startTraining()"
>
Start Training
</button>

<button onclick="refreshTraining()">
Refresh
</button>

</div>


<div class="card">

<h2>Training Settings</h2>

<table>

<tr>
<td>CPU</td>
<td>CPU</td>
</tr>

<tr>
<td>Image Size</td>
<td>320</td>
</tr>

<tr>
<td>Batch</td>
<td>1</td>
</tr>

<tr>
<td>Workers</td>
<td>0</td>
</tr>

<tr>
<td>Epochs</td>
<td>20</td>
</tr>

</table>

<p class="small">
These settings are optimized for Render Free.
</p>

</div>


<div class="card">

<h2>Dataset</h2>

<div id="datasetStats">
Loading...
</div>

<div style="overflow:auto">

<table>

<thead>
<tr>
<th>Split</th>
<th>Filename</th>
<th>Annotations</th>
</tr>
</thead>

<tbody id="datasetTable">
</tbody>

</table>

</div>

</div>

</section>


<section id="history" class="tab">

<div class="card">

<h2>History</h2>

<div id="historyData">
Loading...
</div>

</div>

</section>


<section id="settings" class="tab">

<div class="card">

<h2>Settings</h2>

<p>
NEERIKA BUCKET AI
</p>

<p class="small">
Geology & Mining Services
</p>

</div>

</section>

</main>

<footer>
Geology & Mining Services
</footer>


<script>

let cameraStream = null;

function showTab(name){

document.querySelectorAll('.tab')
.forEach(x => x.classList.remove('active'));

document.querySelectorAll('nav button')
.forEach(x => x.classList.remove('active'));

const tab = document.getElementById(name);

const nav = document.getElementById(
'nav-' + name
);

if(tab){
tab.classList.add('active');
}

if(nav){
nav.classList.add('active');
}

if(name === 'training'){
loadDataset();
refreshTraining();
}

if(name === 'dashboard'){
loadDashboard();
}

}

async function api(url, options){

const response = await fetch(
url,
options
);

let data;

try{
data = await response.json();
}catch(e){
data = {
error:'Invalid server response'
};
}

if(!response.ok){
throw new Error(
data.error ||
data.message ||
'Request failed'
);
}

return data;
}


async function refreshTraining(){

try{

const data = await api(
'/api/training/status'
);

const status = document.getElementById(
'trainingStatus'
);

const progress = document.getElementById(
'trainingProgress'
);

const epoch = document.getElementById(
'epochText'
);

status.className = 'status';

if(data.status === 'error'){
status.classList.add('error');
}

if(data.status === 'completed'){
status.classList.add('successbox');
}

status.innerHTML =
'<strong>Status:</strong> ' +
escapeHtml(data.status || '') +
'<br>' +
escapeHtml(data.message || '');

const p = Number(
data.progress || 0
);

progress.style.width =
p + '%';

progress.textContent =
p + '%';

epoch.textContent =
'Epoch: ' +
Number(data.epoch || 0) +
' / ' +
Number(data.epochs || 20);

const button = document.getElementById(
'startTraining'
);

if(
data.status === 'training' ||
data.status === 'preparing' ||
data.status === 'loading_model' ||
data.status === 'starting'
){

button.disabled = true;
button.textContent =
'Training in progress...';

}else{

button.disabled = false;
button.textContent =
'Start Training';

}

}catch(e){

console.error(e);

}

}


async function startTraining(){

const button = document.getElementById(
'startTraining'
);

button.disabled = true;

try{

const data = await api(
'/api/training/start',
{
method:'POST'
}
);

alert(
data.message ||
'Training started.'
);

}catch(e){

alert(
e.message
);

button.disabled = false;

}

refreshTraining();

}


async function loadDataset(){

try{

const stats = await api(
'/api/dataset/stats'
);

document.getElementById(
'datasetStats'
).innerHTML = `

<p>
Training Images:
<strong>${stats.training_images}</strong>
</p>

<p>
Validation Images:
<strong>${stats.validation_images}</strong>
</p>

<p>
Training Labels:
<strong>${stats.training_labels}</strong>
</p>

<p>
Validation Labels:
<strong>${stats.validation_labels}</strong>
</p>

<p>
Unlabeled Training:
<strong>${stats.unlabeled_training}</strong>
</p>

<p>
Unlabeled Validation:
<strong>${stats.unlabeled_validation}</strong>
</p>

`;

const train = await api(
'/api/dataset/images?split=train'
);

const val = await api(
'/api/dataset/images?split=val'
);

const all = [
...(train.images || []),
...(val.images || [])
];

const tbody = document.getElementById(
'datasetTable'
);

tbody.innerHTML = '';

for(const item of all){

const tr = document.createElement('tr');

tr.innerHTML = `

<td>
${escapeHtml(item.split)}
</td>

<td>
${escapeHtml(item.filename)}
</td>

<td>
<span class="badge">
${item.annotations}
</span>
</td>

`;

tbody.appendChild(tr);

}

}catch(e){

console.error(e);

}

}


document.getElementById(
'uploadForm'
).addEventListener(
'submit',
async function(e){

e.preventDefault();

const form = new FormData(this);

const message =
document.getElementById(
'uploadMessage'
);

message.innerHTML =
'Uploading...';

try{

const data = await api(
'/api/dataset/upload',
{
method:'POST',
body:form
}
);

message.innerHTML =
'<div class="status successbox">' +
escapeHtml(
data.message ||
'Upload successful'
) +
'</div>';

this.reset();

loadDataset();

}catch(err){

message.innerHTML =
'<div class="status error">' +
escapeHtml(err.message) +
'</div>';

}

});


async function loadDashboard(){

try{

const stats = await api(
'/api/dataset/stats'
);

document.getElementById(
'dashboardStats'
).innerHTML = `

<p>
Training images:
<strong>${stats.training_images}</strong>
</p>

<p>
Training labels:
<strong>${stats.training_labels}</strong>
</p>

<p>
Validation images:
<strong>${stats.validation_images}</strong>
</p>

`;

}catch(e){

console.error(e);

}

}


async function startCamera(){

try{

cameraStream =
await navigator.mediaDevices.getUserMedia({
video:{
facingMode:{
ideal:'environment'
}
},
audio:false
});

document.getElementById(
'cameraVideo'
).srcObject =
cameraStream;

}catch(e){

alert(
'Camera error: ' +
e.message
);

}

}


function stopCamera(){

if(cameraStream){

cameraStream
.getTracks()
.forEach(
track => track.stop()
);

cameraStream = null;

document.getElementById(
'cameraVideo'
).srcObject = null;

}

}


function escapeHtml(value){

return String(value ?? '')
.replaceAll('&','&amp;')
.replaceAll('<','&lt;')
.replaceAll('>','&gt;')
.replaceAll('"','&quot;')
.replaceAll("'","&#039;");

}


showTab('training');

loadDataset();

refreshTraining();

setInterval(
refreshTraining,
1500
);

</script>

</body>
</html>
"""


# ============================================================
# HTTP HANDLER
# ============================================================

class Handler(BaseHTTPRequestHandler):

    protocol_version = "HTTP/1.1"

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

    # --------------------------------------------------------
    # HEAD
    # --------------------------------------------------------

    def do_HEAD(self):

        self.send_response(200)
        self.send_header(
            "Content-Length",
            "0"
        )
        self.end_headers()

    # --------------------------------------------------------
    # GET
    # --------------------------------------------------------

    def do_GET(self):

        try:

            parsed = urlparse(
                self.path
            )

            path = parsed.path

            query = parse_qs(
                parsed.query
            )

            # --------------------------------------------
            # HOME
            # --------------------------------------------

            if path == "/":

                html_response(
                    self,
                    HTML
                )

                return

            # --------------------------------------------
            # TRAINING STATUS
            # --------------------------------------------

            if path == "/api/training/status":

                json_response(
                    self,
                    {
                        **TRAINING_STATE
                    }
                )

                return

            # --------------------------------------------
            # DATASET STATS
            # --------------------------------------------

            if path == "/api/dataset/stats":

                json_response(
                    self,
                    dataset_stats()
                )

                return

            # --------------------------------------------
            # DATASET IMAGES
            # --------------------------------------------

            if path == "/api/dataset/images":

                split = query.get(
                    "split",
                    ["train"]
                )[0]

                if split not in [
                    "train",
                    "val"
                ]:
                    split = "train"

                json_response(
                    self,
                    {
                        "split": split,
                        "images": dataset_images(
                            split
                        )
                    }
                )

                return

            # --------------------------------------------
            # IMAGE
            # --------------------------------------------

            if path == "/api/dataset/image":

                split = query.get(
                    "split",
                    ["train"]
                )[0]

                filename = query.get(
                    "filename",
                    [""]
                )[0]

                file_path = image_path_for_split(
                    split,
                    filename
                )

                file_response(
                    self,
                    file_path
                )

                return

            # --------------------------------------------
            # LABELS GET
            # --------------------------------------------

            if path == "/api/dataset/labels":

                split = query.get(
                    "split",
                    ["train"]
                )[0]

                filename = query.get(
                    "filename",
                    [""]
                )[0]

                label_path = label_path_for_split(
                    split,
                    filename
                )

                boxes = []

                if os.path.exists(
                    label_path
                ):

                    with open(
                        label_path,
                        "r",
                        encoding="utf-8"
                    ) as f:

                        for line in f:

                            parts = line.strip().split()

                            if len(parts) < 5:
                                continue

                            try:

                                cls = int(
                                    float(parts[0])
                                )

                                x = float(parts[1])
                                y = float(parts[2])
                                w = float(parts[3])
                                h = float(parts[4])

                                boxes.append({
                                    "class_id": cls,
                                    "x": x,
                                    "y": y,
                                    "w": w,
                                    "h": h
                                })

                            except Exception:
                                continue

                json_response(
                    self,
                    {
                        "filename": filename,
                        "split": split,
                        "boxes": boxes
                    }
                )

                return

            # --------------------------------------------
            # MODEL
            # --------------------------------------------

            if path == "/api/model":

                if os.path.exists(
                    MODEL_OUTPUT
                ):

                    json_response(
                        self,
                        {
                            "exists": True,
                            "path": MODEL_OUTPUT,
                            "size": os.path.getsize(
                                MODEL_OUTPUT
                            )
                        }
                    )

                else:

                    json_response(
                        self,
                        {
                            "exists": False
                        }
                    )

                return

            self.send_error(
                404,
                "Not found"
            )

        except Exception as e:

            traceback.print_exc()

            json_response(
                self,
                {
                    "error": str(e)
                },
                500
            )

    # --------------------------------------------------------
    # POST
    # --------------------------------------------------------

    def do_POST(self):

        try:

            parsed = urlparse(
                self.path
            )

            path = parsed.path

            # --------------------------------------------
            # START TRAINING
            # --------------------------------------------

            if path == "/api/training/start":

                ok, message = start_training()

                if ok:

                    json_response(
                        self,
                        {
                            "success": True,
                            "message": message
                        }
                    )

                else:

                    json_response(
                        self,
                        {
                            "success": False,
                            "message": message
                        },
                        409
                    )

                return

            # --------------------------------------------
            # DATASET UPLOAD
            # --------------------------------------------

            if path == "/api/dataset/upload":

                fields, files = parse_multipart(
                    self
                )

                upload = files.get(
                    "image"
                )

                if not upload:

                    json_response(
                        self,
                        {
                            "error":
                            "Image haijatumwa."
                        },
                        400
                    )

                    return

                original = upload.get(
                    "filename",
                    "image.jpg"
                )

                filename = safe_filename(
                    original
                )

                split = fields.get(
                    "split",
                    "train"
                )

                if split not in [
                    "train",
                    "val"
                ]:
                    split = "train"

                # Add unique prefix so duplicate uploads
                # don't overwrite each other.

                base, ext = os.path.splitext(
                    filename
                )

                unique_name = (
                    f"{base}_"
                    f"{datetime.now().strftime('%Y%m%d%H%M%S')}_"
                    f"{uuid.uuid4().hex[:8]}"
                    f"{ext.lower()}"
                )

                destination = image_path_for_split(
                    split,
                    unique_name
                )

                with open(
                    destination,
                    "wb"
                ) as f:

                    f.write(
                        upload["content"]
                    )

                # Make sure corresponding label directory
                # exists.

                os.makedirs(
                    os.path.dirname(
                        label_path_for_split(
                            split,
                            unique_name
                        )
                    ),
                    exist_ok=True
                )

                # Register in DB if possible.

                if PSYCOPG2_AVAILABLE and DATABASE_URL:

                    try:

                        db_execute(
                            """
                            INSERT INTO training_dataset
                            (filename, split, image_path,
                             label_path, annotations)
                            VALUES (%s,%s,%s,%s,%s)
                            """,
                            (
                                unique_name,
                                split,
                                destination,
                                label_path_for_split(
                                    split,
                                    unique_name
                                ),
                                0
                            )
                        )

                    except Exception as db_error:

                        print(
                            "Dataset DB insert warning:",
                            db_error
                        )

                json_response(
                    self,
                    {
                        "success": True,
                        "message": (
                            "Image uploaded successfully. "
                            "Sasa fungua image kwenye annotation "
                            "workflow na save YOLO labels."
                        ),
                        "filename": unique_name,
                        "split": split
                    }
                )

                return

            # --------------------------------------------
            # SAVE LABELS
            # --------------------------------------------

            if path == "/api/dataset/labels":

                length = int(
                    self.headers.get(
                        "Content-Length",
                        "0"
                    )
                )

                raw = self.rfile.read(
                    length
                )

                data = json.loads(
                    raw.decode(
                        "utf-8"
                    )
                )

                filename = safe_filename(
                    data.get(
                        "filename",
                        ""
                    )
                )

                split = data.get(
                    "split",
                    "train"
                )

                if split not in [
                    "train",
                    "val"
                ]:
                    split = "train"

                boxes = data.get(
                    "boxes",
                    []
                )

                label_path = label_path_for_split(
                    split,
                    filename
                )

                image_path = image_path_for_split(
                    split,
                    filename
                )

                if not os.path.exists(
                    image_path
                ):

                    json_response(
                        self,
                        {
                            "error":
                            "Image haipo kwenye dataset."
                        },
                        404
                    )

                    return

                lines = []

                for box in boxes:

                    try:

                        class_id = int(
                            box.get(
                                "class_id",
                                CLASS_ID
                            )
                        )

                        # Only BUCKET_LOADED class.
                        class_id = CLASS_ID

                        x = float(
                            box["x"]
                        )

                        y = float(
                            box["y"]
                        )

                        w = float(
                            box["w"]
                        )

                        h = float(
                            box["h"]
                        )

                        # Clamp normalized coordinates.

                        x = max(
                            0.0,
                            min(
                                1.0,
                                x
                            )
                        )

                        y = max(
                            0.0,
                            min(
                                1.0,
                                y
                            )
                        )

                        w = max(
                            0.0001,
                            min(
                                1.0,
                                w
                            )
                        )

                        h = max(
                            0.0001,
                            min(
                                1.0,
                                h
                            )
                        )

                        lines.append(
                            f"{class_id} "
                            f"{x:.6f} "
                            f"{y:.6f} "
                            f"{w:.6f} "
                            f"{h:.6f}"
                        )

                    except Exception:
                        continue

                # IMPORTANT:
                # Write labels even if only one box.
                # Training requires at least one valid class 0.

                with open(
                    label_path,
                    "w",
                    encoding="utf-8"
                ) as f:

                    if lines:
                        f.write(
                            "\n".join(lines)
                            + "\n"
                        )

                annotation_count = len(
                    lines
                )

                # Update database.

                if PSYCOPG2_AVAILABLE and DATABASE_URL:

                    try:

                        existing = db_execute(
                            """
                            SELECT id
                            FROM training_dataset
                            WHERE filename=%s
                            AND split=%s
                            ORDER BY id DESC
                            LIMIT 1
                            """,
                            (
                                filename,
                                split
                            ),
                            fetchone=True
                        )

                        if existing:

                            db_execute(
                                """
                                UPDATE training_dataset
                                SET annotations=%s,
                                    image_path=%s,
                                    label_path=%s
                                WHERE id=%s
                                """,
                                (
                                    annotation_count,
                                    image_path,
                                    label_path,
                                    existing["id"]
                                )
                            )

                        else:

                            db_execute(
                                """
                                INSERT INTO training_dataset
                                (filename, split,
                                 image_path, label_path,
                                 annotations)
                                VALUES (%s,%s,%s,%s,%s)
                                """,
                                (
                                    filename,
                                    split,
                                    image_path,
                                    label_path,
                                    annotation_count
                                )
                            )

                    except Exception as db_error:

                        print(
                            "Label DB warning:",
                            db_error
                        )

                if annotation_count == 0:

                    json_response(
                        self,
                        {
                            "success": False,
                            "message":
                            "Hakuna BUCKET_LOADED box iliyohifadhiwa.",
                            "annotations": 0
                        },
                        400
                    )

                    return

                json_response(
                    self,
                    {
                        "success": True,
                        "message":
                        f"Saved successfully: "
                        f"{os.path.basename(label_path)} | "
                        f"{annotation_count} bucket(s).",
                        "annotations":
                        annotation_count
                    }
                )

                return

            # --------------------------------------------
            # BUCKET COUNT
            # --------------------------------------------

            if path == "/api/count":

                length = int(
                    self.headers.get(
                        "Content-Length",
                        "0"
                    )
                )

                raw = self.rfile.read(
                    length
                )

                data = json.loads(
                    raw.decode(
                        "utf-8"
                    )
                )

                bucket_type = data.get(
                    "bucket_type",
                    "BUCKET_LOADED"
                )

                shift = data.get(
                    "shift",
                    ""
                )

                operator_name = data.get(
                    "operator_name",
                    ""
                )

                count = int(
                    data.get(
                        "count",
                        1
                    )
                )

                if PSYCOPG2_AVAILABLE and DATABASE_URL:

                    db_execute(
                        """
                        INSERT INTO bucket_counts
                        (bucket_type,count,shift,operator_name)
                        VALUES (%s,%s,%s,%s)
                        """,
                        (
                            bucket_type,
                            count,
                            shift,
                            operator_name
                        )
                    )

                json_response(
                    self,
                    {
                        "success": True,
                        "message": "Bucket count saved."
                    }
                )

                return

            self.send_error(
                404,
                "Not found"
            )

        except Exception as e:

            traceback.print_exc()

            try:

                json_response(
                    self,
                    {
                        "success": False,
                        "error": str(e),
                        "trace": traceback.format_exc()
                    },
                    500
                )

            except Exception:
                pass


# ============================================================
# START SERVER
# ============================================================

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
    "PSYCOPG2_AVAILABLE:",
    PSYCOPG2_AVAILABLE
)

print(
    "CPU:",
    DEVICE
)

print(
    "IMAGE_SIZE:",
    IMAGE_SIZE
)

print(
    "BATCH_SIZE:",
    BATCH_SIZE
)

print(
    "WORKERS:",
    WORKERS
)

print(
    "EPOCHS:",
    EPOCHS
)

print(
    "PORT:",
    PORT
)

init_database()

server = ThreadingHTTPServer(
    ("0.0.0.0", PORT),
    Handler
)

print(
    f"NEERIKA BUCKET AI running on port {PORT}"
)

server.serve_forever()
