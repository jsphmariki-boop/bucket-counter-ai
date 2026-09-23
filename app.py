import os
import io
import csv
import json
import base64
import traceback
import threading
import time
import uuid
import shutil
from datetime import datetime
from urllib.parse import urlparse
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
except Exception:
    psycopg2 = None
    RealDictCursor = None

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

try:
    import cv2
except Exception:
    cv2 = None

try:
    import numpy as np
except Exception:
    np = None


# ============================================================
# BASIC CONFIGURATION
# ============================================================

HOST = "0.0.0.0"
PORT = int(os.environ.get("PORT", "10000"))

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

REFERENCE_DIR = os.path.join(BASE_DIR, "bucket_images")

MODEL_DIR = os.path.join(AI_DIR, "models")

MODEL_PATH = os.path.join(MODEL_DIR, "bucket_best.pt")
FALLBACK_MODEL_PATH = os.path.join(MODEL_DIR, "best.pt")

DATASET_YAML = os.path.join(DATASET_DIR, "dataset.yaml")

TRAIN_LOCK = threading.Lock()

TRAINING_STATE = {
    "status": "idle",
    "message": "Training has not started.",
    "epoch": 0,
    "epochs": 0,
    "progress": 0,
    "error": "",
    "started_at": None,
    "finished_at": None,
}

MODEL_CACHE = {
    "model": None,
    "path": None,
}


# ============================================================
# CREATE REQUIRED DIRECTORIES
# ============================================================

REQUIRED_DIRS = [
    AI_DIR,
    DATASET_DIR,
    IMAGES_DIR,
    LABELS_DIR,
    TRAIN_IMAGES_DIR,
    VAL_IMAGES_DIR,
    TRAIN_LABELS_DIR,
    VAL_LABELS_DIR,
    REFERENCE_DIR,
    MODEL_DIR,
]

for directory in REQUIRED_DIRS:
    os.makedirs(directory, exist_ok=True)


# ============================================================
# HELPERS
# ============================================================

def json_response(handler, data, status=200):
    body = json.dumps(data, ensure_ascii=False).encode("utf-8")

    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Cache-Control", "no-store")
    handler.end_headers()

    try:
        handler.wfile.write(body)
    except Exception:
        pass


def html_response(handler, html, status=200):
    body = html.encode("utf-8")

    handler.send_response(status)
    handler.send_header("Content-Type", "text/html; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Cache-Control", "no-store")
    handler.end_headers()

    try:
        handler.wfile.write(body)
    except Exception:
        pass


def bytes_response(handler, data, content_type="application/octet-stream"):
    handler.send_response(200)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Content-Length", str(len(data)))
    handler.send_header("Cache-Control", "no-cache")
    handler.end_headers()

    try:
        handler.wfile.write(data)
    except Exception:
        pass


def safe_filename(filename):
    filename = os.path.basename(filename or "")
    filename = filename.replace("\\", "_")
    filename = filename.replace("/", "_")

    allowed = (
        "abcdefghijklmnopqrstuvwxyz"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "0123456789"
        "-_."
    )

    cleaned = "".join(
        c if c in allowed else "_"
        for c in filename
    )

    if not cleaned:
        cleaned = "image"

    return cleaned


def unique_filename(filename):
    filename = safe_filename(filename)

    name, ext = os.path.splitext(filename)

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    return (
        f"{name}_{timestamp}_"
        f"{uuid.uuid4().hex[:8]}"
        f"{ext.lower()}"
    )


def is_image_filename(filename):
    return filename.lower().endswith(
        (
            ".jpg",
            ".jpeg",
            ".png",
            ".webp",
            ".bmp",
        )
    )


def image_content_type(filename):
    lower = filename.lower()

    if lower.endswith(".png"):
        return "image/png"

    if lower.endswith(".webp"):
        return "image/webp"

    if lower.endswith(".bmp"):
        return "image/bmp"

    return "image/jpeg"


def read_json_body(handler):
    try:
        length = int(
            handler.headers.get(
                "Content-Length",
                "0",
            )
        )
    except Exception:
        length = 0

    if length <= 0:
        return {}

    raw = handler.rfile.read(length)

    if not raw:
        return {}

    try:
        return json.loads(
            raw.decode("utf-8")
        )
    except Exception:
        return {}


def send_error_json(handler, message, status=400):
    json_response(
        handler,
        {
            "success": False,
            "error": str(message),
        },
        status,
    )


# ============================================================
# DATABASE
# ============================================================

def get_db_connection():
    if not DATABASE_URL:
        return None

    if psycopg2 is None:
        return None

    return psycopg2.connect(
        DATABASE_URL,
        connect_timeout=10,
    )


def ensure_database():
    if not DATABASE_URL:
        print("DATABASE_URL not configured.")
        return

    if psycopg2 is None:
        print("psycopg2 is not installed.")
        return

    connection = None

    try:
        connection = get_db_connection()

        cursor = connection.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS bucket_counts (
                id BIGSERIAL PRIMARY KEY,
                bucket_count INTEGER NOT NULL DEFAULT 0,
                count_date TEXT,
                shift TEXT,
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
            """
        )

        connection.commit()

        cursor.close()

        print("Database ready.")

    except Exception as exc:
        print("Database initialization error:")
        print(exc)

        if connection:
            connection.rollback()

    finally:
        if connection:
            connection.close()


def save_bucket_count(
    bucket_count,
    shift="",
):
    if not DATABASE_URL:
        return {
            "success": False,
            "error": "DATABASE_URL is not configured.",
        }

    connection = None

    try:
        connection = get_db_connection()

        cursor = connection.cursor()

        today = datetime.now().strftime("%Y-%m-%d")

        cursor.execute(
            """
            INSERT INTO bucket_counts
            (
                bucket_count,
                count_date,
                shift
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
                int(bucket_count),
                today,
                shift or "",
            ),
        )

        row = cursor.fetchone()

        connection.commit()

        cursor.close()

        return {
            "success": True,
            "id": row[0] if row else None,
        }

    except Exception as exc:
        if connection:
            connection.rollback()

        return {
            "success": False,
            "error": str(exc),
        }

    finally:
        if connection:
            connection.close()


def get_history(limit=100):
    if not DATABASE_URL:
        return []

    connection = None

    try:
        connection = get_db_connection()

        cursor = connection.cursor(
            cursor_factory=RealDictCursor
        )

        cursor.execute(
            """
            SELECT
                id,
                bucket_count,
                count_date::text AS count_date,
                shift,
                created_at::text AS created_at
            FROM bucket_counts
            ORDER BY id DESC
            LIMIT %s
            """,
            (int(limit),),
        )

        rows = cursor.fetchall()

        cursor.close()

        return [
            dict(row)
            for row in rows
        ]

    except Exception as exc:
        print("History error:", exc)
        return []

    finally:
        if connection:
            connection.close()


def get_total_count():
    if not DATABASE_URL:
        return 0

    connection = None

    try:
        connection = get_db_connection()

        cursor = connection.cursor()

        cursor.execute(
            """
            SELECT COALESCE(
                SUM(bucket_count),
                0
            )
            FROM bucket_counts
            """
        )

        row = cursor.fetchone()

        cursor.close()

        return int(row[0] or 0)

    except Exception as exc:
        print("Total count error:", exc)
        return 0

    finally:
        if connection:
            connection.close()


# ============================================================
# DATASET
# ============================================================

def list_files(directory, extensions=None):
    if not os.path.isdir(directory):
        return []

    result = []

    for filename in os.listdir(directory):
        path = os.path.join(
            directory,
            filename,
        )

        if not os.path.isfile(path):
            continue

        if extensions:
            if not filename.lower().endswith(
                tuple(extensions)
            ):
                continue

        result.append(filename)

    result.sort()

    return result


def dataset_stats():
    train_images = list_files(
        TRAIN_IMAGES_DIR,
        (
            ".jpg",
            ".jpeg",
            ".png",
            ".webp",
            ".bmp",
        ),
    )

    val_images = list_files(
        VAL_IMAGES_DIR,
        (
            ".jpg",
            ".jpeg",
            ".png",
            ".webp",
            ".bmp",
        ),
    )

    train_labels = list_files(
        TRAIN_LABELS_DIR,
        (".txt",),
    )

    val_labels = list_files(
        VAL_LABELS_DIR,
        (".txt",),
    )

    return {
        "training_images": len(train_images),
        "validation_images": len(val_images),
        "training_labels": len(train_labels),
        "validation_labels": len(val_labels),
        "total_images": (
            len(train_images)
            + len(val_images)
        ),
        "total_labels": (
            len(train_labels)
            + len(val_labels)
        ),
    }


def create_dataset_yaml():
    os.makedirs(
        DATASET_DIR,
        exist_ok=True,
    )

    yaml_content = f"""
path: {DATASET_DIR}
train: images/train
val: images/val

names:
  0: BUCKET_LOADED
""".strip() + "\n"

    with open(
        DATASET_YAML,
        "w",
        encoding="utf-8",
    ) as file:
        file.write(yaml_content)


create_dataset_yaml()


def prepare_validation_data():
    """
    If validation is empty, copy some training images
    into validation so YOLO has a valid validation folder.

    Existing validation files are never deleted.
    """

    train_images = list_files(
        TRAIN_IMAGES_DIR,
        (
            ".jpg",
            ".jpeg",
            ".png",
            ".webp",
            ".bmp",
        ),
    )

    val_images = list_files(
        VAL_IMAGES_DIR,
        (
            ".jpg",
            ".jpeg",
            ".png",
            ".webp",
            ".bmp",
        ),
    )

    if val_images:
        return

    if not train_images:
        return

    amount = max(
        1,
        int(len(train_images) * 0.2),
    )

    selected = train_images[:amount]

    for filename in selected:
        source_image = os.path.join(
            TRAIN_IMAGES_DIR,
            filename,
        )

        target_image = os.path.join(
            VAL_IMAGES_DIR,
            filename,
        )

        if not os.path.exists(target_image):
            shutil.copy2(
                source_image,
                target_image,
            )

        label_name = (
            os.path.splitext(filename)[0]
            + ".txt"
        )

        source_label = os.path.join(
            TRAIN_LABELS_DIR,
            label_name,
        )

        target_label = os.path.join(
            VAL_LABELS_DIR,
            label_name,
        )

        if os.path.exists(source_label):
            shutil.copy2(
                source_label,
                target_label,
            )


# ============================================================
# MULTIPART UPLOAD
# ============================================================

def parse_multipart(handler):
    content_type = handler.headers.get(
        "Content-Type",
        "",
    )

    if "multipart/form-data" not in content_type:
        return {}

    boundary_marker = "boundary="

    if boundary_marker not in content_type:
        return {}

    boundary = content_type.split(
        boundary_marker,
        1,
    )[1].strip()

    boundary = boundary.strip('"')

    try:
        length = int(
            handler.headers.get(
                "Content-Length",
                "0",
            )
        )
    except Exception:
        length = 0

    raw = handler.rfile.read(length)

    boundary_bytes = (
        b"--"
        + boundary.encode("utf-8")
    )

    parts = raw.split(boundary_bytes)

    fields = {}

    for part in parts:
        if not part:
            continue

        if part in (
            b"--",
            b"--\r\n",
        ):
            continue

        if part.startswith(b"\r\n"):
            part = part[2:]

        if part.endswith(b"\r\n"):
            part = part[:-2]

        separator = b"\r\n\r\n"

        if separator not in part:
            continue

        header_bytes, body = part.split(
            separator,
            1,
        )

        headers_text = (
            header_bytes.decode(
                "utf-8",
                errors="ignore",
            )
        )

        name = None
        filename = None

        for line in headers_text.split("\r\n"):
            lower = line.lower()

            if "content-disposition:" not in lower:
                continue

            if 'name="' in line:
                name = line.split(
                    'name="',
                    1,
                )[1].split(
                    '"',
                    1,
                )[0]

            if 'filename="' in line:
                filename = line.split(
                    'filename="',
                    1,
                )[1].split(
                    '"',
                    1,
                )[0]

        if not name:
            continue

        if filename:
            fields[name] = {
                "filename": filename,
                "data": body,
            }
        else:
            fields[name] = body.decode(
                "utf-8",
                errors="ignore",
            )

    return fields


# ============================================================
# MODEL
# ============================================================

def find_model_path():
    candidates = [
        MODEL_PATH,
        FALLBACK_MODEL_PATH,
        os.path.join(
            BASE_DIR,
            "best.pt",
        ),
        os.path.join(
            BASE_DIR,
            "yolo11n.pt",
        ),
    ]

    for path in candidates:
        if os.path.isfile(path):
            return path

    return None


def get_model():
    if YOLO is None:
        return None

    model_path = find_model_path()

    if not model_path:
        return None

    cached_model = MODEL_CACHE.get("model")
    cached_path = MODEL_CACHE.get("path")

    if (
        cached_model is not None
        and cached_path == model_path
    ):
        return cached_model

    try:
        model = YOLO(model_path)

        MODEL_CACHE["model"] = model
        MODEL_CACHE["path"] = model_path

        return model

    except Exception as exc:
        print("Model loading error:", exc)

        MODEL_CACHE["model"] = None
        MODEL_CACHE["path"] = None

        return None


# ============================================================
# YOLO TRAINING
# ============================================================

def training_callback(trainer):
    try:
        epoch = int(
            getattr(
                trainer,
                "epoch",
                0,
            )
        )

        total_epochs = int(
            getattr(
                trainer,
                "epochs",
                TRAINING_STATE["epochs"],
            )
        )

        if total_epochs <= 0:
            total_epochs = 1

        progress = int(
            ((epoch + 1) / total_epochs)
            * 100
        )

        if progress > 100:
            progress = 100

        TRAINING_STATE["epoch"] = (
            epoch + 1
        )

        TRAINING_STATE["epochs"] = (
            total_epochs
        )

        TRAINING_STATE["progress"] = (
            progress
        )

        TRAINING_STATE["message"] = (
            f"Training YOLO: "
            f"Epoch {epoch + 1}/"
            f"{total_epochs}"
        )

    except Exception as exc:
        print(
            "Training callback error:",
            exc,
        )


def train_model(epochs):
    global MODEL_CACHE

    try:
        TRAINING_STATE["status"] = "training"
        TRAINING_STATE["message"] = (
            "Preparing YOLO training..."
        )
        TRAINING_STATE["epoch"] = 0
        TRAINING_STATE["epochs"] = epochs
        TRAINING_STATE["progress"] = 0
        TRAINING_STATE["error"] = ""
        TRAINING_STATE["started_at"] = (
            datetime.now().isoformat()
        )
        TRAINING_STATE["finished_at"] =            datetime.now().isoformat()

        if YOLO is None:
            raise RuntimeError(
                "Ultralytics YOLO is not installed."
            )

        # Hakikisha folders zipo
        for directory in [
            TRAIN_IMAGES_DIR,
            VAL_IMAGES_DIR,
            TRAIN_LABELS_DIR,
            VAL_LABELS_DIR,
        ]:
            os.makedirs(
                directory,
                exist_ok=True,
            )

        # Tengeneza YAML upya
        create_dataset_yaml()

        # Tayarisha validation kama haipo
        prepare_validation_data()

        stats = dataset_stats()

        if stats["training_images"] == 0:
            raise RuntimeError(
                "No training images found. "
                "Upload training images first."
            )

        if stats["training_labels"] == 0:
            raise RuntimeError(
                "No training labels found. "
                "Each training image must have a .txt YOLO label."
            )

        if stats["validation_images"] == 0:
            raise RuntimeError(
                "No validation images found."
            )

        # Hakikisha angalau validation label ipo
        if stats["validation_labels"] == 0:
            raise RuntimeError(
                "No validation labels found. "
                "Validation images must have YOLO labels."
            )

        TRAINING_STATE["message"] = (
            "Loading YOLO model..."
        )

        # Tumia trained model kama ipo,
        # vinginevyo tumia yolo11n.pt
        base_model_path = find_model_path()

        if base_model_path is None:
            raise RuntimeError(
                "No YOLO model found. "
                "Upload yolo11n.pt or bucket_best.pt "
                "into the AI model folder."
            )

        print(
            "YOLO training model:",
            base_model_path,
        )

        model = YOLO(base_model_path)

        TRAINING_STATE["message"] = (
            "Starting YOLO training..."
        )

        # Callback ya progress
        try:
            model.add_callback(
                "on_train_epoch_end",
                training_callback,
            )
        except Exception as callback_error:
            print(
                "Could not attach training callback:",
                callback_error,
            )

        # Training output directory
        project_dir = os.path.join(
            AI_DIR,
            "runs",
        )

        os.makedirs(
            project_dir,
            exist_ok=True,
        )

        TRAINING_STATE["message"] = (
            "Training YOLO for "
            f"{epochs} epochs..."
        )

        print(
            "Starting YOLO training..."
        )

        # Run training
        model.train(
            data=DATASET_YAML,
            epochs=int(epochs),
            imgsz=640,
            batch=2,
            workers=0,
            project=project_dir,
            name="bucket_training",
            exist_ok=True,
            pretrained=True,
            verbose=True,
        )

        # Tafuta best.pt baada ya training
        trained_best = os.path.join(
            project_dir,
            "bucket_training",
            "weights",
            "best.pt",
        )

        trained_last = os.path.join(
            project_dir,
            "bucket_training",
            "weights",
            "last.pt",
        )

        selected_model = None

        if os.path.isfile(trained_best):
            selected_model = trained_best

        elif os.path.isfile(trained_last):
            selected_model = trained_last

        if selected_model is None:
            raise RuntimeError(
                "Training completed but no trained "
                "model weights were found."
            )

        # Hakikisha model directory ipo
        os.makedirs(
            MODEL_DIR,
            exist_ok=True,
        )

        # Hifadhi model mpya kama bucket_best.pt
        shutil.copy2(
            selected_model,
            MODEL_PATH,
        )

        print(
            "Trained model saved to:",
            MODEL_PATH,
        )

        # Clear model cache ili model mpya itumike
        MODEL_CACHE["model"] = None
        MODEL_CACHE["path"] = None

        TRAINING_STATE["status"] = "completed"

        TRAINING_STATE["epoch"] = int(
            epochs
        )

        TRAINING_STATE["epochs"] = int(
            epochs
        )

        TRAINING_STATE["progress"] = 100

        TRAINING_STATE["message"] = (
            "YOLO training completed successfully."
        )

        TRAINING_STATE["finished_at"] = (
            datetime.now().isoformat()
        )

        TRAINING_STATE["error"] = ""

        print(
            "YOLO training completed successfully."
        )

    except Exception as exc:

        traceback.print_exc()

        TRAINING_STATE["status"] = "error"

        TRAINING_STATE["message"] = (
            "YOLO training failed."
        )

        TRAINING_STATE["error"] = str(
            exc
        )

        TRAINING_STATE["finished_at"] = (
            datetime.now().isoformat()
        )

        print(
            "YOLO TRAINING ERROR:",
            exc,
        )

    finally:
        try:
            TRAIN_LOCK.release()
        except Exception:
            pass


def start_training(epochs=20):
    try:
        epochs = int(epochs)
    except Exception:
        epochs = 20

    if epochs < 1:
        epochs = 1

    if epochs > 300:
        epochs = 300

    if TRAIN_LOCK.locked():
        return {
            "success": False,
            "error": "Training is already running.",
        }

    acquired = TRAIN_LOCK.acquire(
        blocking=False
    )

    if not acquired:
        return {
            "success": False,
            "error": "Training is already running.",
        }

    TRAINING_STATE["status"] = "starting"
    TRAINING_STATE["message"] = (
        "Training is starting..."
    )
    TRAINING_STATE["epoch"] = 0
    TRAINING_STATE["epochs"] = epochs
    TRAINING_STATE["progress"] = 0
    TRAINING_STATE["error"] = ""
    TRAINING_STATE["started_at"] = (
        datetime.now().isoformat()
    )
    TRAINING_STATE["finished_at"] = None

    thread = threading.Thread(
        target=train_model,
        args=(epochs,),
        daemon=True,
    )

    thread.start()

    return {
        "success": True,
        "message": (
            f"Training started for "
            f"{epochs} epochs."
        ),
    }


# ============================================================
# CAMERA DETECTION
# ============================================================

def detect_image(image_bytes):
    if YOLO is None:
        return {
            "success": False,
            "error": "YOLO is not installed.",
        }

    if cv2 is None:
        return {
            "success": False,
            "error": "OpenCV is not installed.",
        }

    if np is None:
        return {
            "success": False,
            "error": "NumPy is not installed.",
        }

    model = get_model()

    if model is None:
        return {
            "success": False,
            "error": (
                "No YOLO model is available."
            ),
        }

    try:
        array = np.frombuffer(
            image_bytes,
            dtype=np.uint8,
        )

        image = cv2.imdecode(
            array,
            cv2.IMREAD_COLOR,
        )

        if image is None:
            raise RuntimeError(
                "Could not decode image."
            )

        results = model.predict(
            source=image,
            conf=0.35,
            verbose=False,
        )

        detections = []

        for result in results:

            boxes = getattr(
                result,
                "boxes",
                None,
            )

            if boxes is None:
                continue

            for box in boxes:

                try:
                    confidence = float(
                        box.conf[0]
                    )
                except Exception:
                    confidence = 0.0

                try:
                    class_id = int(
                        box.cls[0]
                    )
                except Exception:
                    class_id = 0

                try:
                    coordinates = (
                        box.xyxy[0]
                        .cpu()
                        .numpy()
                        .tolist()
                    )
                except Exception:
                    coordinates = [
                        0,
                        0,
                        0,
                        0,
                    ]

                class_name = (
                    "BUCKET_LOADED"
                    if class_id == 0
                    else str(class_id)
                )

                detections.append(
                    {
                        "class_id": class_id,
                        "class_name": class_name,
                        "confidence": round(
                            confidence,
                            4,
                        ),
                        "box": coordinates,
                    }
                )

        bucket_count = sum(
            1
            for item in detections
            if item["class_name"]
            == "BUCKET_LOADED"
        )

        return {
            "success": True,
            "count": bucket_count,
            "detections": detections,
        }

    except Exception as exc:

        traceback.print_exc()

        return {
            "success": False,
            "error": str(exc),
        }


# ============================================================
# REFERENCE BUCKET IMAGES
# ============================================================

def get_reference_images():
    files = list_files(
        REFERENCE_DIR,
        (
            ".jpg",
            ".jpeg",
            ".png",
            ".webp",
            ".bmp",
        ),
    )

    result = []

    for filename in files:
        result.append(
            {
                "filename": filename,
                "url": (
                    "/reference/"
                    + filename
                ),
            }
        )

    return result


def save_reference_image(
    filename,
    data,
):
    if not data:
        return {
            "success": False,
            "error": "Empty image.",
        }

    if not is_image_filename(
        filename
    ):
        return {
            "success": False,
            "error": "Invalid image format.",
        }

    new_name = unique_filename(
        filename
    )

    path = os.path.join(
        REFERENCE_DIR,
        new_name,
    )

    with open(
        path,
        "wb",
    ) as file:
        file.write(data)

    return {
        "success": True,
        "filename": new_name,
        "url": (
            "/reference/"
            + new_name
        ),
    }


# ============================================================
# DATASET IMAGE UPLOAD
# ============================================================

def save_dataset_image(
    filename,
    data,
    split="train",
):
    if not data:
        return {
            "success": False,
            "error": "Empty image.",
        }

    if not is_image_filename(
        filename
    ):
        return {
            "success": False,
            "error": (
                "Only image files are allowed."
            ),
        }

    if split not in (
        "train",
        "val",
    ):
        split = "train"

    image_directory = (
        TRAIN_IMAGES_DIR
        if split == "train"
        else VAL_IMAGES_DIR
    )

    label_directory = (
        TRAIN_LABELS_DIR
        if split == "train"
        else VAL_LABELS_DIR
    )

    os.makedirs(
        image_directory,
        exist_ok=True,
    )

    os.makedirs(
        label_directory,
        exist_ok=True,
    )

    new_name = unique_filename(
        filename
    )

    image_path = os.path.join(
        image_directory,
        new_name,
    )

    with open(
        image_path,
        "wb",
    ) as file:
        file.write(data)

    return {
        "success": True,
        "filename": new_name,
        "split": split,
        "path": image_path,
        "label_required": True,
    }


# ============================================================
# YOLO LABEL CREATION
# ============================================================

def save_yolo_label(
    image_filename,
    x_center,
    y_center,
    width,
    height,
    split="train",
):
    if split not in (
        "train",
        "val",
    ):
        split = "train"

    try:
        x_center = float(x_center)
        y_center = float(y_center)
        width = float(width)
        height = float(height)
    except Exception:
        return {
            "success": False,
            "error": "Invalid bounding box.",
        }

    # Normalize values to YOLO range
    x_center = max(
        0.0,
        min(1.0, x_center),
    )

    y_center = max(
        0.0,
        min(1.0, y_center),
    )

    width = max(
        0.0,
        min(1.0, width),
    )

    height = max(
        0.0,
        min(1.0, height),
    )

    label_directory = (
        TRAIN_LABELS_DIR
        if split == "train"
        else VAL_LABELS_DIR
    )

    os.makedirs(
        label_directory,
        exist_ok=True,
    )

    label_filename = (
        os.path.splitext(
            safe_filename(
                image_filename
            )
        )[0]
        + ".txt"
    )

    label_path = os.path.join(
        label_directory,
        label_filename,
    )

    with open(
        label_path,
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            "0 "
            f"{x_center:.6f} "
            f"{y_center:.6f} "
            f"{width:.6f} "
            f"{height:.6f}\n"
        )

    return {
        "success": True,
        "filename": label_filename,
        "path": label_path,
    }


# ============================================================
# HTTP HANDLER
# ============================================================

class Handler(BaseHTTPRequestHandler):

    def log_message(
        self,
        format_string,
        *args,
    ):
        print(
            "%s - %s"
            % (
                self.address_string(),
                format_string % args,
            )
        )

    def do_GET(self):

        parsed = urlparse(
            self.path
        )

        path = parsed.path

        # -----------------------------
        # API
        # -----------------------------

        if path == "/api/status":

            stats = dataset_stats()

            json_response(
                self,
                {
                    "success": True,
                    "training": TRAINING_STATE,
                    "dataset": stats,
                    "total_count": get_total_count(),
                    "database": bool(
                        DATABASE_URL
                    ),
                    "model_available": (
                        find_model_path()
                        is not None
                    ),
                },
            )

            return

        if path == "/api/training/status":

            json_response(
                self,
                {
                    "success": True,
                    "training": TRAINING_STATE,
                    "dataset": dataset_stats(),
                },
            )

            return

        if path == "/api/history":

            json_response(
                self,
                {
                    "success": True,
                    "history": get_history(),
                    "total": get_total_count(),
                },
            )

            return

        if path == "/api/dataset/stats":

            json_response(
                self,
                {
                    "success": True,
                    "stats": dataset_stats(),
                },
            )

            return

        if path == "/api/references":

            json_response(
                self,
                {
                    "success": True,
                    "images": get_reference_images(),
                },
            )

            return

        # -----------------------------
        # Reference images
        # -----------------------------

        if path.startswith(
            "/reference/"
        ):

            filename = safe_filename(
                path.split(
                    "/reference/",
                    1,
                )[1]
            )

            file_path = os.path.join(
                REFERENCE_DIR,
                filename,
            )

            if not os.path.isfile(
                file_path
            ):
                self.send_error(
                    404,
                    "Image not found",
                )
                return

            try:
                with open(
                    file_path,
                    "rb",
                ) as file:
                    data = file.read()

                bytes_response(
                    self,
                    data,
                    image_content_type(
                        filename
                    ),
                )

            except Exception:
                self.send_error(
                    500,
                    "Could not read image",
                )

            return

        # -----------------------------
        # Home page
        # -----------------------------

        if path in (
            "/",
            "/index.html",
        ):

            html_response(
                self,
                HTML_PAGE,
            )

            return

        self.send_error(
            404,
            "Not found",
        )

    def do_POST(self):

        parsed = urlparse(
            self.path
        )

        path = parsed.path

        # ====================================================
        # SAVE BUCKET COUNT
        # ====================================================

        if path == "/api/count":

            data = read_json_body(
                self
            )

            count = data.get(
                "count",
                0,
            )

            shift = data.get(
                "shift",
                "",
            )

            try:
                count = int(count)
            except Exception:
                count = 0

            if count < 0:
                count = 0

            result = save_bucket_count(
                count,
                shift,
            )

            if result["success"]:

                json_response(
                    self,
                    {
                        "success": True,
                        "message": (
                            "Bucket count saved."
                        ),
                        "id": result.get(
                            "id"
                        ),
                    },
                )

            else:

                json_response(
                    self,
                    result,
                    500,
                )

            return

        # ====================================================
        # START TRAINING
        # ====================================================

        if path == "/api/training/start":

            data = read_json_body(
                self
            )

            epochs = data.get(
                "epochs",
                20,
            )

            result = start_training(
                epochs
            )

            json_response(
                self,
                result,
                200
                if result["success"]
                else 409,
            )

            return

        # ====================================================
        # DETECT IMAGE
        # ====================================================

        if path == "/api/detect":

            fields = parse_multipart(
                self
            )

            image_field = fields.get(
                "image"
            )

            if not image_field:
                send_error_json(
                    self,
                    "Image is required.",
                    400,
                )
                return

            image_data = image_field.get(
                "data",
                b"",
            )

            result = detect_image(
                image_data
            )

            json_response(
                self,
                result,
                200
                if result["success"]
                else 500,
            )

            return

        # ====================================================
        # UPLOAD REFERENCE IMAGE
        # ====================================================

        if path == "/api/reference/upload":

            fields = parse_multipart(
                self
            )

            image_field = fields.get(
                "image"
            )

            if not image_field:
                send_error_json(
                    self,
                    "Image is required.",
                    400,
                )
                return

            result = save_reference_image(
                image_field.get(
                    "filename",
                    "bucket.jpg",
                ),
                image_field.get(
                    "data",
                    b"",
                ),
            )

            json_response(
                self,
                result,
                200
                if result["success"]
                else 400,
            )

            return

        # ====================================================
        # UPLOAD TRAINING IMAGE
        # ====================================================

        if path == "/api/dataset/upload":

            fields = parse_multipart(
                self
            )

            image_field = fields.get(
                "image"
            )

            split = fields.get(
                "split",
                "train",
            )

            if not image_field:
                send_error_json(
                    self,
                    "Image is required.",
                    400,
                )
                return

            result = save_dataset_image(
                image_field.get(
                    "filename",
                    "training.jpg",
                ),
                image_field.get(
                    "data",
                    b"",
                ),
                split,
            )

            json_response(
                self,
                result,
                200
                if result["success"]
                else 400,
            )

            return

        # ====================================================
        # SAVE YOLO LABEL
        # ====================================================

        if path == "/api/dataset/label":

            data = read_json_body(
                self
            )

            result = save_yolo_label(
                data.get(
                    "image",
                    "",
                ),
                data.get(
                    "x_center",
                    0,
                ),
                data.get(
                    "y_center",
                    0,
                ),
                data.get(
                    "width",
                    0,
                ),
                data.get(
                    "height",
                    0,
                ),
                data.get(
                    "split",
                    "train",
                ),
            )

            json_response(
                self,
                result,
                200
                if result["success"]
                else 400,
            )

            return

        send_error_json(
            self,
            "API endpoint not found.",
            404,
        )


# ============================================================
# HTML FRONTEND
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
    background: #0f172a;
    color: #e5e7eb;
}

header {
    padding: 18px;
    background: #111827;
    border-bottom:
        1px solid #334155;
}

header h1 {
    margin: 0;
    font-size: 22px;
}

header p {
    margin: 5px 0 0;
    color: #94a3b8;
}

nav {
    display: flex;
    gap: 6px;
    padding: 10px;
    background: #1e293b;
    overflow-x: auto;
}

nav button {
    border: 0;
    background: #334155;
    color: white;
    padding:
        10px 14px;
    border-radius: 8px;
    cursor: pointer;
}

nav button.active {
    background: #2563eb;
}

main {
    max-width: 1100px;
    margin: auto;
    padding: 18px;
}

.tab {
    display: none;
}

.tab.active {
    display: block;
}

.card {
    background: #1e293b;
    border:
        1px solid #334155;
    border-radius: 12px;
    padding: 16px;
    margin-bottom: 16px;
}

.card h2 {
    margin-top: 0;
}

button {
    cursor: pointer;
}

.primary {
    border: 0;
    background: #2563eb;
    color: white;
    padding:
        11px 16px;
    border-radius: 8px;
    font-weight: bold;
}

.success {
    border: 0;
    background: #16a34a;
    color: white;
    padding:
        11px 16px;
    border-radius: 8px;
    font-weight: bold;
}

.danger {
    border: 0;
    background: #dc2626;
    color: white;
    padding:
        11px 16px;
    border-radius: 8px;
}

input,
select {
    width: 100%;
    padding: 11px;
    margin:
        6px 0 12px;
    border-radius: 8px;
    border:
        1px solid #475569;
    background: #0f172a;
    color: white;
}

.camera-box {
    width: 100%;
    max-width: 800px;
    margin: auto;
    background: black;
    border-radius: 12px;
    overflow: hidden;
    position: relative;
}

video {
    width: 100%;
    display: block;
}

canvas {
    width: 100%;
    display: block;
}

.big-count {
    font-size: 55px;
    font-weight: bold;
    text-align: center;
    margin: 10px;
}

.status {
    padding: 10px;
    background: #0f172a;
    border-radius: 8px;
    margin-top: 10px;
}

.grid {
    display: grid;
    grid-template-columns:
        repeat(
            auto-fit,
            minmax(
                200px,
                1fr
            )
        );
    gap: 12px;
}

.stat {
    background: #0f172a;
    border-radius: 10px;
    padding: 15px;
}

.stat strong {
    display: block;
    font-size: 28px;
    margin-top: 5px;
}

.preview {
    display: grid;
    grid-template-columns:
        repeat(
            auto-fit,
            minmax(
                160px,
                1fr
            )
        );
    gap: 10px;
    margin-top: 12px;
}

.preview img {
    width: 100%;
    border-radius: 8px;
}

.progress {
    width: 100%;
    height: 20px;
    background: #334155;
    border-radius: 10px;
    overflow: hidden;
}

.progress-bar {
    height: 100%;
    width: 0%;
    background: #22c55e;
    transition: width 0.3s;
}

footer {
    text-align: center;
    padding: 25px;
    color: #64748b;
}

.small {
    color: #94a3b8;
    font-size: 13px;
}

</style>

</head>

<body>

<header>

<h1>
NEERIKA BUCKET AI
</h1>

<p>
Mining Production Bucket Counter
</p>

</header>


<nav>

<button
    onclick="showTab('dashboard', this)"
    class="active"
>
Dashboard
</button>

<button
    onclick="showTab('camera', this)"
>
Camera
</button>

<button
    onclick="showTab('buckets', this)"
>
Buckets
</button>

<button
    onclick="showTab('training', this)"
>
Training
</button>

<button
    onclick="showTab('history', this)"
>
History
</button>

<button
    onclick="showTab('settings', this)"
>
Settings
</button>

</nav>


<main>

<!-- ===================================================== -->
<!-- DASHBOARD -->
<!-- ===================================================== -->

<section
    id="dashboard"
    class="tab active"
>

<div class="card">

<h2>
Dashboard
</h2>

<div class="grid">

<div class="stat">
Total Buckets
<strong id="dashboardTotal">
0
</strong>
</div>

<div class="stat">
Training Images
<strong id="dashboardTraining">
0
</strong>
</div>

<div class="stat">
Validation Images
<strong id="dashboardValidation">
0
</strong>
</div>

<div class="stat">
Model
<strong id="dashboardModel">
No
</strong>
</div>

</div>

</div>

</section>


<!-- ===================================================== -->
<!-- CAMERA -->
<!-- ===================================================== -->

<section
    id="camera"
    class="tab"
>

<div class="card">

<h2>
Bucket Camera
</h2>

<div class="camera-box">

<video
    id="video"
    autoplay
    playsinline
>
</video>

</div>

<canvas
    id="canvas"
    style="display:none;"
>
</canvas>

<div class="big-count"
     id="cameraCount">
0
</div>

<select id="shift">

<option value="">
Select Shift
</option>

<option value="A">
Shift A
</option>

<option value="B">
Shift B
</option>

<option value="C">
Shift C
</option>

</select>

<button
    class="primary"
    onclick="startCamera()"
>
Start Camera
</button>

<button
    class="success"
    onclick="captureAndDetect()"
>
Detect Buckets
</button>

<button
    class="primary"
    onclick="saveCurrentCount()"
>
Save Count
</button>

<div
    id="cameraStatus"
    class="status"
>
Camera not started.
</div>

</div>

</section>


<!-- ===================================================== -->
<!-- BUCKETS -->
<!-- ===================================================== -->

<section
    id="buckets"
    class="tab"
>

<div class="card">

<h2>
Bucket Reference Images
</h2>

<p class="small">
Upload photos of the loaded bucket type.
</p>

<form
    id="referenceForm"
>

<input
    type="file"
    id="referenceFile"
    accept="image/*"
>

<button
    class="primary"
    type="submit"
>
Upload Reference Image
</button>

</form>

<div
    id="referenceStatus"
    class="status"
>
Ready.
</div>

<div
    id="referencePreview"
    class="preview"
>
</div>

</div>

</section>


<!-- ===================================================== -->
<!-- TRAINING -->
<!-- ===================================================== -->

<section
    id="training"
    class="tab"
>

<div class="card">

<h2>
YOLO Training Dataset
</h2>

<p class="small">
Upload images for training.
</p>

<form
    id="trainingForm"
>

<input
    type="file"
    id="trainingFile"
    accept="image/*"
>

<select id="trainingSplit">

<option value="train">
Training
</option>

<option value="val">
Validation
</option>

</select>

<button
    class="primary"
    type="submit"
>
Upload Image
</button>

</form>

<div
    id="trainingUploadStatus"
    class="status"
>
Ready.
</div>

</div>


<div class="card">

<h2>
Dataset Information
</h2>

<div class="grid">

<div class="stat">
Training Images
<strong id="trainingImages">
0
</strong>
</div>

<div class="stat">
Validation Images
<strong id="validationImages">
0
</strong>
</div>

<div class="stat">
Training Labels
<strong id="trainingLabels">
0
</strong>
</div>

<div class="stat">
Validation Labels
<strong id="validationLabels">
0
</strong>
</div>

</div>

</div>


<div class="card">

<h2>
Training Status
</h2>

<div
    id="trainingMessage"
    class="status"
>
Status: idle
</div>

<p>
Progress:
<span id="trainingProgressText">
0%
</span>
</p>

<div class="progress">

<div
    id="trainingProgress"
    class="progress-bar"
>
</div>

</div>

<p>
Epoch:
<span id="trainingEpoch">
0/0
</span>
</p>

</div>


<div class="card">

<h2>
Start YOLO Training
</h2>

<label>
Epochs
</label>

<input
    type="number"
    id="epochs"
    value="20"
    min="1"
    max="300"
>

<button
    class="success"
    onclick="startTraining()"
>
Start YOLO Training
</button>

</div>

</section>


<!-- ===================================================== -->
<!-- HISTORY -->
<!-- ===================================================== -->

<section
    id="history"
    class="tab"
>

<div class="card">

<h2>
Bucket Count History
</h2>

<div
    id="historyContainer"
>
Loading...
</div>

</div>

</section>


<!-- ===================================================== -->
<!-- SETTINGS -->
<!-- ===================================================== -->

<section
    id="settings"
    class="tab"
>

<div class="card">

<h2>
Settings
</h2>

<p>
NEERIKA BUCKET AI
</p>

<p class="small">
Mining Production Bucket Counter
</p>

<p class="small">
Counts loaded ore/material buckets only.
</p>

</div>

</section>

</main>


<footer>
Geology & Mining Services
</footer>


<script>

let stream = null;

let currentCount = 0;


/* =====================================================
   TABS
===================================================== */

function showTab(
    id,
    button
) {

    document
        .querySelectorAll(".tab")
        .forEach(
            element => {
                element.classList.remove(
                    "active"
                );
            }
        );

    document
        .getElementById(id)
        .classList.add(
            "active"
        );

    document
        .querySelectorAll("nav button")
        .forEach(
            element => {
                element.classList.remove(
                    "active"
                );
            }
        );

    if (button) {
        button.classList.add(
            "active"
        );
    }

    if (id === "history") {
        loadHistory();
    }

    if (id === "buckets") {
        loadReferences();
    }

    if (id === "training") {
        loadDatasetStats();
        loadTrainingStatus();
    }
}


/* =====================================================
   DASHBOARD
===================================================== */

async function loadDashboard() {

    try {

        const response =
            await fetch(
                "/api/status"
            );

        const data =
            await response.json();

        document
            .getElementById(
                "dashboardTotal"
            )
            .textContent =
            data.total_count || 0;

        document
            .getElementById(
                "dashboardTraining"
            )
            .textContent =
            data.dataset
                ?.training_images || 0;

        document
            .getElementById(
                "dashboardValidation"
            )
            .textContent =
            data.dataset
                ?.validation_images || 0;

        document
            .getElementById(
                "dashboardModel"
            )
            .textContent =
            data.model_available
                ? "Ready"
                : "No";

    } catch (error) {

        console.error(error);

    }
}


/* =====================================================
   CAMERA
===================================================== */

async function startCamera() {

    try {

        stream =
            await navigator
                .mediaDevices
                .getUserMedia(
                    {
                        video: {
                            facingMode:
                                "environment"
                        },
                        audio: false
                    }
                );

        document
            .getElementById(
                "video"
            )
            .srcObject =
            stream;

        document
            .getElementById(
                "cameraStatus"
            )
            .textContent =
            "Camera started.";

    } catch (error) {

        document
            .getElementById(
                "cameraStatus"
            )
            .textContent =
            "Camera error: "
            + error.message;

    }
}


async function captureAndDetect() {

    const video =
        document.getElementById(
            "video"
        );

    const canvas =
        document.getElementById(
            "canvas"
        );

    if (
        !video.videoWidth ||
        !video.videoHeight
    ) {

        document
            .getElementById(
                "cameraStatus"
            )
            .textContent =
            "Start the camera first.";

        return;
    }

    canvas.width =
        video.videoWidth;

    canvas.height =
        video.videoHeight;

    const context =
        canvas.getContext(
            "2d"
        );

    context.drawImage(
        video,
        0,
        0,
        canvas.width,
        canvas.height
    );

    document
        .getElementById(
            "cameraStatus"
        )
        .textContent =
        "Detecting...";

    canvas.toBlob(
        async function(blob) {

            try {

                const form =
                    new FormData();

                form.append(
                    "image",
                    blob,
                    "camera.jpg"
                );

                const response =
                    await fetch(
                        "/api/detect",
                        {
                            method:
                                "POST",
                            body:
                                form
                        }
                    );

                const data =
                    await response.json();

                if (!data.success) {

                    throw new Error(
                        data.error ||
                        "Detection failed."
                    );

                }

                currentCount =
                    data.count || 0;

                document
                    .getElementById(
                        "cameraCount"
                    )
                    .textContent =
                    currentCount;

                document
                    .getElementById(
                        "cameraStatus"
                    )
                    .textContent =
                    "Detected "
                    + currentCount
                    + " loaded bucket(s).";

            } catch (error) {

                document
                    .getElementById(
                        "cameraStatus"
                    )
                    .textContent =
                    "Detection error: "
                    + error.message;

            }

        },
        "image/jpeg",
        0.9
    );
}


async function saveCurrentCount() {

    try {

        const shift =
            document
                .getElementById(
                    "shift"
                )
                .value;

        const response =
            await fetch(
                "/api/count",
                {
                    method:
                        "POST",
                    headers: {
                        "Content-Type":
                            "application/json"
                    },
                    body:
                        JSON.stringify(
                            {
                                count:
                                    currentCount,
                                shift:
                                    shift
                            }
                        )
                }
            );

        const data =
            await response.json();

        if (!data.success) {

            throw new Error(
                data.error ||
                "Could not save count."
            );

        }

        document
            .getElementById(
                "cameraStatus"
            )
            .textContent =
            "Count saved successfully.";

        loadDashboard();

    } catch (error) {

        document
            .getElementById(
                "cameraStatus"
            )
            .textContent =
            "Save error: "
            + error.message;

    }
}


/* =====================================================
   REFERENCE IMAGES
===================================================== */

document
    .getElementById(
        "referenceForm"
    )
    .addEventListener(
        "submit",
        async function(event) {

            event.preventDefault();

            const file =
                document
                    .getElementById(
                        "referenceFile"
                    )
                    .files[0];

            if (!file) {

                document
                    .getElementById(
                        "referenceStatus"
                    )
                    .textContent =
                    "Select an image first.";

                return;
            }

            const form =
                new FormData();

            form.append(
                "image",
                file
            );

            try {

                const response =
                    await fetch(
                        "/api/reference/upload",
                        {
                            method:
                                "POST",
                            body:
                                form
                        }
                    );

                const data =
                    await response.json();

                document
                    .getElementById(
                        "referenceStatus"
                    )
                    .textContent =
                    data.success
                        ? "Reference image uploaded."
                        : data.error;

                if (data.success) {
                    loadReferences();
                }

            } catch (error) {

                document
                    .getElementById(
                        "referenceStatus"
                    )
                    .textContent =
                    error.message;

            }

        }
    );


async function loadReferences() {

    try {

        const response =
            await fetch(
                "/api/references"
            );

        const data =
            await response.json();

        const container =
            document
                .getElementById(
                    "referencePreview"
                );

        container.innerHTML = "";

        data.images
            .forEach(
                image => {

                    const img =
                        document.createElement(
                            "img"
                        );

                    img.src =
                        image.url;

                    img.alt =
                        image.filename;

                    container.appendChild(
                        img
                    );

                }
            );

    } catch (error) {

        console.error(error);

    }
}


/* =====================================================
   TRAINING DATASET
===================================================== */

document
    .getElementById(
        "trainingForm"
    )
    .addEventListener(
        "submit",
        async function(event) {

            event.preventDefault();

            const file =
                document
                    .getElementById(
                        "trainingFile"
                    )
                    .files[0];

            const split =
                document
                    .getElementById(
                        "trainingSplit"
                    )
                    .value;

            if (!file) {

                document
                    .getElementById(
                        "trainingUploadStatus"
                    )
                    .textContent =
                    "Select an image first.";

                return;
            }

            const form =
                new FormData();

            form.append(
                "image",
                file
            );

            form.append(
                "split",
                split
            );

            try {

                const response =
                    await fetch(
                        "/api/dataset/upload",
                        {
                            method:
                                "POST",
                            body:
                                form
                        }
                    );

                const data =
                    await response.json();

                if (!data.success) {

                    throw new Error(
                        data.error
                    );

                }

                document
                    .getElementById(
                        "trainingUploadStatus"
                    )
                    .textContent =
                    "Image uploaded: "
                    + data.filename
                    + ". Now create its YOLO label.";

                loadDatasetStats();

            } catch (error) {

                document
                    .getElementById(
                        "trainingUploadStatus"
                    )
                    .textContent =
                    "Upload error: "
                    + error.message;

            }

        }
    );


async function loadDatasetStats() {

    try {

        const response =
            await fetch(
                "/api/dataset/stats"
            );

        const data =
            await response.json();

        const stats =
            data.stats || {};

        document
            .getElementById(
                "trainingImages"
            )
            .textContent =
            stats.training_images || 0;

        document
            .getElementById(
                "validationImages"
            )
            .textContent =
            stats.validation_images || 0;

        document
            .getElementById(
                "trainingLabels"
            )
            .textContent =
            stats.training_labels || 0;

        document
            .getElementById(
                "validationLabels"
            )
            .textContent =
            stats.validation_labels || 0;

    } catch (error) {

        console.error(error);

    }
}


/* =====================================================
   TRAINING
===================================================== */

async function startTraining() {

    const epochs =
        parseInt(
            document
                .getElementById(
                    "epochs"
                )
                .value,
            10
        ) || 20;

    const status =
        document
            .getElementById(
                "trainingMessage"
            );

    status.textContent =
        "Starting training...";

    try {

        const response =
            await fetch(
                "/api/training/start",
                {
                    method:
                        "POST",
                    headers: {
                        "Content-Type":
                            "application/json"
                    },
                    body:
                        JSON.stringify(
                            {
                                epochs:
                                    epochs
                            }
                        )
                }
            );

        const data =
            await response.json();

        status.textContent =
            data.success
                ? "Training started."
                : data.error;

        loadTrainingStatus();

    } catch (error) {

        status.textContent =
            "Training error: "
            + error.message;

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

        const state =
            data.training || {};

        document
            .getElementById(
                "trainingMessage"
            )
            .textContent =
            "Status: "
            + (
                state.status ||
                "idle"
            )
            + " — "
            + (
                state.message ||
                ""
            );

        const progress =
            state.progress || 0;

        document
            .getElementById(
                "trainingProgress"
            )
            .style.width =
            progress + "%";

        document
            .getElementById(
                "trainingProgressText"
            )
            .textContent =
            progress + "%";

        document
            .getElementById(
                "trainingEpoch"
            )
            .textContent =
            (
                state.epoch || 0
            )
            + "/"
            + (
                state.epochs || 0
            );

    } catch (error) {

        console.error(error);

    }
}


/* =====================================================
   HISTORY
===================================================== */

async function loadHistory() {

    const container =
        document
            .getElementById(
                "historyContainer"
            );

    container.innerHTML =
        "Loading...";

    try {

        const response =
            await fetch(
                "/api/history"
            );

        const data =
            await response.json();

        if (
            !data.history ||
            data.history.length === 0
        ) {

            container.innerHTML =
                "<p>No bucket counts yet.</p>";

            return;
        }

        let html =
            "<div class='grid'>";

        data.history
            .forEach(
                item => {

                    html +=
                        "<div class='stat'>"
                        + "<strong>"
                        + item.bucket_count
                        + "</strong>"
                        + "<div>"
                        + (
                            item.count_date ||
                            ""
                        )
                        + "</div>"
                        + "<div>"
                        + (
                            item.shift ||
                            "No shift"
                        )
                        + "</div>"
                        + "</div>";

                }
            );

        html += "</div>";

        container.innerHTML =
            html;

    } catch (error) {

        container.innerHTML =
            "History error: "
            + error.message;

    }
}


/* =====================================================
   AUTO REFRESH
===================================================== */

setInterval(
    function() {

        loadTrainingStatus();

        loadDashboard();

    },
    3000
);


/* =====================================================
   INITIAL LOAD
===================================================== */

loadDashboard();

loadReferences();

loadDatasetStats();

loadTrainingStatus();

</script>

</body>

</html>
"""


# ============================================================
# SERVER START
# ============================================================

def run_server():

    ensure_database()

    server = ThreadingHTTPServer(
        (
            HOST,
            PORT,
        ),
        Handler,
    )

    print(
        "================================================"
    )

    print(
        "NEERIKA BUCKET AI"
    )

    print(
        "Mining Production Bucket Counter"
    )

    print(
        f"Server running on "
        f"http://{HOST}:{PORT}"
    )

    print(
        "================================================"
    )

    try:
        server.serve_forever()

    except KeyboardInterrupt:

        print(
            "Server stopped."
        )

    finally:

        server.server_close()


if __name__ == "__main__":
    run_server()
