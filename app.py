import os
import io
import json
import csv
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

# ============================================================
# NEERIKA BUCKET AI
# Mining Production Bucket Counter
# FULL REPLACEMENT APP
# ============================================================

os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/Ultralytics")

# -----------------------------
# Optional libraries
# -----------------------------
try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
    POSTGRES_AVAILABLE = True
except Exception:
    POSTGRES_AVAILABLE = False

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except Exception:
    YOLO_AVAILABLE = False


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

AI_DIR = os.path.join(BASE_DIR, "ai")

DATASET_DIR = os.path.join(AI_DIR, "dataset")

TRAIN_IMAGES_DIR = os.path.join(
    DATASET_DIR, "images", "train"
)

VAL_IMAGES_DIR = os.path.join(
    DATASET_DIR, "images", "val"
)

TRAIN_LABELS_DIR = os.path.join(
    DATASET_DIR, "labels", "train"
)

VAL_LABELS_DIR = os.path.join(
    DATASET_DIR, "labels", "val"
)

MODEL_DIR = os.path.join(
    AI_DIR, "models"
)

TRAINING_RUN_DIR = os.path.join(
    AI_DIR, "training_runs"
)

DATASET_YAML = os.path.join(
    AI_DIR, "bucket_dataset.yaml"
)

MODEL_OUTPUT = os.path.join(
    MODEL_DIR, "bucket_best.pt"
)

os.makedirs(TRAIN_IMAGES_DIR, exist_ok=True)
os.makedirs(VAL_IMAGES_DIR, exist_ok=True)
os.makedirs(TRAIN_LABELS_DIR, exist_ok=True)
os.makedirs(VAL_LABELS_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(TRAINING_RUN_DIR, exist_ok=True)


# Render Free settings
EPOCHS = int(os.environ.get("YOLO_EPOCHS", "20"))
IMAGE_SIZE = int(os.environ.get("YOLO_IMAGE_SIZE", "320"))
BATCH_SIZE = int(os.environ.get("YOLO_BATCH", "1"))
WORKERS = int(os.environ.get("YOLO_WORKERS", "0"))
DEVICE = os.environ.get("YOLO_DEVICE", "cpu")

MAX_UPLOAD_SIZE = 15 * 1024 * 1024


DATABASE_URL = os.environ.get("DATABASE_URL", "").strip()


# ============================================================
# GLOBAL STATE
# ============================================================

TRAINING_LOCK = threading.Lock()

TRAINING_STATE = {
    "status": "idle",
    "message": "Ready",
    "progress": 0,
    "epoch": 0,
    "epochs": EPOCHS,
    "started_at": None,
    "finished_at": None,
    "error": None
}


# ============================================================
# DATABASE
# ============================================================

def db_connect():
    if not POSTGRES_AVAILABLE:
        return None

    if not DATABASE_URL:
        return None

    return psycopg2.connect(
        DATABASE_URL,
        sslmode="require"
    )


def init_database():
    conn = None

    try:
        conn = db_connect()

        if conn is None:
            return

        cur = conn.cursor()

        cur.execute("""
            CREATE TABLE IF NOT EXISTS bucket_counts (
                id BIGSERIAL PRIMARY KEY,
                bucket_type TEXT DEFAULT 'DEFAULT',
                count INTEGER DEFAULT 1,
                counted_at TIMESTAMPTZ DEFAULT NOW(),
                shift TEXT,
                source TEXT DEFAULT 'camera'
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS training_dataset (
                id BIGSERIAL PRIMARY KEY,
                filename TEXT UNIQUE NOT NULL,
                split TEXT DEFAULT 'train',
                annotations INTEGER DEFAULT 0,
                annotation_data JSONB DEFAULT '[]'::jsonb,
                uploaded_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)

        conn.commit()
        cur.close()

    except Exception:
        if conn:
            conn.rollback()

    finally:
        if conn:
            conn.close()


init_database()


# ============================================================
# DATABASE HELPERS
# ============================================================

def db_upsert_image(
    filename,
    split="train",
    annotations=0,
    annotation_data=None
):
    conn = None

    try:
        conn = db_connect()

        if conn is None:
            return

        if annotation_data is None:
            annotation_data = []

        cur = conn.cursor()

        cur.execute("""
            INSERT INTO training_dataset
                (
                    filename,
                    split,
                    annotations,
                    annotation_data
                )
            VALUES
                (%s, %s, %s, %s::jsonb)
            ON CONFLICT (filename)
            DO UPDATE SET
                split = EXCLUDED.split,
                annotations = EXCLUDED.annotations,
                annotation_data = EXCLUDED.annotation_data
        """, (
            filename,
            split,
            annotations,
            json.dumps(annotation_data)
        ))

        conn.commit()
        cur.close()

    except Exception:
        if conn:
            conn.rollback()

    finally:
        if conn:
            conn.close()


def db_get_image(filename):
    conn = None

    try:
        conn = db_connect()

        if conn is None:
            return None

        cur = conn.cursor(
            cursor_factory=RealDictCursor
        )

        cur.execute("""
            SELECT
                id,
                filename,
                split,
                annotations,
                annotation_data,
                uploaded_at
            FROM training_dataset
            WHERE filename = %s
            LIMIT 1
        """, (filename,))

        row = cur.fetchone()

        cur.close()

        return dict(row) if row else None

    except Exception:
        return None

    finally:
        if conn:
            conn.close()


def db_get_dataset_rows():
    conn = None

    try:
        conn = db_connect()

        if conn is None:
            return []

        cur = conn.cursor(
            cursor_factory=RealDictCursor
        )

        cur.execute("""
            SELECT
                filename,
                split,
                annotations,
                annotation_data,
                uploaded_at
            FROM training_dataset
            ORDER BY uploaded_at DESC
        """)

        rows = cur.fetchall()

        cur.close()

        return [dict(x) for x in rows]

    except Exception:
        return []

    finally:
        if conn:
            conn.close()


# ============================================================
# FILE HELPERS
# ============================================================

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp"
}


def safe_filename(name):
    name = os.path.basename(name)

    allowed = (
        "abcdefghijklmnopqrstuvwxyz"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "0123456789"
        "._-"
    )

    cleaned = ""

    for c in name:
        if c in allowed:
            cleaned += c
        else:
            cleaned += "_"

    if not cleaned:
        cleaned = "image.jpg"

    return cleaned


def image_path(filename, split="train"):
    filename = safe_filename(filename)

    if split == "val":
        return os.path.join(
            VAL_IMAGES_DIR,
            filename
        )

    return os.path.join(
        TRAIN_IMAGES_DIR,
        filename
    )


def label_path(filename, split="train"):
    filename = safe_filename(filename)

    stem = os.path.splitext(filename)[0]

    if split == "val":
        return os.path.join(
            VAL_LABELS_DIR,
            stem + ".txt"
        )

    return os.path.join(
        TRAIN_LABELS_DIR,
        stem + ".txt"
    )


def valid_image_filename(filename):
    ext = os.path.splitext(filename)[1].lower()
    return ext in IMAGE_EXTENSIONS


# ============================================================
# ANNOTATION HELPERS
# ============================================================

def normalize_box(box):
    """
    Accepts:
      {
        x: 0..1,
        y: 0..1,
        width: 0..1,
        height: 0..1
      }

    or pixel-like values.

    Returns normalized YOLO:
      [class_id, cx, cy, w, h]
    """

    try:
        x = float(box.get("x", 0))
        y = float(box.get("y", 0))
        w = float(box.get("width", 0))
        h = float(box.get("height", 0))

        # Already normalized
        if (
            0 <= x <= 1 and
            0 <= y <= 1 and
            0 <= w <= 1 and
            0 <= h <= 1
        ):
            nx = x
            ny = y
            nw = w
            nh = h

        else:
            # If frontend accidentally sends pixels,
            # width/height must also be supplied.
            image_width = float(
                box.get("image_width", 0)
            )

            image_height = float(
                box.get("image_height", 0)
            )

            if image_width <= 0 or image_height <= 0:
                return None

            nx = x / image_width
            ny = y / image_height
            nw = w / image_width
            nh = h / image_height

        if nw <= 0 or nh <= 0:
            return None

        # Convert top-left x/y to YOLO center
        cx = nx + (nw / 2.0)
        cy = ny + (nh / 2.0)

        cx = max(0.0, min(1.0, cx))
        cy = max(0.0, min(1.0, cy))
        nw = max(0.0001, min(1.0, nw))
        nh = max(0.0001, min(1.0, nh))

        return [
            0,
            cx,
            cy,
            nw,
            nh
        ]

    except Exception:
        return None


def boxes_to_yolo(boxes):
    output = []

    if not isinstance(boxes, list):
        return output

    for box in boxes:

        if not isinstance(box, dict):
            continue

        result = normalize_box(box)

        if result is None:
            continue

        output.append(result)

    return output


def write_yolo_label_file(
    filename,
    boxes,
    split="train"
):
    path = label_path(
        filename,
        split
    )

    yolo_boxes = boxes_to_yolo(boxes)

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as f:

        for row in yolo_boxes:
            f.write(
                f"{row[0]} "
                f"{row[1]:.6f} "
                f"{row[2]:.6f} "
                f"{row[3]:.6f} "
                f"{row[4]:.6f}\n"
            )

    return path, len(yolo_boxes)


def read_yolo_label_file(
    filename,
    split="train"
):
    path = label_path(
        filename,
        split
    )

    if not os.path.exists(path):
        return []

    boxes = []

    try:
        with open(
            path,
            "r",
            encoding="utf-8"
        ) as f:

            for line in f:
                parts = line.strip().split()

                if len(parts) != 5:
                    continue

                class_id = int(float(parts[0]))

                cx = float(parts[1])
                cy = float(parts[2])
                w = float(parts[3])
                h = float(parts[4])

                # Convert YOLO center format to
                # top-left format for frontend.
                x = cx - (w / 2.0)
                y = cy - (h / 2.0)

                boxes.append({
                    "x": x,
                    "y": y,
                    "width": w,
                    "height": h,
                    "class_id": class_id
                })

    except Exception:
        return []

    return boxes


# ============================================================
# DATASET SCAN
# ============================================================

def scan_split(split):
    if split == "val":
        image_dir = VAL_IMAGES_DIR
        label_dir = VAL_LABELS_DIR
    else:
        image_dir = TRAIN_IMAGES_DIR
        label_dir = TRAIN_LABELS_DIR

    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)

    images = []
    labels = []

    for filename in os.listdir(image_dir):

        if not valid_image_filename(filename):
            continue

        images.append(filename)

        lp = label_path(
            filename,
            split
        )

        if os.path.exists(lp):

            try:
                if os.path.getsize(lp) > 0:
                    labels.append(filename)
            except Exception:
                pass

    return images, labels


def dataset_stats():
    train_images, train_labels = scan_split("train")
    val_images, val_labels = scan_split("val")

    return {
        "training_images": len(train_images),
        "validation_images": len(val_images),
        "training_labels": len(train_labels),
        "validation_labels": len(val_labels),
        "unlabeled_training": max(
            0,
            len(train_images) - len(train_labels)
        ),
        "unlabeled_validation": max(
            0,
            len(val_images) - len(val_labels)
        )
    }


# ============================================================
# REBUILD LABELS FROM SUPABASE
# ============================================================

def restore_annotations_from_database():
    """
    If Render filesystem lost .txt files but Supabase
    still has annotation_data, rebuild the YOLO labels.
    """

    rows = db_get_dataset_rows()

    restored = 0

    for row in rows:

        filename = row.get("filename")
        split = row.get("split") or "train"

        annotation_data = (
            row.get("annotation_data")
            or []
        )

        if not filename:
            continue

        if not annotation_data:
            continue

        try:
            path, count = write_yolo_label_file(
                filename,
                annotation_data,
                split
            )

            if count > 0:
                restored += 1

        except Exception:
            pass

    return restored


# ============================================================
# DATASET VALIDATION
# ============================================================

def validate_training_dataset():
    restore_annotations_from_database()

    train_images, train_labels = scan_split(
        "train"
    )

    valid = []

    for filename in train_labels:

        path = label_path(
            filename,
            "train"
        )

        try:
            with open(
                path,
                "r",
                encoding="utf-8"
            ) as f:

                found_bucket = False

                for line in f:

                    parts = line.strip().split()

                    if len(parts) != 5:
                        continue

                    if int(float(parts[0])) == 0:
                        found_bucket = True
                        break

                if found_bucket:
                    valid.append(filename)

        except Exception:
            pass

    return valid


# ============================================================
# VALIDATION FALLBACK
# ============================================================

def prepare_validation_dataset():

    train_images, train_labels = scan_split(
        "train"
    )

    if not train_labels:
        return False

    val_images, val_labels = scan_split(
        "val"
    )

    # If validation already exists, use it.
    if val_labels:
        return True

    # Use one training image as validation fallback.
    source_filename = train_labels[0]

    source_image = os.path.join(
        TRAIN_IMAGES_DIR,
        source_filename
    )

    source_label = os.path.join(
        TRAIN_LABELS_DIR,
        os.path.splitext(
            source_filename
        )[0] + ".txt"
    )

    destination_image = os.path.join(
        VAL_IMAGES_DIR,
        source_filename
    )

    destination_label = os.path.join(
        VAL_LABELS_DIR,
        os.path.splitext(
            source_filename
        )[0] + ".txt"
    )

    try:

        shutil.copy2(
            source_image,
            destination_image
        )

        shutil.copy2(
            source_label,
            destination_label
        )

        return True

    except Exception:
        return False


# ============================================================
# YOLO DATASET YAML
# ============================================================

def create_dataset_yaml():

    dataset_path = DATASET_DIR.replace(
        "\\",
        "/"
    )

    text = f"""path: {dataset_path}
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

        f.write(text)

    return DATASET_YAML


# ============================================================
# TRAINING PROGRESS
# ============================================================

def update_training(
    status=None,
    message=None,
    progress=None,
    epoch=None,
    error=None
):

    if status is not None:
        TRAINING_STATE["status"] = status

    if message is not None:
        TRAINING_STATE["message"] = message

    if progress is not None:
        TRAINING_STATE["progress"] = int(
            max(0, min(100, progress))
        )

    if epoch is not None:
        TRAINING_STATE["epoch"] = int(
            epoch
        )

    if error is not None:
        TRAINING_STATE["error"] = error


# ============================================================
# YOLO CALLBACK
# ============================================================

def training_epoch_callback(trainer):

    try:

        epoch = int(
            getattr(
                trainer,
                "epoch",
                0
            )
        )

        total = int(
            getattr(
                trainer,
                "epochs",
                EPOCHS
            )
        )

        progress = int(
            ((epoch + 1) / total) * 100
        )

        update_training(
            status="training",
            message=(
                f"Training epoch "
                f"{epoch + 1}/{total}"
            ),
            progress=progress,
            epoch=epoch + 1
        )

    except Exception:
        pass


# ============================================================
# TRAINING WORKER
# ============================================================

def training_worker():

    with TRAINING_LOCK:

        try:

            update_training(
                status="preparing",
                message="Checking dataset...",
                progress=1,
                epoch=0,
                error=None
            )

            # Restore any labels saved in Supabase.
            restore_annotations_from_database()

            valid_images = (
                validate_training_dataset()
            )

            if not valid_images:

                update_training(
                    status="error",
                    message=(
                        "Hakuna picha yenye "
                        "BUCKET_LOADED annotation. "
                        "Fungua picha, chora box kwenye "
                        "bucket iliyobeba ore, kisha "
                        "bonyeza Save YOLO Labels."
                    ),
                    progress=0,
                    epoch=0,
                    error="NO_BUCKET_LABELS"
                )

                return

            update_training(
                status="preparing",
                message=(
                    f"Dataset ready: "
                    f"{len(valid_images)} labeled image(s)."
                ),
                progress=5,
                epoch=0
            )

            # Prepare validation.
            prepare_validation_dataset()

            create_dataset_yaml()

            if not YOLO_AVAILABLE:

                update_training(
                    status="error",
                    message=(
                        "Ultralytics YOLO haipatikani. "
                        "Hakikisha requirements.txt ina "
                        "ultralytics."
                    ),
                    progress=0,
                    epoch=0,
                    error="YOLO_NOT_AVAILABLE"
                )

                return

            update_training(
                status="loading",
                message=(
                    "Loading YOLO model..."
                ),
                progress=7,
                epoch=0
            )

            model = YOLO(
                "yolo11n.pt"
            )

            # Register callback.
            try:
                model.add_callback(
                    "on_train_epoch_end",
                    training_epoch_callback
                )
            except Exception:
                pass

            update_training(
                status="training",
                message=(
                    "YOLO training inaanza..."
                ),
                progress=8,
                epoch=0
            )

            run_name = (
                "bucket_training_"
                + datetime.now().strftime(
                    "%Y%m%d_%H%M%S"
                )
            )

            project_dir = AI_DIR

            # ------------------------------------------------
            # ACTUAL TRAINING
            # ------------------------------------------------

            results = model.train(

                data=DATASET_YAML,

                epochs=EPOCHS,

                imgsz=IMAGE_SIZE,

                batch=BATCH_SIZE,

                workers=WORKERS,

                device=DEVICE,

                project=project_dir,

                name=run_name,

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

            update_training(
                status="finalizing",
                message=(
                    "Training imekamilika. "
                    "Searching best model..."
                ),
                progress=98,
                epoch=EPOCHS
            )

            run_dir = os.path.join(
                project_dir,
                run_name
            )

            candidates = [

                os.path.join(
                    run_dir,
                    "weights",
                    "best.pt"
                ),

                os.path.join(
                    run_dir,
                    "weights",
                    "last.pt"
                )

            ]

            model_found = None

            for candidate in candidates:

                if os.path.exists(candidate):

                    model_found = candidate
                    break

            if model_found:

                shutil.copy2(
                    model_found,
                    MODEL_OUTPUT
                )

                update_training(
                    status="completed",
                    message=(
                        "Training completed successfully. "
                        "Model saved as bucket_best.pt"
                    ),
                    progress=100,
                    epoch=EPOCHS
                )

            else:

                update_training(
                    status="error",
                    message=(
                        "Training imekwisha lakini "
                        "best.pt/last.pt haikupatikana."
                    ),
                    progress=0,
                    epoch=EPOCHS,
                    error="MODEL_NOT_FOUND"
                )

        except Exception as e:

            error_text = (
                f"{type(e).__name__}: {str(e)}"
            )

            print(
                "TRAINING ERROR:",
                error_text
            )

            traceback.print_exc()

            update_training(
                status="error",
                message=(
                    "Training error: "
                    + error_text
                ),
                progress=0,
                error=error_text
            )


# ============================================================
# HTTP RESPONSE HELPERS
# ============================================================

class Handler(BaseHTTPRequestHandler):

    server_version = "NEERIKA-BUCKET-AI/1.0"

    def log_message(
        self,
        format,
        *args
    ):
        print(
            "%s - - [%s] %s"
            % (
                self.address_string(),
                self.log_date_time_string(),
                format % args
            )
        )

    def send_json(
        self,
        data,
        status=200
    ):

        body = json.dumps(
            data,
            ensure_ascii=False
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

    def send_html(
        self,
        html,
        status=200
    ):

        body = html.encode(
            "utf-8"
        )

        self.send_response(status)

        self.send_header(
            "Content-Type",
            "text/html; charset=utf-8"
        )

        self.send_header(
            "Content-Length",
            str(len(body))
        )

        self.end_headers()

        self.wfile.write(body)

    def send_bytes(
        self,
        data,
        content_type,
        status=200
    ):

        self.send_response(status)

        self.send_header(
            "Content-Type",
            content_type
        )

        self.send_header(
            "Content-Length",
            str(len(data))
        )

        self.end_headers()

        self.wfile.write(data)

    # ========================================================
    # GET
    # ========================================================

    def do_GET(self):

        try:

            parsed = urlparse(
                self.path
            )

            path = parsed.path

            if path == "/":
                self.send_html(
                    MAIN_HTML
                )
                return

            if path == "/api/training/status":

                self.send_json({
                    **TRAINING_STATE,
                    "model_exists":
                        os.path.exists(
                            MODEL_OUTPUT
                        )
                })

                return

            if path == "/api/dataset/stats":

                # Restore DB labels first.
                restore_annotations_from_database()

                self.send_json(
                    dataset_stats()
                )

                return

            if path == "/api/dataset/images":

                rows = []

                for split in [
                    "train",
                    "val"
                ]:

                    image_dir = (
                        TRAIN_IMAGES_DIR
                        if split == "train"
                        else VAL_IMAGES_DIR
                    )

                    for filename in os.listdir(
                        image_dir
                    ):

                        if not valid_image_filename(
                            filename
                        ):
                            continue

                        lp = label_path(
                            filename,
                            split
                        )

                        annotation_count = 0

                        if os.path.exists(lp):

                            try:

                                with open(
                                    lp,
                                    "r",
                                    encoding="utf-8"
                                ) as f:

                                    for line in f:

                                        if line.strip():
                                            annotation_count += 1

                            except Exception:
                                pass

                        rows.append({
                            "filename": filename,
                            "split": split,
                            "annotations":
                                annotation_count,
                            "labeled":
                                annotation_count > 0
                        })

                self.send_json({
                    "images": rows
                })

                return

            if path == "/api/dataset/image":

                qs = parse_qs(
                    parsed.query
                )

                filename = (
                    qs.get(
                        "filename",
                        [""]
                    )[0]
                )

                split = (
                    qs.get(
                        "split",
                        ["train"]
                    )[0]
                )

                filename = safe_filename(
                    filename
                )

                full_path = image_path(
                    filename,
                    split
                )

                if not os.path.exists(
                    full_path
                ):

                    self.send_json({
                        "error":
                            "Image not found"
                    }, 404)

                    return

                with open(
                    full_path,
                    "rb"
                ) as f:

                    data = f.read()

                encoded = base64.b64encode(
                    data
                ).decode("ascii")

                ext = os.path.splitext(
                    filename
                )[1].lower()

                mime = {
                    ".jpg":
                        "image/jpeg",
                    ".jpeg":
                        "image/jpeg",
                    ".png":
                        "image/png",
                    ".webp":
                        "image/webp"
                }.get(
                    ext,
                    "application/octet-stream"
                )

                boxes = read_yolo_label_file(
                    filename,
                    split
                )

                self.send_json({
                    "filename": filename,
                    "split": split,
                    "mime": mime,
                    "data":
                        "data:"
                        + mime
                        + ";base64,"
                        + encoded,
                    "boxes": boxes
                })

                return

            if path == "/api/dataset/labels":

                qs = parse_qs(
                    parsed.query
                )

                filename = safe_filename(
                    qs.get(
                        "filename",
                        [""]
                    )[0]
                )

                split = qs.get(
                    "split",
                    ["train"]
                )[0]

                boxes = read_yolo_label_file(
                    filename,
                    split
                )

                self.send_json({
                    "filename": filename,
                    "split": split,
                    "boxes": boxes,
                    "count": len(boxes)
                })

                return

            if path == "/api/model":

                exists = os.path.exists(
                    MODEL_OUTPUT
                )

                size = 0

                if exists:

                    try:
                        size = os.path.getsize(
                            MODEL_OUTPUT
                        )
                    except Exception:
                        pass

                self.send_json({
                    "exists": exists,
                    "path":
                        "ai/models/bucket_best.pt",
                    "size": size
                })

                return

            self.send_json({
                "error": "Not found"
            }, 404)

        except Exception as e:

            traceback.print_exc()

            self.send_json({
                "error": str(e)
            }, 500)

    # ========================================================
    # POST
    # ========================================================

    def do_POST(self):

        try:

            parsed = urlparse(
                self.path
            )

            path = parsed.path

            if path == "/api/dataset/upload":

                self.handle_upload()
                return

            if path == "/api/dataset/labels":

                self.handle_save_labels()
                return

            if path == "/api/training/start":

                self.handle_training_start()
                return

            if path == "/api/count":

                self.handle_count()
                return

            self.send_json({
                "error": "Not found"
            }, 404)

        except Exception as e:

            traceback.print_exc()

            self.send_json({
                "error": str(e)
            }, 500)

    # ========================================================
    # BODY
    # ========================================================

    def read_body(self):

        length = int(
            self.headers.get(
                "Content-Length",
                "0"
            )
        )

        if length > MAX_UPLOAD_SIZE:

            raise ValueError(
                "Upload too large"
            )

        return self.rfile.read(
            length
        )

    # ========================================================
    # JSON
    # ========================================================

    def read_json(self):

        body = self.read_body()

        if not body:
            return {}

        return json.loads(
            body.decode("utf-8")
        )

    # ========================================================
    # MULTIPART UPLOAD
    # ========================================================

    def handle_upload(self):

        content_type = self.headers.get(
            "Content-Type",
            ""
        )

        if "multipart/form-data" not in content_type:

            self.send_json({
                "error":
                    "multipart/form-data required"
            }, 400)

            return

        boundary_token = None

        for part in content_type.split(";"):

            part = part.strip()

            if part.startswith(
                "boundary="
            ):

                boundary_token = (
                    part.split(
                        "=",
                        1
                    )[1]
                )

        if not boundary_token:

            self.send_json({
                "error":
                    "Multipart boundary missing"
            }, 400)

            return

        if boundary_token.startswith('"'):
            boundary_token = (
                boundary_token
                .strip('"')
            )

        body = self.read_body()

        boundary = (
            b"--"
            + boundary_token.encode()
        )

        parts = body.split(
            boundary
        )

        uploaded = []

        for part in parts:

            if (
                not part
                or part in [b"--", b"\r\n"]
            ):
                continue

            header_end = part.find(
                b"\r\n\r\n"
            )

            if header_end < 0:
                continue

            header_bytes = part[
                :header_end
            ]

            data = part[
                header_end + 4:
            ]

            if data.endswith(
                b"\r\n"
            ):
                data = data[:-2]

            headers = (
                header_bytes
                .decode(
                    "utf-8",
                    errors="ignore"
                )
            )

            if (
                'name="image"'
                not in headers
            ):
                continue

            filename = None

            marker = (
                'filename="'
            )

            if marker in headers:

                filename = (
                    headers
                    .split(
                        marker,
                        1
                    )[1]
                    .split(
                        '"',
                        1
                    )[0]
                )

            if not filename:
                continue

            filename = safe_filename(
                filename
            )

            if not valid_image_filename(
                filename
            ):

                self.send_json({
                    "error":
                        "Only JPG, JPEG, PNG or WEBP images allowed."
                }, 400)

                return

            # Avoid duplicate names.
            stem, ext = os.path.splitext(
                filename
            )

            final_filename = filename

            counter = 1

            while os.path.exists(
                image_path(
                    final_filename,
                    "train"
                )
            ):

                final_filename = (
                    f"{stem}_{counter}{ext}"
                )

                counter += 1

            target = image_path(
                final_filename,
                "train"
            )

            with open(
                target,
                "wb"
            ) as f:

                f.write(data)

            # Make sure empty label does not
            # falsely count as labeled.
            lp = label_path(
                final_filename,
                "train"
            )

            if os.path.exists(lp):

                try:
                    os.remove(lp)
                except Exception:
                    pass

            db_upsert_image(
                final_filename,
                "train",
                0,
                []
            )

            uploaded.append(
                final_filename
            )

        if not uploaded:

            self.send_json({
                "error":
                    "No image was uploaded."
            }, 400)

            return

        self.send_json({
            "success": True,
            "message":
                "Image uploaded successfully. "
                "Fungua image kwenye annotation "
                "workflow na chora BUCKET_LOADED.",
            "files": uploaded
        })

    # ========================================================
    # SAVE LABELS
    # ========================================================

    def handle_save_labels():

        data = self.read_json()

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

        boxes = data.get(
            "boxes",
            []
        )

        if split not in [
            "train",
            "val"
        ]:

            split = "train"

        if not filename:

            self.send_json({
                "error":
                    "Filename is required."
            }, 400)

            return

        if not valid_image_filename(
            filename
        ):

            self.send_json({
                "error":
                    "Invalid image filename."
            }, 400)

            return

        full_image_path = image_path(
            filename,
            split
        )

        if not os.path.exists(
            full_image_path
        ):

            self.send_json({
                "error":
                    "Image not found."
            }, 404)

            return

        yolo_boxes = boxes_to_yolo(
            boxes
        )

        if not yolo_boxes:

            self.send_json({
                "error":
                    "Hakuna box iliyowekwa. "
                    "Chora rectangle kuzunguka "
                    "loaded bucket kwanza."
            }, 400)

            return

        label_file, count = (
            write_yolo_label_file(
                filename,
                boxes,
                split
            )
        )

        # Save annotation data permanently
        # in Supabase too.
        db_upsert_image(
            filename,
            split,
            count,
            boxes
        )

        self.send_json({
            "success": True,
            "message":
                f"Saved successfully: "
                f"{os.path.basename(label_file)} "
                f"| {count} bucket(s).",
            "filename": filename,
            "split": split,
            "annotations": count,
            "label_file":
                label_file
        })

    # ========================================================
    # START TRAINING
    # ========================================================

    def handle_training_start():

        if TRAINING_STATE["status"] in [
            "preparing",
            "loading",
            "training",
            "finalizing"
        ]:

            self.send_json({
                "success": False,
                "message":
                    "Training is already running."
            })

            return

        update_training(
            status="preparing",
            message="Training queued...",
            progress=1,
            epoch=0,
            error=None
        )

        thread = threading.Thread(
            target=training_worker,
            daemon=True
        )

        thread.start()

        self.send_json({
            "success": True,
            "message":
                "Training started."
        })

    # ========================================================
    # COUNT
    # ========================================================

    def handle_count():

        data = self.read_json()

        amount = int(
            data.get(
                "count",
                1
            )
        )

        amount = max(
            1,
            min(
                amount,
                100
            )
        )

        bucket_type = str(
            data.get(
                "bucket_type",
                "DEFAULT"
            )
        )

        shift = str(
            data.get(
                "shift",
                ""
            )
        )

        conn = None

        try:

            conn = db_connect()

            if conn is None:

                self.send_json({
                    "success": False,
                    "error":
                        "DATABASE_URL haijawekwa."
                }, 500)

                return

            cur = conn.cursor()

            cur.execute("""
                INSERT INTO bucket_counts
                    (
                        bucket_type,
                        count,
                        shift,
                        source
                    )
                VALUES
                    (%s, %s, %s, %s)
            """, (
                bucket_type,
                amount,
                shift,
                "camera"
            ))

            conn.commit()

            cur.close()

            self.send_json({
                "success": True,
                "count": amount
            })

        except Exception as e:

            if conn:
                conn.rollback()

            self.send_json({
                "success": False,
                "error": str(e)
            }, 500)

        finally:

            if conn:
                conn.close()


# ============================================================
# FRONTEND
# ============================================================

MAIN_HTML = r"""
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
    font-family: Arial, sans-serif;
    background: #f2f4f7;
    color: #17202a;
}

header {
    background: #111827;
    color: white;
    padding: 16px;
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
    gap: 6px;
    padding: 10px;
    background: white;
    overflow-x: auto;
    border-bottom: 1px solid #ddd;
}

nav button {
    border: 0;
    padding: 10px 15px;
    border-radius: 8px;
    background: #e5e7eb;
    cursor: pointer;
    white-space: nowrap;
}

nav button.active {
    background: #111827;
    color: white;
}

main {
    padding: 15px;
    max-width: 1200px;
    margin: auto;
}

.tab {
    display: none;
}

.tab.active {
    display: block;
}

.card {
    background: white;
    padding: 15px;
    margin-bottom: 15px;
    border-radius: 12px;
    box-shadow:
        0 2px 8px rgba(0,0,0,.08);
}

button {
    cursor: pointer;
}

.primary {
    background: #2563eb;
    color: white;
    border: 0;
    padding: 11px 16px;
    border-radius: 8px;
}

.success {
    background: #16a34a;
    color: white;
    border: 0;
    padding: 11px 16px;
    border-radius: 8px;
}

.danger {
    background: #dc2626;
    color: white;
    border: 0;
    padding: 11px 16px;
    border-radius: 8px;
}

.secondary {
    background: #6b7280;
    color: white;
    border: 0;
    padding: 11px 16px;
    border-radius: 8px;
}

.stats {
    display: grid;
    grid-template-columns:
        repeat(auto-fit,minmax(150px,1fr));
    gap: 10px;
}

.stat {
    background: #f8fafc;
    padding: 15px;
    border-radius: 10px;
}

.stat strong {
    display: block;
    font-size: 25px;
    margin-top: 5px;
}

.progress {
    width: 100%;
    height: 18px;
    background: #e5e7eb;
    border-radius: 20px;
    overflow: hidden;
    margin-top: 10px;
}

.progress-bar {
    height: 100%;
    width: 0%;
    background: #2563eb;
    transition: width .3s;
}

table {
    width: 100%;
    border-collapse: collapse;
}

th,
td {
    border-bottom: 1px solid #ddd;
    padding: 9px;
    text-align: left;
}

.small {
    font-size: 13px;
    color: #6b7280;
}

#annotationArea {
    position: relative;
    display: inline-block;
    max-width: 100%;
    margin-top: 12px;
    background: #111;
    touch-action: none;
    user-select: none;
    -webkit-user-select: none;
}

#annotationImage {
    display: block;
    max-width: 100%;
    max-height: 65vh;
    object-fit: contain;
}

#annotationCanvas {
    position: absolute;
    left: 0;
    top: 0;
    width: 100%;
    height: 100%;
    touch-action: none;
}

.annotation-controls {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-top: 10px;
}

#selectedImageName {
    font-weight: bold;
    margin-top: 10px;
}

.message {
    padding: 10px;
    border-radius: 8px;
    background: #eef2ff;
    margin-top: 10px;
}

.camera-box {
    text-align: center;
}

video {
    width: 100%;
    max-width: 700px;
    background: black;
    border-radius: 12px;
}

footer {
    text-align: center;
    padding: 25px;
    color: #777;
}

.file-list button {
    margin: 3px;
}

@media(max-width:600px) {

    main {
        padding: 9px;
    }

    table {
        font-size: 12px;
    }

    .card {
        padding: 11px;
    }

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
    class="tabBtn active"
    onclick="showTab('dashboard',this)"
>
Dashboard
</button>

<button
    class="tabBtn"
    onclick="showTab('camera',this)"
>
Camera
</button>

<button
    class="tabBtn"
    onclick="showTab('buckets',this)"
>
Buckets
</button>

<button
    class="tabBtn"
    onclick="showTab('training',this)"
>
Training
</button>

<button
    class="tabBtn"
    onclick="showTab('history',this)"
>
History
</button>

<button
    class="tabBtn"
    onclick="showTab('settings',this)"
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
    class="tab active"
>

<div class="card">

<h2>
Dashboard
</h2>

<div class="stats">

<div class="stat">
Training Images
<strong id="dashTrainImages">0</strong>
</div>

<div class="stat">
Training Labels
<strong id="dashTrainLabels">0</strong>
</div>

<div class="stat">
Unlabeled
<strong id="dashUnlabeled">0</strong>
</div>

<div class="stat">
Model
<strong id="dashModel">No</strong>
</div>

</div>

</div>

</section>


<!-- =====================================================
     CAMERA
===================================================== -->

<section
    id="camera"
    class="tab"
>

<div class="card camera-box">

<h2>
Camera
</h2>

<video
    id="cameraVideo"
    autoplay
    playsinline
>
</video>

<br><br>

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

<div
    class="message"
    id="cameraMessage"
>
Camera ready.
</div>

</div>

</section>


<!-- =====================================================
     BUCKETS
===================================================== -->

<section
    id="buckets"
    class="tab"
>

<div class="card">

<h2>
Bucket Type
</h2>

<p>
Active detection class:
<strong>
BUCKET_LOADED
</strong>
</p>

<p class="small">
Only loaded ore/material buckets should be
annotated. Do not annotate empty buckets,
people or equipment.
</p>

</div>

</section>


<!-- =====================================================
     TRAINING
===================================================== -->

<section
    id="training"
    class="tab"
>

<div class="card">

<h2>
YOLO Training Dataset
</h2>

<p>
Upload images and annotate them as
<strong>
BUCKET_LOADED
</strong>.
</p>

<input
    id="imageUpload"
    type="file"
    accept="image/jpeg,image/png,image/webp"
>

<br><br>

<button
    class="primary"
    onclick="uploadImage()"
>
Upload Image
</button>

<div
    id="uploadMessage"
    class="message"
>
No file uploaded yet.
</div>

</div>


<div class="card">

<h3>
Annotation Workflow
</h3>

<div
    id="selectedImageName"
>
Chagua picha kwenye dataset hapa chini.
</div>

<div id="annotationArea">

<img
    id="annotationImage"
    draggable="false"
>

<canvas
    id="annotationCanvas"
>
</canvas>

</div>

<div class="annotation-controls">

<button
    class="secondary"
    onclick="undoBox()"
>
Undo Last
</button>

<button
    class="danger"
    onclick="clearBoxes()"
>
Clear All
</button>

<button
    class="success"
    onclick="saveLabels()"
>
Save YOLO Labels
</button>

</div>

<div
    id="annotationMessage"
    class="message"
>
Chagua image kisha chora rectangle
kuzunguka loaded bucket.
</div>

</div>


<div class="card">

<h3>
Training Status
</h3>

<p>
Status:
<strong id="trainingStatus">
idle
</strong>
</p>

<p id="trainingMessage">
Ready
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
<strong id="trainingEpoch">
0
</strong>
/
<span id="trainingEpochTotal">
20
</span>
</p>

<button
    class="primary"
    onclick="startTraining()"
>
Start Training
</button>

<button
    class="secondary"
    onclick="refreshTraining()"
>
Refresh
</button>

</div>


<div class="card">

<h3>
Training Settings
</h3>

<div class="stats">

<div class="stat">
CPU
<strong>CPU</strong>
</div>

<div class="stat">
Image Size
<strong>320</strong>
</div>

<div class="stat">
Batch
<strong>1</strong>
</div>

<div class="stat">
Workers
<strong>0</strong>
</div>

<div class="stat">
Epochs
<strong>20</strong>
</div>

</div>

<p class="small">
These settings are optimized for Render Free.
</p>

</div>


<div class="card">

<h3>
Dataset
</h3>

<div
    class="stats"
    id="datasetStats"
>
</div>

<br>

<table>

<thead>

<tr>
<th>Split</th>
<th>Filename</th>
<th>Annotations</th>
<th>Action</th>
</tr>

</thead>

<tbody
    id="datasetTable"
>
</tbody>

</table>

</div>

</section>


<!-- =====================================================
     HISTORY
===================================================== -->

<section
    id="history"
    class="tab"
>

<div class="card">

<h2>
History
</h2>

<p>
Production bucket count history will appear
here after counting is connected to the
camera model.
</p>

</div>

</section>


<!-- =====================================================
     SETTINGS
===================================================== -->

<section
    id="settings"
    class="tab"
>

<div class="card">

<h2>
Settings
</h2>

<p>
Detection class:
<strong>
BUCKET_LOADED
</strong>
</p>

<p>
Image size:
<strong>
320
</strong>
</p>

<p>
Training device:
<strong>
CPU
</strong>
</p>

<p>
Workers:
<strong>
0
</strong>
</p>

</div>

</section>

</main>


<footer>
Geology & Mining Services
</footer>


<script>

/* =========================================================
   GLOBAL STATE
========================================================= */

let currentFilename = "";
let currentSplit = "train";

let boxes = [];

let drawing = false;

let startX = 0;
let startY = 0;

let currentBox = null;

let cameraStream = null;


/* =========================================================
   TAB
========================================================= */

function showTab(name, button) {

    document
        .querySelectorAll(".tab")
        .forEach(function(tab) {
            tab.classList.remove("active");
        });

    document
        .querySelectorAll(".tabBtn")
        .forEach(function(btn) {
            btn.classList.remove("active");
        });

    const target =
        document.getElementById(name);

    if (target) {
        target.classList.add("active");
    }

    if (button) {
        button.classList.add("active");
    }

    if (name === "training") {
        loadDataset();
        refreshTraining();
    }

}


/* =========================================================
   UPLOAD
========================================================= */

async function uploadImage() {

    const input =
        document.getElementById(
            "imageUpload"
        );

    const message =
        document.getElementById(
            "uploadMessage"
        );

    if (!input.files.length) {

        message.textContent =
            "Chagua picha kwanza.";

        return;
    }

    const formData =
        new FormData();

    formData.append(
        "image",
        input.files[0]
    );

    message.textContent =
        "Uploading...";

    try {

        const response =
            await fetch(
                "/api/dataset/upload",
                {
                    method: "POST",
                    body: formData
                }
            );

        const data =
            await response.json();

        if (!response.ok) {

            throw new Error(
                data.error ||
                "Upload failed"
            );
        }

        message.textContent =
            data.message ||
            "Image uploaded successfully.";

        input.value = "";

        await loadDataset();

        if (
            data.files &&
            data.files.length
        ) {

            openImage(
                data.files[0],
                "train"
            );

        }

    } catch (error) {

        message.textContent =
            "Upload error: "
            + error.message;
    }

}


/* =========================================================
   DATASET
========================================================= */

async function loadDataset() {

    try {

        const statsResponse =
            await fetch(
                "/api/dataset/stats"
            );

        const stats =
            await statsResponse.json();

        document.getElementById(
            "datasetStats"
        ).innerHTML = `

            <div class="stat">
                Training Images
                <strong>
                    ${stats.training_images}
                </strong>
            </div>

            <div class="stat">
                Validation Images
                <strong>
                    ${stats.validation_images}
                </strong>
            </div>

            <div class="stat">
                Training Labels
                <strong>
                    ${stats.training_labels}
                </strong>
            </div>

            <div class="stat">
                Validation Labels
                <strong>
                    ${stats.validation_labels}
                </strong>
            </div>

            <div class="stat">
                Unlabeled Training
                <strong>
                    ${stats.unlabeled_training}
                </strong>
            </div>

            <div class="stat">
                Unlabeled Validation
                <strong>
                    ${stats.unlabeled_validation}
                </strong>
            </div>

        `;

        document.getElementById(
            "dashTrainImages"
        ).textContent =
            stats.training_images;

        document.getElementById(
            "dashTrainLabels"
        ).textContent =
            stats.training_labels;

        document.getElementById(
            "dashUnlabeled"
        ).textContent =
            stats.unlabeled_training;


        const response =
            await fetch(
                "/api/dataset/images"
            );

        const data =
            await response.json();

        const tbody =
            document.getElementById(
                "datasetTable"
            );

        tbody.innerHTML = "";

        (data.images || [])
            .forEach(function(item) {

                const tr =
                    document.createElement(
                        "tr"
                    );

                tr.innerHTML = `

                    <td>
                        ${escapeHtml(
                            item.split
                        )}
                    </td>

                    <td>
                        ${escapeHtml(
                            item.filename
                        )}
                    </td>

                    <td>
                        ${item.annotations}
                    </td>

                    <td>

                        <button
                            class="primary"
                            onclick='openImage(
                                ${JSON.stringify(
                                    item.filename
                                )},
                                ${JSON.stringify(
                                    item.split
                                )}
                            )'
                        >
                            Annotate
                        </button>

                    </td>

                `;

                tbody.appendChild(tr);

            });

    } catch (error) {

        console.error(
            "Dataset error",
            error
        );
    }

}


/* =========================================================
   OPEN IMAGE
========================================================= */

async function openImage(
    filename,
    split
) {

    currentFilename =
        filename;

    currentSplit =
        split || "train";

    const message =
        document.getElementById(
            "annotationMessage"
        );

    message.textContent =
        "Loading image...";

    try {

        const response =
            await fetch(
                "/api/dataset/image"
                + "?filename="
                + encodeURIComponent(
                    filename
                )
                + "&split="
                + encodeURIComponent(
                    currentSplit
                )
            );

        const data =
            await response.json();

        if (!response.ok) {

            throw new Error(
                data.error ||
                "Image not found"
            );
        }

        document.getElementById(
            "selectedImageName"
        ).textContent =
            filename
            + " | "
            + currentSplit;

        const img =
            document.getElementById(
                "annotationImage"
            );

        img.onload = function() {

            resizeCanvas();

            boxes = Array.isArray(
                data.boxes
            )
                ? data.boxes
                : [];

            drawBoxes();

            message.textContent =
                "Chora rectangle kuzunguka "
                + "BUCKET_LOADED. "
                + "Box itabaki baada ya kutoa kidole.";

        };

        img.src =
            data.data;

    } catch (error) {

        message.textContent =
            "Image error: "
            + error.message;
    }

}


/* =========================================================
   CANVAS
========================================================= */

const canvas =
    document.getElementById(
        "annotationCanvas"
    );

const ctx =
    canvas.getContext(
        "2d"
    );

const image =
    document.getElementById(
        "annotationImage"
    );


function resizeCanvas() {

    if (!image.naturalWidth) {
        return;
    }

    const rect =
        image.getBoundingClientRect();

    canvas.width =
        image.clientWidth;

    canvas.height =
        image.clientHeight;

    canvas.style.width =
        image.clientWidth + "px";

    canvas.style.height =
        image.clientHeight + "px";

    drawBoxes();
}


window.addEventListener(
    "resize",
    resizeCanvas
);


/* =========================================================
   POINTER POSITION
========================================================= */

function getPointerPosition(event) {

    const rect =
        canvas.getBoundingClientRect();

    let x =
        event.clientX
        - rect.left;

    let y =
        event.clientY
        - rect.top;

    x = Math.max(
        0,
        Math.min(
            canvas.clientWidth,
            x
        )
    );

    y = Math.max(
        0,
        Math.min(
            canvas.clientHeight,
            y
        )
    );

    return {
        x: x,
        y: y
    };

}


/* =========================================================
   POINTER DOWN
========================================================= */

canvas.addEventListener(
    "pointerdown",
    function(event) {

        if (!currentFilename) {
            return;
        }

        event.preventDefault();

        canvas.setPointerCapture(
            event.pointerId
        );

        const pos =
            getPointerPosition(
                event
            );

        startX =
            pos.x;

        startY =
            pos.y;

        currentBox = {
            x: startX,
            y: startY,
            width: 0,
            height: 0
        };

        drawing = true;

    }
);


/* =========================================================
   POINTER MOVE
========================================================= */

canvas.addEventListener(
    "pointermove",
    function(event) {

        if (!drawing) {
            return;
        }

        event.preventDefault();

        const pos =
            getPointerPosition(
                event
            );

        let x =
            Math.min(
                startX,
                pos.x
            );

        let y =
            Math.min(
                startY,
                pos.y
            );

        let width =
            Math.abs(
                pos.x
                - startX
            );

        let height =
            Math.abs(
                pos.y
                - startY
            );

        currentBox = {
            x: x,
            y: y,
            width: width,
            height: height
        };

        drawBoxes();

        drawCurrentBox();

    }
);


/* =========================================================
   POINTER UP
========================================================= */

canvas.addEventListener(
    "pointerup",
    finishDrawing
);

canvas.addEventListener(
    "pointercancel",
    finishDrawing
);


function finishDrawing(event) {

    if (!drawing) {
        return;
    }

    event.preventDefault();

    drawing = false;

    try {

        canvas.releasePointerCapture(
            event.pointerId
        );

    } catch (e) {}

    if (
        currentBox &&
        currentBox.width >= 8 &&
        currentBox.height >= 8
    ) {

        const normalized =
            pixelBoxToNormalized(
                currentBox
            );

        boxes.push(
            normalized
        );

    }

    currentBox = null;

    drawBoxes();

}


/* =========================================================
   PIXEL → NORMALIZED
========================================================= */

function pixelBoxToNormalized(
    box
) {

    const width =
        canvas.clientWidth;

    const height =
        canvas.clientHeight;

    return {
        x:
            box.x / width,

        y:
            box.y / height,

        width:
            box.width / width,

        height:
            box.height / height,

        class_id: 0
    };

}


/* =========================================================
   NORMALIZED → PIXEL
========================================================= */

function normalizedBoxToPixel(
    box
) {

    return {
        x:
            box.x
            * canvas.clientWidth,

        y:
            box.y
            * canvas.clientHeight,

        width:
            box.width
            * canvas.clientWidth,

        height:
            box.height
            * canvas.clientHeight
    };

}


/* =========================================================
   DRAW BOXES
========================================================= */

function drawBoxes() {

    ctx.clearRect(
        0,
        0,
        canvas.width,
        canvas.height
    );

    boxes.forEach(
        function(box, index) {

            const b =
                normalizedBoxToPixel(
                    box
                );

            ctx.lineWidth = 3;

            ctx.strokeStyle =
                "#00ff00";

            ctx.strokeRect(
                b.x,
                b.y,
                b.width,
                b.height
            );

            ctx.fillStyle =
                "rgba(0,255,0,.18)";

            ctx.fillRect(
                b.x,
                b.y,
                b.width,
                b.height
            );

            ctx.fillStyle =
                "#00ff00";

            ctx.font =
                "bold 16px Arial";

            ctx.fillText(
                "BUCKET_LOADED "
                + (index + 1),
                b.x + 5,
                b.y + 20
            );

        }
    );

}


/* =========================================================
   DRAW CURRENT BOX
========================================================= */

function drawCurrentBox() {

    if (!currentBox) {
        return;
    }

    ctx.lineWidth = 3;

    ctx.strokeStyle =
        "#ffff00";

    ctx.strokeRect(
        currentBox.x,
        currentBox.y,
        currentBox.width,
        currentBox.height
    );

}


/* =========================================================
   UNDO
========================================================= */

function undoBox() {

    if (!boxes.length) {

        document.getElementById(
            "annotationMessage"
        ).textContent =
            "Hakuna box ya kuondoa.";

        return;
    }

    boxes.pop();

    drawBoxes();

    document.getElementById(
        "annotationMessage"
    ).textContent =
        "Box ya mwisho imeondolewa.";

}


/* =========================================================
   CLEAR
========================================================= */

function clearBoxes() {

    boxes = [];

    drawBoxes();

    document.getElementById(
        "annotationMessage"
    ).textContent =
        "Boxes zote zimeondolewa. "
        + "Chora tena kwenye bucket.";

}


/* =========================================================
   SAVE LABELS
========================================================= */

async function saveLabels() {

    const message =
        document.getElementById(
            "annotationMessage"
        );

    if (!currentFilename) {

        message.textContent =
            "Chagua image kwanza.";

        return;
    }

    if (!boxes.length) {

        message.textContent =
            "Hakuna box. "
            + "Chora rectangle kwenye loaded bucket.";

        return;
    }

    message.textContent =
        "Saving YOLO labels...";

    try {

        const response =
            await fetch(
                "/api/dataset/labels",
                {
                    method: "POST",
                    headers: {
                        "Content-Type":
                            "application/json"
                    },
                    body: JSON.stringify({
                        filename:
                            currentFilename,

                        split:
                            currentSplit,

                        boxes:
                            boxes
                    })
                }
            );

        const data =
            await response.json();

        if (!response.ok) {

            throw new Error(
                data.error ||
                "Save failed"
            );
        }

        message.textContent =
            data.message;

        await loadDataset();

    } catch (error) {

        message.textContent =
            "Save error: "
            + error.message;
    }

}


/* =========================================================
   TRAINING
========================================================= */

async function startTraining() {

    try {

        const statsResponse =
            await fetch(
                "/api/dataset/stats"
            );

        const stats =
            await statsResponse.json();

        if (
            stats.training_labels < 1
        ) {

            alert(
                "Huwezi kuanza training. "
                + "Training Labels = 0. "
                + "Annotate image kwanza."
            );

            return;
        }

        const response =
            await fetch(
                "/api/training/start",
                {
                    method: "POST"
                }
            );

        const data =
            await response.json();

        alert(
            data.message ||
            "Training started."
        );

        refreshTraining();

    } catch (error) {

        alert(
            "Training error: "
            + error.message
        );

    }

}


/* =========================================================
   TRAINING STATUS
========================================================= */

async function refreshTraining() {

    try {

        const response =
            await fetch(
                "/api/training/status"
            );

        const data =
            await response.json();

        document.getElementById(
            "trainingStatus"
        ).textContent =
            data.status || "idle";

        document.getElementById(
            "trainingMessage"
        ).textContent =
            data.message || "Ready";

        document.getElementById(
            "trainingProgress"
        ).style.width =
            (
                data.progress || 0
            ) + "%";

        document.getElementById(
            "trainingEpoch"
        ).textContent =
            data.epoch || 0;

        document.getElementById(
            "trainingEpochTotal"
        ).textContent =
            data.epochs || 20;

        document.getElementById(
            "dashModel"
        ).textContent =
            data.model_exists
                ? "Yes"
                : "No";

    } catch (error) {

        console.error(
            error
        );

    }

}


/* =========================================================
   AUTO STATUS POLLING
========================================================= */

setInterval(
    refreshTraining,
    3000
);


/* =========================================================
   CAMERA
========================================================= */

async function startCamera() {

    const video =
        document.getElementById(
            "cameraVideo"
        );

    const message =
        document.getElementById(
            "cameraMessage"
        );

    try {

        cameraStream =
            await navigator.mediaDevices
                .getUserMedia({
                    video: {
                        facingMode:
                            "environment"
                    },
                    audio: false
                });

        video.srcObject =
            cameraStream;

        message.textContent =
            "Camera started.";

    } catch (error) {

        message.textContent =
            "Camera error: "
            + error.message;

    }

}


function stopCamera() {

    const video =
        document.getElementById(
            "cameraVideo"
        );

    if (cameraStream) {

        cameraStream
            .getTracks()
            .forEach(
                function(track) {
                    track.stop();
                }
            );

        cameraStream = null;
    }

    video.srcObject = null;

    document.getElementById(
        "cameraMessage"
    ).textContent =
        "Camera stopped.";

}


/* =========================================================
   ESCAPE HTML
========================================================= */

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


/* =========================================================
   INITIAL LOAD
========================================================= */

loadDataset();

refreshTraining();

</script>

</body>
</html>
"""


# ============================================================
# SERVER
# ============================================================

def run_server():

    port = int(
        os.environ.get(
            "PORT",
            "10000"
        )
    )

    server = ThreadingHTTPServer(
        ("0.0.0.0", port),
        Handler
    )

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
        "Server running on port:",
        port
    )

    print(
        "YOLO_AVAILABLE:",
        YOLO_AVAILABLE
    )

    print(
        "POSTGRES_AVAILABLE:",
        POSTGRES_AVAILABLE
    )

    print(
        "EPOCHS:",
        EPOCHS
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
        "=========================================="
    )

    server.serve_forever()


if __name__ == "__main__":
    run_server()
