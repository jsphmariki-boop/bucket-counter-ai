import os
import io
import json
import traceback
import threading
import shutil
import uuid
from datetime import datetime
from urllib.parse import urlparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# ============================================================
# NEERIKA BUCKET AI
# Mining Production Bucket Counter
# FULL app.py
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


# ============================================================
# CONFIGURATION
# ============================================================

PORT = int(os.environ.get("PORT", "8080"))

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATABASE_URL = os.environ.get("DATABASE_URL", "").strip()

AI_DIR = os.path.join(BASE_DIR, "ai")
DATASET_DIR = os.path.join(AI_DIR, "dataset")

IMAGES_DIR = os.path.join(DATASET_DIR, "images")
LABELS_DIR = os.path.join(DATASET_DIR, "labels")

TRAIN_IMAGES_DIR = os.path.join(IMAGES_DIR, "train")
VAL_IMAGES_DIR = os.path.join(IMAGES_DIR, "val")

TRAIN_LABELS_DIR = os.path.join(LABELS_DIR, "train")
VAL_LABELS_DIR = os.path.join(LABELS_DIR, "val")

MODELS_DIR = os.path.join(AI_DIR, "models")

REFERENCE_DIR = os.path.join(BASE_DIR, "reference_images")

UPLOAD_DIR = os.path.join(BASE_DIR, "bucket_images")

YOLO_YAML = os.path.join(
    DATASET_DIR,
    "dataset.yaml"
)

TRAINED_MODEL = os.path.join(
    MODELS_DIR,
    "bucket_best.pt"
)

DEFAULT_MODEL = os.environ.get(
    "YOLO_MODEL",
    "yolo11n.pt"
)


# ============================================================
# CREATE DIRECTORIES
# ============================================================

DIRECTORIES = [
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
    UPLOAD_DIR,
]

for directory in DIRECTORIES:
    os.makedirs(directory, exist_ok=True)


# ============================================================
# TRAINING STATE
# ============================================================

TRAINING_LOCK = threading.Lock()

TRAINING_STATE = {
    "status": "idle",
    "message": "Training has not started.",
    "progress": 0,
    "epoch": 0,
    "total_epochs": 0,
    "images": 0,
    "labels": 0,
    "started_at": None,
    "finished_at": None,
    "error": None,
}


# ============================================================
# DATABASE
# ============================================================

def db_connect():

    if not DATABASE_URL:
        return None

    if psycopg2 is None:
        return None

    try:
        return psycopg2.connect(
            DATABASE_URL,
            connect_timeout=10
        )
    except Exception as e:
        print("DATABASE ERROR:", e)
        return None


def init_database():

    conn = db_connect()

    if conn is None:
        print("Database not configured.")
        return

    try:

        cur = conn.cursor()

        cur.execute("""
            CREATE TABLE IF NOT EXISTS bucket_counts (
                id BIGSERIAL PRIMARY KEY,
                bucket_type TEXT DEFAULT 'default',
                count INTEGER DEFAULT 1,
                shift TEXT DEFAULT '',
                operator_name TEXT DEFAULT '',
                source TEXT DEFAULT 'camera',
                confidence DOUBLE PRECISION DEFAULT 0,
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)

        conn.commit()

        cur.close()
        conn.close()

        print("Database initialized.")

    except Exception as e:

        print("DATABASE INIT ERROR:", e)

        try:
            conn.rollback()
            conn.close()
        except Exception:
            pass


def save_bucket_count(
    bucket_type="default",
    count=1,
    shift="",
    operator_name="",
    source="camera",
    confidence=0
):

    conn = db_connect()

    if conn is None:
        return False

    try:

        cur = conn.cursor()

        cur.execute("""
            INSERT INTO bucket_counts
            (
                bucket_type,
                count,
                shift,
                operator_name,
                source,
                confidence
            )
            VALUES (%s,%s,%s,%s,%s,%s)
        """, (
            bucket_type,
            int(count),
            shift,
            operator_name,
            source,
            float(confidence)
        ))

        conn.commit()

        cur.close()
        conn.close()

        return True

    except Exception as e:

        print("SAVE COUNT ERROR:", e)

        try:
            conn.rollback()
            conn.close()
        except Exception:
            pass

        return False


def get_history():

    conn = db_connect()

    if conn is None:
        return []

    try:

        cur = conn.cursor(
            cursor_factory=RealDictCursor
        )

        cur.execute("""
            SELECT
                id,
                bucket_type,
                count,
                shift,
                operator_name,
                source,
                confidence,
                created_at
            FROM bucket_counts
            ORDER BY created_at DESC
            LIMIT 200
        """)

        rows = cur.fetchall()

        result = []

        for row in rows:

            item = dict(row)

            if item.get("created_at"):
                item["created_at"] = (
                    item["created_at"].isoformat()
                )

            result.append(item)

        cur.close()
        conn.close()

        return result

    except Exception as e:

        print("HISTORY ERROR:", e)

        try:
            conn.close()
        except Exception:
            pass

        return []


def get_total_count():

    conn = db_connect()

    if conn is None:
        return 0

    try:

        cur = conn.cursor()

        cur.execute("""
            SELECT COALESCE(SUM(count),0)
            FROM bucket_counts
        """)

        value = cur.fetchone()[0]

        cur.close()
        conn.close()

        return int(value or 0)

    except Exception:

        try:
            conn.close()
        except Exception:
            pass

        return 0


# ============================================================
# FILE / DATASET HELPERS
# ============================================================

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp"
}


def is_image(filename):

    extension = os.path.splitext(
        filename
    )[1].lower()

    return extension in IMAGE_EXTENSIONS


def list_images(directory):

    os.makedirs(
        directory,
        exist_ok=True
    )

    result = []

    for filename in os.listdir(directory):

        path = os.path.join(
            directory,
            filename
        )

        if (
            os.path.isfile(path)
            and is_image(filename)
        ):
            result.append(path)

    return result


def list_labels(directory):

    os.makedirs(
        directory,
        exist_ok=True
    )

    result = []

    for filename in os.listdir(directory):

        if filename.lower().endswith(".txt"):

            path = os.path.join(
                directory,
                filename
            )

            if os.path.isfile(path):
                result.append(path)

    return result


def dataset_statistics():

    train_images = list_images(
        TRAIN_IMAGES_DIR
    )

    val_images = list_images(
        VAL_IMAGES_DIR
    )

    train_labels = list_labels(
        TRAIN_LABELS_DIR
    )

    val_labels = list_labels(
        VAL_LABELS_DIR
    )

    return {
        "train_images": len(train_images),
        "val_images": len(val_images),
        "train_labels": len(train_labels),
        "val_labels": len(val_labels),
        "images": (
            len(train_images)
            + len(val_images)
        ),
        "labels": (
            len(train_labels)
            + len(val_labels)
        )
    }


def create_dataset_yaml():

    os.makedirs(
        DATASET_DIR,
        exist_ok=True
    )

    root = DATASET_DIR.replace(
        "\\",
        "/"
    )

    yaml_text = (
        "path: " + root + "\n"
        "train: images/train\n"
        "val: images/val\n"
        "\n"
        "names:\n"
        "  0: BUCKET_LOADED\n"
    )

    with open(
        YOLO_YAML,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(yaml_text)

    return YOLO_YAML


# ============================================================
# UPLOAD IMAGE
# ============================================================

def save_training_image(
    original_filename,
    data
):

    extension = os.path.splitext(
        original_filename
    )[1].lower()

    if extension not in IMAGE_EXTENSIONS:
        extension = ".jpg"

    filename = (
        datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )
        + "_"
        + uuid.uuid4().hex[:8]
        + extension
    )

    path = os.path.join(
        TRAIN_IMAGES_DIR,
        filename
    )

    with open(
        path,
        "wb"
    ) as f:

        f.write(data)

    return filename


# ============================================================
# ANNOTATION
# ============================================================

def save_annotation(
    filename,
    boxes
):

    image_path = os.path.join(
        TRAIN_IMAGES_DIR,
        filename
    )

    if not os.path.isfile(
        image_path
    ):
        raise FileNotFoundError(
            "Training image not found."
        )

    image_name = os.path.splitext(
        filename
    )[0]

    label_filename = (
        image_name + ".txt"
    )

    label_path = os.path.join(
        TRAIN_LABELS_DIR,
        label_filename
    )

    lines = []

    for box in boxes:

        try:

            x = float(box["x"])
            y = float(box["y"])
            w = float(box["width"])
            h = float(box["height"])

        except Exception:

            continue

        # Clamp values
        x = max(0.0, min(1.0, x))
        y = max(0.0, min(1.0, y))
        w = max(0.0, min(1.0, w))
        h = max(0.0, min(1.0, h))

        if w <= 0 or h <= 0:
            continue

        # YOLO:
        # class center_x center_y width height

        lines.append(
            "0 %.6f %.6f %.6f %.6f"
            % (
                x,
                y,
                w,
                h
            )
        )

    if not lines:

        raise ValueError(
            "No valid annotation boxes supplied."
        )

    with open(
        label_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "\n".join(lines)
            + "\n"
        )

    create_dataset_yaml()

    return label_filename


def get_training_images():

    result = []

    for path in list_images(
        TRAIN_IMAGES_DIR
    ):

        filename = os.path.basename(
            path
        )

        label_filename = (
            os.path.splitext(filename)[0]
            + ".txt"
        )

        label_exists = os.path.isfile(
            os.path.join(
                TRAIN_LABELS_DIR,
                label_filename
            )
        )

        result.append({
            "filename": filename,
            "url":
                "/training-image/"
                + filename,
            "annotated":
                label_exists
        })

    return result


# ============================================================
# VALIDATE DATASET
# ============================================================

def validate_dataset():

    stats = dataset_statistics()

    if stats["train_images"] == 0:

        return (
            False,
            "No training images found. Upload images first.",
            stats
        )

    if stats["train_labels"] == 0:

        return (
            False,
            "No YOLO labels found. Annotate your images first.",
            stats
        )

    image_names = set()

    for path in list_images(
        TRAIN_IMAGES_DIR
    ):

        image_names.add(
            os.path.splitext(
                os.path.basename(path)
            )[0]
        )

    label_names = set()

    for path in list_labels(
        TRAIN_LABELS_DIR
    ):

        label_names.add(
            os.path.splitext(
                os.path.basename(path)
            )[0]
        )

    matched = image_names.intersection(
        label_names
    )

    if not matched:

        return (
            False,
            "No image has a matching YOLO label.",
            stats
        )

    return (
        True,
        "Dataset is ready.",
        stats
    )


# ============================================================
# CREATE VALIDATION DATA
# ============================================================

def prepare_validation_data():

    train_images = list_images(
        TRAIN_IMAGES_DIR
    )

    if len(train_images) < 2:

        return

    # Clear old validation set
    for path in list_images(
        VAL_IMAGES_DIR
    ):

        try:
            os.remove(path)
        except Exception:
            pass

    for path in list_labels(
        VAL_LABELS_DIR
    ):

        try:
            os.remove(path)
        except Exception:
            pass

    # Use approximately 20% for validation.
    # At least one image if there are 2+ images.
    val_count = max(
        1,
        int(len(train_images) * 0.2)
    )

    selected = train_images[
        -val_count:
    ]

    for image_path in selected:

        filename = os.path.basename(
            image_path
        )

        label_name = (
            os.path.splitext(filename)[0]
            + ".txt"
        )

        label_path = os.path.join(
            TRAIN_LABELS_DIR,
            label_name
        )

        if not os.path.isfile(
            label_path
        ):
            continue

        shutil.copy2(
            image_path,
            os.path.join(
                VAL_IMAGES_DIR,
                filename
            )
        )

        shutil.copy2(
            label_path,
            os.path.join(
                VAL_LABELS_DIR,
                label_name
            )
        )


# ============================================================
# TRAINING
# ============================================================

def set_training(**kwargs):

    with TRAINING_LOCK:

        TRAINING_STATE.update(
            kwargs
        )


def run_training(epochs):

    try:

        set_training(
            status="preparing",
            message="Preparing dataset...",
            progress=1,
            total_epochs=epochs,
            error=None
        )

        valid, message, stats = (
            validate_dataset()
        )

        set_training(
            images=stats["images"],
            labels=stats["labels"]
        )

        if not valid:

            set_training(
                status="error",
                message=message,
                progress=0,
                error=message,
                finished_at=datetime.now().isoformat()
            )

            return

        if YOLO is None:

            message = (
                "Ultralytics is not installed. "
                "Add ultralytics to requirements.txt."
            )

            set_training(
                status="error",
                message=message,
                error=message
            )

            return

        prepare_validation_data()

        create_dataset_yaml()

        set_training(
            status="loading_model",
            message="Loading YOLO model...",
            progress=3
        )

        model_path = DEFAULT_MODEL

        if os.path.isfile(
            TRAINED_MODEL
        ):

            model_path = TRAINED_MODEL

        print(
            "YOLO MODEL:",
            model_path
        )

        model = YOLO(
            model_path
        )

        set_training(
            status="training",
            message=(
                "Training YOLO for "
                + str(epochs)
                + " epochs..."
            ),
            progress=5
        )

        model.train(
            data=YOLO_YAML,
            epochs=int(epochs),
            imgsz=640,
            batch=4,
            workers=0,
            project=MODELS_DIR,
            name="bucket_training",
            exist_ok=True,
            pretrained=True,
            verbose=True
        )

        best_model = os.path.join(
            MODELS_DIR,
            "bucket_training",
            "weights",
            "best.pt"
        )

        if os.path.isfile(
            best_model
        ):

            shutil.copy2(
                best_model,
                TRAINED_MODEL
            )

        set_training(
            status="completed",
            message=(
                "YOLO training completed successfully."
            ),
            progress=100,
            epoch=epochs,
            total_epochs=epochs,
            finished_at=datetime.now().isoformat(),
            error=None
        )

    except Exception as e:

        error = traceback.format_exc()

        print(
            "TRAINING ERROR:"
        )

        print(error)

        set_training(
            status="error",
            message=str(e),
            progress=0,
            error=str(e),
            finished_at=datetime.now().isoformat()
        )


def start_training(epochs):

    with TRAINING_LOCK:

        if TRAINING_STATE["status"] in [
            "preparing",
            "loading_model",
            "training"
        ]:

            return (
                False,
                "Training is already running."
            )

        TRAINING_STATE.update({
            "status": "preparing",
            "message": "Preparing training...",
            "progress": 0,
            "epoch": 0,
            "total_epochs": epochs,
            "started_at":
                datetime.now().isoformat(),
            "finished_at": None,
            "error": None
        })

    thread = threading.Thread(
        target=run_training,
        args=(epochs,),
        daemon=True
    )

    thread.start()

    return (
        True,
        "Training started."
    )


# ============================================================
# REFERENCE IMAGES
# ============================================================

def save_reference(
    filename,
    data
):

    extension = os.path.splitext(
        filename
    )[1].lower()

    if extension not in IMAGE_EXTENSIONS:
        extension = ".jpg"

    new_name = (
        datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )
        + "_"
        + uuid.uuid4().hex[:8]
        + extension
    )

    path = os.path.join(
        REFERENCE_DIR,
        new_name
    )

    with open(
        path,
        "wb"
    ) as f:

        f.write(data)

    return new_name


def reference_images():

    result = []

    for filename in os.listdir(
        REFERENCE_DIR
    ):

        path = os.path.join(
            REFERENCE_DIR,
            filename
        )

        if (
            os.path.isfile(path)
            and is_image(filename)
        ):

            result.append({
                "name": filename,
                "url":
                    "/reference/"
                    + filename
            })

    return result


# ============================================================
# MULTIPART PARSER
# ============================================================

def parse_multipart(handler):

    content_type = handler.headers.get(
        "Content-Type",
        ""
    )

    if "multipart/form-data" not in content_type:
        return {}

    boundary = None

    for part in content_type.split(";"):

        part = part.strip()

        if part.startswith(
            "boundary="
        ):

            boundary = part.split(
                "=",
                1
            )[1].strip('"')

    if not boundary:
        return {}

    length = int(
        handler.headers.get(
            "Content-Length",
            "0"
        )
    )

    raw = handler.rfile.read(
        length
    )

    boundary_bytes = (
        b"--"
        + boundary.encode()
    )

    result = {}

    for block in raw.split(
        boundary_bytes
    ):

        if not block:
            continue

        if block in [
            b"--",
            b"--\r\n"
        ]:
            continue

        block = block.strip(
            b"\r\n-"
        )

        header_end = block.find(
            b"\r\n\r\n"
        )

        if header_end < 0:
            continue

        headers = block[
            :header_end
        ].decode(
            "utf-8",
            errors="ignore"
        )

        data = block[
            header_end + 4:
        ]

        disposition = ""

        for line in headers.split(
            "\r\n"
        ):

            if line.lower().startswith(
                "content-disposition:"
            ):

                disposition = line

        name = None
        filename = None

        if 'name="' in disposition:

            name = disposition.split(
                'name="',
                1
            )[1].split(
                '"',
                1
            )[0]

        if 'filename="' in disposition:

            filename = disposition.split(
                'filename="',
                1
            )[1].split(
                '"',
                1
            )[0]

        if name:

            result[name] = {
                "filename": filename,
                "data": data
            }

    return result


# ============================================================
# RESPONSE HELPERS
# ============================================================

def send_json(
    handler,
    data,
    status=200
):

    body = json.dumps(
        data,
        ensure_ascii=False,
        default=str
    ).encode("utf-8")

    handler.send_response(
        status
    )

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
        "no-cache"
    )

    handler.end_headers()

    handler.wfile.write(
        body
    )


def send_html(
    handler,
    html,
    status=200
):

    body = html.encode(
        "utf-8"
    )

    handler.send_response(
        status
    )

    handler.send_header(
        "Content-Type",
        "text/html; charset=utf-8"
    )

    handler.send_header(
        "Content-Length",
        str(len(body))
    )

    handler.end_headers()

    handler.wfile.write(
        body
    )


def send_not_found(handler):

    send_json(
        handler,
        {
            "ok": False,
            "error": "Not found"
        },
        404
    )


# ============================================================
# HTML
# ============================================================

HTML = r"""
<!DOCTYPE html>
<html lang="en">

<head>

<meta charset="UTF-8">

<meta name="viewport"
content="width=device-width,initial-scale=1.0">

<title>NEERIKA BUCKET AI</title>

<style>

* {
    box-sizing: border-box;
}

body {
    margin: 0;
    font-family: Arial, sans-serif;
    background: #f4f6f8;
    color: #1e293b;
}

header {
    background: #111827;
    color: white;
    padding: 18px;
}

header h1 {
    margin: 0;
    font-size: 24px;
}

header p {
    margin: 5px 0 0;
    color: #cbd5e1;
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
    padding: 14px 18px;
    cursor: pointer;
    font-weight: bold;
}

nav button.active {
    background: #111827;
    color: white;
}

main {
    max-width: 1200px;
    margin: auto;
    padding: 18px;
}

.section {
    display: none;
}

.section.active {
    display: block;
}

.card {
    background: white;
    border-radius: 12px;
    padding: 18px;
    margin-bottom: 18px;
    box-shadow: 0 2px 8px rgba(0,0,0,.08);
}

.stats {
    display: grid;
    grid-template-columns:
        repeat(auto-fit,minmax(180px,1fr));
    gap: 15px;
}

.stat {
    background: white;
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0 2px 8px rgba(0,0,0,.08);
}

.stat h3 {
    margin: 0;
    color: #64748b;
}

.stat strong {
    display: block;
    margin-top: 8px;
    font-size: 30px;
}

button.primary {
    border: 0;
    background: #111827;
    color: white;
    padding: 12px 18px;
    border-radius: 8px;
    cursor: pointer;
    font-weight: bold;
}

button.primary:hover {
    opacity: .9;
}

button.danger {
    border: 0;
    background: #b91c1c;
    color: white;
    padding: 10px 15px;
    border-radius: 8px;
    cursor: pointer;
}

input,
select {
    padding: 10px;
    border: 1px solid #cbd5e1;
    border-radius: 7px;
}

.status {
    background: #f1f5f9;
    padding: 14px;
    border-radius: 8px;
    margin-top: 12px;
}

.success {
    color: #15803d;
}

.error {
    color: #b91c1c;
}

.warning {
    color: #a16207;
}

.progress {
    width: 100%;
    height: 20px;
    background: #e2e8f0;
    border-radius: 10px;
    overflow: hidden;
    margin-top: 12px;
}

.progressBar {
    width: 0%;
    height: 100%;
    background: #111827;
    transition: width .3s;
}

table {
    width: 100%;
    border-collapse: collapse;
}

th,
td {
    padding: 10px;
    border-bottom: 1px solid #ddd;
    text-align: left;
}

.preview-grid {
    display: grid;
    grid-template-columns:
        repeat(auto-fill,minmax(170px,1fr));
    gap: 12px;
}

.preview {
    background: #f8fafc;
    padding: 8px;
    border-radius: 8px;
}

.preview img {
    width: 100%;
    height: 140px;
    object-fit: cover;
    border-radius: 6px;
}

.image-item {
    border: 2px solid transparent;
    cursor: pointer;
}

.image-item.selected {
    border-color: #111827;
}

.annotation-layout {
    display: grid;
    grid-template-columns:
        260px 1fr;
    gap: 18px;
}

@media(max-width:800px) {

    .annotation-layout {
        grid-template-columns: 1fr;
    }

}

.image-list {
    max-height: 500px;
    overflow-y: auto;
}

.annotation-image-wrapper {
    position: relative;
    width: 100%;
    max-width: 900px;
    margin: auto;
    background: #111;
    overflow: hidden;
}

#annotationImage {
    display: block;
    width: 100%;
    height: auto;
}

#annotationCanvas {
    position: absolute;
    left: 0;
    top: 0;
    width: 100%;
    height: 100%;
    cursor: crosshair;
}

.annotation-help {
    background: #f8fafc;
    padding: 12px;
    border-radius: 8px;
    margin-bottom: 12px;
}

.box-list {
    margin-top: 12px;
}

.box-item {
    display: flex;
    justify-content: space-between;
    padding: 8px;
    background: #f1f5f9;
    margin-bottom: 5px;
    border-radius: 6px;
}

video {
    width: 100%;
    max-width: 700px;
}

footer {
    text-align: center;
    padding: 30px;
    color: #64748b;
}

</style>

</head>

<body>

<header>

<h1>NEERIKA BUCKET AI</h1>

<p>Mining Production Bucket Counter</p>

</header>

<nav>

<button class="active"
onclick="showSection('dashboard',this)">
Dashboard
</button>

<button
onclick="showSection('camera',this)">
Camera
</button>

<button
onclick="showSection('buckets',this)">
Buckets
</button>

<button
onclick="showSection('training',this)">
Training
</button>

<button
onclick="showSection('history',this)">
History
</button>

<button
onclick="showSection('settings',this)">
Settings
</button>

</nav>

<main>


<!-- ================================================= -->
<!-- DASHBOARD -->
<!-- ================================================= -->

<section id="dashboard"
class="section active">

<div class="stats">

<div class="stat">

<h3>Total Buckets</h3>

<strong id="totalCount">
0
</strong>

</div>

<div class="stat">

<h3>Training Images</h3>

<strong id="dashboardImages">
0
</strong>

</div>

<div class="stat">

<h3>YOLO Labels</h3>

<strong id="dashboardLabels">
0
</strong>

</div>

<div class="stat">

<h3>Training</h3>

<strong id="dashboardTraining">
Idle
</strong>

</div>

</div>

<div class="card">

<h2>System</h2>

<p id="systemMessage">
Loading...
</p>

</div>

</section>


<!-- ================================================= -->
<!-- CAMERA -->
<!-- ================================================= -->

<section id="camera"
class="section">

<div class="card">

<h2>Camera</h2>

<video id="cameraVideo"
autoplay
playsinline></video>

<br><br>

<button class="primary"
onclick="startCamera()">
Start Camera
</button>

</div>

</section>


<!-- ================================================= -->
<!-- BUCKET REFERENCE -->
<!-- ================================================= -->

<section id="buckets"
class="section">

<div class="card">

<h2>Bucket Reference Images</h2>

<p>
Store reference photos of the bucket type.
</p>

<input
type="file"
id="referenceFile"
accept="image/*">

<br><br>

<button
class="primary"
onclick="uploadReference()">

Upload Reference Image

</button>

<p id="referenceMessage"></p>

</div>

<div class="card">

<div id="referenceGrid"
class="preview-grid">
</div>

</div>

</section>


<!-- ================================================= -->
<!-- TRAINING -->
<!-- ================================================= -->

<section id="training"
class="section">

<div class="card">

<h2>YOLO Training Dataset</h2>

<p>
Step 1: Upload a clear photo of a loaded ore bucket.
</p>

<input
type="file"
id="trainingFile"
accept="image/*">

<br><br>

<button
class="primary"
onclick="uploadTrainingImage()">

Upload Image

</button>

<div id="uploadMessage"
class="status">
</div>

</div>


<div class="card">

<h2>Dataset Information</h2>

<table>

<tr>
<th>Training Images</th>
<td id="trainImages">0</td>
</tr>

<tr>
<th>Validation Images</th>
<td id="valImages">0</td>
</tr>

<tr>
<th>Training Labels</th>
<td id="trainLabels">0</td>
</tr>

<tr>
<th>Validation Labels</th>
<td id="valLabels">0</td>
</tr>

<tr>
<th>Total Images</th>
<td id="allImages">0</td>
</tr>

<tr>
<th>Total Labels</th>
<td id="allLabels">0</td>
</tr>

</table>

</div>


<!-- ================================================= -->
<!-- ANNOTATION -->
<!-- ================================================= -->

<div class="card">

<h2>YOLO Annotation</h2>

<div class="annotation-help">

<strong>Step 2:</strong>

Select an uploaded image, then drag your finger or mouse
around the <strong>loaded ore bucket</strong>.

Do not draw boxes around people, empty buckets,
equipment or other objects.

</div>

<div class="annotation-layout">

<div>

<h3>Images</h3>

<div
id="imageList"
class="image-list">
</div>

</div>


<div>

<div class="annotation-image-wrapper">

<img
id="annotationImage"
alt="Select an image">

<canvas
id="annotationCanvas">
</canvas>

</div>

<br>

<button
class="primary"
onclick="saveAnnotation()">

Save Annotation

</button>

<button
class="danger"
onclick="clearBoxes()">

Clear Boxes

</button>

<div id="annotationMessage"
class="status">

Select an image.

</div>

<div class="box-list"
id="boxList">
</div>

</div>

</div>

</div>


<!-- ================================================= -->
<!-- TRAINING STATUS -->
<!-- ================================================= -->

<div class="card">

<h2>Training Status</h2>

<div
id="trainingStatus"
class="status">

Status: idle

</div>

<div class="progress">

<div
id="progressBar"
class="progressBar">
</div>

</div>

<p id="progressText">
0%
</p>

</div>


<div class="card">

<h2>Start YOLO Training</h2>

<label>
Epochs:
</label>

<select id="epochs">

<option value="10">
10
</option>

<option value="20"
selected>
20
</option>

<option value="30">
30
</option>

<option value="50">
50
</option>

</select>

<br><br>

<button
class="primary"
onclick="startTraining()">

Start YOLO Training

</button>

</div>

</section>


<!-- ================================================= -->
<!-- HISTORY -->
<!-- ================================================= -->

<section id="history"
class="section">

<div class="card">

<h2>Bucket History</h2>

<button
class="primary"
onclick="loadHistory()">

Refresh History

</button>

<br><br>

<div style="overflow-x:auto">

<table>

<thead>

<tr>

<th>ID</th>
<th>Bucket Type</th>
<th>Count</th>
<th>Shift</th>
<th>Operator</th>
<th>Confidence</th>
<th>Date</th>

</tr>

</thead>

<tbody
id="historyBody">
</tbody>

</table>

</div>

</div>

</section>


<!-- ================================================= -->
<!-- SETTINGS -->
<!-- ================================================= -->

<section id="settings"
class="section">

<div class="card">

<h2>Settings</h2>

<p>
Application: NEERIKA BUCKET AI
</p>

<p>
Detection class:
<strong>BUCKET_LOADED</strong>
</p>

<p>
Database:
<span id="databaseStatus">
Checking...
</span>
</p>

<p>
YOLO:
<span id="yoloStatus">
Checking...
</span>
</p>

</div>

</section>

</main>

<footer>
Geology & Mining Services
</footer>


<script>

let currentImage = null;

let boxes = [];

let drawing = false;

let startX = 0;

let startY = 0;

let cameraStream = null;


/* =====================================================
   NAVIGATION
===================================================== */

function showSection(id, button) {

    document
        .querySelectorAll(".section")
        .forEach(
            x => x.classList.remove("active")
        );

    document
        .getElementById(id)
        .classList.add("active");

    document
        .querySelectorAll("nav button")
        .forEach(
            x => x.classList.remove("active")
        );

    if (button) {
        button.classList.add("active");
    }

    if (id === "training") {

        loadDataset();
        loadTrainingImages();
        loadTrainingStatus();

    }

    if (id === "history") {
        loadHistory();
    }

    if (id === "buckets") {
        loadReferences();
    }

}


/* =====================================================
   DASHBOARD
===================================================== */

async function loadDashboard() {

    try {

        const r =
            await fetch("/api/dashboard");

        const d =
            await r.json();

        document.getElementById(
            "totalCount"
        ).textContent =
            d.total_count;

        document.getElementById(
            "dashboardImages"
        ).textContent =
            d.dataset.images;

        document.getElementById(
            "dashboardLabels"
        ).textContent =
            d.dataset.labels;

        document.getElementById(
            "dashboardTraining"
        ).textContent =
            d.training.status;

        document.getElementById(
            "systemMessage"
        ).textContent =
            d.message;

    } catch (e) {

        document.getElementById(
            "systemMessage"
        ).textContent =
            "Server connection error.";

    }

}


/* =====================================================
   DATASET
===================================================== */

async function loadDataset() {

    try {

        const r =
            await fetch("/api/dataset");

        const d =
            await r.json();

        document.getElementById(
            "trainImages"
        ).textContent =
            d.train_images;

        document.getElementById(
            "valImages"
        ).textContent =
            d.val_images;

        document.getElementById(
            "trainLabels"
        ).textContent =
            d.train_labels;

        document.getElementById(
            "valLabels"
        ).textContent =
            d.val_labels;

        document.getElementById(
            "allImages"
        ).textContent =
            d.images;

        document.getElementById(
            "allLabels"
        ).textContent =
            d.labels;

    } catch (e) {

        console.error(e);

    }

}


/* =====================================================
   UPLOAD TRAINING IMAGE
===================================================== */

async function uploadTrainingImage() {

    const input =
        document.getElementById(
            "trainingFile"
        );

    const message =
        document.getElementById(
            "uploadMessage"
        );

    if (!input.files.length) {

        message.innerHTML =
            "<span class='error'>" +
            "Choose an image first." +
            "</span>";

        return;
    }

    const form =
        new FormData();

    form.append(
        "image",
        input.files[0]
    );

    message.textContent =
        "Uploading...";

    try {

        const r =
            await fetch(
                "/api/training/upload",
                {
                    method: "POST",
                    body: form
                }
            );

        const d =
            await r.json();

        if (!d.ok) {

            message.innerHTML =
                "<span class='error'>" +
                d.error +
                "</span>";

            return;
        }

        message.innerHTML =
            "<span class='success'>" +
            "Image uploaded successfully." +
            "</span>";

        input.value = "";

        await loadDataset();

        await loadTrainingImages();

    } catch (e) {

        message.innerHTML =
            "<span class='error'>" +
            e.message +
            "</span>";

    }

}


/* =====================================================
   LOAD TRAINING IMAGES
===================================================== */

async function loadTrainingImages() {

    const list =
        document.getElementById(
            "imageList"
        );

    try {

        const r =
            await fetch(
                "/api/training/images"
            );

        const d =
            await r.json();

        list.innerHTML = "";

        if (!d.images.length) {

            list.innerHTML =
                "<p>No training images yet.</p>";

            return;
        }

        d.images.forEach(
            image => {

                const div =
                    document.createElement(
                        "div"
                    );

                div.className =
                    "preview image-item";

                if (image.annotated) {

                    div.style.borderColor =
                        "#15803d";

                }

                div.innerHTML =
                    `
                    <img
                    src="${image.url}"
                    alt="${image.filename}">

                    <div>
                    ${image.filename}
                    </div>

                    <small>
                    ${
                        image.annotated
                        ? "✓ Annotated"
                        : "Not annotated"
                    }
                    </small>
                    `;

                div.onclick =
                    function() {

                        selectImage(
                            image.filename
                        );

                    };

                list.appendChild(div);

            }
        );

    } catch (e) {

        list.innerHTML =
            "<p>Could not load images.</p>";

    }

}


/* =====================================================
   SELECT IMAGE
===================================================== */

function selectImage(filename) {

    currentImage =
        filename;

    boxes = [];

    document.getElementById(
        "boxList"
    ).innerHTML = "";

    const img =
        document.getElementById(
            "annotationImage"
        );

    img.onload =
        function() {

            resizeCanvas();

        };

    img.src =
        "/training-image/"
        + encodeURIComponent(filename);

    document.getElementById(
        "annotationMessage"
    ).innerHTML =
        "Selected: <strong>"
        + filename
        + "</strong><br>" +
        "Draw a box around the loaded bucket.";

}


/* =====================================================
   CANVAS
===================================================== */

const canvas =
    document.getElementById(
        "annotationCanvas"
    );

const ctx =
    canvas.getContext("2d");

const image =
    document.getElementById(
        "annotationImage"
    );


function resizeCanvas() {

    const rect =
        image.getBoundingClientRect();

    canvas.width =
        rect.width;

    canvas.height =
        rect.height;

    redraw();

}


window.addEventListener(
    "resize",
    resizeCanvas
);


/* =====================================================
   MOUSE / TOUCH COORDINATES
===================================================== */

function getPosition(event) {

    const rect =
        canvas.getBoundingClientRect();

    let clientX;
    let clientY;

    if (
        event.touches &&
        event.touches.length
    ) {

        clientX =
            event.touches[0].clientX;

        clientY =
            event.touches[0].clientY;

    } else {

        clientX =
            event.clientX;

        clientY =
            event.clientY;

    }

    return {
        x: clientX - rect.left,
        y: clientY - rect.top
    };

}


/* =====================================================
   START DRAWING
===================================================== */

function beginDrawing(event) {

    if (!currentImage) {
        return;
    }

    event.preventDefault();

    const p =
        getPosition(event);

    startX = p.x;
    startY = p.y;

    drawing = true;

}


function moveDrawing(event) {

    if (!drawing) {
        return;
    }

    event.preventDefault();

    const p =
        getPosition(event);

    redraw();

    ctx.strokeStyle =
        "#ff0000";

    ctx.lineWidth = 3;

    ctx.strokeRect(
        startX,
        startY,
        p.x - startX,
        p.y - startY
    );

}


function finishDrawing(event) {

    if (!drawing) {
        return;
    }

    event.preventDefault();

    const p =
        getPosition(event);

    drawing = false;

    let x =
        Math.min(
            startX,
            p.x
        );

    let y =
        Math.min(
            startY,
            p.y
        );

    let width =
        Math.abs(
            p.x - startX
        );

    let height =
        Math.abs(
            p.y - startY
        );

    if (
        width < 10 ||
        height < 10
    ) {

        redraw();

        return;
    }

    boxes.push({
        x: x / canvas.width,
        y: y / canvas.height,
        width:
            width / canvas.width,
        height:
            height / canvas.height
    });

    redraw();

    updateBoxList();

}


/* =====================================================
   CANVAS EVENTS
===================================================== */

canvas.addEventListener(
    "mousedown",
    beginDrawing
);

canvas.addEventListener(
    "mousemove",
    moveDrawing
);

canvas.addEventListener(
    "mouseup",
    finishDrawing
);

canvas.addEventListener(
    "mouseleave",
    finishDrawing
);

canvas.addEventListener(
    "touchstart",
    beginDrawing,
    {passive:false}
);

canvas.addEventListener(
    "touchmove",
    moveDrawing,
    {passive:false}
);

canvas.addEventListener(
    "touchend",
    finishDrawing,
    {passive:false}
);


/* =====================================================
   REDRAW
===================================================== */

function redraw() {

    ctx.clearRect(
        0,
        0,
        canvas.width,
        canvas.height
    );

    boxes.forEach(
        (box, index) => {

            const x =
                box.x * canvas.width;

            const y =
                box.y * canvas.height;

            const w =
                box.width * canvas.width;

            const h =
                box.height * canvas.height;

            ctx.strokeStyle =
                "#00ff00";

            ctx.lineWidth = 3;

            ctx.strokeRect(
                x,
                y,
                w,
                h
            );

            ctx.fillStyle =
                "#00ff00";

            ctx.font =
                "bold 16px Arial";

            ctx.fillText(
                "BUCKET_LOADED "
                + (index + 1),
                x,
                Math.max(
                    18,
                    y - 5
                )
            );

        }
    );

}


/* =====================================================
   BOX LIST
===================================================== */

function updateBoxList() {

    const list =
        document.getElementById(
            "boxList"
        );

    list.innerHTML = "";

    boxes.forEach(
        (box, index) => {

            const div =
                document.createElement(
                    "div"
                );

            div.className =
                "box-item";

            div.innerHTML =
                `
                <span>
                BUCKET_LOADED ${index + 1}
                </span>

                <button
                onclick="deleteBox(${index})">
                Delete
                </button>
                `;

            list.appendChild(div);

        }
    );

}


function deleteBox(index) {

    boxes.splice(
        index,
        1
    );

    redraw();

    updateBoxList();

}


function clearBoxes() {

    boxes = [];

    redraw();

    updateBoxList();

}


/* =====================================================
   SAVE ANNOTATION
===================================================== */

async function saveAnnotation() {

    const message =
        document.getElementById(
            "annotationMessage"
        );

    if (!currentImage) {

        message.innerHTML =
            "<span class='error'>" +
            "Select an image first." +
            "</span>";

        return;
    }

    if (!boxes.length) {

        message.innerHTML =
            "<span class='error'>" +
            "Draw at least one box around the bucket." +
            "</span>";

        return;
    }

    message.textContent =
        "Saving annotation...";

    try {

        const r =
            await fetch(
                "/api/annotation/save",
                {
                    method: "POST",
                    headers: {
                        "Content-Type":
                            "application/json"
                    },
                    body: JSON.stringify({
                        filename:
                            currentImage,
                        boxes:
                            boxes
                    })
                }
            );

        const d =
            await r.json();

        if (!d.ok) {

            message.innerHTML =
                "<span class='error'>" +
                d.error +
                "</span>";

            return;
        }

        message.innerHTML =
            "<span class='success'>" +
            "Annotation saved successfully." +
            "<br>Label file: "
            + d.label +
            "</span>";

        await loadDataset();

        await loadTrainingImages();

    } catch (e) {

        message.innerHTML =
            "<span class='error'>" +
            e.message +
            "</span>";

    }

}


/* =====================================================
   TRAINING STATUS
===================================================== */

async function loadTrainingStatus() {

    try {

        const r =
            await fetch(
                "/api/training/status"
            );

        const d =
            await r.json();

        const status =
            document.getElementById(
                "trainingStatus"
            );

        status.innerHTML =
            "<strong>Status:</strong> "
            + d.status
            + "<br><br>"
            + "<strong>Message:</strong> "
            + d.message;

        if (d.error) {

            status.innerHTML +=
                "<br><br><span class='error'>"
                + d.error
                + "</span>";

        }

        const progress =
            Number(
                d.progress || 0
            );

        document.getElementById(
            "progressBar"
        ).style.width =
            progress + "%";

        document.getElementById(
            "progressText"
        ).textContent =
            progress + "%";

    } catch (e) {

        console.error(e);

    }

}


/* =====================================================
   START TRAINING
===================================================== */

async function startTraining() {

    const epochs =
        Number(
            document.getElementById(
                "epochs"
            ).value
        );

    const status =
        document.getElementById(
            "trainingStatus"
        );

    status.textContent =
        "Checking dataset...";

    try {

        const check =
            await fetch(
                "/api/training/check"
            );

        const checkData =
            await check.json();

        if (!checkData.ok) {

            status.innerHTML =
                "<span class='error'>"
                + checkData.message
                + "</span>";

            await loadDataset();

            return;
        }

        status.textContent =
            "Starting training...";

        const r =
            await fetch(
                "/api/training/start",
                {
                    method: "POST",
                    headers: {
                        "Content-Type":
                            "application/json"
                    },
                    body: JSON.stringify({
                        epochs: epochs
                    })
                }
            );

        const d =
            await r.json();

        if (!d.ok) {

            status.innerHTML =
                "<span class='error'>"
                + d.message
                + "</span>";

            return;
        }

        status.textContent =
            d.message;

        loadTrainingStatus();

    } catch (e) {

        status.innerHTML =
            "<span class='error'>"
            + e.message
            + "</span>";

    }

}


/* =====================================================
   REFERENCE UPLOAD
===================================================== */

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

        message.textContent =
            "Choose an image.";

        return;
    }

    const form =
        new FormData();

    form.append(
        "image",
        input.files[0]
    );

    message.textContent =
        "Uploading...";

    try {

        const r =
            await fetch(
                "/api/reference/upload",
                {
                    method: "POST",
                    body: form
                }
            );

        const d =
            await r.json();

        if (!d.ok) {

            message.textContent =
                d.error;

            return;
        }

        message.innerHTML =
            "<span class='success'>" +
            "Reference uploaded successfully."
            + "</span>";

        input.value = "";

        loadReferences();

    } catch (e) {

        message.textContent =
            e.message;

    }

}


async function loadReferences() {

    const grid =
        document.getElementById(
            "referenceGrid"
        );

    try {

        const r =
            await fetch(
                "/api/reference/list"
            );

        const d =
            await r.json();

        grid.innerHTML = "";

        d.images.forEach(
            image => {

                const div =
                    document.createElement(
                        "div"
                    );

                div.className =
                    "preview";

                div.innerHTML =
                    `
                    <img src="${image.url}">
                    <p>${image.name}</p>
                    `;

                grid.appendChild(
                    div
                );

            }
        );

    } catch (e) {

        console.error(e);

    }

}


/* =====================================================
   HISTORY
===================================================== */

async function loadHistory() {

    try {

        const r =
            await fetch(
                "/api/history"
            );

        const d =
            await r.json();

        const body =
            document.getElementById(
                "historyBody"
            );

        body.innerHTML = "";

        d.history.forEach(
            item => {

                const row =
                    document.createElement(
                        "tr"
                    );

                row.innerHTML =
                    `
                    <td>${item.id}</td>
                    <td>${item.bucket_type || ""}</td>
                    <td>${item.count || 0}</td>
                    <td>${item.shift || ""}</td>
                    <td>${item.operator_name || ""}</td>
                    <td>
                    ${Number(
                        item.confidence || 0
                    ).toFixed(3)}
                    </td>
                    <td>${item.created_at || ""}</td>
                    `;

                body.appendChild(
                    row
                );

            }
        );

    } catch (e) {

        console.error(e);

    }

}


/* =====================================================
   CAMERA
===================================================== */

async function startCamera() {

    try {

        if (cameraStream) {

            cameraStream
                .getTracks()
                .forEach(
                    track =>
                        track.stop()
                );

        }

        cameraStream =
            await navigator.mediaDevices
                .getUserMedia({
                    video: {
                        facingMode:
                            "environment"
                    },
                    audio: false
                });

        document.getElementById(
            "cameraVideo"
        ).srcObject =
            cameraStream;

    } catch (e) {

        alert(
            "Camera error: "
            + e.message
        );

    }

}


/* =====================================================
   AUTO REFRESH
===================================================== */

setInterval(
    loadTrainingStatus,
    3000
);

setInterval(
    loadDashboard,
    5000
);


/* =====================================================
   INITIAL
===================================================== */

loadDashboard();

loadDataset();

loadTrainingStatus();

</script>

</body>
</html>
"""


# ============================================================
# HTTP HANDLER
# ============================================================

class Handler(
    BaseHTTPRequestHandler
):

    server_version = (
        "NEERIKA-BUCKET-AI/2.0"
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


    # ========================================================
    # GET
    # ========================================================

    def do_GET(self):

        parsed = urlparse(
            self.path
        )

        path = parsed.path


        # ----------------------------------------------------
        # HOME
        # ----------------------------------------------------

        if path == "/":

            send_html(
                self,
                HTML
            )

            return


        # ----------------------------------------------------
        # DASHBOARD
        # ----------------------------------------------------

        if path == "/api/dashboard":

            stats = (
                dataset_statistics()
            )

            with TRAINING_LOCK:

                training = dict(
                    TRAINING_STATE
                )

            send_json(
                self,
                {
                    "ok": True,
                    "total_count":
                        get_total_count(),
                    "dataset":
                        stats,
                    "training":
                        training,
                    "message":
                        "NEERIKA BUCKET AI is running."
                }
            )

            return


        # ----------------------------------------------------
        # DATASET
        # ----------------------------------------------------

        if path == "/api/dataset":

            send_json(
                self,
                dataset_statistics()
            )

            return


        # ----------------------------------------------------
        # TRAINING IMAGES
        # ----------------------------------------------------

        if path == "/api/training/images":

            send_json(
                self,
                {
                    "ok": True,
                    "images":
                        get_training_images()
                }
            )

            return


        # ----------------------------------------------------
        # TRAINING STATUS
        # ----------------------------------------------------

        if path == "/api/training/status":

            with TRAINING_LOCK:

                state = dict(
                    TRAINING_STATE
                )

            send_json(
                self,
                state
            )

            return


        # ----------------------------------------------------
        # TRAINING CHECK
        # ----------------------------------------------------

        if path == "/api/training/check":

            valid, message, stats = (
                validate_dataset()
            )

            send_json(
                self,
                {
                    "ok": valid,
                    "message": message,
                    "dataset": stats
                }
            )

            return


        # ----------------------------------------------------
        # HISTORY
        # ----------------------------------------------------

        if path == "/api/history":

            send_json(
                self,
                {
                    "ok": True,
                    "history":
                        get_history()
                }
            )

            return


        # ----------------------------------------------------
        # REFERENCE LIST
        # ----------------------------------------------------

        if path == "/api/reference/list":

            send_json(
                self,
                {
                    "ok": True,
                    "images":
                        reference_images()
                }
            )

            return


        # ----------------------------------------------------
        # TRAINING IMAGE FILE
        # ----------------------------------------------------

        if path.startswith(
            "/training-image/"
        ):

            filename = os.path.basename(
                path[
                    len("/training-image/"):
                ]
            )

            file_path = os.path.join(
                TRAIN_IMAGES_DIR,
                filename
            )

            self.serve_file(
                file_path
            )

            return


        # ----------------------------------------------------
        # REFERENCE FILE
        # ----------------------------------------------------

        if path.startswith(
            "/reference/"
        ):

            filename = os.path.basename(
                path[
                    len("/reference/"):
                ]
            )

            file_path = os.path.join(
                REFERENCE_DIR,
                filename
            )

            self.serve_file(
                file_path
            )

            return


        # ----------------------------------------------------
        # MODEL STATUS
        # ----------------------------------------------------

        if path == "/api/model/status":

            send_json(
                self,
                {
                    "ok": True,
                    "ultralytics":
                        YOLO is not None,
                    "trained_model":
                        os.path.isfile(
                            TRAINED_MODEL
                        )
                }
            )

            return


        send_not_found(
            self
        )


    # ========================================================
    # POST
    # ========================================================

    def do_POST(self):

        parsed = urlparse(
            self.path
        )

        path = parsed.path


        # ----------------------------------------------------
        # TRAINING IMAGE UPLOAD
        # ----------------------------------------------------

        if path == "/api/training/upload":

            try:

                fields =parse_multipart(
                        self
                    )

                image =  fields.get(
                        "image"
                    )

                if not image:

                    send_json(
                        self,
                        {
                            "ok": False,
                            "error":
                                "No image uploaded."
                        },
                        400
                    )

                    return

                filename = (
                    image.get(
                        "filename"
                    )
                    or "image.jpg"
                )

                data = image.get(
                    "data",
                    b""
                )

                if not is_image(
                    filename
                ):

                    send_json(
                        self,
                        {
                            "ok": False,
                            "error":
                                "Only image files are allowed."
                        },
                        400
                    )

                    return

                if not data:

                    send_json(
                        self,
                        {
                            "ok": False,
                            "error":
                                "Image is empty."
                        },
                        400
                    )

                    return

                saved = (
                    save_training_image(
                        filename,
                        data
                    )
                )

                create_dataset_yaml()

                send_json(
                    self,
                    {
                        "ok": True,
                        "filename": saved,
                        "message":
                            "Image uploaded successfully.",
                        "dataset":
                            dataset_statistics()
                    }
                )

                return

            except Exception as e:

                print(
                    traceback.format_exc()
                )

                send_json(
                    self,
                    {
                        "ok": False,
                        "error": str(e)
                    },
                    500
                )

                return


        # ----------------------------------------------------
        # SAVE ANNOTATION
        # ----------------------------------------------------

        if path == "/api/annotation/save":

            try:

                length = int(
                    self.headers.get(
                        "Content-Length",
                        "0"
                    )
                )

                raw = self.rfile.read(
                    length
                )

                payload = json.loads(
                    raw.decode(
                        "utf-8"
                    )
                )

                filename = payload.get(
                    "filename"
                )

                boxes = payload.get(
                    "boxes",
                    []
                )

                if not filename:

                    send_json(
                        self,
                        {
                            "ok": False,
                            "error":
                                "Filename is required."
                        },
                        400
                    )

                    return

                if not boxes:

                    send_json(
                        self,
                        {
                            "ok": False,
                            "error":
                                "At least one box is required."
                        },
                        400
                    )

                    return

                label = save_annotation(
                    filename,
                    boxes
                )

                send_json(
                    self,
                    {
                        "ok": True,
                        "label": label,
                        "message":
                            "Annotation saved successfully.",
                        "dataset":
                            dataset_statistics()
                    }
                )

                return

            except Exception as e:

                print(
                    traceback.format_exc()
                )

                send_json(
                    self,
                    {
                        "ok": False,
                        "error": str(e)
                    },
                    500
                )

                return


        # ----------------------------------------------------
        # START TRAINING
        # ----------------------------------------------------

        if path == "/api/training/start":

            try:

                length = int(
                    self.headers.get(
                        "Content-Length",
                        "0"
                    )
                )

                raw = self.rfile.read(
                    length
                )

                payload = {}

                if raw:

                    try:

                        payload = json.loads(
                            raw.decode(
                                "utf-8"
                            )
                        )

                    except Exception:

                        payload = {}

                epochs = int(
                    payload.get(
                        "epochs",
                        20
                    )
                )

                epochs = max(
                    1,
                    min(
                        100,
                        epochs
                    )
                )

                valid, message, stats = (
                    validate_dataset()
                )

                if not valid:

                    set_training(
                        status="error",
                        message=message,
                        progress=0,
                        error=message
                    )

                    send_json(
                        self,
                        {
                            "ok": False,
                            "message": message,
                            "dataset": stats
                        },
                        400
                    )

                    return

                started, start_message = (
                    start_training(
                        epochs
                    )
                )

                if not started:

                    send_json(
                        self,
                        {
                            "ok": False,
                            "message":
                                start_message
                        },
                        409
                    )

                    return

                send_json(
                    self,
                    {
                        "ok": True,
                        "message":
                            start_message,
                        "epochs":
                            epochs
                    }
                )

                return

            except Exception as e:

                print(
                    traceback.format_exc()
                )

                send_json(
                    self,
                    {
                        "ok": False,
                        "message":
                            str(e)
                    },
                    500
                )

                return


        # ----------------------------------------------------
        # REFERENCE UPLOAD
        # ----------------------------------------------------

        if path == "/api/reference/upload":

            try:

                fields =parse_multipart(
                        self
                    )

                image = fields.get(
                        "image"
                    )

                if not image:

                    send_json(
                        self,
                        {
                            "ok": False,
                            "error":
                                "No image uploaded."
                        },
                        400
                    )

                    return

                filename = (
                    image.get(
                        "filename"
                    )
                    or "reference.jpg"
                )

                data = image.get(
                    "data",
                    b""
                )

                if not is_image(
                    filename
                ):

                    send_json(
                        self,
                        {
                            "ok": False,
                            "error":
                                "Only images are allowed."
                        },
                        400
                    )

                    return

                saved = save_reference(
                    filename,
                    data
                )

                send_json(
                    self,
                    {
                        "ok": True,
                        "filename": saved
                    }
                )

                return

            except Exception as e:

                send_json(
                    self,
                    {
                        "ok": False,
                        "error": str(e)
                    },
                    500
                )

                return


        # ----------------------------------------------------
        # SAVE BUCKET COUNT
        # ----------------------------------------------------

        if path == "/api/count":

            try:

                length = int(
                    self.headers.get(
                        "Content-Length",
                        "0"
                    )
                )

                raw = self.rfile.read(
                    length
                )

                payload = json.loads(
                    raw.decode(
                        "utf-8"
                    )
                )

                success = save_bucket_count(
                    bucket_type=
                        payload.get(
                            "bucket_type",
                            "default"
                        ),
                    count=int(
                        payload.get(
                            "count",
                            1
                        )
                    ),
                    shift=
                        payload.get(
                            "shift",
                            ""
                        ),
                    operator_name=
                        payload.get(
                            "operator_name",
                            ""
                        ),
                    source=
                        payload.get(
                            "source",
                            "camera"
                        ),
                    confidence=float(
                        payload.get(
                            "confidence",
                            0
                        )
                    )
                )

                send_json(
                    self,
                    {
                        "ok": success,
                        "message":
                            (
                                "Count saved."
                                if success
                                else
                                "Could not save count."
                            )
                    }
                )

                return

            except Exception as e:

                send_json(
                    self,
                    {
                        "ok": False,
                        "error": str(e)
                    },
                    500
                )

                return


        send_not_found(
            self
        )


    # ========================================================
    # SERVE FILE
    # ========================================================

    def serve_file(
        self,
        file_path
    ):

        if not os.path.isfile(
            file_path
        ):

            send_not_found(
                self
            )

            return

        extension = os.path.splitext(
            file_path
        )[1].lower()

        content_types = {
            ".jpg":
                "image/jpeg",
            ".jpeg":
                "image/jpeg",
            ".png":
                "image/png",
            ".webp":
                "image/webp",
            ".bmp":
                "image/bmp"
        }

        content_type = (
            content_types.get(
                extension,
                "application/octet-stream"
            )
        )

        try:

            with open(
                file_path,
                "rb"
            ) as f:

                data = f.read()

            self.send_response(
                200
            )

            self.send_header(
                "Content-Type",
                content_type
            )

            self.send_header(
                "Content-Length",
                str(len(data))
            )

            self.end_headers()

            self.wfile.write(
                data
            )

        except Exception as e:

            print(
                "FILE ERROR:",
                e
            )

            send_not_found(
                self
            )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)

    print(
        "NEERIKA BUCKET AI"
    )

    print(
        "Mining Production Bucket Counter"
    )

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
        "TRAIN_IMAGES_DIR:",
        TRAIN_IMAGES_DIR
    )

    print(
        "TRAIN_LABELS_DIR:",
        TRAIN_LABELS_DIR
    )

    print(
        "YOLO:",
        YOLO is not None
    )

    print(
        "DATABASE:",
        bool(DATABASE_URL)
    )

    # Make sure folders exist.
    for directory in DIRECTORIES:

        os.makedirs(
            directory,
            exist_ok=True
        )

    # Always create dataset.yaml.
    create_dataset_yaml()

    init_database()

    server = ThreadingHTTPServer(
        (
            "0.0.0.0",
            PORT
        ),
        Handler
    )

    print(
        "Server running on port",
        PORT
    )

    print("=" * 60)

    try:

        server.serve_forever()

    except KeyboardInterrupt:

        print(
            "Server stopped."
        )

    finally:

        server.server_close()


if __name__ == "__main__":

    main()
