import os
import io
import json
import base64
import traceback
import tempfile
import shutil
import threading
import time
import re

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/Ultralytics")

import psycopg2
from psycopg2.extras import RealDictCursor

from ultralytics import YOLO

from PIL import Image
import numpy as np


# ============================================================
# NEERIKA BUCKET AI
# Mining Production Bucket Counter
# ============================================================

HOST = "0.0.0.0"

PORT = int(
    os.environ.get(
        "PORT",
        "8080"
    )
)

DATABASE_URL = (
    os.environ.get("DATABASE_URL")
    or os.environ.get("SUPABASE_DB_URL")
    or os.environ.get("POSTGRES_URL")
)

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

TRAIN_BATCH = int(
    os.environ.get(
        "TRAIN_BATCH",
        "1"
    )
)

TRAIN_IMAGE_SIZE = int(
    os.environ.get(
        "TRAIN_IMAGE_SIZE",
        "320"
    )
)

MAX_JSON_BYTES = 15 * 1024 * 1024

MAX_UPLOAD_BYTES = 15 * 1024 * 1024

COUNT_COOLDOWN_SECONDS = 4.0


# ============================================================
# GLOBALS
# ============================================================

MODEL = None

MODEL_ERROR = ""

MODEL_LOCK = threading.Lock()

TRAIN_LOCK = threading.Lock()

TRAINING = False

LAST_COUNT_TIME = 0.0


CLASS_IDS = {
    0: "BUCKET_LOADED",
    1: "BUCKET_EMPTY",
    2: "PEOPLE",
    3: "EQUIPMENT",
}


# ============================================================
# DATABASE
# ============================================================

def db():

    if not DATABASE_URL:

        raise RuntimeError(
            "DATABASE_URL haijawekwa kwenye "
            "Render Environment Variables."
        )

    return psycopg2.connect(
        DATABASE_URL,
        cursor_factory=RealDictCursor,
        connect_timeout=15
    )


# ============================================================
# DATABASE INITIALIZATION
# ============================================================

def init_db():

    conn = db()
    cur = conn.cursor()

    try:

        cur.execute("""
            CREATE TABLE IF NOT EXISTS buckets (
                id SERIAL PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT DEFAULT '',
                active BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS dataset_images (
                id SERIAL PRIMARY KEY,
                filename TEXT NOT NULL,
                image_data BYTEA NOT NULL,
                mime_type TEXT DEFAULT 'image/jpeg',
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS annotations (
                id SERIAL PRIMARY KEY,
                image_id INTEGER
                    REFERENCES dataset_images(id)
                    ON DELETE CASCADE,
                class_name TEXT NOT NULL,
                x_center DOUBLE PRECISION NOT NULL,
                y_center DOUBLE PRECISION NOT NULL,
                box_width DOUBLE PRECISION NOT NULL,
                box_height DOUBLE PRECISION NOT NULL,
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS training_state (
                id INTEGER PRIMARY KEY,
                status TEXT DEFAULT 'idle',
                message TEXT DEFAULT '',
                progress INTEGER DEFAULT 0,
                updated_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS trained_model (
                id INTEGER PRIMARY KEY,
                model_data BYTEA NOT NULL,
                filename TEXT DEFAULT 'best.pt',
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS daily_counts (
                id SERIAL PRIMARY KEY,
                count_date DATE UNIQUE NOT NULL,
                bucket_count INTEGER DEFAULT 0,
                updated_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS detection_events (
                id SERIAL PRIMARY KEY,
                class_name TEXT NOT NULL,
                confidence DOUBLE PRECISION DEFAULT 0,
                counted BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)

        migrations = [

            """
            ALTER TABLE dataset_images
            ADD COLUMN IF NOT EXISTS filename TEXT
            """,

            """
            ALTER TABLE dataset_images
            ADD COLUMN IF NOT EXISTS image_data BYTEA
            """,

            """
            ALTER TABLE dataset_images
            ADD COLUMN IF NOT EXISTS mime_type TEXT
            DEFAULT 'image/jpeg'
            """,

            """
            ALTER TABLE dataset_images
            ADD COLUMN IF NOT EXISTS created_at
            TIMESTAMPTZ DEFAULT NOW()
            """,

            """
            ALTER TABLE annotations
            ADD COLUMN IF NOT EXISTS image_id INTEGER
            """,

            """
            ALTER TABLE annotations
            ADD COLUMN IF NOT EXISTS class_name TEXT
            DEFAULT 'BUCKET_LOADED'
            """,

            """
            ALTER TABLE annotations
            ADD COLUMN IF NOT EXISTS x_center
            DOUBLE PRECISION DEFAULT 0
            """,

            """
            ALTER TABLE annotations
            ADD COLUMN IF NOT EXISTS y_center
            DOUBLE PRECISION DEFAULT 0
            """,

            """
            ALTER TABLE annotations
            ADD COLUMN IF NOT EXISTS box_width
            DOUBLE PRECISION DEFAULT 0
            """,

            """
            ALTER TABLE annotations
            ADD COLUMN IF NOT EXISTS box_height
            DOUBLE PRECISION DEFAULT 0
            """,

            """
            ALTER TABLE annotations
            ADD COLUMN IF NOT EXISTS created_at
            TIMESTAMPTZ DEFAULT NOW()
            """,

            """
            ALTER TABLE daily_counts
            ADD COLUMN IF NOT EXISTS count_date DATE
            """,

            """
            ALTER TABLE daily_counts
            ADD COLUMN IF NOT EXISTS bucket_count INTEGER
            DEFAULT 0
            """,

            """
            ALTER TABLE daily_counts
            ADD COLUMN IF NOT EXISTS updated_at
            TIMESTAMPTZ DEFAULT NOW()
            """,

            """
            ALTER TABLE detection_events
            ADD COLUMN IF NOT EXISTS class_name TEXT
            DEFAULT 'BUCKET_LOADED'
            """,

            """
            ALTER TABLE detection_events
            ADD COLUMN IF NOT EXISTS confidence
            DOUBLE PRECISION DEFAULT 0
            """,

            """
            ALTER TABLE detection_events
            ADD COLUMN IF NOT EXISTS counted
            BOOLEAN DEFAULT FALSE
            """,

            """
            ALTER TABLE detection_events
            ADD COLUMN IF NOT EXISTS created_at
            TIMESTAMPTZ DEFAULT NOW()
            """,

            """
            ALTER TABLE training_state
            ADD COLUMN IF NOT EXISTS status TEXT
            DEFAULT 'idle'
            """,

            """
            ALTER TABLE training_state
            ADD COLUMN IF NOT EXISTS message TEXT
            DEFAULT ''
            """,

            """
            ALTER TABLE training_state
            ADD COLUMN IF NOT EXISTS progress INTEGER
            DEFAULT 0
            """,

            """
            ALTER TABLE training_state
            ADD COLUMN IF NOT EXISTS updated_at
            TIMESTAMPTZ DEFAULT NOW()
            """,

            """
            ALTER TABLE trained_model
            ADD COLUMN IF NOT EXISTS model_data BYTEA
            """,

            """
            ALTER TABLE trained_model
            ADD COLUMN IF NOT EXISTS filename TEXT
            DEFAULT 'best.pt'
            """,

            """
            ALTER TABLE trained_model
            ADD COLUMN IF NOT EXISTS created_at
            TIMESTAMPTZ DEFAULT NOW()
            """,

            """
            ALTER TABLE buckets
            ADD COLUMN IF NOT EXISTS name TEXT
            """,

            """
            ALTER TABLE buckets
            ADD COLUMN IF NOT EXISTS description TEXT
            DEFAULT ''
            """,

            """
            ALTER TABLE buckets
            ADD COLUMN IF NOT EXISTS active BOOLEAN
            DEFAULT FALSE
            """,

            """
            ALTER TABLE buckets
            ADD COLUMN IF NOT EXISTS created_at
            TIMESTAMPTZ DEFAULT NOW()
            """
        ]

        for sql in migrations:

            cur.execute(sql)

        cur.execute("""
            UPDATE dataset_images
            SET mime_type = 'image/jpeg'
            WHERE mime_type IS NULL
        """)

        cur.execute("""
            UPDATE daily_counts
            SET bucket_count = 0
            WHERE bucket_count IS NULL
        """)

        cur.execute("""
            UPDATE training_state
            SET status = 'idle'
            WHERE status IS NULL
        """)

        cur.execute("""
            UPDATE training_state
            SET message = ''
            WHERE message IS NULL
        """)

        cur.execute("""
            UPDATE training_state
            SET progress = 0
            WHERE progress IS NULL
        """)

        cur.execute("""
            INSERT INTO training_state(
                id,
                status,
                message,
                progress
            )
            VALUES(
                1,
                'idle',
                'Ready',
                0
            )
            ON CONFLICT(id) DO NOTHING
        """)

        conn.commit()

    except Exception:

        conn.rollback()

        raise

    finally:

        cur.close()
        conn.close()


# ============================================================
# RESET TRAINING STATE
# ============================================================

def reset_stale_training_state():

    conn = db()
    cur = conn.cursor()

    try:

        cur.execute("""
            UPDATE training_state

            SET
                status = 'idle',
                message = 'Ready',
                progress = 0,
                updated_at = NOW()

            WHERE
                id = 1
                AND status IN (
                    'preparing',
                    'training',
                    'saving'
                )
        """)

        conn.commit()

    except Exception:

        conn.rollback()

        traceback.print_exc()

    finally:

        cur.close()
        conn.close()


# ============================================================
# HTML HELPERS
# ============================================================

def esc(value):

    text = "" if value is None else str(value)

    return (
        text
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#039;")
    )


def json_out(handler, data, status=200):

    raw = json.dumps(
        data,
        ensure_ascii=False,
        default=str
    ).encode("utf-8")

    try:

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
            "Cache-Control",
            "no-store"
        )

        handler.end_headers()

        handler.wfile.write(raw)

    except (
        BrokenPipeError,
        ConnectionResetError
    ):

        pass


def html_out(handler, text, status=200):

    raw = text.encode("utf-8")

    try:

        handler.send_response(status)

        handler.send_header(
            "Content-Type",
            "text/html; charset=utf-8"
        )

        handler.send_header(
            "Content-Length",
            str(len(raw))
        )

        handler.send_header(
            "Cache-Control",
            "no-store"
        )

        handler.end_headers()

        handler.wfile.write(raw)

    except (
        BrokenPipeError,
        ConnectionResetError
    ):

        pass


def error_out(handler, message, status=500):

    json_out(
        handler,
        {
            "error": str(message)
        },
        status
    )


def read_body(handler, limit=None):

    if limit is None:

        limit = MAX_JSON_BYTES

    try:

        size = int(
            handler.headers.get(
                "Content-Length",
                "0"
            )
        )

    except Exception:

        size = 0

    if size > limit:

        raise ValueError(
            "Request ni kubwa sana."
        )

    if size <= 0:

        return b""

    data = handler.rfile.read(size)

    if len(data) > limit:

        raise ValueError(
            "Request ni kubwa sana."
        )

    return data


def parse_json(handler):

    raw = read_body(handler)

    if not raw:

        return {}

    return json.loads(
        raw.decode("utf-8")
    )


# ============================================================
# MULTIPART PARSER
# Python 3.14 compatible - no cgi
# ============================================================

def parse_multipart(handler):

    content_type = handler.headers.get(
        "Content-Type",
        ""
    )

    if not content_type.startswith(
        "multipart/form-data"
    ):

        raise ValueError(
            "Expected multipart/form-data."
        )

    match = re.search(
        r'boundary="?([^";]+)"?',
        content_type
    )

    if not match:

        raise ValueError(
            "Multipart boundary haijapatikana."
        )

    boundary = match.group(1).encode(
        "utf-8"
    )

    raw = read_body(
        handler,
        MAX_UPLOAD_BYTES
    )

    delimiter = b"--" + boundary

    parts = raw.split(
        delimiter
    )

    result = {}

    for part in parts:

        part = part.strip()

        if not part:

            continue

        if part == b"--":

            continue

        if part.endswith(b"--"):

            part = part[:-2]

        if part.startswith(b"\r\n"):

            part = part[2:]

        separator = b"\r\n\r\n"

        if separator not in part:

            continue

        header_bytes, body = part.split(
            separator,
            1
        )

        body = body.rstrip(b"\r\n")

        headers = {}

        for line in header_bytes.split(
            b"\r\n"
        ):

            if b":" not in line:

                continue

            key, value = line.split(
                b":",
                1
            )

            headers[
                key.decode(
                    "utf-8",
                    errors="ignore"
                ).lower()
            ] = value.decode(
                "utf-8",
                errors="ignore"
            ).strip()

        disposition = headers.get(
            "content-disposition",
            ""
        )

        name_match = re.search(
            r'name="([^"]+)"',
            disposition
        )

        if not name_match:

            continue

        name = name_match.group(1)

        filename_match = re.search(
            r'filename="([^"]*)"',
            disposition
        )

        filename = ""

        if filename_match:

            filename = filename_match.group(1)

        result[name] = {
            "filename": filename,
            "content_type": headers.get(
                "content-type",
                "application/octet-stream"
            ),
            "data": body
        }

    return result


# ============================================================
# LAYOUT
# ============================================================

def layout(title, body, active):

    items = [
        ("Dashboard", "/"),
        ("Camera", "/camera"),
        ("Buckets", "/buckets"),
        ("Training", "/training"),
        ("History", "/history"),
        ("Settings", "/settings")
    ]

    nav = ""

    for name, path in items:

        cls = (
            "active"
            if name == active
            else ""
        )

        nav += (
            '<a class="nav-item %s" href="%s">%s</a>'
            % (
                cls,
                path,
                name
            )
        )

    page = r"""
<!doctype html>

<html>

<head>

<meta charset="utf-8">

<meta name="viewport"
content="width=device-width,initial-scale=1">

<title>
__TITLE__ - NEERIKA BUCKET AI
</title>

<style>

*{
box-sizing:border-box
}

body{
margin:0;
font-family:Arial,sans-serif;
background:#f4f6f8;
color:#17202a
}

header{
background:#111827;
color:white;
padding:16px
}

.brand{
font-size:22px;
font-weight:800
}

.sub{
font-size:13px;
opacity:.75;
margin-top:4px
}

.nav{
display:flex;
gap:7px;
overflow:auto;
background:#1f2937;
padding:8px
}

.nav-item{
color:white;
text-decoration:none;
padding:10px 13px;
border-radius:8px;
white-space:nowrap
}

.nav-item.active,
.nav-item:hover{
background:#374151
}

main{
max-width:1200px;
margin:auto;
padding:16px
}

.card,
.stat{
background:white;
border-radius:14px;
padding:18px;
margin-bottom:16px;
box-shadow:0 2px 10px rgba(0,0,0,.07)
}

.grid{
display:grid;
grid-template-columns:
repeat(auto-fit,minmax(210px,1fr));
gap:14px
}

.num{
font-size:38px;
font-weight:800;
margin-top:7px
}

button{
border:0;
border-radius:9px;
padding:10px 15px;
background:#111827;
color:white;
cursor:pointer;
margin:3px
}

button.secondary{
background:#6b7280
}

button.success{
background:#166534
}

button.danger{
background:#991b1b
}

button:disabled{
opacity:.55;
cursor:not-allowed
}

input,
select,
textarea{
width:100%;
padding:10px;
border:1px solid #d1d5db;
border-radius:8px
}

textarea{
min-height:100px
}

label{
font-weight:700;
display:block;
margin-bottom:5px
}

.row{
margin-bottom:13px
}

.status{
padding:10px;
background:#f3f4f6;
border-radius:8px;
margin-top:10px
}

table{
width:100%;
border-collapse:collapse
}

th,
td{
text-align:left;
padding:9px;
border-bottom:1px solid #e5e7eb
}

.footer{
text-align:center;
color:#6b7280;
font-size:12px;
padding:25px
}

.progress{
height:20px;
background:#e5e7eb;
border-radius:10px;
overflow:hidden
}

.bar{
height:100%;
background:#111827;
width:0%;
transition:width .3s
}

.small{
font-size:13px;
color:#6b7280
}

video{
width:100%;
max-height:520px;
background:#111;
border-radius:12px
}

.err{
color:#991b1b;
font-weight:700
}

.ok{
color:#166534;
font-weight:700
}

img.preview{
max-width:220px;
max-height:160px;
border-radius:8px;
margin-top:8px
}

@media(max-width:600px){

main{
padding:10px
}

.num{
font-size:30px
}

}

</style>

</head>

<body>

<header>

<div class="brand">
NEERIKA BUCKET AI
</div>

<div class="sub">
Mining Production Bucket Counter
</div>

</header>

<nav class="nav">

__NAV__

</nav>

<main>

__BODY__

</main>

<div class="footer">
Geology &amp; Mining Services
</div>

</body>

</html>
"""

    return (
        page
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
# BYTEA
# ============================================================

def _image_bytes(value):

    if value is None:

        return b""

    if isinstance(value, bytes):

        return value

    if isinstance(value, memoryview):

        return value.tobytes()

    if isinstance(value, bytearray):

        return bytes(value)

    if isinstance(value, str):

        text = value.strip()

        if text.startswith("\\x"):

            try:

                return bytes.fromhex(
                    text[2:]
                )

            except Exception:

                pass

        try:

            return base64.b64decode(
                text,
                validate=True
            )

        except Exception:

            pass

        return text.encode(
            "utf-8"
        )

    try:

        return bytes(value)

    except Exception:

        return b""


# ============================================================
# MODEL
# ============================================================

def restore_model():

    if os.path.exists(
        MODEL_PATH
    ):

        return True

    try:

        conn = db()
        cur = conn.cursor()

        try:

            cur.execute("""
                SELECT model_data
                FROM trained_model
                WHERE id=1
            """)

            row = cur.fetchone()

        finally:

            cur.close()
            conn.close()

        if row and row["model_data"]:

            data = _image_bytes(
                row["model_data"]
            )

            if data:

                os.makedirs(
                    os.path.dirname(
                        MODEL_PATH
                    ),
                    exist_ok=True
                )

                with open(
                    MODEL_PATH,
                    "wb"
                ) as f:

                    f.write(data)

                print(
                    "Restored best.pt from Supabase."
                )

                return True

    except Exception:

        traceback.print_exc()

    return False


def load_model():

    global MODEL
    global MODEL_ERROR

    with MODEL_LOCK:

        if MODEL is not None:

            return MODEL

        try:

            restore_model()

            if os.path.exists(
                MODEL_PATH
            ):

                path = MODEL_PATH

            elif os.path.exists(
                MODEL_BACKUP_PATH
            ):

                path = MODEL_BACKUP_PATH

            else:

                path = None

            if path:

                MODEL = YOLO(path)

                MODEL_ERROR = ""

                print(
                    "Loaded model:",
                    path
                )

                return MODEL

            MODEL = YOLO(
                "yolo11n.pt"
            )

            MODEL_ERROR = (
                "Fallback yolo11n.pt loaded. "
                "Train your bucket model."
            )

            print(MODEL_ERROR)

            return MODEL

        except Exception as e:

            MODEL_ERROR = str(e)

            traceback.print_exc()

            return None


def save_model_db(path):

    with open(
        path,
        "rb"
    ) as f:

        model_data = f.read()

    conn = db()
    cur = conn.cursor()

    try:

        cur.execute("""
            INSERT INTO trained_model(
                id,
                model_data,
                filename
            )

            VALUES(
                1,
                %s,
                %s
            )

            ON CONFLICT(id)
            DO UPDATE SET
                model_data = EXCLUDED.model_data,
                filename = EXCLUDED.filename,
                created_at = NOW()
        """, (
            psycopg2.Binary(
                model_data
            ),
            "best.pt"
        ))

        conn.commit()

    except Exception:

        conn.rollback()

        raise

    finally:

        cur.close()
        conn.close()


# ============================================================
# CLASS
# ============================================================

def normalize(class_id, name=""):

    if class_id in CLASS_IDS:

        return CLASS_IDS[
            class_id
        ]

    text = str(
        name
    ).upper()

    if "EMPTY" in text:

        return "BUCKET_EMPTY"

    if (
        "PEOPLE" in text
        or "PERSON" in text
    ):

        return "PEOPLE"

    if (
        "EQUIPMENT" in text
        or "MACHINE" in text
    ):

        return "EQUIPMENT"

    if "LOADED" in text:

        return "BUCKET_LOADED"

    return "UNKNOWN"


# ============================================================
# DETECTION
# ============================================================

def detect(data):

    model = load_model()

    if model is None:

        raise RuntimeError(
            "YOLO model haijapatikana: "
            + MODEL_ERROR
        )

    image = Image.open(
        io.BytesIO(data)
    ).convert("RGB")

    array = np.array(image)

    results = model.predict(
        source=array,
        conf=YOLO_CONFIDENCE,
        imgsz=YOLO_IMAGE_SIZE,
        verbose=False
    )

    output = []

    if not results:

        return output

    result = results[0]

    if result.boxes is None:

        return output

    names = getattr(
        result,
        "names",
        {}
    )

    for box in result.boxes:

        try:

            xy = (
                box.xyxy[0]
                .cpu()
                .numpy()
                .tolist()
            )

            confidence = float(
                box.conf[0]
                .cpu()
                .item()
            )

            class_id = int(
                box.cls[0]
                .cpu()
                .item()
            )

            raw_name = names.get(
                class_id,
                ""
            )

            output.append({

                "class_id":
                    class_id,

                "class_name":
                    normalize(
                        class_id,
                        raw_name
                    ),

                "raw_class_name":
                    str(raw_name),

                "confidence":
                    round(
                        confidence,
                        4
                    ),

                "x1":
                    round(
                        float(xy[0]),
                        2
                    ),

                "y1":
                    round(
                        float(xy[1]),
                        2
                    ),

                "x2":
                    round(
                        float(xy[2]),
                        2
                    ),

                "y2":
                    round(
                        float(xy[3]),
                        2
                    )
            })

        except Exception:

            traceback.print_exc()

    return output


# ============================================================
# COUNT
# ============================================================

def count_loaded(detections):

    global LAST_COUNT_TIME

    loaded = [
        item
        for item in detections
        if item.get("class_name")
        == "BUCKET_LOADED"
    ]

    if not loaded:

        return {
            "counted": False,
            "reason":
                "No loaded bucket detected.",
            "count": 0
        }

    now = time.time()

    if (
        now - LAST_COUNT_TIME
        < COUNT_COOLDOWN_SECONDS
    ):

        return {
            "counted": False,
            "reason":
                "Cooldown: possible same bucket.",
            "count": 0
        }

    best = max(
        loaded,
        key=lambda item:
            item.get(
                "confidence",
                0
            )
    )

    confidence = float(
        best.get(
            "confidence",
            0
        )
    )

    conn = db()
    cur = conn.cursor()

    try:

        cur.execute("""
            SELECT id
            FROM daily_counts
            WHERE count_date = CURRENT_DATE
            LIMIT 1
        """)

        row = cur.fetchone()

        if row:

            cur.execute("""
                UPDATE daily_counts

                SET
                    bucket_count =
                        COALESCE(
                            bucket_count,
                            0
                        ) + 1,

                    updated_at = NOW()

                WHERE id=%s
            """, (
                row["id"],
            ))

        else:

            cur.execute("""
                INSERT INTO daily_counts(
                    count_date,
                    bucket_count,
                    updated_at
                )

                VALUES(
                    CURRENT_DATE,
                    1,
                    NOW()
                )
            """)

        cur.execute("""
            INSERT INTO detection_events(
                class_name,
                confidence,
                counted
            )

            VALUES(
                %s,
                %s,
                TRUE
            )
        """, (
            "BUCKET_LOADED",
            confidence
        ))

        conn.commit()

    except Exception:

        conn.rollback()

        raise

    finally:

        cur.close()
        conn.close()

    LAST_COUNT_TIME = now

    return {
        "counted": True,
        "reason":
            "Loaded bucket counted.",
        "count": 1,
        "confidence": confidence
    }


def today():

    conn = db()
    cur = conn.cursor()

    try:

        cur.execute("""
            SELECT bucket_count
            FROM daily_counts
            WHERE count_date = CURRENT_DATE
            LIMIT 1
        """)

        row = cur.fetchone()

        if row:

            return int(
                row["bucket_count"]
                or 0
            )

        return 0

    finally:

        cur.close()
        conn.close()


# ============================================================
# DATASET
# ============================================================

def dataset_rows():

    conn = db()
    cur = conn.cursor()

    try:

        cur.execute("""
            SELECT
                di.id,
                di.filename,
                di.mime_type,
                di.created_at,
                COUNT(a.id)
                AS annotation_count

            FROM dataset_images di

            LEFT JOIN annotations a
                ON a.image_id = di.id

            GROUP BY
                di.id,
                di.filename,
                di.mime_type,
                di.created_at

            ORDER BY di.id DESC
        """)

        return cur.fetchall()

    finally:

        cur.close()
        conn.close()


def save_image(
    filename,
    mime,
    data
):

    if len(data) > MAX_UPLOAD_BYTES:

        raise ValueError(
            "Picha ni kubwa sana. "
            "Maximum ni 15 MB."
        )

    conn = db()
    cur = conn.cursor()

    try:

        cur.execute("""
            INSERT INTO dataset_images(
                filename,
                image_data,
                mime_type
            )

            VALUES(
                %s,
                %s,
                %s
            )

            RETURNING id
        """, (
            filename,
            psycopg2.Binary(data),
            mime or "image/jpeg"
        ))

        image_id = cur.fetchone()[
            "id"
        ]

        conn.commit()

        return image_id

    except Exception:

        conn.rollback()

        raise

    finally:

        cur.close()
        conn.close()


def save_annotations(
    image_id,
    annotations
):

    conn = db()
    cur = conn.cursor()

    try:

        cur.execute("""
            DELETE FROM annotations
            WHERE image_id=%s
        """, (
            image_id,
        ))

        for item in annotations:

            cur.execute("""
                INSERT INTO annotations(
                    image_id,
                    class_name,
                    x_center,
                    y_center,
                    box_width,
                    box_height
                )

                VALUES(
                    %s,
                    %s,
                    %s,
                    %s,
                    %s,
                    %s
                )
            """, (

                image_id,

                str(
                    item.get(
                        "class_name",
                        "BUCKET_LOADED"
                    )
                ),

                float(
                    item.get(
                        "x_center",
                        0
                    )
                ),

                float(
                    item.get(
                        "y_center",
                        0
                    )
                ),

                float(
                    item.get(
                        "box_width",
                        0
                    )
                ),

                float(
                    item.get(
                        "box_height",
                        0
                    )
                )
            ))

        conn.commit()

    except Exception:

        conn.rollback()

        raise

    finally:

        cur.close()
        conn.close()


def delete_image(image_id):

    conn = db()
    cur = conn.cursor()

    try:

        cur.execute("""
            DELETE FROM dataset_images
            WHERE id=%s
        """, (
            image_id,
        ))

        deleted = (
            cur.rowcount > 0
        )

        conn.commit()

        return deleted

    except Exception:

        conn.rollback()

        raise

    finally:

        cur.close()
        conn.close()


# ============================================================
# TRAINING STATE
# ============================================================

def set_training(
    status,
    message,
    progress
):

    conn = db()
    cur = conn.cursor()

    try:

        cur.execute("""
            UPDATE training_state

            SET
                status=%s,
                message=%s,
                progress=%s,
                updated_at=NOW()

            WHERE id=1
        """, (
            status,
            message,
            int(progress)
        ))

        conn.commit()

    except Exception:

        conn.rollback()

        raise

    finally:

        cur.close()
        conn.close()


# ============================================================
# BUILD DATASET
# ============================================================

def build_dataset():

    rows = dataset_rows()

    if len(rows) < 2:

        raise RuntimeError(
            "Training inahitaji angalau "
            "picha 2."
        )

    root = tempfile.mkdtemp(
        prefix="neerika_train_"
    )

    images_dir = os.path.join(
        root,
        "images"
    )

    labels_dir = os.path.join(
        root,
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

    usable = 0

    conn = db()
    cur = conn.cursor()

    try:

        for row in rows:

            cur.execute("""
                SELECT
                    image_data,
                    mime_type,
                    filename

                FROM dataset_images

                WHERE id=%s
            """, (
                row["id"],
            ))

            image_row = cur.fetchone()

            if not image_row:

                continue

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
                row["id"],
            ))

            annotations = []

            for item in cur.fetchall():

                class_name = str(
                    item.get(
                        "class_name"
                    )
                    or ""
                ).upper()

                if class_name == "BUCKET_LOADED":

                    annotations.append(item)

            if not annotations:

                continue

            raw = _image_bytes(
                image_row.get(
                    "image_data"
                )
            )

            if not raw:

                continue

            filename = str(
                image_row.get(
                    "filename"
                )
                or row.get(
                    "filename"
                )
                or ""
            ).lower()

            mime = str(
                image_row.get(
                    "mime_type"
                )
                or ""
            ).lower()

            if (
                "png" in mime
                or filename.endswith(".png")
            ):

                extension = ".png"

            elif (
                "webp" in mime
                or filename.endswith(".webp")
            ):

                extension = ".webp"

            else:

                extension = ".jpg"

            stem = (
                "image_"
                + str(row["id"])
            )

            image_path = os.path.join(
                images_dir,
                stem + extension
            )

            with open(
                image_path,
                "wb"
            ) as f:

                f.write(raw)

            label_path = os.path.join(
                labels_dir,
                stem + ".txt"
            )

            with open(
                label_path,
                "w",
                encoding="utf-8"
            ) as f:

                for annotation in annotations:

                    try:

                        xc = max(
                            0,
                            min(
                                1,
                                float(
                                    annotation[
                                        "x_center"
                                    ]
                                )
                            )
                        )

                        yc = max(
                            0,
                            min(
                                1,
                                float(
                                    annotation[
                                        "y_center"
                                    ]
                                )
                            )
                        )

                        bw = max(
                            0,
                            min(
                                1,
                                float(
                                    annotation[
                                        "box_width"
                                    ]
                                )
                            )
                        )

                        bh = max(
                            0,
                            min(
                                1,
                                float(
                                    annotation[
                                        "box_height"
                                    ]
                                )
                            )
                        )

                        f.write(
                            "0 "
                            + f"{xc:.6f} "
                            + f"{yc:.6f} "
                            + f"{bw:.6f} "
                            + f"{bh:.6f}\n"
                        )

                    except Exception:

                        continue

            if os.path.getsize(
                label_path
            ) > 0:

                usable += 1

    finally:

        cur.close()
        conn.close()

    if usable < 2:

        shutil.rmtree(
            root,
            ignore_errors=True
        )

        raise RuntimeError(
            "Angalau picha 2 zenye "
            "BUCKET_LOADED annotations "
            "zinahitajika."
        )

    yaml_path = os.path.join(
        root,
        "data.yaml"
    )

    with open(
        yaml_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "path: "
            + root.replace(
                "\\",
                "/"
            )
            + "\n"
        )

        f.write(
            "train: images\n"
        )

        f.write(
            "val: images\n"
        )

        f.write(
            "names:\n"
        )

        f.write(
            "  0: BUCKET_LOADED\n"
        )

    return (
        root,
        yaml_path,
        usable
    )


# ============================================================
# TRAINING
# ============================================================

def training_worker_body():

    global MODEL
    global MODEL_ERROR

    root = None

    try:

        set_training(
            "preparing",
            "Preparing dataset...",
            5
        )

        (
            root,
            yaml_path,
            usable
        ) = build_dataset()

        set_training(
            "training",
            "Dataset ready: "
            + str(usable)
            + " images. Starting YOLO "
            + str(TRAIN_EPOCHS)
            + " epochs...",
            18
        )

        print(
            "NEERIKA TRAINING:",
            "images=",
            usable,
            "epochs=",
            TRAIN_EPOCHS,
            "batch=",
            TRAIN_BATCH,
            "imgsz=",
            TRAIN_IMAGE_SIZE
        )

        model = YOLO(
            "yolo11n.pt"
        )

        output_dir = os.path.join(
            root,
            "runs"
        )

        os.makedirs(
            output_dir,
            exist_ok=True
        )

        def on_epoch_start(trainer):

            try:

                epoch = (
                    int(
                        getattr(
                            trainer,
                            "epoch",
                            0
                        )
                    )
                    + 1
                )

                total = max(
                    1,
                    int(
                        getattr(
                            trainer,
                            "epochs",
                            TRAIN_EPOCHS
                        )
                        or TRAIN_EPOCHS
                    )
                )

                progress = min(
                    90,
                    20
                    + int(
                        (
                            (epoch - 1)
                            / total
                        )
                        * 70
                    )
                )

                set_training(
                    "training",
                    "Training epoch "
                    + str(epoch)
                    + "/"
                    + str(total)
                    + " - running...",
                    progress
                )

                print(
                    "NEERIKA TRAINING:",
                    "epoch",
                    epoch,
                    "/",
                    total,
                    "started"
                )

            except Exception:

                traceback.print_exc()

        def on_epoch_end(trainer):

            try:

                epoch = (
                    int(
                        getattr(
                            trainer,
                            "epoch",
                            0
                        )
                    )
                    + 1
                )

                total = max(
                    1,
                    int(
                        getattr(
                            trainer,
                            "epochs",
                            TRAIN_EPOCHS
                        )
                        or TRAIN_EPOCHS
                    )
                )

                progress = min(
                    90,
                    20
                    + int(
                        (
                            epoch
                            / total
                        )
                        * 70
                    )
                )

                set_training(
                    "training",
                    "Training epoch "
                    + str(epoch)
                    + "/"
                    + str(total)
                    + " completed.",
                    progress
                )

                print(
                    "NEERIKA TRAINING:",
                    "epoch",
                    epoch,
                    "/",
                    total,
                    "completed"
                )

            except Exception:

                traceback.print_exc()

        model.add_callback(
            "on_train_epoch_start",
            on_epoch_start
        )

        model.add_callback(
            "on_train_epoch_end",
            on_epoch_end
        )

        set_training(
            "training",
            "Training YOLO for "
            + str(TRAIN_EPOCHS)
            + " epochs...",
            20
        )

        print(
            "NEERIKA TRAINING START"
        )

        model.train(

            data=yaml_path,

            epochs=TRAIN_EPOCHS,

            imgsz=TRAIN_IMAGE_SIZE,

            project=output_dir,

            name="neerika_bucket",

            exist_ok=True,

            verbose=True,

            workers=0,

            batch=TRAIN_BATCH,

            cache=False,

            plots=False,

            amp=False,

            patience=TRAIN_EPOCHS,

            device="cpu"
        )

        weights_dir = os.path.join(
            output_dir,
            "neerika_bucket",
            "weights"
        )

        best_path = os.path.join(
            weights_dir,
            "best.pt"
        )

        last_path = os.path.join(
            weights_dir,
            "last.pt"
        )

        if os.path.exists(
            best_path
        ):

            trained_path = best_path

        elif os.path.exists(
            last_path
        ):

            trained_path = last_path

        else:

            raise RuntimeError(
                "Training imekwisha lakini "
                "best.pt/last.pt "
                "haikupatikana."
            )

        set_training(
            "saving",
            "Training complete. "
            "Saving best model...",
            93
        )

        os.makedirs(
            os.path.dirname(
                MODEL_PATH
            ),
            exist_ok=True
        )

        shutil.copy2(
            trained_path,
            MODEL_PATH
        )

        os.makedirs(
            os.path.dirname(
                MODEL_BACKUP_PATH
            ),
            exist_ok=True
        )

        shutil.copy2(
            trained_path,
            MODEL_BACKUP_PATH
        )

        set_training(
            "saving",
            "Saving trained model to Supabase...",
            96
        )

        save_model_db(
            MODEL_PATH
        )

        with MODEL_LOCK:

            MODEL = YOLO(
                MODEL_PATH
            )

            MODEL_ERROR = ""

        set_training(
            "completed",
            "Training completed successfully. "
            + str(usable)
            + " images, "
            + str(TRAIN_EPOCHS)
            + " epochs.",
            100
        )

        print(
            "NEERIKA TRAINING COMPLETE"
        )

    except Exception as e:

        traceback.print_exc()

        try:

            set_training(
                "error",
                "Training error: "
                + str(e),
                0
            )

        except Exception:

            traceback.print_exc()

    finally:

        if root:

            shutil.rmtree(
                root,
                ignore_errors=True
            )


def reserved_training_worker():

    global TRAINING

    try:

        training_worker_body()

    finally:

        with TRAIN_LOCK:

            TRAINING = False

        print(
            "NEERIKA TRAINING WORKER STOPPED"
        )


# ============================================================
# BUCKETS
# ============================================================

def list_buckets():

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

            ORDER BY id DESC
        """)

        return cur.fetchall()

    finally:

        cur.close()
        conn.close()


def create_bucket(
    name,
    description
):

    name = name.strip()

    description = description.strip()

    if not name:

        raise ValueError(
            "Bucket name haipaswi kuwa tupu."
        )

    conn = db()
    cur = conn.cursor()

    try:

        cur.execute("""
            UPDATE buckets
            SET active=FALSE
        """)

        cur.execute("""
            INSERT INTO buckets(
                name,
                description,
                active
            )

            VALUES(
                %s,
                %s,
                TRUE
            )

            RETURNING id
        """, (
            name,
            description
        ))

        bucket_id = cur.fetchone()[
            "id"
        ]

        conn.commit()

        return bucket_id

    except Exception:

        conn.rollback()

        raise

    finally:

        cur.close()
        conn.close()


def activate_bucket(bucket_id):

    conn = db()
    cur = conn.cursor()

    try:

        cur.execute("""
            UPDATE buckets
            SET active=FALSE
        """)

        cur.execute("""
            UPDATE buckets
            SET active=TRUE
            WHERE id=%s
        """, (
            bucket_id,
        ))

        if cur.rowcount == 0:

            raise ValueError(
                "Bucket haijapatikana."
            )

        conn.commit()

    except Exception:

        conn.rollback()

        raise

    finally:

        cur.close()
        conn.close()


# ============================================================
# DASHBOARD
# ============================================================

def dashboard_page():

    try:

        rows = dataset_rows()

        image_count = len(rows)

        annotation_count = sum(
            int(
                row[
                    "annotation_count"
                ]
                or 0
            )
            for row in rows
        )

        current_count = today()

    except Exception:

        image_count = 0

        annotation_count = 0

        current_count = 0

    body = """
<div class="grid">

<div class="stat">

Today's Loaded Buckets

<div class="num">
__COUNT__
</div>

</div>

<div class="stat">

Dataset Images

<div class="num">
__IMAGES__
</div>

</div>

<div class="stat">

Annotations

<div class="num">
__ANNOTATIONS__
</div>

</div>

</div>

<div class="card">

<h2>
NEERIKA BUCKET AI
</h2>

<p>
System ya kutambua na kuhesabu
loaded ore/material buckets
kutoka shaft.
</p>

<p class="small">
Model inahitaji kufundishwa kwa
picha za bucket yako halisi.
</p>

<a href="/camera">
<button class="success">
Open Camera
</button>
</a>

<a href="/training">
<button>
Training
</button>
</a>

</div>
"""

    body = body.replace(
        "__COUNT__",
        str(current_count)
    )

    body = body.replace(
        "__IMAGES__",
        str(image_count)
    )

    body = body.replace(
        "__ANNOTATIONS__",
        str(annotation_count)
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
<div class="card">

<h2>
Camera Detection
</h2>

<video
id="video"
autoplay
playsinline>
</video>

<div>

<button onclick="startCamera()">
Start Camera
</button>

<button
onclick="captureAndDetect()"
class="success">
Detect Bucket
</button>

<button
onclick="stopCamera()"
class="secondary">
Stop
</button>

</div>

<canvas
id="canvas"
style="display:none">
</canvas>

<div
id="status"
class="status">
Camera haijaanza.
</div>

<div
id="result"
class="status">
Detection result itaonekana hapa.
</div>

</div>

<script>

let stream = null;

async function startCamera(){

    try{

        stream =
            await navigator.mediaDevices.getUserMedia({
                video:{
                    facingMode:{
                        ideal:"environment"
                    }
                },
                audio:false
            });

        document.getElementById("video")
            .srcObject = stream;

        document.getElementById("status")
            .textContent =
            "Camera imeanza.";

    }catch(error){

        document.getElementById("status")
            .textContent =
            "Camera error: "
            + error.message;
    }
}


function stopCamera(){

    if(stream){

        stream.getTracks().forEach(
            function(track){
                track.stop();
            }
        );
    }

    stream = null;

    document.getElementById("video")
        .srcObject = null;

    document.getElementById("status")
        .textContent =
        "Camera imesimama.";
}


async function captureAndDetect(){

    const video =
        document.getElementById("video");

    const canvas =
        document.getElementById("canvas");

    const result =
        document.getElementById("result");

    if(!video.videoWidth){

        result.textContent =
            "Anza camera kwanza.";

        return;
    }

    canvas.width =
        video.videoWidth;

    canvas.height =
        video.videoHeight;

    const ctx =
        canvas.getContext("2d");

    ctx.drawImage(
        video,
        0,
        0,
        canvas.width,
        canvas.height
    );

    result.textContent =
        "Inatambua...";

    canvas.toBlob(
        async function(blob){

            const form =
                new FormData();

            form.append(
                "image",
                blob,
                "camera.jpg"
            );

            try{

                const response =
                    await fetch(
                        "/api/detect",
                        {
                            method:"POST",
                            body:form
                        }
                    );

                const data =
                    await response.json();

                if(!response.ok){

                    result.innerHTML =
                        "<span class='err'>"
                        + (
                            data.error
                            ||
                            "Detection error"
                        )
                        + "</span>";

                    return;
                }

                let html =
                    "<b>Detections:</b> "
                    +
                    data.detections.length
                    +
                    "<br>";

                data.detections.forEach(
                    function(item){

                        html +=
                            item.class_name
                            +
                            " — "
                            +
                            item.confidence
                            +
                            "<br>";
                    }
                );

                html +=
                    "<br><b>Count:</b> "
                    +
                    data.count_result.count
                    +
                    "<br>"
                    +
                    data.count_result.reason;

                result.innerHTML =
                    html;

            }catch(error){

                result.textContent =
                    "Error: "
                    +
                    error.message;
            }

        },
        "image/jpeg",
        0.90
    );
}

</script>
"""

    return layout(
        "Camera",
        body,
        "Camera"
    )


# ============================================================
# BUCKET PAGE
# IMPORTANT:
# NOT an f-string.
# Therefore JavaScript {} are SAFE.
# ============================================================

def buckets_page():

    rows = list_buckets()

    table = """
<table>

<tr>
<th>ID</th>
<th>Name</th>
<th>Description</th>
<th>Active</th>
<th>Action</th>
</tr>
"""

    for row in rows:

        active = (
            "YES"
            if row["active"]
            else "NO"
        )

        table += """
<tr>

<td>__ID__</td>

<td>__NAME__</td>

<td>__DESCRIPTION__</td>

<td>__ACTIVE__</td>

<td>

<button
onclick="activateBucket(__ID__)">
Activate
</button>

</td>

</tr>
"""

        table = table.replace(
            "__ID__",
            str(row["id"]),
            1
        )

        table = table.replace(
            "__NAME__",
            esc(row["name"]),
            1
        )

        table = table.replace(
            "__DESCRIPTION__",
            esc(row["description"]),
            1
        )

        table = table.replace(
            "__ACTIVE__",
            active,
            1
        )

    table += """
</table>
"""

    body = """
<div class="card">

<h2>
Bucket Registration
</h2>

<form
onsubmit="createBucket(event)">

<div class="row">

<label>
Bucket Type Name
</label>

<input
id="bucketName"
placeholder="NEERIKA BUCKET 1"
required>

</div>

<div class="row">

<label>
Description
</label>

<textarea
id="bucketDescription"
placeholder="Maelezo ya bucket...">
</textarea>

</div>

<button
class="success"
type="submit">
Register Bucket
</button>

</form>

<div
id="bucketStatus"
class="status">
</div>

</div>

<div class="card">

<h2>
Registered Buckets
</h2>

__TABLE__

</div>

<script>

async function createBucket(event){

    event.preventDefault();

    const data = {
        name:
            document.getElementById(
                "bucketName"
            ).value,

        description:
            document.getElementById(
                "bucketDescription"
            ).value
    };

    try{

        const response =
            await fetch(
                "/api/buckets/create",
                {
                    method:"POST",

                    headers:{
                        "Content-Type":
                            "application/json"
                    },

                    body:
                        JSON.stringify(data)
                }
            );

        const result =
            await response.json();

        document.getElementById(
            "bucketStatus"
        ).textContent =
            result.error
            ||
            result.message
            ||
            "Done";

        if(response.ok){

            setTimeout(
                function(){
                    location.reload();
                },
                500
            );
        }

    }catch(error){

        document.getElementById(
            "bucketStatus"
        ).textContent =
            "Error: "
            +
            error.message;
    }
}


async function activateBucket(id){

    try{

        const response =
            await fetch(
                "/api/buckets/activate",
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

        const result =
            await response.json();

        if(!response.ok){

            alert(
                result.error
                ||
                "Error"
            );

            return;
        }

        location.reload();

    }catch(error){

        alert(
            "Error: "
            +
            error.message
        );
    }
}

</script>
"""

    body = body.replace(
        "__TABLE__",
        table
    )

    return layout(
        "Buckets",
        body,
        "Buckets"
    )


# ============================================================
# TRAINING PAGE
# NOT f-string.
# ============================================================

def training_page():

    body = """
<div class="card">

<h2>
YOLO Training
</h2>

<p>
Training hutumia picha na annotations
zilizohifadhiwa kwenye Supabase.
</p>

<p class="small">

Render Free CPU settings:

batch =
__BATCH__

image size =
__IMAGE_SIZE__

workers =
0

</p>

<button
id="startBtn"
class="success"
onclick="startTraining()">

Start Training

</button>

<div class="status">

<b id="trainStatus">
Checking...
</b>

<div
style="margin-top:10px"
class="progress">

<div
id="bar"
class="bar">
</div>

</div>

<div
id="trainMessage"
style="margin-top:8px">
</div>

</div>

</div>

<div class="card">

<h3>
Training Settings
</h3>

<p>
Epochs:
<b>
__EPOCHS__
</b>
</p>

<p>
Batch:
<b>
__BATCH__
</b>
</p>

<p>
Image size:
<b>
__IMAGE_SIZE__
</b>
</p>

<p>
Workers:
<b>
0
</b>
</p>

</div>

<script>

async function refreshTraining(){

    try{

        const response =
            await fetch(
                "/api/training/status",
                {
                    cache:"no-store"
                }
            );

        const data =
            await response.json();

        document.getElementById(
            "trainStatus"
        ).textContent =
            "Status: "
            +
            data.status;

        document.getElementById(
            "trainMessage"
        ).textContent =
            data.message
            ||
            "";

        document.getElementById(
            "bar"
        ).style.width =
            Math.max(
                0,
                Math.min(
                    100,
                    Number(
                        data.progress
                        ||
                        0
                    )
                )
            )
            +
            "%";

        const running =
            data.status === "training"
            ||
            data.status === "preparing"
            ||
            data.status === "saving";

        const button =
            document.getElementById(
                "startBtn"
            );

        button.disabled =
            running;

        button.textContent =
            running
            ?
            "Training is running..."
            :
            "Start Training";

    }catch(error){

        document.getElementById(
            "trainMessage"
        ).textContent =
            "Status error: "
            +
            error.message;
    }
}


async function startTraining(){

    const button =
        document.getElementById(
            "startBtn"
        );

    button.disabled = true;

    try{

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
                "Training haikuanza."
            );
        }

    }catch(error){

        alert(
            "Connection error: "
            +
            error.message
        );
    }

    refreshTraining();
}


refreshTraining();

setInterval(
    refreshTraining,
    2000
);

</script>
"""

    body = body.replace(
        "__BATCH__",
        str(TRAIN_BATCH)
    )

    body = body.replace(
        "__IMAGE_SIZE__",
        str(TRAIN_IMAGE_SIZE)
    )

    body = body.replace(
        "__EPOCHS__",
        str(TRAIN_EPOCHS)
    )

    return layout(
        "Training",
        body,
        "Training"
    )


# ============================================================
# HISTORY
# ============================================================

def history_page():

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

            LIMIT 60
        """)

        rows = cur.fetchall()

    finally:

        cur.close()
        conn.close()

    table = """
<table>

<tr>
<th>Date</th>
<th>Loaded Buckets</th>
<th>Updated</th>
</tr>
"""

    for row in rows:

        table += """
<tr>

<td>__DATE__</td>

<td>__COUNT__</td>

<td>__UPDATED__</td>

</tr>
"""

        table = table.replace(
            "__DATE__",
            esc(row["count_date"]),
            1
        )

        table = table.replace(
            "__COUNT__",
            esc(row["bucket_count"]),
            1
        )

        table = table.replace(
            "__UPDATED__",
            esc(row["updated_at"]),
            1
        )

    table += """
</table>
"""

    body = """
<div class="card">

<h2>
Production History
</h2>

__TABLE__

</div>
"""

    body = body.replace(
        "__TABLE__",
        table
    )

    return layout(
        "History",
        body,
        "History"
    )


# ============================================================
# SETTINGS
# ============================================================

def settings_page():

    model_exists = (
        os.path.exists(MODEL_PATH)
        or
        os.path.exists(
            MODEL_BACKUP_PATH
        )
    )

    model_status = (
        "Available"
        if model_exists
        else "Not available"
    )

    body = """
<div class="card">

<h2>
Settings
</h2>

<p>

<b>
Model file:
</b>

__MODEL__

</p>

<p>

<b>
Model message:
</b>

__MESSAGE__

</p>

<p>

<b>
Confidence:
</b>

__CONFIDENCE__

</p>

<p>

<b>
Detection image size:
</b>

__SIZE__

</p>

<p class="small">

DATABASE_URL inabaki
server-side.

Usiiweke kwenye
browser/frontend.

</p>

</div>
"""

    body = body.replace(
        "__MODEL__",
        esc(model_status)
    )

    body = body.replace(
        "__MESSAGE__",
        esc(MODEL_ERROR)
    )

    body = body.replace(
        "__CONFIDENCE__",
        str(YOLO_CONFIDENCE)
    )

    body = body.replace(
        "__SIZE__",
        str(YOLO_IMAGE_SIZE)
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
        format_string,
        *args
    ):

        try:

            print(
                "%s - %s"
                % (
                    self.address_string(),
                    format_string % args
                )
            )

        except Exception:

            pass


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

                return html_out(
                    self,
                    dashboard_page()
                )

            if path == "/camera":

                return html_out(
                    self,
                    camera_page()
                )

            if path == "/buckets":

                return html_out(
                    self,
                    buckets_page()
                )

            if path == "/training":

                return html_out(
                    self,
                    training_page()
                )

            if path == "/history":

                return html_out(
                    self,
                    history_page()
                )

            if path == "/settings":

                return html_out(
                    self,
                    settings_page()
                )

            if path == "/api/training/status":

                conn = db()
                cur = conn.cursor()

                try:

                    cur.execute("""
                        SELECT
                            status,
                            message,
                            progress,
                            updated_at

                        FROM training_state

                        WHERE id=1
                    """)

                    row = cur.fetchone()

                finally:

                    cur.close()
                    conn.close()

                if not row:

                    return json_out(
                        self,
                        {
                            "status":
                                "idle",

                            "message":
                                "Ready",

                            "progress":
                                0
                        }
                    )

                return json_out(
                    self,
                    {
                        "status":
                            row["status"],

                        "message":
                            row["message"],

                        "progress":
                            int(
                                row["progress"]
                                or 0
                            ),

                        "updated_at":
                            row["updated_at"]
                    }
                )

            if path == "/api/count/today":

                return json_out(
                    self,
                    {
                        "count":
                            today()
                    }
                )

            if path == "/api/dataset":

                return json_out(
                    self,
                    {
                        "images":
                            dataset_rows()
                    }
                )

            if path == "/api/buckets":

                return json_out(
                    self,
                    {
                        "buckets":
                            list_buckets()
                    }
                )

            return error_out(
                self,
                "Not found",
                404
            )

        except Exception as e:

            traceback.print_exc()

            return error_out(
                self,
                e,
                500
            )


    # ========================================================
    # POST
    # ========================================================

    def do_POST(self):

        global TRAINING

        try:

            parsed = urlparse(
                self.path
            )

            path = parsed.path

            # ------------------------------------------------
            # DETECTION
            # ------------------------------------------------

            if path == "/api/detect":

                form = parse_multipart(
                    self
                )

                if "image" not in form:

                    return error_out(
                        self,
                        "Image haikutumwa.",
                        400
                    )

                item = form["image"]

                data = item["data"]

                if len(data) > MAX_UPLOAD_BYTES:

                    return error_out(
                        self,
                        "Image ni kubwa sana.",
                        413
                    )

                detections = detect(
                    data
                )

                count_result = count_loaded(
                    detections
                )

                return json_out(
                    self,
                    {
                        "detections":
                            detections,

                        "count_result":
                            count_result
                    }
                )


            # ------------------------------------------------
            # START TRAINING
            # ------------------------------------------------

            if path == "/api/training/start":

                with TRAIN_LOCK:

                    if TRAINING:

                        return error_out(
                            self,
                            "Training is already running.",
                            409
                        )

                    TRAINING = True

                try:

                    set_training(
                        "preparing",
                        "Training requested...",
                        5
                    )

                except Exception:

                    with TRAIN_LOCK:

                        TRAINING = False

                    raise

                worker = threading.Thread(
                    target=
                        reserved_training_worker,
                    daemon=True
                )

                worker.start()

                return json_out(
                    self,
                    {
                        "message":
                            "Training started."
                    }
                )


            # ------------------------------------------------
            # CREATE BUCKET
            # ------------------------------------------------

            if path == "/api/buckets/create":

                data = parse_json(
                    self
                )

                bucket_id = create_bucket(
                    str(
                        data.get(
                            "name",
                            ""
                        )
                    ),
                    str(
                        data.get(
                            "description",
                            ""
                        )
                    )
                )

                return json_out(
                    self,
                    {
                        "message":
                            "Bucket registered.",

                        "id":
                            bucket_id
                    }
                )


            # ------------------------------------------------
            # ACTIVATE BUCKET
            # ------------------------------------------------

            if path == "/api/buckets/activate":

                data = parse_json(
                    self
                )

                if "id" not in data:

                    raise ValueError(
                        "Bucket ID haipo."
                    )

                bucket_id = int(
                    data["id"]
                )

                activate_bucket(
                    bucket_id
                )

                return json_out(
                    self,
                    {
                        "message":
                            "Bucket activated."
                    }
                )


            # ------------------------------------------------
            # DATASET UPLOAD
            # ------------------------------------------------

            if path == "/api/dataset/upload":

                form = parse_multipart(
                    self
                )

                if "image" not in form:

                    return error_out(
                        self,
                        "Image haikutumwa.",
                        400
                    )

                item = form["image"]

                data = item["data"]

                if len(data) > MAX_UPLOAD_BYTES:

                    return error_out(
                        self,
                        "Image ni kubwa sana.",
                        413
                    )

                filename = (
                    item.get(
                        "filename"
                    )
                    or
                    "uploaded.jpg"
                )

                mime = (
                    item.get(
                        "content_type"
                    )
                    or
                    "image/jpeg"
                )

                image_id = save_image(
                    filename,
                    mime,
                    data
                )

                return json_out(
                    self,
                    {
                        "message":
                            "Image saved.",

                        "image_id":
                            image_id
                    }
                )


            # ------------------------------------------------
            # SAVE ANNOTATIONS
            # ------------------------------------------------

            if path == "/api/dataset/annotations":

                data = parse_json(
                    self
                )

                if "image_id" not in data:

                    raise ValueError(
                        "image_id haipo."
                    )

                image_id = int(
                    data["image_id"]
                )

                annotations = data.get(
                    "annotations",
                    []
                )

                if not isinstance(
                    annotations,
                    list
                ):

                    raise ValueError(
                        "annotations lazima "
                        "iwe list."
                    )

                save_annotations(
                    image_id,
                    annotations
                )

                return json_out(
                    self,
                    {
                        "message":
                            "Annotations saved."
                    }
                )


            # ------------------------------------------------
            # DELETE IMAGE
            # ------------------------------------------------

            if path == "/api/dataset/delete":

                data = parse_json(
                    self
                )

                if "id" not in data:

                    raise ValueError(
                        "Image ID haipo."
                    )

                image_id = int(
                    data["id"]
                )

                deleted = delete_image(
                    image_id
                )

                return json_out(
                    self,
                    {
                        "deleted":
                            deleted
                    }
                )


            return error_out(
                self,
                "Not found",
                404
            )

        except Exception as e:

            traceback.print_exc()

            return error_out(
                self,
                e,
                500
            )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "========================================"
    )

    print(
        "       NEERIKA BUCKET AI"
    )

    print(
        "========================================"
    )

    print(
        "Starting server..."
    )

    if not DATABASE_URL:

        print(
            "WARNING: DATABASE_URL "
            "haijawekwa."
        )

    else:

        try:

            init_db()

            reset_stale_training_state()

            print(
                "Database initialized."
            )

        except Exception as e:

            print(
                "DATABASE ERROR:",
                str(e)
            )

            traceback.print_exc()

    try:

        load_model()

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
        "Running on port "
        + str(PORT)
    )

    try:

        server.serve_forever()

    except KeyboardInterrupt:

        print(
            "Server stopped."
        )

    finally:

        server.server_close()


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    main()
