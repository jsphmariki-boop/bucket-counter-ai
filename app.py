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
from urllib.parse import urlparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/Ultralytics")

import psycopg2
from psycopg2.extras import RealDictCursor
from ultralytics import YOLO
from PIL import Image
import numpy as np

HOST = "0.0.0.0"
PORT = int(os.environ.get("PORT", "8080"))
DATABASE_URL = (
    os.environ.get("DATABASE_URL")
    or os.environ.get("SUPABASE_DB_URL")
    or os.environ.get("POSTGRES_URL")
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best.pt")
MODEL_BACKUP_PATH = os.path.join(BASE_DIR, "ai", "models", "best.pt")

YOLO_CONFIDENCE = float(os.environ.get("YOLO_CONFIDENCE", "0.25"))
YOLO_IMAGE_SIZE = int(os.environ.get("YOLO_IMAGE_SIZE", "640"))
TRAIN_EPOCHS = int(os.environ.get("TRAIN_EPOCHS", "20"))

MAX_JSON_BYTES = 15 * 1024 * 1024
MAX_UPLOAD_BYTES = 15 * 1024 * 1024
COUNT_COOLDOWN_SECONDS = 4.0

MODEL = None
MODEL_ERROR = ""
MODEL_LOCK = threading.Lock()

TRAINING = False
TRAINING_RUN_ID = None
TRAINING_LOCK_CONN = None
TRAINING_LOCK_KEY = 918273645

LAST_COUNT_TIME = 0.0
SERVER_INSTANCE_ID = uuid.uuid4().hex

CLASS_IDS = {
    0: "BUCKET_LOADED",
    1: "BUCKET_EMPTY",
    2: "PEOPLE",
    3: "EQUIPMENT",
}


def db():
    if not DATABASE_URL:
        raise RuntimeError(
            "DATABASE_URL haijawekwa kwenye Render Environment Variables."
        )
    return psycopg2.connect(
        DATABASE_URL,
        cursor_factory=RealDictCursor,
        connect_timeout=15,
    )


def init_db():
    c = db()
    x = c.cursor()
    try:
        x.execute("""
            CREATE TABLE IF NOT EXISTS buckets (
                id SERIAL PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT DEFAULT '',
                active BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)

        x.execute("""
            CREATE TABLE IF NOT EXISTS dataset_images (
                id SERIAL PRIMARY KEY,
                filename TEXT NOT NULL,
                image_data BYTEA NOT NULL,
                mime_type TEXT DEFAULT 'image/jpeg',
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)

        x.execute("""
            CREATE TABLE IF NOT EXISTS annotations (
                id SERIAL PRIMARY KEY,
                image_id INTEGER REFERENCES dataset_images(id) ON DELETE CASCADE,
                class_name TEXT NOT NULL,
                x_center DOUBLE PRECISION NOT NULL,
                y_center DOUBLE PRECISION NOT NULL,
                box_width DOUBLE PRECISION NOT NULL,
                box_height DOUBLE PRECISION NOT NULL,
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)

        x.execute("""
            CREATE TABLE IF NOT EXISTS training_state (
                id INTEGER PRIMARY KEY,
                status TEXT DEFAULT 'idle',
                message TEXT DEFAULT '',
                progress INTEGER DEFAULT 0,
                updated_at TIMESTAMPTZ DEFAULT NOW(),
                run_id TEXT,
                server_id TEXT,
                started_at TIMESTAMPTZ,
                finished_at TIMESTAMPTZ
            )
        """)

        x.execute("""
            CREATE TABLE IF NOT EXISTS trained_model (
                id INTEGER PRIMARY KEY,
                model_data BYTEA NOT NULL,
                filename TEXT DEFAULT 'best.pt',
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)

        x.execute("""
            CREATE TABLE IF NOT EXISTS daily_counts (
                id SERIAL PRIMARY KEY,
                count_date DATE UNIQUE NOT NULL,
                bucket_count INTEGER DEFAULT 0,
                updated_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)

        x.execute("""
            CREATE TABLE IF NOT EXISTS detection_events (
                id SERIAL PRIMARY KEY,
                class_name TEXT NOT NULL,
                confidence DOUBLE PRECISION DEFAULT 0,
                counted BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)

        x.execute("""
            INSERT INTO training_state
                (id, status, message, progress)
            VALUES
                (1, 'idle', 'Ready', 0)
            ON CONFLICT (id) DO NOTHING
        """)

        c.commit()
    except Exception:
        c.rollback()
        raise
    finally:
        x.close()
        c.close()


def esc(v):
    return (
        "" if v is None else str(v)
    ).replace("&", "&amp;").replace("<", "&lt;").replace(
        ">", "&gt;"
    ).replace('"', "&quot;").replace("'", "&#039;")


def json_out(h, data, status=200):
    b = json.dumps(data, ensure_ascii=False, default=str).encode()
    h.send_response(status)
    h.send_header("Content-Type", "application/json; charset=utf-8")
    h.send_header("Content-Length", str(len(b)))
    h.send_header("Cache-Control", "no-store")
    h.end_headers()
    h.wfile.write(b)


def html_out(h, text, status=200):
    b = text.encode()
    h.send_response(status)
    h.send_header("Content-Type", "text/html; charset=utf-8")
    h.send_header("Content-Length", str(len(b)))
    h.send_header("Cache-Control", "no-store")
    h.end_headers()
    h.wfile.write(b)


def error_out(h, msg, status=500):
    return json_out(h, {"error": str(msg)}, status)


def layout(title, body, active):
    items = [
        ("Dashboard", "/"),
        ("Camera", "/camera"),
        ("Buckets", "/buckets"),
        ("Training", "/training"),
        ("History", "/history"),
        ("Settings", "/settings"),
    ]

    nav = "".join(
        '<a class="nav-item {}" href="{}">{}</a>'.format(
            "active" if n == active else "", p, n
        )
        for n, p in items
    )

    page = r"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__ - NEERIKA BUCKET AI</title>
<style>
*{box-sizing:border-box}
body{margin:0;font-family:Arial,sans-serif;background:#f4f6f8;color:#17202a}
header{background:#111827;color:#fff;padding:16px}
.brand{font-size:22px;font-weight:800}
.sub{font-size:13px;opacity:.75;margin-top:4px}
.nav{display:flex;gap:7px;overflow:auto;background:#1f2937;padding:8px}
.nav-item{color:#fff;text-decoration:none;padding:10px 13px;border-radius:8px;white-space:nowrap}
.nav-item.active,.nav-item:hover{background:#374151}
main{max-width:1200px;margin:auto;padding:16px}
.card,.stat{background:#fff;border-radius:14px;padding:18px;margin-bottom:16px;box-shadow:0 2px 10px rgba(0,0,0,.07)}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:14px}
.num{font-size:38px;font-weight:800;margin-top:7px}
button{border:0;border-radius:9px;padding:10px 15px;background:#111827;color:#fff;cursor:pointer;margin:3px}
button.secondary{background:#6b7280}
button.success{background:#166534}
button.danger{background:#991b1b}
button:disabled{opacity:.55;cursor:not-allowed}
input,select{width:100%;padding:10px;border:1px solid #d1d5db;border-radius:8px}
label{font-weight:700;display:block;margin-bottom:5px}
.row{margin-bottom:13px}
.status{padding:10px;background:#f3f4f6;border-radius:8px;margin-top:10px}
table{width:100%;border-collapse:collapse}
th,td{text-align:left;padding:9px;border-bottom:1px solid #e5e7eb}
.footer{text-align:center;color:#6b7280;font-size:12px;padding:25px}
.progress{height:24px;background:#e5e7eb;border-radius:12px;overflow:hidden}
.bar{height:100%;background:#111827;width:0%;transition:width .3s;text-align:center;color:#fff;font-size:12px;line-height:24px}
.preview{max-width:100%;max-height:500px;border-radius:10px}
.small{font-size:13px;color:#6b7280}
.epoch{font-size:28px;font-weight:800;margin:8px 0}
@media(max-width:600px){
 main{padding:10px}
 .num{font-size:30px}
}
</style>
</head>
<body>
<header>
<div class="brand">NEERIKA BUCKET AI</div>
<div class="sub">Mining Production Bucket Counter</div>
</header>
<nav class="nav">__NAV__</nav>
<main>__BODY__</main>
<div class="footer">Geology &amp; Mining Services</div>
</body>
</html>"""

    return (
        page.replace("__TITLE__", esc(title))
        .replace("__NAV__", nav)
        .replace("__BODY__", body)
    )


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
        q = value.strip()
        if q.startswith("\\x"):
            try:
                return bytes.fromhex(q[2:])
            except Exception:
                pass
        try:
            return base64.b64decode(q, validate=True)
        except Exception:
            return q.encode("utf-8")
    try:
        return bytes(value)
    except Exception:
        return b""


def restore_model():
    if os.path.exists(MODEL_PATH):
        return True

    try:
        c = db()
        x = c.cursor()
        x.execute("SELECT model_data FROM trained_model WHERE id=1")
        r = x.fetchone()
        x.close()
        c.close()

        if r and r["model_data"]:
            os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
            with open(MODEL_PATH, "wb") as f:
                f.write(_image_bytes(r["model_data"]))
            return True
    except Exception:
        traceback.print_exc()

    return False


def load_model():
    global MODEL, MODEL_ERROR

    with MODEL_LOCK:
        if MODEL is not None:
            return MODEL

        try:
            restore_model()

            if os.path.exists(MODEL_PATH):
                p = MODEL_PATH
            elif os.path.exists(MODEL_BACKUP_PATH):
                p = MODEL_BACKUP_PATH
            else:
                p = None

            if p:
                MODEL = YOLO(p)
                MODEL_ERROR = ""
                print("Loaded model:", p)
                return MODEL

            MODEL = YOLO("yolo11n.pt")
            MODEL_ERROR = (
                "Fallback yolo11n.pt loaded. "
                "Start training your bucket model."
            )
            return MODEL

        except Exception as e:
            MODEL_ERROR = str(e)
            traceback.print_exc()
            return None


def save_model_db(path):
    with open(path, "rb") as f:
        b = f.read()

    c = db()
    x = c.cursor()

    try:
        x.execute(
            """
            INSERT INTO trained_model(id,model_data,filename)
            VALUES(1,%s,%s)
            ON CONFLICT(id)
            DO UPDATE SET
                model_data=EXCLUDED.model_data,
                filename=EXCLUDED.filename,
                created_at=NOW()
            """,
            (psycopg2.Binary(b), "best.pt"),
        )
        c.commit()
    finally:
        x.close()
        c.close()


def normalize(cid, name=""):
    if cid in CLASS_IDS:
        return CLASS_IDS[cid]

    s = str(name).upper()

    if "EMPTY" in s:
        return "BUCKET_EMPTY"
    if "PEOPLE" in s or "PERSON" in s:
        return "PEOPLE"
    if "EQUIPMENT" in s or "MACHINE" in s:
        return "EQUIPMENT"
    if "LOADED" in s or "BUCKET" in s:
        return "BUCKET_LOADED"

    return "UNKNOWN"


def detect(data):
    model = load_model()

    if model is None:
        raise RuntimeError("YOLO model haijapatikana: " + MODEL_ERROR)

    im = Image.open(io.BytesIO(data)).convert("RGB")
    arr = np.array(im)

    results = model.predict(
        source=arr,
        conf=YOLO_CONFIDENCE,
        imgsz=YOLO_IMAGE_SIZE,
        verbose=False,
    )

    out = []

    if not results or results[0].boxes is None:
        return out

    names = getattr(results[0], "names", {})

    for box in results[0].boxes:
        try:
            xy = box.xyxy[0].cpu().numpy().tolist()
            conf = float(box.conf[0].cpu().item())
            cid = int(box.cls[0].cpu().item())
            raw = names.get(cid, "")

            out.append(
                {
                    "class_id": cid,
                    "class_name": normalize(cid, raw),
                    "raw_class_name": str(raw),
                    "confidence": round(conf, 4),
                    "x1": round(float(xy[0]), 2),
                    "y1": round(float(xy[1]), 2),
                    "x2": round(float(xy[2]), 2),
                    "y2": round(float(xy[3]), 2),
                }
            )
        except Exception:
            pass

    return out


def count_loaded(dets):
    global LAST_COUNT_TIME

    loaded = [
        d for d in dets
        if d["class_name"] == "BUCKET_LOADED"
    ]

    if not loaded:
        return {
            "counted": False,
            "reason": "No loaded bucket detected.",
            "count": 0,
        }

    if time.time() - LAST_COUNT_TIME < COUNT_COOLDOWN_SECONDS:
        return {
            "counted": False,
            "reason": "Cooldown: possible same bucket.",
            "count": 0,
        }

    best = max(loaded, key=lambda d: d["confidence"])
    conf = float(best["confidence"])

    c = db()
    x = c.cursor()

    try:
        x.execute(
            """
            SELECT id
            FROM daily_counts
            WHERE count_date = CURRENT_DATE
            LIMIT 1
            """
        )

        existing = x.fetchone()

        if existing:
            x.execute(
                """
                UPDATE daily_counts
                SET bucket_count=COALESCE(bucket_count,0)+1,
                    updated_at=NOW()
                WHERE id=%s
                """,
                (existing["id"],),
            )
        else:
            x.execute(
                """
                INSERT INTO daily_counts
                    (count_date,bucket_count,updated_at)
                VALUES
                    (CURRENT_DATE,1,NOW())
                """
            )

        x.execute(
            """
            INSERT INTO detection_events
                (class_name,confidence,counted)
            VALUES
                (%s,%s,TRUE)
            """,
            ("BUCKET_LOADED", conf),
        )

        c.commit()

    finally:
        x.close()
        c.close()

    LAST_COUNT_TIME = time.time()

    return {
        "counted": True,
        "reason": "Loaded bucket counted.",
        "count": 1,
        "confidence": conf,
    }


def today():
    c = db()
    x = c.cursor()

    try:
        x.execute(
            """
            SELECT bucket_count
            FROM daily_counts
            WHERE count_date = CURRENT_DATE
            LIMIT 1
            """
        )
        r = x.fetchone()
        return int(r["bucket_count"]) if r else 0
    finally:
        x.close()
        c.close()


def dataset_rows():
    c = db()
    x = c.cursor()

    try:
        x.execute(
            """
            SELECT
                di.id,
                di.filename,
                di.mime_type,
                di.created_at,
                COUNT(a.id) AS annotation_count
            FROM dataset_images di
            LEFT JOIN annotations a
                ON a.image_id=di.id
            GROUP BY
                di.id,
                di.filename,
                di.mime_type,
                di.created_at
            ORDER BY di.id DESC
            """
        )
        return x.fetchall()
    finally:
        x.close()
        c.close()


def save_image(filename, mime, data):
    c = db()
    x = c.cursor()

    try:
        x.execute(
            """
            INSERT INTO dataset_images
                (filename,image_data,mime_type)
            VALUES
                (%s,%s,%s)
            RETURNING id
            """,
            (filename, psycopg2.Binary(data), mime),
        )
        i = x.fetchone()["id"]
        c.commit()
        return i
    finally:
        x.close()
        c.close()


def save_annotations(image_id, anns):
    c = db()
    x = c.cursor()

    try:
        x.execute(
            "DELETE FROM annotations WHERE image_id=%s",
            (image_id,),
        )

        for a in anns:
            x.execute(
                """
                INSERT INTO annotations
                    (image_id,class_name,x_center,y_center,box_width,box_height)
                VALUES
                    (%s,%s,%s,%s,%s,%s)
                """,
                (
                    image_id,
                    str(a.get("class_name", "BUCKET_LOADED")),
                    float(a.get("x_center", 0)),
                    float(a.get("y_center", 0)),
                    float(a.get("box_width", 0)),
                    float(a.get("box_height", 0)),
                ),
            )

        c.commit()
    finally:
        x.close()
        c.close()


def delete_image(i):
    c = db()
    x = c.cursor()

    try:
        x.execute(
            "DELETE FROM dataset_images WHERE id=%s",
            (i,),
        )
        ok = x.rowcount > 0
        c.commit()
        return ok
    finally:
        x.close()
        c.close()


def set_training(
    status,
    msg,
    progress,
    run_id=None,
    finished=False,
):
    progress = max(0, min(100, int(progress)))

    c = db()
    x = c.cursor()

    try:
        if run_id:
            if finished:
                x.execute(
                    """
                    UPDATE training_state
                    SET status=%s,
                        message=%s,
                        progress=%s,
                        updated_at=NOW(),
                        finished_at=NOW()
                    WHERE id=1 AND run_id=%s
                    """,
                    (status, msg, progress, str(run_id)),
                )
            else:
                x.execute(
                    """
                    UPDATE training_state
                    SET status=%s,
                        message=%s,
                        progress=%s,
                        updated_at=NOW()
                    WHERE id=1 AND run_id=%s
                    """,
                    (status, msg, progress, str(run_id)),
                )
        else:
            x.execute(
                """
                UPDATE training_state
                SET status=%s,
                    message=%s,
                    progress=%s,
                    updated_at=NOW()
                WHERE id=1
                """,
                (status, msg, progress),
            )

        c.commit()

    finally:
        x.close()
        c.close()


def mark_interrupted_on_startup():
    c = db()
    x = c.cursor()

    try:
        x.execute(
            """
            UPDATE training_state
            SET status='interrupted',
                message='Training interrupted because Render restarted. Start a new training run.',
                progress=0,
                updated_at=NOW(),
                finished_at=NOW()
            WHERE id=1
              AND status IN ('preparing','training','saving')
            """
        )

        if x.rowcount:
            print(
                "NEERIKA TRAINING: previous active run marked INTERRUPTED."
            )

        c.commit()

    finally:
        x.close()
        c.close()


def acquire_training_lock():
    global TRAINING_LOCK_CONN

    conn = psycopg2.connect(
        DATABASE_URL,
        cursor_factory=RealDictCursor,
        connect_timeout=15,
    )
    conn.autocommit = True

    cur = conn.cursor()
    cur.execute(
        "SELECT pg_try_advisory_lock(%s) AS locked",
        (TRAINING_LOCK_KEY,),
    )
    row = cur.fetchone()
    cur.close()

    if not row or not row["locked"]:
        conn.close()
        return None

    TRAINING_LOCK_CONN = conn
    return conn


def release_training_lock(conn=None):
    global TRAINING_LOCK_CONN

    conn = conn or TRAINING_LOCK_CONN

    if not conn:
        return

    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT pg_advisory_unlock(%s)",
            (TRAINING_LOCK_KEY,),
        )
        cur.close()
    except Exception:
        pass

    try:
        conn.close()
    except Exception:
        pass

    if conn is TRAINING_LOCK_CONN:
        TRAINING_LOCK_CONN = None


def claim_training_run():
    global TRAINING, TRAINING_RUN_ID

    lock_conn = acquire_training_lock()

    if not lock_conn:
        return None

    run_id = uuid.uuid4().hex

    c = db()
    x = c.cursor()

    try:
        x.execute(
            """
            UPDATE training_state
            SET status='preparing',
                message='Training accepted. Preparing dataset...',
                progress=1,
                updated_at=NOW(),
                run_id=%s,
                server_id=%s,
                started_at=NOW(),
                finished_at=NULL
            WHERE id=1
              AND status NOT IN ('preparing','training','saving')
            RETURNING id
            """,
            (run_id, SERVER_INSTANCE_ID),
        )

        if not x.fetchone():
            c.rollback()
            release_training_lock(lock_conn)
            return None

        c.commit()

        TRAINING = True
        TRAINING_RUN_ID = run_id

        return run_id

    except Exception:
        c.rollback()
        release_training_lock(lock_conn)
        raise

    finally:
        x.close()
        c.close()


def build_dataset():
    rows = dataset_rows()

    if len(rows) < 2:
        raise RuntimeError(
            "Training inahitaji angalau picha 2."
        )

    root = tempfile.mkdtemp(prefix="neerika_train_")
    imgs = os.path.join(root, "images")
    labs = os.path.join(root, "labels")

    os.makedirs(imgs, exist_ok=True)
    os.makedirs(labs, exist_ok=True)

    usable = 0

    c = db()
    x = c.cursor()

    try:
        for r in rows:
            x.execute(
                """
                SELECT image_data,mime_type,filename
                FROM dataset_images
                WHERE id=%s
                """,
                (r["id"],),
            )
            ir = x.fetchone()

            if not ir:
                continue

            x.execute(
                """
                SELECT
                    class_name,
                    x_center,
                    y_center,
                    box_width,
                    box_height
                FROM annotations
                WHERE image_id=%s
                ORDER BY id
                """,
                (r["id"],),
            )

            anns = [
                a for a in x.fetchall()
                if str(a.get("class_name") or "").upper()
                == "BUCKET_LOADED"
            ]

            if not anns:
                continue

            raw = _image_bytes(ir.get("image_data"))

            if not raw:
                print(
                    "Skipping image",
                    r["id"],
                    "- invalid image_data",
                )
                continue

            filename = str(
                ir.get("filename")
                or r.get("filename")
                or ""
            ).lower()

            mime = str(
                ir.get("mime_type") or ""
            ).lower()

            if "png" in mime or filename.endswith(".png"):
                ext = ".png"
            elif "webp" in mime or filename.endswith(".webp"):
                ext = ".webp"
            else:
                ext = ".jpg"

            stem = "image_" + str(r["id"])

            with open(
                os.path.join(imgs, stem + ext),
                "wb",
            ) as f:
                f.write(raw)

            label_path = os.path.join(
                labs,
                stem + ".txt",
            )

            with open(
                label_path,
                "w",
                encoding="utf-8",
            ) as f:
                for a in anns:
                    try:
                        xc = max(
                            0,
                            min(1, float(a["x_center"])),
                        )
                        yc = max(
                            0,
                            min(1, float(a["y_center"])),
                        )
                        bw = max(
                            0,
                            min(1, float(a["box_width"])),
                        )
                        bh = max(
                            0,
                            min(1, float(a["box_height"])),
                        )

                        f.write(
                            f"0 {xc:.6f} {yc:.6f} "
                            f"{bw:.6f} {bh:.6f}\n"
                        )
                    except Exception:
                        pass

            if os.path.getsize(label_path) > 0:
                usable += 1

    finally:
        x.close()
        c.close()

    if usable < 2:
        shutil.rmtree(root, ignore_errors=True)
        raise RuntimeError(
            "Angalau picha 2 zenye BUCKET_LOADED "
            "annotations zinahitajika."
        )

    yaml = os.path.join(root, "data.yaml")

    with open(yaml, "w", encoding="utf-8") as f:
        f.write(
            "path: " + root.replace("\\", "/") + "\n"
            "train: images\n"
            "val: images\n"
            "names:\n"
            "  0: BUCKET_LOADED\n"
        )

    return root, yaml, usable


def read_epoch_from_results(results_csv):
    """
    YOLO huandika results.csv baada ya kila epoch.
    Hii ndiyo njia kuu ya kuonyesha epoch kwenye UI,
    badala ya kutegemea callback pekee.
    """
    if not os.path.exists(results_csv):
        return None

    try:
        with open(
            results_csv,
            "r",
            encoding="utf-8",
            errors="ignore",
        ) as f:
            rows = list(csv.DictReader(f))

        if not rows:
            return None

        last = rows[-1]

        epoch_value = last.get("epoch")

        if epoch_value is None:
            return None

        epoch = int(float(epoch_value)) + 1

        return epoch

    except Exception:
        return None


def epoch_monitor(run_id, results_csv, total_epochs, stop_event):
    """
    Background monitor:
    Inasoma results.csv kila sekunde 2.
    Hivyo hata kama callback ya Ultralytics haifanyi kazi,
    UI bado itaonyesha Epoch 1/20, 2/20, ... 20/20.
    """
    last_epoch = 0

    while not stop_event.is_set():
        try:
            epoch = read_epoch_from_results(results_csv)

            if epoch is not None and epoch > last_epoch:
                last_epoch = epoch

                epoch = min(epoch, total_epochs)

                progress = min(
                    90,
                    20 + int(
                        (epoch / max(1, total_epochs)) * 70
                    ),
                )

                set_training(
                    "training",
                    f"Training YOLO... Epoch {epoch}/{total_epochs}",
                    progress,
                    run_id,
                )

                print(
                    f"NEERIKA TRAINING: "
                    f"Epoch {epoch}/{total_epochs} "
                    f"-> {progress}%"
                )

        except Exception:
            traceback.print_exc()

        stop_event.wait(2)


def training_worker(run_id, lock_conn):
    global TRAINING
    global TRAINING_RUN_ID
    global MODEL
    global MODEL_ERROR

    root = None
    monitor_stop = threading.Event()
    monitor_thread = None

    try:
        set_training(
            "preparing",
            "Preparing dataset...",
            5,
            run_id,
        )

        root, yaml, usable = build_dataset()

        set_training(
            "training",
            f"Dataset ready: {usable} images. Starting YOLO {TRAIN_EPOCHS} epochs...",
            18,
            run_id,
        )

        model = YOLO("yolo11n.pt")

        out = os.path.join(root, "runs")
        os.makedirs(out, exist_ok=True)

        run_dir = os.path.join(
            out,
            "neerika_bucket",
        )

        results_csv = os.path.join(
            run_dir,
            "results.csv",
        )

        # Callback ya ziada kwa compatibility na matoleo tofauti ya Ultralytics.
        def on_epoch_end(trainer):
            try:
                epoch = int(
                    getattr(trainer, "epoch", 0)
                ) + 1

                total = int(
                    getattr(
                        trainer,
                        "epochs",
                        TRAIN_EPOCHS,
                    )
                    or TRAIN_EPOCHS
                )

                progress = min(
                    90,
                    20 + int(
                        (epoch / max(1, total)) * 70
                    ),
                )

                set_training(
                    "training",
                    f"Training YOLO... Epoch {epoch}/{total}",
                    progress,
                    run_id,
                )

                print(
                    f"NEERIKA CALLBACK: "
                    f"Epoch {epoch}/{total}"
                )

            except Exception:
                traceback.print_exc()

        try:
            model.add_callback(
                "on_train_epoch_end",
                on_epoch_end,
            )
        except Exception:
            pass

        try:
            model.add_callback(
                "on_fit_epoch_end",
                on_epoch_end,
            )
        except Exception:
            pass

        set_training(
            "training",
            f"Training YOLO for {TRAIN_EPOCHS} epochs... Epoch 0/{TRAIN_EPOCHS}",
            20,
            run_id,
        )

        print(
            "=" * 60
        )
        print(
            "NEERIKA TRAINING START"
        )
        print(
            "Run:", run_id
        )
        print(
            "Epochs:", TRAIN_EPOCHS
        )
        print(
            "Images:", usable
        )
        print(
            "Workers: 0"
        )
        print(
            "Batch: 4"
        )
        print(
            "=" * 60
        )

        # Monitor huanza kabla ya model.train().
        monitor_thread = threading.Thread(
            target=epoch_monitor,
            args=(
                run_id,
                results_csv,
                TRAIN_EPOCHS,
                monitor_stop,
            ),
            daemon=True,
        )
        monitor_thread.start()

        model.train(
            data=yaml,
            epochs=TRAIN_EPOCHS,
            imgsz=YOLO_IMAGE_SIZE,
            project=out,
            name="neerika_bucket",
            exist_ok=True,
            verbose=True,
            workers=0,
            batch=4,
            cache=False,
            plots=False,
        )

        monitor_stop.set()

        if monitor_thread:
            monitor_thread.join(timeout=5)

        # Hakikisha UI inaonyesha mwisho wa training.
        set_training(
            "saving",
            "Training complete. Saving best.pt...",
            93,
            run_id,
        )

        best = os.path.join(
            out,
            "neerika_bucket",
            "weights",
            "best.pt",
        )

        if not os.path.exists(best):
            raise RuntimeError(
                "Training imekwisha lakini best.pt haikupatikana."
            )

        os.makedirs(
            os.path.dirname(MODEL_PATH),
            exist_ok=True,
        )

        shutil.copy2(
            best,
            MODEL_PATH,
        )

        os.makedirs(
            os.path.dirname(MODEL_BACKUP_PATH),
            exist_ok=True,
        )

        shutil.copy2(
            best,
            MODEL_BACKUP_PATH,
        )

        print(
            "NEERIKA TRAINING: best.pt copied."
        )

        set_training(
            "saving",
            "best.pt saved locally. Saving best.pt to Supabase...",
            96,
            run_id,
        )

        save_model_db(MODEL_PATH)

        print(
            "NEERIKA TRAINING: best.pt saved to Supabase."
        )

        with MODEL_LOCK:
            MODEL = YOLO(MODEL_PATH)
            MODEL_ERROR = ""

        set_training(
            "completed",
            (
                f"Training completed successfully. "
                f"{usable} images, {TRAIN_EPOCHS} epochs. "
                f"best.pt saved."
            ),
            100,
            run_id,
            finished=True,
        )

        print(
            "=" * 60
        )
        print(
            "NEERIKA TRAINING COMPLETE"
        )
        print(
            "Run:", run_id
        )
        print(
            "=" * 60
        )

    except Exception as e:
        traceback.print_exc()

        monitor_stop.set()

        try:
            set_training(
                "error",
                "Training error: " + str(e),
                0,
                run_id,
                finished=True,
            )
        except Exception:
            traceback.print_exc()

    finally:
        monitor_stop.set()

        if monitor_thread:
            try:
                monitor_thread.join(timeout=2)
            except Exception:
                pass

        if root:
            shutil.rmtree(
                root,
                ignore_errors=True,
            )

        TRAINING = False
        TRAINING_RUN_ID = None

        release_training_lock(
            lock_conn
        )


def start_training_request():
    run_id = claim_training_run()

    if not run_id:
        return None

    lock_conn = TRAINING_LOCK_CONN

    try:
        threading.Thread(
            target=training_worker,
            args=(run_id, lock_conn),
            daemon=True,
        ).start()

        return run_id

    except Exception:
        try:
            set_training(
                "error",
                "Training worker could not be started.",
                0,
                run_id,
                finished=True,
            )
        finally:
            global TRAINING
            global TRAINING_RUN_ID

            TRAINING = False
            TRAINING_RUN_ID = None

            release_training_lock(
                lock_conn
            )

        raise


def dashboard():
    body = """
    <div class="card">
        <h2>Mining Production Dashboard</h2>
        <p>
            NEERIKA BUCKET AI counts loaded ore/material
            buckets coming from the mine shaft.
        </p>
    </div>

    <div class="grid">
        <div class="stat">
            <div class="small">Today's Loaded Buckets</div>
            <div class="num">__COUNT__</div>
        </div>

        <div class="stat">
            <div class="small">AI System</div>
            <div class="num">READY</div>
        </div>

        <div class="stat">
            <div class="small">Counting Target</div>
            <div class="num">LOADED</div>
        </div>
    </div>

    <div class="card">
        <h3>Not counted</h3>
        <ul>
            <li>Empty buckets</li>
            <li>People</li>
            <li>Equipment</li>
        </ul>
    </div>

    <div class="card">
        <a href="/camera">
            <button>Open Camera</button>
        </a>

        <a href="/training">
            <button class="secondary">Training</button>
        </a>
    </div>
    """.replace(
        "__COUNT__",
        esc(today()),
    )

    return layout(
        "Dashboard",
        body,
        "Dashboard",
    )


def camera():
    body = r"""
    <div class="card">
        <h2>Bucket Camera</h2>

        <p class="small">
            Take a camera frame and send it to YOLO.
            Only BUCKET_LOADED is eligible for counting.
        </p>

        <video
            id="v"
            autoplay
            playsinline
            style="width:100%;border-radius:12px;background:#000">
        </video>

        <canvas id="c" style="display:none"></canvas>

        <p>
            <button onclick="start()">Start Camera</button>
            <button class="secondary" onclick="stop()">Stop</button>
            <button class="success" onclick="detectNow()">
                Detect & Count
            </button>
        </p>

        <div id="r" class="status">
            Camera not started.
        </div>

        <div class="num" id="n">0</div>
    </div>

    <script>
    let stream=null;

    async function start(){
        try{
            stream=await navigator.mediaDevices.getUserMedia({
                video:{facingMode:'environment'},
                audio:false
            });

            v.srcObject=stream;
            r.innerText='Camera started.';
        }catch(e){
            r.innerText='Camera error: '+e.message;
        }
    }

    function stop(){
        if(stream){
            stream.getTracks().forEach(t=>t.stop());
        }

        stream=null;
        r.innerText='Camera stopped.';
    }

    async function detectNow(){
        if(!v.videoWidth){
            alert('Start camera first.');
            return;
        }

        c.width=v.videoWidth;
        c.height=v.videoHeight;

        c.getContext('2d').drawImage(
            v,
            0,
            0,
            c.width,
            c.height
        );

        r.innerText='Detecting...';

        try{
            let q=await fetch('/api/detect',{
                method:'POST',
                headers:{
                    'Content-Type':'application/json'
                },
                body:JSON.stringify({
                    image:c.toDataURL('image/jpeg',.85),
                    count:true
                })
            });

            let d=await q.json();

            if(!q.ok){
                throw Error(d.error||'Detection failed');
            }

            let loaded=d.detections.filter(
                x=>x.class_name==='BUCKET_LOADED'
            ).length;

            r.innerText=
                'Detected: '+d.detections.length+
                ' | Loaded: '+loaded+
                ' | '+
                (d.count_result
                    ? d.count_result.reason
                    : '');

            refresh();

        }catch(e){
            r.innerText='Error: '+e.message;
        }
    }

    async function refresh(){
        try{
            let q=await fetch(
                '/api/history',
                {cache:'no-store'}
            );

            let d=await q.json();
            n.innerText=d.today_count||0;
        }catch(e){}
    }

    refresh();
    </script>
    """

    return layout(
        "Camera",
        body,
        "Camera",
    )


def buckets():
    body = r"""
    <div class="card">
        <h2>Bucket Registration</h2>

        <div class="row">
            <label>Bucket Name</label>
            <input
                id="name"
                placeholder="NEERIKA Loaded Bucket">
        </div>

        <div class="row">
            <label>Description</label>
            <input
                id="desc"
                placeholder="Bucket description">
        </div>

        <button onclick="add()">
            Register Bucket
        </button>

        <div id="msg" class="status"></div>
    </div>

    <div class="card">
        <h3>Registered Buckets</h3>
        <div id="list">Loading...</div>
    </div>

    <script>
    async function load(){
        let q=await fetch('/api/buckets');
        let d=await q.json();

        if(!d.buckets.length){
            list.innerHTML='<p>No buckets registered.</p>';
            return;
        }

        let h=
            '<table>'+
            '<tr>'+
            '<th>Name</th>'+
            '<th>Description</th>'+
            '<th>Status</th>'+
            '<th>Action</th>'+
            '</tr>';

        d.buckets.forEach(b=>{
            h+='<tr>'+
                '<td>'+e(b.name)+'</td>'+
                '<td>'+e(b.description||'')+'</td>'+
                '<td>'+(b.active?'ACTIVE':'')+'</td>'+
                '<td>'+
                (b.active
                    ? '<button disabled>Active</button>'
                    : '<button onclick="activate('+b.id+')">Activate</button>'
                )+
                '</td>'+
                '</tr>';
        });

        list.innerHTML=h+'</table>';
    }

    async function add(){
        let nameValue=
            document.getElementById('name').value.trim();

        let descValue=
            document.getElementById('desc').value.trim();

        if(!nameValue){
            alert('Enter bucket name');
            return;
        }

        let q=await fetch('/api/buckets',{
            method:'POST',
            headers:{
                'Content-Type':'application/json'
            },
            body:JSON.stringify({
                name:nameValue,
                description:descValue
            })
        });

        let d=await q.json();

        msg.innerText=
            d.message||d.error||'';

        if(q.ok){
            document.getElementById('name').value='';
            document.getElementById('desc').value='';
            load();
        }
    }

    async function activate(id){
        let q=await fetch('/api/buckets/active',{
            method:'POST',
            headers:{
                'Content-Type':'application/json'
            },
            body:JSON.stringify({id})
        });

        let d=await q.json();

        if(!q.ok){
            alert(d.error);
            return;
        }

        load();
    }

    function e(s){
        return String(s)
            .replaceAll('&','&amp;')
            .replaceAll('<','&lt;')
            .replaceAll('>','&gt;')
            .replaceAll('"','&quot;')
            .replaceAll("'","&#039;");
    }

    load();
    </script>
    """

    return layout(
        "Buckets",
        body,
        "Buckets",
    )


def training():
    c = db()
    x = c.cursor()

    try:
        x.execute(
            """
            SELECT
                status,
                message,
                progress,
                updated_at,
                run_id
            FROM training_state
            WHERE id=1
            """
        )

        s = x.fetchone() or {
            "status": "idle",
            "message": "Ready",
            "progress": 0,
            "updated_at": None,
            "run_id": None,
        }

    finally:
        x.close()
        c.close()

    rows = dataset_rows()

    tr = ""

    for r in rows:
        tr += (
            "<tr>"
            "<td>" + str(r["id"]) + "</td>"
            "<td>" + esc(r["filename"]) + "</td>"
            "<td>" + str(r["annotation_count"]) + "</td>"
            "<td>"
            '<button class="danger" '
            'onclick="delimg(' + str(r["id"]) + ')">'
            "Delete"
            "</button>"
            "</td>"
            "</tr>"
        )

    if not tr:
        tr = (
            '<tr><td colspan="4">'
            "No dataset images."
            "</td></tr>"
        )

    body = r"""
    <div class="card">
        <h2>YOLO Training Dataset</h2>

        <p class="small">
            Upload images. Save BUCKET_LOADED annotations
            before training.
        </p>

        <div class="row">
            <label>Image</label>
            <input
                id="file"
                type="file"
                accept="image/*">
        </div>

        <button onclick="upload()">
            Upload Image
        </button>

        <div id="up" class="status"></div>
    </div>

    <div class="card">
        <h3>Training Status</h3>

        <p>
            Status:
            <b id="st">__STATUS__</b>
        </p>

        <p id="ms">__MESSAGE__</p>

        <p class="small">
            Last update:
            <span id="upd">__UPDATED__</span>
        </p>

        <div class="epoch" id="epoch">
            Epoch 0/20
        </div>

        <div class="progress">
            <div
                id="bar"
                class="bar"
                style="width:__PROGRESS__%">
                __PROGRESS__%
            </div>
        </div>

        <br>

        <button
            id="startBtn"
            class="success"
            onclick="startTrain()">
            Start Training
        </button>

        <button
            class="secondary"
            onclick="poll()">
            Refresh Status
        </button>
    </div>

    <div class="card">
        <h3>Dataset</h3>

        <table>
            <tr>
                <th>ID</th>
                <th>Filename</th>
                <th>Annotations</th>
                <th>Action</th>
            </tr>

            __ROWS__
        </table>
    </div>

    <script>
    let pollTimer=null;

    const activeStates=[
        'preparing',
        'training',
        'saving'
    ];

    function setStatus(d){

        st.innerText=d.status||'idle';

        ms.innerText=d.message||'';

        let p=Number(d.progress||0);

        bar.style.width=p+'%';
        bar.innerText=p+'%';

        upd.innerText=d.updated_at||'-';

        const message=
            String(d.message||'');

        const match=
            message.match(/Epoch\s+(\d+)\s*\/\s*(\d+)/i);

        if(match){
            epoch.innerText=
                'Epoch '+match[1]+'/'+match[2];
        }else if(
            (d.status||'').toLowerCase()
            ==='completed'
        ){
            epoch.innerText='Epoch 20/20';
        }

        const active=
            activeStates.includes(
                (d.status||'').toLowerCase()
            );

        startBtn.disabled=active;

        startBtn.innerText=
            active
            ? 'Training Running...'
            : 'Start Training';

        if(active){
            if(pollTimer)
                clearTimeout(pollTimer);

            pollTimer=
                setTimeout(poll,2000);
        }
    }

    async function upload(){

        let f=file.files[0];

        if(!f){
            alert('Choose image');
            return;
        }

        let rd=new FileReader();

        rd.onload=async()=>{

            try{

                let q=await fetch(
                    '/api/dataset/upload',
                    {
                        method:'POST',
                        headers:{
                            'Content-Type':
                            'application/json'
                        },
                        body:JSON.stringify({
                            filename:f.name,
                            mime_type:f.type,
                            data:rd.result
                        })
                    }
                );

                let d=await q.json();

                up.innerText=
                    d.message||d.error||'';

                if(q.ok){
                    setTimeout(
                        ()=>location.reload(),
                        500
                    );
                }

            }catch(e){
                up.innerText=
                    'Upload error: '+e.message;
            }
        };

        rd.readAsDataURL(f);
    }

    async function delimg(id){

        if(!confirm('Delete image?'))
            return;

        let q=await fetch(
            '/api/dataset/delete',
            {
                method:'POST',
                headers:{
                    'Content-Type':
                    'application/json'
                },
                body:JSON.stringify({id})
            }
        );

        let d=await q.json();

        if(!q.ok){
            alert(d.error);
            return;
        }

        location.reload();
    }

    async function startTrain(){

        startBtn.disabled=true;

        try{

            let q=await fetch(
                '/api/training/start',
                {
                    method:'POST',
                    headers:{
                        'Content-Type':
                        'application/json'
                    }
                }
            );

            let d=await q.json();

            if(!q.ok){
                alert(
                    d.error||
                    'Training could not be started.'
                );
            }

            await poll();

        }catch(e){

            alert(
                'Connection error: '+
                e.message
            );

            await poll();
        }
    }

    async function poll(){

        try{

            let q=await fetch(
                '/api/training/status',
                {cache:'no-store'}
            );

            let d=await q.json();

            setStatus(d);

        }catch(e){

            ms.innerText=
                'Status connection error. Retrying...';

            if(pollTimer)
                clearTimeout(pollTimer);

            pollTimer=
                setTimeout(
                    poll,
                    3000
                );
        }
    }

    poll();
    </script>
    """

    body = (
        body
        .replace("__STATUS__", esc(s["status"]))
        .replace("__MESSAGE__", esc(s["message"]))
        .replace(
            "__PROGRESS__",
            esc(s["progress"]),
        )
        .replace(
            "__UPDATED__",
            esc(s["updated_at"] or "-"),
        )
        .replace("__ROWS__", tr)
    )

    return layout(
        "Training",
        body,
        "Training",
    )


def history():
    body = r"""
    <div class="card">
        <h2>Production History</h2>

        <button onclick="load()">
            Refresh
        </button>

        <a href="/api/history.csv">
            <button class="secondary">
                Export CSV
            </button>
        </a>
    </div>

    <div class="card" id="table">
        Loading...
    </div>

    <script>
    async function load(){

        let q=await fetch(
            '/api/history',
            {cache:'no-store'}
        );

        let d=await q.json();

        let h=
            '<table>'+
            '<tr>'+
            '<th>Date</th>'+
            '<th>Loaded Buckets</th>'+
            '</tr>';

        d.history.forEach(r=>{
            h+=
                '<tr>'+
                '<td>'+r.count_date+'</td>'+
                '<td>'+r.bucket_count+'</td>'+
                '</tr>';
        });

        h+='</table>';

        table.innerHTML=
            d.history.length
            ? h
            : '<p>No history yet.</p>';
    }

    load();
    </script>
    """

    return layout(
        "History",
        body,
        "History",
    )


def settings():
    body = """
    <div class="card">
        <h2>Settings</h2>

        <p>
            YOLO confidence:
            <b>__CONF__</b>
        </p>

        <p>
            Image size:
            <b>__SIZE__</b>
        </p>

        <p>
            Training epochs:
            <b>__EPOCHS__</b>
        </p>

        <p>
            Count cooldown:
            <b>__COOLDOWN__ seconds</b>
        </p>
    </div>

    <div class="card">
        <h3>Model</h3>

        <p>__MODEL__</p>

        <p class="small">
            DATABASE_URL remains in Render Environment Variables
            and is not written into this source code.
        </p>
    </div>
    """.replace(
        "__CONF__",
        esc(YOLO_CONFIDENCE),
    ).replace(
        "__SIZE__",
        esc(YOLO_IMAGE_SIZE),
    ).replace(
        "__EPOCHS__",
        esc(TRAIN_EPOCHS),
    ).replace(
        "__COOLDOWN__",
        esc(COUNT_COOLDOWN_SECONDS),
    ).replace(
        "__MODEL__",
        esc(
            "Loaded"
            if MODEL is not None
            else "Not loaded yet"
        ),
    )

    return layout(
        "Settings",
        body,
        "Settings",
    )


class Handler(BaseHTTPRequestHandler):

    def log_message(self, fmt, *args):
        print(
            f"{self.address_string()} - "
            f"{fmt % args}"
        )

    def body_json(self):
        n = int(
            self.headers.get(
                "Content-Length",
                "0",
            )
        )

        if n > MAX_JSON_BYTES:
            raise ValueError(
                "Request too large."
            )

        raw = self.rfile.read(n)

        return (
            json.loads(
                raw.decode()
            )
            if raw
            else {}
        )

    def do_HEAD(self):
        self.send_response(200)
        self.send_header(
            "Content-Type",
            "text/html; charset=utf-8",
        )
        self.send_header(
            "Cache-Control",
            "no-store",
        )
        self.end_headers()

    def do_GET(self):

        try:
            p = urlparse(self.path).path

            pages = {
                "/": dashboard,
                "/camera": camera,
                "/buckets": buckets,
                "/training": training,
                "/history": history,
                "/settings": settings,
            }

            if p in pages:
                return html_out(
                    self,
                    pages[p](),
                )

            if p == "/health":
                return json_out(
                    self,
                    {
                        "status": "ok",
                        "database_configured":
                            bool(DATABASE_URL),
                        "model_loaded":
                            MODEL is not None,
                        "training":
                            TRAINING,
                    },
                )

            if p == "/api/buckets":
                c = db()
                x = c.cursor()

                try:
                    x.execute(
                        """
                        SELECT
                            id,
                            name,
                            description,
                            active,
                            created_at
                        FROM buckets
                        ORDER BY id DESC
                        """
                    )

                    r = x.fetchall()

                finally:
                    x.close()
                    c.close()

                return json_out(
                    self,
                    {"buckets": r},
                )

            if p == "/api/training/status":
                c = db()
                x = c.cursor()

                try:
                    x.execute(
                        """
                        SELECT
                            status,
                            message,
                            progress,
                            updated_at,
                            run_id,
                            server_id,
                            started_at,
                            finished_at
                        FROM training_state
                        WHERE id=1
                        """
                    )

                    r = x.fetchone()

                finally:
                    x.close()
                    c.close()

                return json_out(
                    self,
                    r or {
                        "status": "idle",
                        "message": "",
                        "progress": 0,
                    },
                )

            if p == "/api/history":
                c = db()
                x = c.cursor()

                try:
                    x.execute(
                        """
                        SELECT
                            count_date,
                            bucket_count,
                            updated_at
                        FROM daily_counts
                        ORDER BY count_date DESC
                        LIMIT 100
                        """
                    )

                    r = x.fetchall()

                finally:
                    x.close()
                    c.close()

                return json_out(
                    self,
                    {
                        "history": r,
                        "today_count": today(),
                    },
                )

            if p == "/api/history.csv":
                c = db()
                x = c.cursor()

                try:
                    x.execute(
                        """
                        SELECT
                            count_date,
                            bucket_count,
                            updated_at
                        FROM daily_counts
                        ORDER BY count_date DESC
                        """
                    )

                    r = x.fetchall()

                finally:
                    x.close()
                    c.close()

                o = io.StringIO()
                w = csv.writer(o)

                w.writerow(
                    [
                        "Date",
                        "Loaded Buckets",
                        "Updated At",
                    ]
                )

                for z in r:
                    w.writerow(
                        [
                            z["count_date"],
                            z["bucket_count"],
                            z["updated_at"],
                        ]
                    )

                b = o.getvalue().encode()

                self.send_response(200)
                self.send_header(
                    "Content-Type",
                    "text/csv; charset=utf-8",
                )
                self.send_header(
                    "Content-Length",
                    str(len(b)),
                )
                self.send_header(
                    "Content-Disposition",
                    "attachment; filename=neerika_history.csv",
                )
                self.end_headers()
                self.wfile.write(b)
                return

            if p.startswith("/api/dataset/image/"):

                i = int(
                    p.rsplit("/", 1)[1]
                )

                c = db()
                x = c.cursor()

                try:
                    x.execute(
                        """
                        SELECT
                            image_data,
                            mime_type
                        FROM dataset_images
                        WHERE id=%s
                        """,
                        (i,),
                    )

                    r = x.fetchone()

                finally:
                    x.close()
                    c.close()

                if not r:
                    return error_out(
                        self,
                        "Image not found.",
                        404,
                    )

                b = _image_bytes(
                    r["image_data"]
                )

                self.send_response(200)
                self.send_header(
                    "Content-Type",
                    r["mime_type"]
                    or "image/jpeg",
                )
                self.send_header(
                    "Content-Length",
                    str(len(b)),
                )
                self.end_headers()
                self.wfile.write(b)
                return

            return error_out(
                self,
                "Not found.",
                404,
            )

        except Exception as e:
            traceback.print_exc()
            return error_out(
                self,
                e,
                500,
            )

    def do_POST(self):

        try:
            p = urlparse(self.path).path

            if p == "/api/detect":

                d = self.body_json()

                s = d.get("image", "")

                if "," in s:
                    s = s.split(",", 1)[1]

                b = base64.b64decode(s)

                det = detect(b)

                cr = (
                    count_loaded(det)
                    if d.get("count", True)
                    else None
                )

                return json_out(
                    self,
                    {
                        "detections": det,
                        "count_result": cr,
                    },
                )

            if p == "/api/buckets":

                d = self.body_json()

                name = str(
                    d.get("name", "")
                ).strip()

                desc = str(
                    d.get("description", "")
                ).strip()

                if not name:
                    raise ValueError(
                        "Bucket name is required."
                    )

                c = db()
                x = c.cursor()

                try:
                    x.execute(
                        """
                        INSERT INTO buckets
                            (name,description)
                        VALUES
                            (%s,%s)
                        RETURNING id
                        """,
                        (name, desc),
                    )

                    i = x.fetchone()["id"]

                    c.commit()

                finally:
                    x.close()
                    c.close()

                return json_out(
                    self,
                    {
                        "message":
                            "Bucket registered successfully.",
                        "id": i,
                    },
                )

            if p == "/api/buckets/active":

                d = self.body_json()
                i = int(d["id"])

                c = db()
                x = c.cursor()

                try:
                    x.execute(
                        "UPDATE buckets SET active=FALSE"
                    )

                    x.execute(
                        """
                        UPDATE buckets
                        SET active=TRUE
                        WHERE id=%s
                        """,
                        (i,),
                    )

                    ok = x.rowcount > 0
                    c.commit()

                finally:
                    x.close()
                    c.close()

                return json_out(
                    self,
                    (
                        {"message": "Bucket activated."}
                        if ok
                        else {"error": "Bucket not found."}
                    ),
                    200 if ok else 404,
                )

            if p == "/api/dataset/upload":

                d = self.body_json()

                s = d.get("data", "")

                if "," in s:
                    s = s.split(",", 1)[1]

                b = base64.b64decode(s)

                if len(b) > MAX_UPLOAD_BYTES:
                    raise ValueError(
                        "Image too large."
                    )

                i = save_image(
                    str(
                        d.get(
                            "filename",
                            "image.jpg",
                        )
                    ),
                    str(
                        d.get(
                            "mime_type",
                            "image/jpeg",
                        )
                    ),
                    b,
                )

                return json_out(
                    self,
                    {
                        "message":
                            "Image uploaded successfully.",
                        "id": i,
                    },
                )

            if p == "/api/dataset/delete":

                i = int(
                    self.body_json()["id"]
                )

                ok = delete_image(i)

                return json_out(
                    self,
                    (
                        {
                            "message":
                                "Image deleted successfully."
                        }
                        if ok
                        else {
                            "error":
                                "Image not found."
                        }
                    ),
                    200 if ok else 404,
                )

            if p == "/api/dataset/annotations":

                d = self.body_json()

                save_annotations(
                    int(d["image_id"]),
                    d.get("annotations", []),
                )

                return json_out(
                    self,
                    {
                        "message":
                            "Annotations saved successfully."
                    },
                )

            if p == "/api/training/start":

                run_id = start_training_request()

                if not run_id:
                    return error_out(
                        self,
                        (
                            "Training is already running. "
                            "Wait for the current run to finish."
                        ),
                        409,
                    )

                return json_out(
                    self,
                    {
                        "message":
                            "Training started.",
                        "run_id":
                            run_id,
                    },
                    202,
                )

            return error_out(
                self,
                "Not found.",
                404,
            )

        except json.JSONDecodeError:
            return error_out(
                self,
                "Invalid JSON.",
                400,
            )

        except ValueError as e:
            return error_out(
                self,
                e,
                400,
            )

        except Exception as e:
            traceback.print_exc()
            return error_out(
                self,
                e,
                500,
            )


def main():

    print("=" * 60)
    print("NEERIKA BUCKET AI")
    print("Starting server...")
    print("=" * 60)

    if DATABASE_URL:
        try:
            init_db()
            print("Database initialized.")

            mark_interrupted_on_startup()

        except Exception:
            traceback.print_exc()

    else:
        print(
            "WARNING: DATABASE_URL is not configured."
        )

    try:
        load_model()
    except Exception:
        traceback.print_exc()

    server = ThreadingHTTPServer(
        (HOST, PORT),
        Handler,
    )

    print(
        "Running on port",
        PORT,
    )

    server.serve_forever()


if __name__ == "__main__":
    main()
