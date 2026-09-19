import os
import io
import csv
import json
import base64
import threading
import traceback
from datetime import datetime, date
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

import psycopg2
from psycopg2.extras import RealDictCursor


# =========================================================
# SERVER
# =========================================================

HOST = "0.0.0.0"
PORT = int(os.environ.get("PORT", "8080"))

DATABASE_URL = (
    os.environ.get("DATABASE_URL")
    or os.environ.get("SUPABASE_DB_URL")
    or os.environ.get("POSTGRES_URL")
)

APP_NAME = "BUCKET COUNTER AI"

CLASSES = [
    "BUCKET_LOADED",
    "BUCKET_EMPTY",
    "PEOPLE",
    "EQUIPMENT",
]


# =========================================================
# DATABASE
# =========================================================

def db_connect():
    if not DATABASE_URL:
        raise RuntimeError(
            "DATABASE_URL is not configured in Render Environment Variables."
        )

    return psycopg2.connect(
        DATABASE_URL,
        connect_timeout=15
    )


def init_db():
    conn = db_connect()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS buckets (
            id BIGSERIAL PRIMARY KEY,
            name TEXT NOT NULL,
            capacity DOUBLE PRECISION DEFAULT 0,
            active BOOLEAN DEFAULT TRUE,
            created_at TIMESTAMPTZ DEFAULT NOW()
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS dataset_images (
            id BIGSERIAL PRIMARY KEY,
            filename TEXT,
            image_data BYTEA NOT NULL,
            labeled BOOLEAN DEFAULT FALSE,
            created_at TIMESTAMPTZ DEFAULT NOW()
        )
    """)

    # Existing table may have been created before "labeled"
    # was added. This keeps old data safe.
    cur.execute("""
        ALTER TABLE dataset_images
        ADD COLUMN IF NOT EXISTS labeled BOOLEAN DEFAULT FALSE
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS annotations (
            id BIGSERIAL PRIMARY KEY,
            image_id BIGINT REFERENCES dataset_images(id)
                ON DELETE CASCADE,
            class_name TEXT NOT NULL,
            x_center DOUBLE PRECISION NOT NULL,
            y_center DOUBLE PRECISION NOT NULL,
            width DOUBLE PRECISION NOT NULL,
            height DOUBLE PRECISION NOT NULL,
            created_at TIMESTAMPTZ DEFAULT NOW()
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS training_state (
            id INTEGER PRIMARY KEY,
            status TEXT DEFAULT 'NOT_STARTED',
            progress INTEGER DEFAULT 0,
            message TEXT DEFAULT '',
            updated_at TIMESTAMPTZ DEFAULT NOW()
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS trained_model (
            id INTEGER PRIMARY KEY,
            model_name TEXT,
            model_data BYTEA,
            created_at TIMESTAMPTZ DEFAULT NOW()
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS daily_counts (
            id BIGSERIAL PRIMARY KEY,
            count_date DATE NOT NULL,
            loaded_count INTEGER DEFAULT 0,
            empty_count INTEGER DEFAULT 0,
            notes TEXT,
            created_at TIMESTAMPTZ DEFAULT NOW()
        )
    """)

    cur.execute("""
        INSERT INTO training_state
        (id, status, progress, message)
        VALUES (
            1,
            'WAITING',
            0,
            'Ready for external YOLO training'
        )
        ON CONFLICT (id) DO NOTHING
    """)

    # Mark images with existing annotations as labeled.
    cur.execute("""
        UPDATE dataset_images
        SET labeled = TRUE
        WHERE id IN (
            SELECT DISTINCT image_id
            FROM annotations
        )
        AND labeled IS NOT TRUE
    """)

    conn.commit()
    cur.close()
    conn.close()


# =========================================================
# DATABASE HELPERS
# =========================================================

def db_query(sql, params=()):
    conn = db_connect()

    try:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute(sql, params)
        rows = cur.fetchall()
        return [dict(row) for row in rows]

    finally:
        conn.close()


def db_one(sql, params=()):
    rows = db_query(sql, params)
    return rows[0] if rows else None


def db_execute(sql, params=()):
    conn = db_connect()

    try:
        cur = conn.cursor()
        cur.execute(sql, params)
        conn.commit()

    finally:
        conn.close()


# =========================================================
# TRAINING STATE
# =========================================================

def set_training_state(status, progress, message):
    db_execute("""
        UPDATE training_state
        SET status = %s,
            progress = %s,
            message = %s,
            updated_at = NOW()
        WHERE id = 1
    """, (
        status,
        int(progress),
        message
    ))


def get_training_state():
    row = db_one("""
        SELECT status, progress, message, updated_at
        FROM training_state
        WHERE id = 1
    """)

    if not row:
        return {
            "status": "WAITING",
            "progress": 0,
            "message": "Ready for external YOLO training"
        }

    return row


# =========================================================
# DATASET
# =========================================================

def dataset_summary():
    total = db_one("""
        SELECT COUNT(*) AS count
        FROM dataset_images
    """)

    labeled = db_one("""
        SELECT COUNT(*) AS count
        FROM dataset_images
        WHERE labeled = TRUE
    """)

    annotations = db_one("""
        SELECT COUNT(*) AS count
        FROM annotations
    """)

    return {
        "total_images": int(total["count"]),
        "labeled_images": int(labeled["count"]),
        "annotations": int(annotations["count"])
    }


# =========================================================
# MODEL STATUS
# =========================================================

def model_ready():
    row = db_one("""
        SELECT id
        FROM trained_model
        WHERE id = 1
    """)

    return row is not None


# =========================================================
# JSON
# =========================================================

def send_json(handler, data, status=200):

    body = json.dumps(
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
            str(len(body))
        )

        handler.send_header(
            "Cache-Control",
            "no-store"
        )

        handler.end_headers()

        handler.wfile.write(body)

    except BrokenPipeError:
        pass

    except ConnectionResetError:
        pass


def read_json(handler):

    length = int(
        handler.headers.get(
            "Content-Length",
            "0"
        )
    )

    if length <= 0:
        return {}

    raw = handler.rfile.read(length)

    if not raw:
        return {}

    return json.loads(
        raw.decode("utf-8")
    )


# =========================================================
# IMAGE DATA
# =========================================================

def decode_image_data(data_url):

    if not data_url:
        raise ValueError("Image data is missing.")

    if "," in data_url:
        data_url = data_url.split(",", 1)[1]

    try:
        return base64.b64decode(data_url)
    except Exception:
        raise ValueError("Invalid image data.")


# =========================================================
# HTML LAYOUT
# =========================================================

def layout(title, content):

    return f"""
<!DOCTYPE html>
<html>

<head>

<meta charset="UTF-8">

<meta
name="viewport"
content="width=device-width, initial-scale=1.0">

<title>{title} - {APP_NAME}</title>

<style>

* {{
    box-sizing: border-box;
}}

body {{
    margin: 0;
    font-family: Arial, sans-serif;
    background: #f4f6f8;
    color: #222;
}}

header {{
    background: #111827;
    color: white;
    padding: 18px;
}}

header h1 {{
    margin: 0;
    font-size: 22px;
}}

header p {{
    margin: 5px 0 0;
    color: #d1d5db;
}}

nav {{
    background: white;
    padding: 12px;
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    border-bottom: 1px solid #ddd;
}}

nav a {{
    text-decoration: none;
    color: #111827;
    padding: 9px 12px;
    border-radius: 7px;
    background: #eef2f7;
}}

main {{
    max-width: 1100px;
    margin: auto;
    padding: 20px;
}}

.card {{
    background: white;
    padding: 20px;
    margin-bottom: 18px;
    border-radius: 12px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
}}

.grid {{
    display: grid;
    grid-template-columns:
        repeat(auto-fit, minmax(200px, 1fr));
    gap: 15px;
}}

.stat {{
    background: white;
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
}}

.stat h3 {{
    margin-top: 0;
}}

.big {{
    font-size: 32px;
    font-weight: bold;
}}

input,
select,
textarea,
button {{
    width: 100%;
    padding: 12px;
    margin: 7px 0;
    border-radius: 7px;
    border: 1px solid #ccc;
    font-size: 15px;
}}

button {{
    background: #111827;
    color: white;
    border: none;
    cursor: pointer;
}}

button:hover {{
    opacity: 0.9;
}}

button.secondary {{
    background: #4b5563;
}}

button.danger {{
    background: #b91c1c;
}}

table {{
    width: 100%;
    border-collapse: collapse;
}}

th,
td {{
    padding: 10px;
    border-bottom: 1px solid #ddd;
    text-align: left;
}}

.badge {{
    display: inline-block;
    padding: 6px 10px;
    border-radius: 20px;
    background: #e5e7eb;
}}

.progress {{
    width: 100%;
    background: #e5e7eb;
    border-radius: 20px;
    overflow: hidden;
    height: 24px;
}}

.progress-bar {{
    height: 24px;
    background: #111827;
    color: white;
    text-align: center;
    line-height: 24px;
}}

footer {{
    text-align: center;
    padding: 25px;
    color: #666;
}}

.training-canvas {{
    width: 100%;
    max-width: 900px;
    border: 2px solid #111827;
    border-radius: 10px;
    display: block;
    background: #111;
    touch-action: none;
}}

.info {{
    padding: 12px;
    background: #eef2ff;
    border-radius: 8px;
    margin: 10px 0;
}}

.success {{
    padding: 12px;
    background: #dcfce7;
    border-radius: 8px;
    margin: 10px 0;
}}

.warning {{
    padding: 12px;
    background: #fef3c7;
    border-radius: 8px;
    margin: 10px 0;
}}

.small {{
    font-size: 13px;
    color: #666;
}}

.box-list {{
    margin-top: 15px;
}}

.box-item {{
    background: #f3f4f6;
    padding: 10px;
    border-radius: 7px;
    margin: 6px 0;
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 10px;
}}

.box-item button {{
    width: auto;
    margin: 0;
    padding: 7px 10px;
}}

.camera-preview {{
    width: 100%;
    max-width: 900px;
    background: #000;
    border-radius: 10px;
}}

</style>

</head>

<body>

<header>

<h1>🪣 BUCKET COUNTER AI</h1>

<p>Underground production monitoring</p>

</header>

<nav>

<a href="/">Dashboard</a>
<a href="/camera">Camera</a>
<a href="/buckets">Buckets</a>
<a href="/training">AI Training</a>
<a href="/history">History</a>
<a href="/settings">Settings</a>

</nav>

<main>

{content}

</main>

<footer>

Geology & Mining Services

</footer>

</body>

</html>
"""


# =========================================================
# DASHBOARD
# =========================================================

def dashboard_page():

    summary = dataset_summary()
    state = get_training_state()

    content = f"""
<h2>📊 Dashboard</h2>

<div class="grid">

<div class="stat">
<h3>Dataset Images</h3>
<div class="big">{summary["total_images"]}</div>
</div>

<div class="stat">
<h3>Labeled Images</h3>
<div class="big">{summary["labeled_images"]}</div>
</div>

<div class="stat">
<h3>Annotations</h3>
<div class="big">{summary["annotations"]}</div>
</div>

<div class="stat">
<h3>AI Model</h3>
<div class="big">
{"READY" if model_ready() else "NOT READY"}
</div>
</div>

</div>

<div class="card">

<h3>Training Status</h3>

<p>{state["message"]}</p>

<div class="progress">

<div
class="progress-bar"
style="width:{state["progress"]}%">

{state["progress"]}%

</div>

</div>

</div>

<div class="card">

<h3>AI Classes</h3>

<ul>

<li>🪣 BUCKET_LOADED = count</li>
<li>🪣 BUCKET_EMPTY = no count</li>
<li>👤 PEOPLE = no count</li>
<li>⚙️ EQUIPMENT = no count</li>

</ul>

</div>
"""

    return layout(
        "Dashboard",
        content
    )


# =========================================================
# BUCKETS PAGE
# =========================================================

def buckets_page():

    rows_data = db_query("""
        SELECT id, name, capacity, active
        FROM buckets
        ORDER BY id DESC
    """)

    rows = ""

    for row in rows_data:

        rows += f"""
<tr>
<td>{row["id"]}</td>
<td>{row["name"]}</td>
<td>{row["capacity"]}</td>
<td>{"YES" if row["active"] else "NO"}</td>
</tr>
"""

    content = """
<h2>🪣 Buckets</h2>

<div class="card">

<h3>Add Bucket</h3>

<input
id="name"
placeholder="Bucket name">

<input
id="capacity"
type="number"
placeholder="Capacity">

<button onclick="addBucket()">
➕ ADD BUCKET
</button>

</div>

<div class="card">

<table>

<tr>
<th>ID</th>
<th>Name</th>
<th>Capacity</th>
<th>Active</th>
</tr>

{ROWS}

</table>

</div>

<script>

async function addBucket() {{

    const name =
        document.getElementById("name").value;

    const capacity =
        document.getElementById("capacity").value;

    if (!name) {{
        alert("Enter bucket name");
        return;
    }}

    const r = await fetch(
        "/api/buckets",
        {{
            method: "POST",
            headers: {{
                "Content-Type": "application/json"
            }},
            body: JSON.stringify({{
                name: name,
                capacity: capacity
            }})
        }}
    );

    const d = await r.json();

    if (d.ok) {{
        window.location.reload();
    }} else {{
        alert(
            d.error ||
            "Failed to add bucket"
        );
    }}

}}

</script>
""".replace(
        "{ROWS}",
        rows
    )

    return layout(
        "Buckets",
        content
    )


# =========================================================
# TRAINING PAGE
# =========================================================

def training_page():

    summary = dataset_summary()
    state = get_training_state()

    model_status = (
        "READY"
        if model_ready()
        else "NOT READY"
    )

    content = f"""
<h2>🤖 AI Training</h2>

<div class="card">

<p>
Database:
<b>Supabase PostgreSQL</b>
|
YOLO:
<b>AVAILABLE FOR MODEL USE</b>
|
Model:
<b>{model_status}</b>
</p>

</div>

<div class="card">

<h3>Classes</h3>

<ul>

<li>BUCKET_LOADED = count</li>
<li>BUCKET_EMPTY = no count</li>
<li>PEOPLE = no count</li>
<li>EQUIPMENT = no count</li>

</ul>

</div>


<!-- =====================================================
     ADD TRAINING IMAGE
====================================================== -->

<div class="card">

<h3>➕ Add Training Image</h3>

<div class="info">

<p>
Piga picha au chagua picha kutoka kwenye simu/computer.
Kisha chora box kuzunguka object unayotaka AI ijifunze.
</p>

<p class="small">
Unaweza kuweka objects nyingi kwenye picha moja.
Mfano: bucket loaded 1 + bucket loaded 2 + person 1.
</p>

</div>

<input
type="file"
id="imageFile"
accept="image/*"
capture="environment"
onchange="loadTrainingImage(event)">

<button
type="button"
onclick="openCamera()">

📷 OPEN CAMERA

</button>

<video
id="trainingVideo"
class="camera-preview"
autoplay
playsinline
style="display:none;">

</video>

<button
id="captureButton"
type="button"
style="display:none;"
onclick="captureCameraPhoto()">

📸 CAPTURE PHOTO

</button>

<canvas
id="trainingCanvas"
class="training-canvas"
style="display:none;">

</canvas>

<div
id="trainingInfo"
class="info"
style="display:none;">

Chagua class kisha chora box kwenye picha.

</div>

<label>
<b>Class:</b>
</label>

<select id="classSelect">

<option value="BUCKET_LOADED">
BUCKET_LOADED — COUNT
</option>

<option value="BUCKET_EMPTY">
BUCKET_EMPTY — NO COUNT
</option>

<option value="PEOPLE">
PEOPLE — NO COUNT
</option>

<option value="EQUIPMENT">
EQUIPMENT — NO COUNT
</option>

</select>

<button
type="button"
onclick="clearBoxes">

🗑️ CLEAR ALL BOXES
</button>

<div id="boxList" class="box-list"></div>

<button
id="saveTrainingButton"
type="button"
onclick="saveTrainingImage()"
style="display:none;">

💾 SAVE TRAINING IMAGE

</button>

<p
id="saveResult">
</p>

</div>


<!-- =====================================================
     DATASET
====================================================== -->

<div class="card">

<h3>Dataset</h3>

<p>
Total images:
<b id="totalImages">
{summary["total_images"]}
</b>
</p>

<p>
Labeled:
<b id="labeledImages">
{summary["labeled_images"]}
</b>
</p>

<p>
Annotations:
<b id="annotationCount">
{summary["annotations"]}
</b>
</p>

</div>


<!-- =====================================================
     TRAINING STATUS
====================================================== -->

<div class="card">

<h3>Training status</h3>

<p id="statusMessage">
{state["message"]}
</p>

<div class="progress">

<div
id="progressBar"
class="progress-bar"
style="width:{state["progress"]}%">

{state["progress"]}%

</div>

</div>

</div>


<div class="card">

<h3>⚠️ Training</h3>

<p>
YOLO training is not executed inside the Render
Web Service. This prevents the web server from
restarting during CPU-heavy training.
</p>

<p>
Your images and labels are safely stored in
Supabase PostgreSQL.
</p>

<p>
After a trained model is uploaded, the system
will use it for bucket detection.
</p>

</div>


<script>

let trainingImage = null;
let trainingImageName = "";
let canvas = null;
let ctx = null;
let boxes = [];

let drawing = false;
let startX = 0;
let startY = 0;
let currentX = 0;
let currentY = 0;

let cameraStream = null;


// =====================================================
// LOAD IMAGE
// =====================================================

function loadTrainingImage(event) {{

    const file =
        event.target.files[0];

    if (!file) {{
        return;
    }}

    trainingImageName =
        file.name || "training_image.jpg";

    const reader =
        new FileReader();

    reader.onload = function(e) {{

        const img =
            new Image();

        img.onload = function() {{

            trainingImage = img;

            showTrainingCanvas();

        }};

        img.src = e.target.result;

    }};

    reader.readAsDataURL(file);

}}


// =====================================================
// SHOW CANVAS
// =====================================================

function showTrainingCanvas() {{

    canvas =
        document.getElementById(
            "trainingCanvas"
        );

    ctx =
        canvas.getContext("2d");

    let maxWidth =
        Math.min(
            window.innerWidth - 50,
            900
        );

    let scale =
        maxWidth /
        trainingImage.width;

    if (scale > 1) {{
        scale = 1;
    }}

    canvas.width =
        Math.round(
            trainingImage.width * scale
        );

    canvas.height =
        Math.round(
            trainingImage.height * scale
        );

    canvas.style.display =
        "block";

    document.getElementById(
        "trainingInfo"
    ).style.display =
        "block";

    document.getElementById(
        "saveTrainingButton"
    ).style.display =
        "block";

    boxes = [];

    redrawCanvas();

    updateBoxList();

}}


// =====================================================
// DRAW IMAGE + BOXES
// =====================================================

function redrawCanvas() {{

    if (!canvas || !trainingImage) {{
        return;
    }}

    ctx.clearRect(
        0,
        0,
        canvas.width,
        canvas.height
    );

    ctx.drawImage(
        trainingImage,
        0,
        0,
        canvas.width,
        canvas.height
    );

    for (
        let i = 0;
        i < boxes.length;
        i++
    ) {{

        const b = boxes[i];

        const x =
            b.x1 * canvas.width;

        const y =
            b.y1 * canvas.height;

        const w =
            (b.x2 - b.x1) *
            canvas.width;

        const h =
            (b.y2 - b.y1) *
            canvas.height;

        ctx.strokeStyle =
            "#ff0000";

        ctx.lineWidth = 3;

        ctx.strokeRect(
            x,
            y,
            w,
            h
        );

        ctx.fillStyle =
            "rgba(255,0,0,0.25)";

        ctx.fillRect(
            x,
            y,
            w,
            h
        );

        ctx.fillStyle =
            "#ffffff";

        ctx.font =
            "bold 14px Arial";

        ctx.fillText(
            b.class_name,
            x + 5,
            y + 18
        );

    }}

    if (drawing) {{

        const x =
            Math.min(
                startX,
                currentX
            );

        const y =
            Math.min(
                startY,
                currentY
            );

        const w =
            Math.abs(
                currentX - startX
            );

        const h =
            Math.abs(
                currentY - startY
            );

        ctx.strokeStyle =
            "#00ff00";

        ctx.lineWidth = 3;

        ctx.strokeRect(
            x,
            y,
            w,
            h
        );

    }}

}}


// =====================================================
// POINTER POSITION
// =====================================================

function getPointerPosition(event) {{

    const rect =
        canvas.getBoundingClientRect();

    let clientX;
    let clientY;

    if (
        event.touches &&
        event.touches.length
    ) {{

        clientX =
            event.touches[0].clientX;

        clientY =
            event.touches[0].clientY;

    }} else {{

        clientX =
            event.clientX;

        clientY =
            event.clientY;

    }}

    let x =
        (clientX - rect.left) /
        rect.width;

    let y =
        (clientY - rect.top) /
        rect.height;

    x =
        Math.max(
            0,
            Math.min(1, x)
        );

    y =
        Math.max(
            0,
            Math.min(1, y)
        );

    return {{
        x: x,
        y: y
    }};

}}


// =====================================================
// START DRAW
// =====================================================

function startDrawing(event) {{

    if (!canvas) {{
        return;
    }}

    event.preventDefault();

    const p =
        getPointerPosition(event);

    startX =
        p.x * canvas.width;

    startY =
        p.y * canvas.height;

    currentX =
        startX;

    currentY =
        startY;

    drawing = true;

}}


// =====================================================
// MOVE DRAW
// =====================================================

function moveDrawing(event) {{

    if (!drawing) {{
        return;
    }}

    event.preventDefault();

    const p =
        getPointerPosition(event);

    currentX =
        p.x * canvas.width;

    currentY =
        p.y * canvas.height;

    redrawCanvas();

}}


// =====================================================
// FINISH DRAW
// =====================================================

function finishDrawing(event) {{

    if (!drawing) {{
        return;
    }}

    event.preventDefault();

    const p =
        getPointerPosition(event);

    currentX =
        p.x * canvas.width;

    currentY =
        p.y * canvas.height;

    drawing = false;

    let x1 =
        Math.min(
            startX,
            currentX
        ) / canvas.width;

    let y1 =
        Math.min(
            startY,
            currentY
        ) / canvas.height;

    let x2 =
        Math.max(
            startX,
            currentX
        ) / canvas.width;

    let y2 =
        Math.max(
            startY,
            currentY
        ) / canvas.height;

    const minSize = 0.01;

    if (
        (x2 - x1) < minSize ||
        (y2 - y1) < minSize
    ) {{

        redrawCanvas();

        return;
    }}

    boxes.push({{
        class_name:
            document.getElementById(
                "classSelect"
            ).value,

        x1: x1,
        y1: y1,
        x2: x2,
        y2: y2
    }});

    redrawCanvas();

    updateBoxList();

}}


// =====================================================
// CANVAS EVENTS
// =====================================================

function setupCanvasEvents() {{

    canvas.addEventListener(
        "mousedown",
        startDrawing
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
        startDrawing,
        {{ passive: false }}
    );

    canvas.addEventListener(
        "touchmove",
        moveDrawing,
        {{ passive: false }}
    );

    canvas.addEventListener(
        "touchend",
        finishDrawing,
        {{ passive: false }}
    );

}}

document.addEventListener(
    "DOMContentLoaded",
    function() {{

        canvas =
            document.getElementById(
                "trainingCanvas"
            );

        setupCanvasEvents();

    }}
);


// =====================================================
// BOX LIST
// =====================================================

function updateBoxList() {{

    const list =
        document.getElementById(
            "boxList"
        );

    if (!boxes.length) {{

        list.innerHTML =
            "<p class='small'>No boxes yet.</p>";

        return;
    }}

    let html = "";

    for (
        let i = 0;
        i < boxes.length;
        i++
    ) {{

        html +=
            "<div class='box-item'>" +

            "<span>" +
            (i + 1) +
            ". " +
            boxes[i].class_name +
            "</span>" +

            "<button " +
            "type='button' " +
            "onclick='deleteBox(" +
            i +
            ")'>" +

            "Delete" +

            "</button>" +

            "</div>";

    }}

    list.innerHTML = html;

}}


// =====================================================
// DELETE BOX
// =====================================================

function deleteBox(index) {{

    boxes.splice(
        index,
        1
    );

    redrawCanvas();

    updateBoxList();

}}


// =====================================================
// CLEAR BOXES
// =====================================================

function clearBoxes() {{

    boxes = [];

    redrawCanvas();

    updateBoxList();

}}


// =====================================================
// CAMERA
// =====================================================

async function openCamera() {{

    try {{

        const video =
            document.getElementById(
                "trainingVideo"
            );

        cameraStream =
            await navigator.mediaDevices.getUserMedia({{
                video: {{
                    facingMode: {{
                        ideal: "environment"
                    }}
                }},
                audio: false
            }});

        video.srcObject =
            cameraStream;

        video.style.display =
            "block";

        document.getElementById(
            "captureButton"
        ).style.display =
            "block";

    }} catch (e) {{

        alert(
            "Camera failed: " +
            e.message
        );

    }}

}}


// =====================================================
// CAPTURE CAMERA PHOTO
// =====================================================

function captureCameraPhoto() {{

    const video =
        document.getElementById(
            "trainingVideo"
        );

    if (
        !video.videoWidth ||
        !video.videoHeight
    ) {{

        alert(
            "Camera is not ready yet."
        );

        return;
    }}

    const temp =
        document.createElement(
            "canvas"
        );

    temp.width =
        video.videoWidth;

    temp.height =
        video.videoHeight;

    const tempCtx =
        temp.getContext("2d");

    tempCtx.drawImage(
        video,
        0,
        0,
        temp.width,
        temp.height
    );

    const dataUrl =
        temp.toDataURL(
            "image/jpeg",
            0.90
        );

    const img =
        new Image();

    img.onload = function() {{

        trainingImage =
            img;

        trainingImageName =
            "camera_" +
            Date.now() +
            ".jpg";

        showTrainingCanvas();

        if (cameraStream) {{

            cameraStream
                .getTracks()
                .forEach(
                    track => track.stop()
                );

            cameraStream = null;

        }}

        video.style.display =
            "none";

        document.getElementById(
            "captureButton"
        ).style.display =
            "none";

    }};

    img.src =
        dataUrl;

}}


// =====================================================
// SAVE TRAINING IMAGE
// =====================================================

async function saveTrainingImage() {{

    const result =
        document.getElementById(
            "saveResult"
        );

    if (!trainingImage) {{

        result.innerHTML =
            "<div class='warning'>" +
            "Choose or capture an image first." +
            "</div>";

        return;
    }}

    if (!boxes.length) {{

        result.innerHTML =
            "<div class='warning'>" +
            "Draw at least one box first." +
            "</div>";

        return;
    }}

    // Create full-resolution image
    const temp =
        document.createElement(
            "canvas"
        );

    temp.width =
        trainingImage.naturalWidth ||
        trainingImage.width;

    temp.height =
        trainingImage.naturalHeight ||
        trainingImage.height;

    const tempCtx =
        temp.getContext("2d");

    tempCtx.drawImage(
        trainingImage,
        0,
        0,
        temp.width,
        temp.height
    );

    const imageData =
        temp.toDataURL(
            "image/jpeg",
            0.90
        );

    // Convert x1/y1/x2/y2 to YOLO
    // x_center, y_center, width, height
    const annotations =
        boxes.map(
            function(b) {{

                return {{
                    class_name:
                        b.class_name,

                    x_center:
                        (b.x1 + b.x2) / 2,

                    y_center:
                        (b.y1 + b.y2) / 2,

                    width:
                        b.x2 - b.x1,

                    height:
                        b.y2 - b.y1
                }};

            }}
        );

    result.innerHTML =
        "<div class='info'>" +
        "Saving..." +
        "</div>";

    try {{

        const response =
            await fetch(
                "/api/training/image",
                {{
                    method: "POST",

                    headers: {{
                        "Content-Type":
                            "application/json"
                    }},

                    body:
                        JSON.stringify({{
                            filename:
                                trainingImageName,

                            image_data:
                                imageData,

                            annotations:
                                annotations
                        }})
                }}
            );

        const data =
            await response.json();

        if (!data.ok) {{

            result.innerHTML =
                "<div class='warning'>" +
                (
                    data.error ||
                    "Save failed."
                ) +
                "</div>";

            return;
        }}

        result.innerHTML =
            "<div class='success'>" +
            "✅ Image and annotations saved successfully." +
            "</div>";

        document.getElementById(
            "totalImages"
        ).innerText =
            data.summary.total_images;

        document.getElementById(
            "labeledImages"
        ).innerText =
            data.summary.labeled_images;

        document.getElementById(
            "annotationCount"
        ).innerText =
            data.summary.annotations;

        // Reset current image
        trainingImage = null;
        trainingImageName = "";
        boxes = [];

        if (canvas) {{

            ctx.clearRect(
                0,
                0,
                canvas.width,
                canvas.height
            );

            canvas.style.display =
                "none";

        }}

        document.getElementById(
            "saveTrainingButton"
        ).style.display =
            "none";

        document.getElementById(
            "trainingInfo"
        ).style.display =
            "none";

        document.getElementById(
            "boxList"
        ).innerHTML = "";

        document.getElementById(
            "imageFile"
        ).value = "";

    }} catch (e) {{

        result.innerHTML =
            "<div class='warning'>" +
            "Save failed: " +
            e.message +
            "</div>";

    }}

}}


// =====================================================
// TRAINING STATUS
// =====================================================

async function refreshTraining() {{

    try {{

        const r =
            await fetch(
                "/api/training/summary"
            );

        const d =
            await r.json();

        document.getElementById(
            "statusMessage"
        ).innerText =
            d.message;

        const bar =
            document.getElementById(
                "progressBar"
            );

        bar.style.width =
            d.progress + "%";

        bar.innerText =
            d.progress + "%";

        document.getElementById(
            "totalImages"
        ).innerText =
            d.total_images;

        document.getElementById(
            "labeledImages"
        ).innerText =
            d.labeled_images;

        document.getElementById(
            "annotationCount"
        ).innerText =
            d.annotations;

    }} catch (e) {{

        console.log(e);

    }}

}}

setInterval(
    refreshTraining,
    5000
);

</script>
"""

    return layout(
        "AI Training",
        content
    )


# =========================================================
# CAMERA PAGE
# =========================================================

def camera_page():

    content = """
<h2>📷 Camera</h2>

<div class="card">

<video
id="video"
autoplay
playsinline
style="
width:100%;
max-width:800px;
background:#000;
border-radius:10px;
">

</video>

<button onclick="startCamera()">
📷 START CAMERA
</button>

<p id="result">
Model detection will appear here.
</p>

</div>

<script>

async function startCamera() {{

    try {{

        const stream =
            await navigator.mediaDevices.getUserMedia({{
                video: true,
                audio: false
            }});

        document.getElementById(
            "video"
        ).srcObject =
            stream;

    }} catch (e) {{

        alert(
            "Camera permission failed: " +
            e.message
        );

    }}

}}

</script>
"""

    return layout(
        "Camera",
        content
    )


# =========================================================
# HISTORY
# =========================================================

def history_page():

    rows = db_query("""
        SELECT
            count_date,
            loaded_count,
            empty_count,
            notes
        FROM daily_counts
        ORDER BY count_date DESC
        LIMIT 100
    """)

    html_rows = ""

    for row in rows:

        html_rows += f"""
<tr>

<td>{row["count_date"]}</td>

<td>{row["loaded_count"]}</td>

<td>{row["empty_count"]}</td>

<td>{row["notes"] or ""}</td>

</tr>
"""

    content = f"""
<h2>📜 History</h2>

<div class="card">

<table>

<tr>
<th>Date</th>
<th>Loaded</th>
<th>Empty</th>
<th>Notes</th>
</tr>

{html_rows}

</table>

</div>
"""

    return layout(
        "History",
        content
    )


# =========================================================
# SETTINGS
# =========================================================

def settings_page():

    content = """
<h2>⚙️ Settings</h2>

<div class="card">

<h3>System</h3>

<p>
Database:
<b>Supabase PostgreSQL</b>
</p>

<p>
Web Service:
<b>Render</b>
</p>

<p>
AI:
<b>YOLO</b>
</p>

</div>

<div class="card">

<h3>Detection Rules</h3>

<ul>

<li>BUCKET_LOADED → COUNT</li>

<li>BUCKET_EMPTY → DO NOT COUNT</li>

<li>PEOPLE → DO NOT COUNT</li>

<li>EQUIPMENT → DO NOT COUNT</li>

</ul>

</div>
"""

    return layout(
        "Settings",
        content
    )


# =========================================================
# API STATUS
# =========================================================

def api_status():

    state = get_training_state()
    summary = dataset_summary()

    return {
        "ok": True,
        "database": "Supabase PostgreSQL",
        "yolo": True,
        "model_ready": model_ready(),
        "training_status": state["status"],
        "training_progress": state["progress"],
        "training_message": state["message"],
        "dataset": summary
    }


def api_training_summary():

    state = get_training_state()
    summary = dataset_summary()

    return {
        "ok": True,
        **summary,
        "status": state["status"],
        "progress": state["progress"],
        "message": state["message"],
        "model_ready": model_ready()
    }


# =========================================================
# HTTP HANDLER
# =========================================================

class Handler(
    BaseHTTPRequestHandler
):

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


    # =====================================================
    # GET
    # =====================================================

    def do_GET(self):

        try:

            parsed =  urlparse(self.path)

            path = parsed.path

            # -------------------------
            # PAGES
            # -------------------------

            if path == "/":
                self.send_html(
                    dashboard_page()
                )
                return

            if path == "/camera":
                self.send_html(
                    camera_page()
                )
                return

            if path == "/buckets":
                self.send_html(
                    buckets_page()
                )
                return

            if path == "/training":
                self.send_html(
                    training_page()
                )
                return

            if path == "/history":
                self.send_html(
                    history_page()
                )
                return

            if path == "/settings":
                self.send_html(
                    settings_page()
                )
                return


            # -------------------------
            # API
            # -------------------------

            if path == "/api/status":

                send_json(
                    self,
                    api_status()
                )

                return


            if path == "/api/training/summary":

                send_json(
                    self,
                    api_training_summary()
                )

                return


            if path == "/api/buckets":

                rows = db_query("""
                    SELECT
                        id,
                        name,
                        capacity,
                        active
                    FROM buckets
                    ORDER BY id DESC
                """)

                send_json(
                    self,
                    {
                        "ok": True,
                        "buckets": rows
                    }
                )

                return


            if path == "/api/dataset/summary":

                send_json(
                    self,
                    {
                        "ok": True,
                        **dataset_summary()
                    }
                )

                return


            if path == "/api/model/status":

                send_json(
                    self,
                    {
                        "ok": True,
                        "ready": model_ready()
                    }
                )

                return


            send_json(
                self,
                {
                    "ok": False,
                    "error": "Not found"
                },
                404
            )


        except BrokenPipeError:
            pass

        except ConnectionResetError:
            pass

        except Exception as e:

            print(
                traceback.format_exc()
            )

            try:

                send_json(
                    self,
                    {
                        "ok": False,
                        "error": str(e)
                    },
                    500
                )

            except Exception:
                pass


    # =====================================================
    # POST
    # =====================================================

    def do_POST(self):

        try:

            parsed =  urlparse(self.path)

            path = parsed.path


            # =================================================
            # ADD BUCKET
            # =================================================

            if path == "/api/buckets":

                data =  read_json(self)

                name = str(
                    data.get(
                        "name",
                        ""
                    )
                ).strip()

                capacity = float(
                    data.get(
                        "capacity",
                        0
                    ) or 0
                )

                if not name:

                    send_json(
                        self,
                        {
                            "ok": False,
                            "error":
                                "Bucket name is required"
                        },
                        400
                    )

                    return

                db_execute("""
                    INSERT INTO buckets
                    (name, capacity)
                    VALUES (%s, %s)
                """, (
                    name,
                    capacity
                ))

                send_json(
                    self,
                    {
                        "ok": True,
                        "message":
                            "Bucket added successfully"
                    }
                )

                return


            # =================================================
            # SAVE TRAINING IMAGE
            # =================================================

            if path == "/api/training/image":

                data = read_json(self)

                filename = str(
                        data.get(
                            "filename",
                            "training_image.jpg"
                        )
                    ).strip()

                image_data_url = data.get(
                        "image_data"
                    )

                annotations =data.get(
                        "annotations",
                        []
                    )

                if not image_data_url:

                    send_json(
                        self,
                        {
                            "ok": False,
                            "error":
                                "Image is required."
                        },
                        400
                    )

                    return


                if not isinstance(
                    annotations,
                    list
                ) or len(annotations) == 0:

                    send_json(
                        self,
                        {
                            "ok": False,
                            "error":
                                "At least one annotation is required."
                        },
                        400
                    )

                    return


                image_bytes =decode_image_data(
                        image_data_url
                    )


                if len(image_bytes) > 10 * 1024 * 1024:

                    send_json(
                        self,
                        {
                            "ok": False,
                            "error":
                                "Image is too large. Maximum 10 MB."
                        },
                        400
                    )

                    return


                # Validate annotations
                clean_annotations = []


                for item in annotations:

                    class_name =  str(
                            item.get(
                                "class_name",
                                ""
                            )
                        ).strip()


                    if class_name not in CLASSES:

                        send_json(
                            self,
                            {
                                "ok": False,
                                "error":
                                    "Invalid class: " +
                                    class_name
                            },
                            400
                        )

                        return


                    x_center =  float(
                            item.get(
                                "x_center",
                                0
                            )
                        )

                    y_center =  float(
                            item.get(
                                "y_center",
                                0
                            )
                        )

                    width =  float(
                            item.get(
                                "width",
                                0
                            )
                        )

                    height =   float(
                            item.get(
                                "height",
                                0
                            )
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

                        send_json(
                            self,
                            {
                                "ok": False,
                                "error":
                                    "Invalid bounding box."
                            },
                            400
                        )

                        return


                    clean_annotations.append(
                        (
                            class_name,
                            x_center,
                            y_center,
                            width,
                            height
                        )
                    )


                # Save everything in one transaction
                conn =  db_connect()

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
                            TRUE
                        )
                        RETURNING id
                    """, (
                        filename,
                        psycopg2.Binary(
                            image_bytes
                        )
                    ))

                    image_id =  cur.fetchone()[0]


                    for item in clean_annotations:

                        cur.execute("""
                            INSERT INTO annotations
                            (
                                image_id,
                                class_name,
                                x_center,
                                y_center,
                                width,
                                height
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
                            item[0],
                            item[1],
                            item[2],
                            item[3],
                            item[4]
                        ))


                    conn.commit()


                except Exception:

                    conn.rollback()

                    raise


                finally:

                    cur.close()
                    conn.close()


                summary = dataset_summary()


                send_json(
                    self,
                    {
                        "ok": True,
                        "image_id":
                            image_id,
                        "message":
                            "Training image saved successfully.",
                        "summary":
                            summary
                    }
                )

                return


            # =================================================
            # START TRAINING
            # =================================================

            if path == "/api/train":

                summary = dataset_summary()


                if summary[
                    "labeled_images"
                ] < 5:

                    send_json(
                        self,
                        {
                            "ok": False,
                            "error":
                                "At least 5 labeled images are required."
                        },
                        400
                    )

                    return


                set_training_state(
                    "WAITING",
                    0,
                    "Dataset is ready for external YOLO training."
                )


                send_json(
                    self,
                    {
                        "ok": True,
                        "status":
                            "WAITING",
                        "message":
                            "Dataset is ready for external YOLO training."
                    }
                )

                return


            # =================================================
            # SAVE TRAINING STATE
            # =================================================

            if path == "/api/training/state":

                data = read_json(self)

                status = str(
                        data.get(
                            "status",
                            "NOT_STARTED"
                        )
                    )

                progress =  int(
                        data.get(
                            "progress",
                            0
                        )
                    )

                message =  str(
                        data.get(
                            "message",
                            ""
                        )
                    )

                set_training_state(
                    status,
                    progress,
                    message
                )

                send_json(
                    self,
                    {
                        "ok": True
                    }
                )

                return


            send_json(
                self,
                {
                    "ok": False,
                    "error": "Not found"
                },
                404
            )


        except BrokenPipeError:
            pass

        except ConnectionResetError:
            pass

        except Exception as e:

            print(
                traceback.format_exc()
            )

            try:

                send_json(
                    self,
                    {
                        "ok": False,
                        "error": str(e)
                    },
                    500
                )

            except Exception:
                pass


    # =====================================================
    # HTML
    # =====================================================

    def send_html(
        self,
        html
    ):

        body =  html.encode(
                "utf-8"
            )

        try:

            self.send_response(
                200
            )

            self.send_header(
                "Content-Type",
                "text/html; charset=utf-8"
            )

            self.send_header(
                "Content-Length",
                str(len(body))
            )

            self.end_headers()

            self.wfile.write(
                body
            )

        except BrokenPipeError:
            pass

        except ConnectionResetError:
            pass


# =========================================================
# START
# =========================================================

def main():

    print(
        "===================================="
    )

    print(
        " BUCKET COUNTER AI"
    )

    print(
        " Underground production monitoring"
    )

    print(
        "===================================="
    )


    try:

        init_db()

        print(
            "Database: Supabase PostgreSQL"
        )

    except Exception as e:

        print(
            "DATABASE ERROR:"
        )

        print(
            str(e)
        )

        raise


    print(
        "YOLO: AVAILABLE FOR MODEL USE"
    )

    print(
        "Model:",
        "READY"
        if model_ready()
        else "NOT READY"
    )


    server = ThreadingHTTPServer(
            (HOST, PORT),
            Handler
        )


    print(
        f"Server running on {HOST}:{PORT}"
    )


    server.serve_forever()


if __name__ == "__main__":
    main()
