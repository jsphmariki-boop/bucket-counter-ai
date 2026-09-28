from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
import os
import hashlib
import secrets
import html
import psycopg2
from urllib.parse import parse_qs
from datetime import date, datetime, timedelta


# ============================================================
# NEERIKA MINE
# Mining Management System
# PostgreSQL / Supabase
# ============================================================

DATABASE_URL = os.environ.get("DATABASE_URL")

HOST = "0.0.0.0"
PORT = int(os.environ.get("PORT", "8080"))

SESSION_DAYS = 7


# ============================================================
# DATABASE WRAPPER
# ============================================================

class PGCursor:
    def __init__(self, cursor):
        self.cursor = cursor

    def execute(self, sql, params=None):
        # Existing application uses ? placeholders.
        # PostgreSQL uses %s.
        sql = sql.replace("?", "%s")

        if params is None:
            self.cursor.execute(sql)
        else:
            self.cursor.execute(sql, params)

        return self

    def fetchone(self):
        return self.cursor.fetchone()

    def fetchall(self):
        return self.cursor.fetchall()

    def close(self):
        self.cursor.close()


class PGConnection:
    def __init__(self, url):
        if not url:
            raise RuntimeError(
                "DATABASE_URL haijawekwa kwenye Render."
            )

        self.conn = psycopg2.connect(url)

    def cursor(self):
        return PGCursor(self.conn.cursor())

    def execute(self, sql, params=None):
        cur = PGCursor(self.conn.cursor())
        cur.execute(sql, params)
        return cur

    def commit(self):
        self.conn.commit()

    def rollback(self):
        self.conn.rollback()

    def close(self):
        self.conn.close()


def get_db():
    return PGConnection(DATABASE_URL)


# ============================================================
# SECURITY / HELPERS
# ============================================================

def hash_password(password):
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def esc(value):
    if value is None:
        return ""
    return html.escape(str(value), quote=True)


def safe_int(value, default=0, minimum=0):
    try:
        n = int(str(value))
        if n < minimum:
            raise ValueError
        return n
    except Exception:
        return default


def safe_float(value, default=0.0, minimum=0.0):
    try:
        n = float(str(value))
        if n < minimum:
            raise ValueError
        return n
    except Exception:
        return default


def valid_date(value):
    try:
        datetime.strptime(value, "%Y-%m-%d")
        return True
    except Exception:
        return False


def today_string():
    return str(date.today())


# ============================================================
# DATABASE INITIALIZATION + SAFE MIGRATION
# ============================================================

def init_db():
    c = get_db()
    x = c.cursor()

    # --------------------------------------------------------
    # USERS
    # --------------------------------------------------------
    x.execute("""
        CREATE TABLE IF NOT EXISTS users(
            id SERIAL PRIMARY KEY,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            role TEXT NOT NULL
        )
    """)

    # --------------------------------------------------------
    # SESSIONS
    # --------------------------------------------------------
    x.execute("""
        CREATE TABLE IF NOT EXISTS sessions(
            session_id TEXT PRIMARY KEY,
            user_id INTEGER NOT NULL
        )
    """)

    # Safe migration: expiration column
    x.execute("""
        ALTER TABLE sessions
        ADD COLUMN IF NOT EXISTS expires_at TIMESTAMP
    """)

    # --------------------------------------------------------
    # DRILLING
    # Existing columns are preserved.
    # New columns are added safely.
    # --------------------------------------------------------
    x.execute("""
        CREATE TABLE IF NOT EXISTS drilling_data(
            id SERIAL PRIMARY KEY,
            report_date TEXT NOT NULL,
            drilled_holes INTEGER NOT NULL,
            charged_holes INTEGER NOT NULL,
            nonels_used INTEGER NOT NULL,
            buckets_used INTEGER NOT NULL DEFAULT 0,
            fuel_used REAL NOT NULL,
            total_length REAL NOT NULL,
            username TEXT NOT NULL
        )
    """)

    # NEW DRILLING COLUMNS
    x.execute("""
        ALTER TABLE drilling_data
        ADD COLUMN IF NOT EXISTS shift TEXT DEFAULT 'Day Shift'
    """)

    x.execute("""
        ALTER TABLE drilling_data
        ADD COLUMN IF NOT EXISTS bits_used INTEGER DEFAULT 0
    """)

    x.execute("""
        ALTER TABLE drilling_data
        ADD COLUMN IF NOT EXISTS cortex_wire_used REAL DEFAULT 0
    """)

    x.execute("""
        ALTER TABLE drilling_data
        ADD COLUMN IF NOT EXISTS fuse_used INTEGER DEFAULT 0
    """)

    # Make sure old records have valid defaults.
    x.execute("""
        UPDATE drilling_data
        SET shift = 'Day Shift'
        WHERE shift IS NULL OR TRIM(shift) = ''
    """)

    x.execute("""
        UPDATE drilling_data
        SET bits_used = 0
        WHERE bits_used IS NULL
    """)

    x.execute("""
        UPDATE drilling_data
        SET cortex_wire_used = 0
        WHERE cortex_wire_used IS NULL
    """)

    x.execute("""
        UPDATE drilling_data
        SET fuse_used = 0
        WHERE fuse_used IS NULL
    """)

    # --------------------------------------------------------
    # PRODUCTION
    # --------------------------------------------------------
    x.execute("""
        CREATE TABLE IF NOT EXISTS production_data(
            id SERIAL PRIMARY KEY,
            production_date TEXT NOT NULL,
            material_type TEXT NOT NULL,
            buckets INTEGER NOT NULL,
            location TEXT NOT NULL,
            remarks TEXT,
            username TEXT NOT NULL
        )
    """)

    # --------------------------------------------------------
    # SHIFT DATA
    # --------------------------------------------------------
    x.execute("""
        CREATE TABLE IF NOT EXISTS shift_data(
            id SERIAL PRIMARY KEY,
            shift_date TEXT NOT NULL,
            shift_name TEXT NOT NULL,
            leader TEXT NOT NULL,
            drillers TEXT NOT NULL,
            blasters TEXT NOT NULL,
            start_time TEXT NOT NULL,
            end_time TEXT NOT NULL,
            location TEXT NOT NULL,
            remarks TEXT,
            username TEXT NOT NULL
        )
    """)

    # --------------------------------------------------------
    # SHIFT LEADERS
    # --------------------------------------------------------
    x.execute("""
        CREATE TABLE IF NOT EXISTS shift_leaders(
            id SERIAL PRIMARY KEY,
            shift_name TEXT UNIQUE NOT NULL,
            leader TEXT NOT NULL
        )
    """)

    # --------------------------------------------------------
    # COSTS
    # --------------------------------------------------------
    x.execute("""
        CREATE TABLE IF NOT EXISTS costs(
            id SERIAL PRIMARY KEY,
            cost_date TEXT NOT NULL,
            cost_type TEXT NOT NULL,
            item_name TEXT NOT NULL,
            quantity REAL NOT NULL DEFAULT 1,
            amount REAL NOT NULL,
            remarks TEXT,
            username TEXT NOT NULL
        )
    """)

    # --------------------------------------------------------
    # DEFAULT SHIFT LEADERS
    # --------------------------------------------------------
    default_leaders = [
        ("Shift A", "Omary Mashamba"),
        ("Shift B", "Masumbuko"),
        ("Shift C", "Mwamnyange"),
    ]

    for shift_name, leader in default_leaders:
        x.execute("""
            INSERT INTO shift_leaders(shift_name, leader)
            VALUES(?, ?)
            ON CONFLICT (shift_name) DO NOTHING
        """, (shift_name, leader))

    # --------------------------------------------------------
    # DEFAULT USERS
    # --------------------------------------------------------
    default_users = [
        ("Joseph", "boyo", "Geologist"),
        ("manager", "boyo", "Manager"),
        ("director", "boyo", "Director"),
    ]

    for username, password, role in default_users:
        x.execute("""
            INSERT INTO users(username, password, role)
            VALUES(?, ?, ?)
            ON CONFLICT (username) DO NOTHING
        """, (
            username,
            hash_password(password),
            role
        ))

    # Give existing sessions an expiration.
    x.execute("""
        UPDATE sessions
        SET expires_at = CURRENT_TIMESTAMP + INTERVAL '7 days'
        WHERE expires_at IS NULL
    """)

    c.commit()
    c.close()


# ============================================================
# SHIFT LEADERS
# ============================================================

def get_shift_leaders():
    c = get_db()

    rows = c.execute("""
        SELECT shift_name, leader
        FROM shift_leaders
        ORDER BY id
    """).fetchall()

    c.close()

    result = {
        "Shift A": "Omary Mashamba",
        "Shift B": "Masumbuko",
        "Shift C": "Mwamnyange",
    }

    for shift_name, leader in rows:
        result[shift_name] = leader

    return result


# ============================================================
# HTML PAGE
# ============================================================

def page(title, content, user=None):

    nav = ""

    if user:
        nav = f"""
        <header class="topbar">
            <div class="brand">
                <div class="brand-title">NEERIKA MINE</div>
                <div class="brand-sub">Geology & Mining Services</div>
            </div>

            <div class="user-area">
                <span class="user-name">{esc(user[1])}</span>
                <span class="role-badge">{esc(user[2])}</span>

                <button class="lang-btn" onclick="setLanguage('sw')">
                    🇹🇿 Swahili
                </button>

                <button class="lang-btn" onclick="setLanguage('en')">
                    🇬🇧 English
                </button>

                <a class="logout-btn" href="/logout">
                    Logout
                </a>
            </div>
        </header>
        """

    return f"""<!DOCTYPE html>
<html lang="sw">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">

<title>{esc(title)} - NEERIKA MINE</title>

<style>

* {{
    box-sizing: border-box;
}}

body {{
    margin: 0;
    font-family: Arial, Helvetica, sans-serif;
    background: #f3f4f6;
    color: #111827;
}}

.topbar {{
    background: #111827;
    color: white;
    padding: 14px 20px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 15px;
    flex-wrap: wrap;
}}

.brand-title {{
    font-size: 22px;
    font-weight: 800;
}}

.brand-sub {{
    font-size: 12px;
    color: #d1d5db;
    margin-top: 3px;
}}

.user-area {{
    display: flex;
    align-items: center;
    gap: 8px;
    flex-wrap: wrap;
}}

.user-name {{
    font-weight: bold;
}}

.role-badge {{
    background: #2563eb;
    padding: 5px 9px;
    border-radius: 15px;
    font-size: 12px;
}}

.lang-btn,
.logout-btn {{
    border: 0;
    padding: 7px 10px;
    border-radius: 7px;
    text-decoration: none;
    cursor: pointer;
    font-size: 12px;
}}

.lang-btn {{
    background: #374151;
    color: white;
}}

.logout-btn {{
    background: #dc2626;
    color: white;
}}

main {{
    max-width: 1400px;
    margin: auto;
    padding: 20px;
}}

h1 {{
    color: #111827;
}}

h2 {{
    margin-top: 28px;
    color: #1f2937;
}}

.card-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
    gap: 15px;
}}

.card {{
    background: white;
    border-radius: 12px;
    padding: 18px;
    margin-bottom: 15px;
    box-shadow: 0 2px 10px rgba(0,0,0,.06);
}}

.card h3 {{
    margin-top: 0;
}}

.number {{
    font-size: 30px;
    font-weight: 800;
    margin: 8px 0;
}}

.small {{
    color: #6b7280;
    font-size: 13px;
}}

.green {{ border-left: 5px solid #16a34a; }}
.blue {{ border-left: 5px solid #2563eb; }}
.red {{ border-left: 5px solid #dc2626; }}
.gold {{ border-left: 5px solid #d97706; }}
.purple {{ border-left: 5px solid #7c3aed; }}
.orange {{ border-left: 5px solid #ea580c; }}

.actions {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 12px;
}}

.action {{
    display: block;
    padding: 16px;
    background: white;
    border-radius: 10px;
    text-decoration: none;
    color: #111827;
    font-weight: bold;
    box-shadow: 0 2px 8px rgba(0,0,0,.06);
}}

.action:hover {{
    background: #eff6ff;
}}

.filter {{
    background: white;
    padding: 15px;
    border-radius: 10px;
    margin: 15px 0;
}}

.filter-row {{
    display: flex;
    gap: 12px;
    align-items: end;
    flex-wrap: wrap;
}}

form {{
    max-width: 850px;
}}

label {{
    display: block;
    font-weight: bold;
    margin-top: 12px;
    margin-bottom: 5px;
}}

input,
select,
textarea {{
    width: 100%;
    padding: 11px;
    border: 1px solid #d1d5db;
    border-radius: 7px;
    font-size: 15px;
    background: white;
}}

textarea {{
    min-height: 90px;
    resize: vertical;
}}

button {{
    margin-top: 15px;
    padding: 11px 16px;
    border: 0;
    border-radius: 7px;
    background: #2563eb;
    color: white;
    font-weight: bold;
    cursor: pointer;
}}

button:hover {{
    opacity: .9;
}}

.table-container {{
    width: 100%;
    overflow-x: auto;
    background: white;
    border-radius: 10px;
}}

table {{
    width: 100%;
    border-collapse: collapse;
    min-width: 850px;
}}

th,
td {{
    border-bottom: 1px solid #e5e7eb;
    padding: 10px;
    text-align: left;
    white-space: nowrap;
}}

th {{
    background: #111827;
    color: white;
}}

tr:hover td {{
    background: #f9fafb;
}}

.edit-btn,
.delete-btn {{
    display: inline-block;
    padding: 6px 9px;
    margin: 2px;
    border-radius: 5px;
    text-decoration: none;
    font-size: 12px;
}}

.edit-btn {{
    background: #2563eb;
    color: white;
}}

.delete-btn {{
    background: #dc2626;
    color: white;
}}

.alert {{
    background: #fef2f2;
    color: #991b1b;
    border-left: 4px solid #dc2626;
    padding: 12px;
    margin: 8px 0;
    border-radius: 6px;
}}

.success {{
    background: #ecfdf5;
    color: #166534;
    border-left: 4px solid #16a34a;
    padding: 12px;
    border-radius: 6px;
}}

.status {{
    display: inline-block;
    padding: 8px 13px;
    border-radius: 20px;
    font-weight: bold;
}}

.status.good {{
    background: #dcfce7;
    color: #166534;
}}

.status.attention {{
    background: #fef3c7;
    color: #92400e;
}}

.status.danger {{
    background: #fee2e2;
    color: #991b1b;
}}

.shift-badge {{
    display: inline-block;
    padding: 5px 9px;
    border-radius: 15px;
    font-size: 12px;
    font-weight: bold;
}}

.day {{
    background: #fef3c7;
    color: #92400e;
}}

.night {{
    background: #e0e7ff;
    color: #3730a3;
}}

.chart {{
    background: white;
    border-radius: 10px;
    padding: 18px;
    margin-top: 15px;
}}

.bar-row {{
    display: grid;
    grid-template-columns: 65px 1fr 100px;
    gap: 8px;
    align-items: center;
    margin: 9px 0;
}}

.bar-bg {{
    height: 14px;
    background: #e5e7eb;
    border-radius: 20px;
    overflow: hidden;
}}

.bar {{
    height: 100%;
    border-radius: 20px;
}}

.bar.prod {{
    background: #16a34a;
}}

.bar.drill {{
    background: #2563eb;
}}

.bar.cost {{
    background: #dc2626;
}}

.shift-card {{
    background: white;
    padding: 15px;
    margin: 10px 0;
    border-radius: 10px;
    border-left: 5px solid #2563eb;
}}

.login-box {{
    max-width: 450px;
    margin: 70px auto;
    padding: 20px;
}}

.login-box .card {{
    padding: 30px;
}}

.btn-download {{
    background: #059669;
}}

.btn-print {{
    background: #111827;
}}

footer {{
    text-align: center;
    color: #6b7280;
    padding: 30px 10px;
    font-size: 12px;
}}

.report-header {{
    display: flex;
    justify-content: space-between;
    gap: 20px;
    border-bottom: 2px solid #111827;
    padding-bottom: 12px;
}}

.total-box {{
    background: #111827;
    color: white;
    padding: 15px;
    border-radius: 10px;
}}

@media(max-width: 700px) {{
    main {{
        padding: 12px;
    }}

    .topbar {{
        padding: 12px;
    }}

    .user-area {{
        width: 100%;
    }}

    .bar-row {{
        grid-template-columns: 55px 1fr 70px;
        font-size: 12px;
    }}

    .report-header {{
        display: block;
    }}
}}

@media print {{
    .no-print,
    .topbar,
    footer {{
        display: none !important;
    }}

    body {{
        background: white;
    }}

    main {{
        max-width: none;
        padding: 0;
    }}

    .card {{
        box-shadow: none;
    }}
}}

</style>
</head>

<body>

{nav}

<main>
{content}
</main>

<footer>
    <strong>NEERIKA MINE</strong><br>
    Geology & Mining Services<br>
    © 2026 Haki Zote Zimehifadhiwa / All Rights Reserved
</footer>


<script>

function setLanguage(lang) {{

    document.querySelectorAll("[data-sw]").forEach(function(el) {{
        if (lang === "en") {{
            el.innerHTML = el.getAttribute("data-en");
        }} else {{
            el.innerHTML = el.getAttribute("data-sw");
        }}
    }});

    localStorage.setItem("neerika_language", lang);
}}

document.addEventListener("DOMContentLoaded", function() {{
    const lang = localStorage.getItem("neerika_language") || "sw";
    setLanguage(lang);
}});


/*
    Simple browser PDF printing/export.
    It uses the browser print dialog so there is
    no server-side PDF dependency.
*/
function exportReportPDF(elementId, filename) {{

    const area = document.getElementById(elementId);

    if (!area) {{
        alert("Report area haipatikani.");
        return;
    }}

    const original = document.body.innerHTML;

    document.body.innerHTML = area.outerHTML;

    window.print();

    document.body.innerHTML = original;

    window.location.reload();
}}

</script>

</body>
</html>
"""


# ============================================================
# HTTP HANDLER
# ============================================================

class MyWebsite(BaseHTTPRequestHandler):

    # --------------------------------------------------------
    # LOGGING
    # --------------------------------------------------------
    def log_message(self, format, *args):
        print(
            "%s - - [%s] %s"
            % (
                self.address_string(),
                self.log_date_time_string(),
                format % args
            ),
            flush=True
        )

    # --------------------------------------------------------
    # CURRENT USER
    # --------------------------------------------------------
    def current_user(self):

        cookie = self.headers.get("Cookie", "")
        session_id = None

        for part in cookie.split(";"):
            part = part.strip()

            if part.startswith("session_id="):
                session_id = part.split("=", 1)[1]

        if not session_id:
            return None

        c = get_db()

        try:
            user = c.execute("""
                SELECT
                    users.id,
                    users.username,
                    users.role
                FROM sessions
                JOIN users
                    ON users.id = sessions.user_id
                WHERE sessions.session_id = ?
                  AND (
                      sessions.expires_at IS NULL
                      OR sessions.expires_at > CURRENT_TIMESTAMP
                  )
            """, (session_id,)).fetchone()

            return user

        finally:
            c.close()

    # Compatibility with old code
    def user(self):
        return self.current_user()

    # --------------------------------------------------------
    # SEND HTML
    # --------------------------------------------------------
    def send_html(self, content, status=200):

        body = content.encode("utf-8")

        self.send_response(status)

        self.send_header(
            "Content-Type",
            "text/html; charset=utf-8"
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

    # --------------------------------------------------------
    # REDIRECT
    # --------------------------------------------------------
    def redirect(self, location):

        self.send_response(302)
        self.send_header("Location", location)
        self.end_headers()

    # --------------------------------------------------------
    # QUERY STRING
    # --------------------------------------------------------
    def q(self):

        if "?" not in self.path:
            return {}

        return parse_qs(
            self.path.split("?", 1)[1]
        )

    # --------------------------------------------------------
    # ID
    # --------------------------------------------------------
    def gid(self):

        return self.q().get("id", [None])[0]

    # --------------------------------------------------------
    # DATE
    # --------------------------------------------------------
    def selected_date(self):

        d = self.q().get(
            "date",
            [today_string()]
        )[0]

        if valid_date(d):
            return d

        return today_string()

    # --------------------------------------------------------
    # ACCESS DENIED
    # --------------------------------------------------------
    def deny(self, u):

        content = """
        <div class="card">
            <h1>Access Denied</h1>
            <div class="alert">
                Huna ruhusa ya kufanya operation hii.
            </div>
            <a href="/dashboard">
                ← Rudi Dashboard
            </a>
        </div>
        """

        self.send_html(
            page("Access Denied", content, u),
            403
        )

    # ========================================================
    # GET
    # ========================================================

    def do_GET(self):

        path = self.path.split("?", 1)[0]

        # ----------------------------------------------------
        # LOGIN
        # ----------------------------------------------------
        if path == "/login":

            self.send_html(
                page(
                    "Login",
                    """
                    <div class="login-box">
                        <div class="card">

                            <h1>NEERIKA MINE</h1>

                            <p>
                                Mining Management System
                            </p>

                            <form method="POST" action="/login">

                                <label
                                    data-sw="Username"
                                    data-en="Username">
                                    Username
                                </label>

                                <input
                                    name="username"
                                    autocomplete="username"
                                    required
                                >

                                <label
                                    data-sw="Password"
                                    data-en="Password">
                                    Password
                                </label>

                                <input
                                    type="password"
                                    name="password"
                                    autocomplete="current-password"
                                    required
                                >

                                <button
                                    type="submit"
                                    data-sw="Login"
                                    data-en="Login">
                                    Login
                                </button>

                            </form>

                        </div>
                    </div>
                    """
                )
            )

            return

        # ----------------------------------------------------
        # LOGOUT
        # ----------------------------------------------------
        if path == "/logout":
            self.logout()
            return

        # ----------------------------------------------------
        # AUTHENTICATION
        # ----------------------------------------------------
        u = self.current_user()

        if not u:
            self.redirect("/login")
            return

        # ----------------------------------------------------
        # ROUTES
        # ----------------------------------------------------
        routes = {

            "/add_drilling":
                (["Geologist", "Manager"], self.add_drilling_form),

            "/edit_drilling":
                (["Geologist", "Manager"], self.edit_drilling_form),

            "/delete_drilling":
                (["Geologist", "Manager"], self.delete_drilling),

            "/add_production":
                (["Geologist", "Manager"], self.add_production_form),

            "/edit_production":
                (["Geologist", "Manager"], self.edit_production_form),

            "/delete_production":
                (["Geologist", "Manager"], self.delete_production),

            "/add_shift":
                (["Geologist", "Manager"], self.add_shift_form),

            "/edit_shift":
                (["Geologist", "Manager"], self.edit_shift_form),

            "/delete_shift":
                (["Geologist", "Manager"], self.delete_shift),

            "/add_cost":
                (["Geologist", "Manager"], self.add_cost_form),

            "/edit_cost":
                (["Geologist", "Manager"], self.edit_cost_form),

            "/delete_cost":
                (["Geologist", "Manager"], self.delete_cost),
        }

        if path == "/dashboard":
            self.dashboard(u)
            return

        if path == "/weekly_report":

            if u[2] not in ["Manager", "Director"]:
                self.deny(u)
            else:
                self.weekly_report(u)

            return

        if path == "/monthly_report":

            if u[2] not in ["Manager", "Director"]:
                self.deny(u)
            else:
                self.monthly_report(u)

            return

        if path == "/shift_leaders":

            if u[2] not in [
                "Manager",
                "Director",
                "Geologist"
            ]:
                self.deny(u)
            else:
                self.shift_leaders_form(u)

            return

        if path in routes:

            allowed_roles, function = routes[path]

            if u[2] not in allowed_roles:
                self.deny(u)
            else:
                function(u)

            return

        self.redirect("/dashboard")

    # ========================================================
    # POST
    # ========================================================

    def do_POST(self):

        path = self.path.split("?", 1)[0]

        try:
            length = int(
                self.headers.get(
                    "Content-Length",
                    "0"
                )
            )
        except Exception:
            length = 0

        if length > 2_000_000:

            self.send_html(
                page(
                    "Request Too Large",
                    """
                    <div class="alert">
                        Request ni kubwa sana.
                    </div>
                    """
                ),
                413
            )

            return

        raw = self.rfile.read(length)

        try:
            data = parse_qs(
                raw.decode("utf-8"),
                keep_blank_values=True
            )
        except Exception:

            self.send_html(
                page(
                    "Bad Request",
                    """
                    <div class="alert">
                        Data ya request haijasomeka.
                    </div>
                    """
                ),
                400
            )

            return

        # ----------------------------------------------------
        # LOGIN
        # ----------------------------------------------------
        if path == "/login":

            username = data.get(
                "username",
                [""]
            )[0].strip()

            password = data.get(
                "password",
                [""]
            )[0]

            c = get_db()

            try:

                user = c.execute("""
                    SELECT id, username, role
                    FROM users
                    WHERE LOWER(username) = LOWER(?)
                      AND password = ?
                """, (
                    username,
                    hash_password(password)
                )).fetchone()

                if not user:

                    self.send_html(
                        page(
                            "Login Error",
                            """
                            <div class="login-box">
                                <div class="alert">
                                    Username au password sio sahihi.
                                </div>

                                <a href="/login">
                                    ← Jaribu tena
                                </a>
                            </div>
                            """
                        ),
                        401
                    )

                    return

                session_id = secrets.token_hex(32)

                c.execute("""
                    INSERT INTO sessions(
                        session_id,
                        user_id,
                        expires_at
                    )
                    VALUES(
                        ?,
                        ?,
                        CURRENT_TIMESTAMP + INTERVAL '7 days'
                    )
                """, (
                    session_id,
                    user[0]
                ))

                c.commit()

            finally:
                c.close()

            self.send_response(302)
            self.send_header(
                "Location",
                "/dashboard"
            )

            secure_cookie = ""

            if os.environ.get("RENDER"):
                secure_cookie = " Secure;"

            self.send_header(
                "Set-Cookie",
                "session_id=%s; HttpOnly;%s SameSite=Lax; Path=/; Max-Age=%s"
                % (
                    session_id,
                    secure_cookie,
                    SESSION_DAYS * 24 * 60 * 60
                )
            )

            self.end_headers()

            return

        # ----------------------------------------------------
        # AUTHENTICATION
        # ----------------------------------------------------
        u = self.current_user()

        if not u:
            self.redirect("/login")
            return

        # ----------------------------------------------------
        # POST ROUTES
        # ----------------------------------------------------
        handlers = {

            "/add_drilling":
                (["Geologist", "Manager"], self.save_drilling),

            "/edit_drilling":
                (["Geologist", "Manager"], self.update_drilling),

            "/add_production":
                (["Geologist", "Manager"], self.save_production),

            "/edit_production":
                (["Geologist", "Manager"], self.update_production),

            "/add_shift":
                (["Geologist", "Manager"], self.save_shift),

            "/edit_shift":
                (["Geologist", "Manager"], self.update_shift),

            "/shift_leaders":
                (["Manager"], self.save_shift_leaders),

            "/add_cost":
                (["Geologist", "Manager"], self.save_cost),

            "/edit_cost":
                (["Geologist", "Manager"], self.update_cost),
        }

        if path in handlers:

            allowed_roles, function = handlers[path]

            if u[2] not in allowed_roles:
                self.deny(u)
                return

            function(data, u)
            return

        self.redirect("/dashboard")

    # ========================================================
    # DASHBOARD
    # ========================================================

    def dashboard(self, u):

        selected = self.selected_date()

        c = get_db()
        x = c.cursor()

        # ----------------------------------------------------
        # DAILY DRILLING TOTAL
        # ----------------------------------------------------
        x.execute("""
            SELECT
                COALESCE(SUM(drilled_holes),0),
                COALESCE(SUM(charged_holes),0),
                COALESCE(SUM(nonels_used),0),
                COALESCE(SUM(buckets_used),0),
                COALESCE(SUM(fuel_used),0),
                COALESCE(SUM(total_length),0),
                COALESCE(SUM(bits_used),0),
                COALESCE(SUM(cortex_wire_used),0),
                COALESCE(SUM(fuse_used),0)
            FROM drilling_data
            WHERE report_date = ?
        """, (selected,))

        (
            drilled,
            charged,
            nonels,
            drill_buckets,
            fuel,
            length,
            bits,
            cortex,
            fuse
        ) = x.fetchone()

        # ----------------------------------------------------
        # DAY SHIFT DRILLING
        # ----------------------------------------------------
        x.execute("""
            SELECT
                COALESCE(SUM(drilled_holes),0),
                COALESCE(SUM(charged_holes),0),
                COALESCE(SUM(nonels_used),0),
                COALESCE(SUM(buckets_used),0),
                COALESCE(SUM(fuel_used),0),
                COALESCE(SUM(total_length),0),
                COALESCE(SUM(bits_used),0),
                COALESCE(SUM(cortex_wire_used),0),
                COALESCE(SUM(fuse_used),0)
            FROM drilling_data
            WHERE report_date = ?
              AND shift = 'Day Shift'
        """, (selected,))

        day_values = x.fetchone()

        # ----------------------------------------------------
        # NIGHT SHIFT DRILLING
        # ----------------------------------------------------
        x.execute("""
            SELECT
                COALESCE(SUM(drilled_holes),0),
                COALESCE(SUM(charged_holes),0),
                COALESCE(SUM(nonels_used),0),
                COALESCE(SUM(buckets_used),0),
                COALESCE(SUM(fuel_used),0),
                COALESCE(SUM(total_length),0),
                COALESCE(SUM(bits_used),0),
                COALESCE(SUM(cortex_wire_used),0),
                COALESCE(SUM(fuse_used),0)
            FROM drilling_data
            WHERE report_date = ?
              AND shift = 'Night Shift'
        """, (selected,))

        night_values = x.fetchone()

        # ----------------------------------------------------
        # PRODUCTION
        # ----------------------------------------------------
        x.execute("""
            SELECT
                COALESCE(SUM(buckets),0)
            FROM production_data
            WHERE production_date = ?
        """, (selected,))

        production = x.fetchone()[0]

        x.execute("""
            SELECT
                COALESCE(SUM(buckets),0)
            FROM production_data
        """)

        cumulative_production = x.fetchone()[0]

        # ----------------------------------------------------
        # COST
        # ----------------------------------------------------
        x.execute("""
            SELECT
                COALESCE(
                    SUM(
                        CASE
                            WHEN cost_type='Gharama za Vifaa'
                            THEN amount
                            ELSE 0
                        END
                    ),0
                ),
                COALESCE(
                    SUM(
                        CASE
                            WHEN cost_type='Gharama Nyinginezo'
                            THEN amount
                            ELSE 0
                        END
                    ),0
                ),
                COALESCE(SUM(amount),0)
            FROM costs
            WHERE cost_date = ?
        """, (selected,))

        equipment_cost, other_cost, total_cost = x.fetchone()

        x.execute("""
            SELECT COALESCE(SUM(amount),0)
            FROM costs
        """)

        cumulative_cost = x.fetchone()[0]

        # ----------------------------------------------------
        # SHIFTS
        # ----------------------------------------------------
        x.execute("""
            SELECT
                id,
                shift_date,
                shift_name,
                leader,
                drillers,
                blasters,
                start_time,
                end_time,
                location,
                remarks,
                username
            FROM shift_data
            WHERE shift_date = ?
            ORDER BY id DESC
        """, (selected,))

        shifts = x.fetchall()

        # ----------------------------------------------------
        # 7 DAY TREND
        # ----------------------------------------------------
        dt = datetime.strptime(
            selected,
            "%Y-%m-%d"
        ).date()

        trend = []

        for i in range(6, -1, -1):

            current = str(
                dt - timedelta(days=i)
            )

            x.execute("""
                SELECT COALESCE(SUM(buckets),0)
                FROM production_data
                WHERE production_date = ?
            """, (current,))

            p = x.fetchone()[0]

            x.execute("""
                SELECT
                    COALESCE(SUM(drilled_holes),0),
                    COALESCE(SUM(charged_holes),0),
                    COALESCE(SUM(total_length),0)
                FROM drilling_data
                WHERE report_date = ?
            """, (current,))

            dvals = x.fetchone()

            x.execute("""
                SELECT COALESCE(SUM(amount),0)
                FROM costs
                WHERE cost_date = ?
            """, (current,))

            cost = x.fetchone()[0]

            trend.append((
                current[5:],
                p,
                dvals[0],
                dvals[1],
                dvals[2],
                cost
            ))

        # ----------------------------------------------------
        # HISTORY
        # ----------------------------------------------------
        x.execute("""
            SELECT
                id,
                production_date,
                material_type,
                buckets,
                location,
                remarks,
                username
            FROM production_data
            ORDER BY id DESC
            LIMIT 30
        """)

        production_history = x.fetchall()

        x.execute("""
            SELECT
                id,
                report_date,
                shift,
                drilled_holes,
                charged_holes,
                nonels_used,
                buckets_used,
                fuel_used,
                total_length,
                bits_used,
                cortex_wire_used,
                fuse_used,
                username
            FROM drilling_data
            ORDER BY id DESC
            LIMIT 50
        """)

        drilling_history = x.fetchall()

        x.execute("""
            SELECT
                id,
                cost_date,
                cost_type,
                item_name,
                quantity,
                amount,
                remarks,
                username
            FROM costs
            ORDER BY id DESC
            LIMIT 30
        """)

        cost_history = x.fetchall()

        x.execute("""
            SELECT
                id,
                shift_date,
                shift_name,
                leader,
                drillers,
                blasters,
                start_time,
                end_time,
                location,
                username
            FROM shift_data
            ORDER BY id DESC
            LIMIT 30
        """)

        shift_history = x.fetchall()

        c.close()

        # ----------------------------------------------------
        # CHARTS
        # ----------------------------------------------------
        max_production = max(
            [r[1] for r in trend] + [1]
        )

        max_drilling = max(
            [r[2] for r in trend] + [1]
        )

        max_cost = max(
            [r[5] for r in trend] + [1]
        )

        production_chart = ""

        for r in trend:

            width = int(
                r[1] / max_production * 100
            )

            production_chart += f"""
            <div class="bar-row">
                <span>{esc(r[0])}</span>
                <div class="bar-bg">
                    <div
                        class="bar prod"
                        style="width:{width}%">
                    </div>
                </div>
                <strong>{r[1]:,}</strong>
            </div>
            """

        drilling_chart = ""

        for r in trend:

            width = int(
                r[2] / max_drilling * 100
            )

            drilling_chart += f"""
            <div class="bar-row">
                <span>{esc(r[0])}</span>
                <div class="bar-bg">
                    <div
                        class="bar drill"
                        style="width:{width}%">
                    </div>
                </div>
                <strong>{r[2]:,}</strong>
            </div>
            """

        cost_chart = ""

        for r in trend:

            width = int(
                r[5] / max_cost * 100
            )

            cost_chart += f"""
            <div class="bar-row">
                <span>{esc(r[0])}</span>
                <div class="bar-bg">
                    <div
                        class="bar cost"
                        style="width:{width}%">
                    </div>
                </div>
                <strong>
                    TSh {r[5]:,.0f}
                </strong>
            </div>
            """

        # ----------------------------------------------------
        # PERFORMANCE
        # ----------------------------------------------------
        if production > 0 and drilled > 0:

            status = "GOOD PERFORMANCE"
            status_class = "good"

        elif production > 0 or drilled > 0:

            status = "NEEDS ATTENTION"
            status_class = "attention"

        else:

            status = "NO ACTIVITY RECORDED"
            status_class = "danger"

        alerts = []

        if charged > drilled:

            alerts.append(
                f"🚨 Charged holes ({charged:,}) "
                f"ni nyingi kuliko drilled holes "
                f"({drilled:,})."
            )

        if production == 0:

            alerts.append(
                f"⚠️ Hakuna production "
                f"iliyorekodiwa tarehe {selected}."
            )

        if drilled == 0:

            alerts.append(
                f"⚠️ Hakuna drilling report "
                f"iliyorekodiwa tarehe {selected}."
            )

        if not shifts:

            alerts.append(
                f"⚠️ Hakuna shift "
                f"iliyorekodiwa tarehe {selected}."
            )

        if production > 0 and total_cost / production > 10000:

            alerts.append(
                f"⚠️ Cost per bucket ni "
                f"TSh {total_cost / production:,.2f}."
            )

        # ====================================================
        # CONTENT
        # ====================================================

        content = f"""

        <h1>
            📊 NEERIKA MINE Dashboard
        </h1>

        <p>
            Karibu
            <strong>{esc(u[1])}</strong>
            — {esc(u[2])}
        </p>

        <div class="filter">

            <form
                method="GET"
                action="/dashboard">

                <div class="filter-row">

                    <div>

                        <label
                            data-sw="📅 Chagua Tarehe"
                            data-en="📅 Select Date">
                            📅 Chagua Tarehe
                        </label>

                        <input
                            type="date"
                            name="date"
                            value="{esc(selected)}"
                            required>

                    </div>

                    <button
                        type="submit"
                        data-sw="🔎 Angalia"
                        data-en="🔎 Filter">
                        🔎 Angalia
                    </button>

                </div>

            </form>

        </div>
        """

        # ====================================================
        # QUICK ACTIONS
        # ====================================================

        if u[2] in ["Geologist", "Manager"]:

            content += """

            <h2>
                ⚡ Quick Actions
            </h2>

            <div class="actions">

                <a
                    class="action"
                    href="/add_drilling">
                    🕳️ Weka Drilling Report
                </a>

                <a
                    class="action"
                    href="/add_production">
                    🪣 Weka Production
                </a>

                <a
                    class="action"
                    href="/add_shift">
                    👷 Weka Shift
                </a>

                <a
                    class="action"
                    href="/add_cost">
                    💰 Weka Gharama
                </a>

            </div>
            """

        # ====================================================
        # MANAGEMENT
        # ====================================================

        if u[2] in ["Manager", "Director"]:

            content += """

            <h2>
                ⚙️ Management Controls
            </h2>

            <div class="actions">

                <a
                    class="action"
                    href="/shift_leaders">
                    ⚙️ Shift Leaders
                </a>

                <a
                    class="action"
                    href="/weekly_report">
                    📆 Weekly Report
                </a>

                <a
                    class="action"
                    href="/monthly_report">
                    📅 Monthly Report
                </a>

            </div>
            """

        # ====================================================
        # STATUS
        # ====================================================

        content += f"""

        <h2>
            🚦 Performance Status
        </h2>

        <div class="card">

            <span class="status {status_class}">
                {status}
            </span>

            <p class="small">
                Tarehe {esc(selected)}
                |
                Production {production:,} buckets
                |
                Drilled {drilled:,} holes
                |
                Cost TSh {total_cost:,.2f}
            </p>

        </div>

        <!-- ============================================= -->
        <!-- PRODUCTION -->
        <!-- ============================================= -->

        <h2>
            🪣 Production Performance
        </h2>

        <div class="card-grid">

            <div class="card green">
                <h3>Material Produced</h3>
                <div class="number">
                    {production:,}
                </div>
                <div class="small">
                    BUCKETS
                </div>
            </div>

            <div class="card blue">
                <h3>Cumulative Production</h3>
                <div class="number">
                    {cumulative_production:,}
                </div>
                <div class="small">
                    BUCKETS
                </div>
            </div>

            <div class="card gold">
                <h3>7-Day Production</h3>
                <div class="number">
                    {sum(r[1] for r in trend):,}
                </div>
                <div class="small">
                    BUCKETS
                </div>
            </div>

        </div>

        <div class="chart">

            <h3>
                📈 Production Trend — 7 Days
            </h3>

            {production_chart}

        </div>

        <!-- ============================================= -->
        <!-- DRILLING DAILY TOTAL -->
        <!-- ============================================= -->

        <h2>
            🕳️ Daily Drilling Total
        </h2>

        <div class="card-grid">

            <div class="card blue">
                <h3>Drilled Holes</h3>
                <div class="number">
                    {drilled:,}
                </div>
                <div class="small">
                    HOLES
                </div>
            </div>

            <div class="card red">
                <h3>Charged Holes</h3>
                <div class="number">
                    {charged:,}
                </div>
                <div class="small">
                    HOLES
                </div>
            </div>

            <div class="card purple">
                <h3>Nonels</h3>
                <div class="number">
                    {nonels:,}
                </div>
                <div class="small">
                    PCS
                </div>
            </div>

            <div class="card gold">
                <h3>Bits Used</h3>
                <div class="number">
                    {bits:,}
                </div>
                <div class="small">
                    PCS
                </div>
            </div>

            <div class="card blue">
                <h3>Cortex Wire</h3>
                <div class="number">
                    {cortex:,.2f}
                </div>
                <div class="small">
                    METERS
                </div>
            </div>

            <div class="card red">
                <h3>Fuse Used</h3>
                <div class="number">
                    {fuse:,}
                </div>
                <div class="small">
                    PCS
                </div>
            </div>

            <div class="card green">
                <h3>Buckets</h3>
                <div class="number">
                    {drill_buckets:,}
                </div>
                <div class="small">
                    BUCKETS
                </div>
            </div>

            <div class="card gold">
                <h3>Fuel</h3>
                <div class="number">
                    {fuel:,.2f}
                </div>
                <div class="small">
                    PCS
                </div>
            </div>

            <div class="card purple">
                <h3>Drilling Length</h3>
                <div class="number">
                    {length:,.2f}
                </div>
                <div class="small">
                    FT
                </div>
            </div>

        </div>

        <!-- ============================================= -->
        <!-- DAY / NIGHT -->
        <!-- ============================================= -->

        <h2>
            ☀️ Day Shift + 🌙 Night Shift
        </h2>

        <div class="card-grid">

            <div class="card">

                <h3>
                    ☀️ Day Shift
                </h3>

                <p>
                    <b>Drilled:</b>
                    {day_values[0]:,}
                </p>

                <p>
                    <b>Charged:</b>
                    {day_values[1]:,}
                </p>

                <p>
                    <b>Nonels:</b>
                    {day_values[2]:,}
                </p>

                <p>
                    <b>Bits:</b>
                    {day_values[6]:,} pcs
                </p>

                <p>
                    <b>Cortex Wire:</b>
                    {day_values[7]:,.2f} m
                </p>

                <p>
                    <b>Fuse:</b>
                    {day_values[8]:,} pcs
                </p>

                <p>
                    <b>Buckets:</b>
                    {day_values[3]:,}
                </p>

                <p>
                    <b>Fuel:</b>
                    {day_values[4]:,.2f}
                </p>

                <p>
                    <b>Length:</b>
                    {day_values[5]:,.2f} ft
                </p>

            </div>

            <div class="card">

                <h3>
                    🌙 Night Shift
                </h3>

                <p>
                    <b>Drilled:</b>
                    {night_values[0]:,}
                </p>

                <p>
                    <b>Charged:</b>
                    {night_values[1]:,}
                </p>

                <p>
                    <b>Nonels:</b>
                    {night_values[2]:,}
                </p>

                <p>
                    <b>Bits:</b>
                    {night_values[6]:,} pcs
                </p>

                <p>
                    <b>Cortex Wire:</b>
                    {night_values[7]:,.2f} m
                </p>

                <p>
                    <b>Fuse:</b>
                    {night_values[8]:,} pcs
                </p>

                <p>
                    <b>Buckets:</b>
                    {night_values[3]:,}
                </p>

                <p>
                    <b>Fuel:</b>
                    {night_values[4]:,.2f}
                </p>

                <p>
                    <b>Length:</b>
                    {night_values[5]:,.2f} ft
                </p>

            </div>

        </div>

        <!-- ============================================= -->
        <!-- DRILLING TREND -->
        <!-- ============================================= -->

        <div class="chart">

            <h3>
                🕳️ Drilling Performance Trend — 7 Days
            </h3>

            {drilling_chart}

        </div>

        <!-- ============================================= -->
        <!-- COST -->
        <!-- ============================================= -->

        <h2>
            💰 Cost Analysis
        </h2>

        <div class="card-grid">

            <div class="card gold">
                <h3>Equipment Cost</h3>
                <div class="number">
                    TSh {equipment_cost:,.2f}
                </div>
            </div>

            <div class="card red">
                <h3>Other Cost</h3>
                <div class="number">
                    TSh {other_cost:,.2f}
                </div>
            </div>

            <div class="card blue">
                <h3>Total Cost</h3>
                <div class="number">
                    TSh {total_cost:,.2f}
                </div>
            </div>

            <div class="card purple">
                <h3>Cumulative Cost</h3>
                <div class="number">
                    TSh {cumulative_cost:,.2f}
                </div>
            </div>

        </div>

        <div class="chart">

            <h3>
                💰 Cost Trend — 7 Days
            </h3>

            {cost_chart}

        </div>

        <!-- ============================================= -->
        <!-- ALERTS -->
        <!-- ============================================= -->

        <h2>
            🚨 Issues / Alerts
        </h2>
        """

        if alerts:

            for alert in alerts:

                content += f"""
                <div class="alert">
                    {esc(alert)}
                </div>
                """

        else:

            content += """
            <div class="success">
                ✅ Hakuna issue kubwa iliyogunduliwa
                kwenye data ya tarehe hii.
            </div>
            """

        # ====================================================
        # SHIFT SELECTED DATE
        # ====================================================

        content += """
        <h2>
            👷 Shift ya Tarehe Iliyochaguliwa
        </h2>
        """

        if shifts:

            for row in shifts:

                edit_buttons = ""

                if u[2] in ["Geologist", "Manager"]:

                    edit_buttons = f"""
                    <a
                        href="/edit_shift?id={row[0]}"
                        class="edit-btn">
                        ✏️ Edit
                    </a>

                    <a
                        href="/delete_shift?id={row[0]}"
                        class="delete-btn"
                        onclick="return confirm('Una uhakika unataka kufuta shift hii?')">
                        🗑️ Delete
                    </a>
                    """

                content += f"""

                <div class="shift-card">

                    <h3>
                        {esc(row[2])}
                        —
                        {esc(row[3])}
                    </h3>

                    <p>
                        <b>Drillers:</b>
                        {esc(row[4])}
                    </p>

                    <p>
                        <b>Blasters:</b>
                        {esc(row[5])}
                    </p>

                    <p>
                        <b>Muda:</b>
                        {esc(row[6])}
                        -
                        {esc(row[7])}
                    </p>

                    <p>
                        <b>Location:</b>
                        {esc(row[8])}
                    </p>

                    <p>
                        <b>Remarks:</b>
                        {esc(row[9] or "-")}
                    </p>

                    {edit_buttons}

                </div>
                """

        else:

            content += """
            <div class="card">
                Hakuna shift iliyorekodiwa kwenye tarehe hii.
            </div>
            """

                # ====================================================
        # PRODUCTION HISTORY
        # ====================================================

        content += """
        <h2>
            🪣 Production History
        </h2>

        <div class="table-container">

        <table>

            <tr>
                <th>Tarehe</th>
                <th>Material</th>
                <th>Buckets</th>
                <th>Location</th>
                <th>Remarks</th>
                <th>Aliyeweka</th>
                <th>Action</th>
            </tr>
        """

        for row in production_history:

            actions = ""

            if u[2] in ["Geologist", "Manager"]:

                actions = f"""
                <a
                    href="/edit_production?id={row[0]}"
                    class="edit-btn">
                    ✏️ Edit
                </a>

                <a
                    href="/delete_production?id={row[0]}"
                    class="delete-btn"
                    onclick="return confirm('Una uhakika unataka kufuta production hii?')">
                    🗑️ Delete
                </a>
                """

            content += f"""
            <tr>

                <td>{esc(row[1])}</td>

                <td>{esc(row[2])}</td>

                <td>{row[3]:,}</td>

                <td>{esc(row[4])}</td>

                <td>{esc(row[5] or "-")}</td>

                <td>{esc(row[6])}</td>

                <td>{actions}</td>

            </tr>
            """

        content += """
        </table>

        </div>
        """


        # ====================================================
        # DRILLING HISTORY
        # ====================================================

        content += """
        <h2>
            🕳️ Drilling History
        </h2>

        <div class="table-container">

        <table>

            <tr>

                <th>Tarehe</th>
                <th>Shift</th>
                <th>Drilled</th>
                <th>Charged</th>
                <th>Nonels</th>
                <th>Buckets</th>
                <th>Fuel</th>
                <th>Length</th>
                <th>Bits</th>
                <th>Cortex Wire</th>
                <th>Fuse</th>
                <th>Aliyeweka</th>
                <th>Action</th>

            </tr>
        """

        for row in drilling_history:

            actions = ""

            if u[2] in ["Geologist", "Manager"]:

                actions = f"""
                <a
                    href="/edit_drilling?id={row[0]}"
                    class="edit-btn">
                    ✏️ Edit
                </a>

                <a
                    href="/delete_drilling?id={row[0]}"
                    class="delete-btn"
                    onclick="return confirm('Una uhakika unataka kufuta drilling report hii?')">
                    🗑️ Delete
                </a>
                """

            shift_class = "day"

            if row[2] == "Night Shift":
                shift_class = "night"

            content += f"""
            <tr>

                <td>{esc(row[1])}</td>

                <td>
                    <span class="shift-badge {shift_class}">
                        {esc(row[2])}
                    </span>
                </td>

                <td>{row[3]:,}</td>

                <td>{row[4]:,}</td>

                <td>{row[5]:,}</td>

                <td>{row[6]:,}</td>

                <td>{row[7]:,.2f}</td>

                <td>{row[8]:,.2f}</td>

                <td>{row[9]:,}</td>

                <td>{row[10]:,.2f} m</td>

                <td>{row[11]:,}</td>

                <td>{esc(row[12])}</td>

                <td>{actions}</td>

            </tr>
            """

        content += """
        </table>

        </div>
        """


        # ====================================================
        # COST HISTORY
        # ====================================================

        content += """
        <h2>
            💰 Gharama History
        </h2>

        <div class="table-container">

        <table>

            <tr>

                <th>Tarehe</th>
                <th>Aina</th>
                <th>Kifaa/Gharama</th>
                <th>Idadi</th>
                <th>Gharama</th>
                <th>Maelezo</th>
                <th>Aliyeweka</th>
                <th>Action</th>

            </tr>
        """

        for row in cost_history:

            actions = ""

            if u[2] in ["Geologist", "Manager"]:

                actions = f"""
                <a
                    href="/edit_cost?id={row[0]}"
                    class="edit-btn">
                    ✏️ Edit
                </a>

                <a
                    href="/delete_cost?id={row[0]}"
                    class="delete-btn"
                    onclick="return confirm('Una uhakika unataka kufuta gharama hii?')">
                    🗑️ Delete
                </a>
                """

            content += f"""
            <tr>

                <td>{esc(row[1])}</td>

                <td>{esc(row[2])}</td>

                <td>{esc(row[3])}</td>

                <td>{row[4]:,.2f}</td>

                <td>TSh {row[5]:,.2f}</td>

                <td>{esc(row[6] or "-")}</td>

                <td>{esc(row[7])}</td>

                <td>{actions}</td>

            </tr>
            """

        content += """
        </table>

        </div>
        """


        # ====================================================
        # SHIFT HISTORY
        # ====================================================

        content += """
        <h2>
            👷 Shift History
        </h2>

        <div class="table-container">

        <table>

            <tr>

                <th>Tarehe</th>
                <th>Shift</th>
                <th>Kiongozi</th>
                <th>Drillers</th>
                <th>Blasters</th>
                <th>Muda</th>
                <th>Location</th>
                <th>Aliyeweka</th>
                <th>Action</th>

            </tr>
        """

        for row in shift_history:

            actions = ""

            if u[2] in ["Geologist", "Manager"]:

                actions = f"""
                <a
                    href="/edit_shift?id={row[0]}"
                    class="edit-btn">
                    ✏️ Edit
                </a>

                <a
                    href="/delete_shift?id={row[0]}"
                    class="delete-btn"
                    onclick="return confirm('Una uhakika unataka kufuta shift hii?')">
                    🗑️ Delete
                </a>
                """

            content += f"""
            <tr>

                <td>{esc(row[1])}</td>

                <td>{esc(row[2])}</td>

                <td>{esc(row[3])}</td>

                <td>{esc(row[4])}</td>

                <td>{esc(row[5])}</td>

                <td>
                    {esc(row[6])} -
                    {esc(row[7])}
                </td>

                <td>{esc(row[8])}</td>

                <td>{esc(row[9])}</td>

                <td>{actions}</td>

            </tr>
            """

        content += """
        </table>

        </div>
        """


        # ====================================================
        # DASHBOARD END
        # ====================================================

        self.send_html(
            page(
                "Dashboard",
                content,
                u
            )
        )


    # ========================================================
    # WEEKLY MANAGEMENT REPORT
    # ========================================================

    def weekly_report(self, u):

        selected_start = self.q().get(
            "start_date",
            [str(date.today() - timedelta(days=6))]
        )[0]

        if not valid_date(selected_start):

            selected_start = str(
                date.today() - timedelta(days=6)
            )

        start_date = datetime.strptime(
            selected_start,
            "%Y-%m-%d"
        ).date()

        end_date = start_date + timedelta(days=6)

        selected_end = str(end_date)

        c = get_db()

        # ----------------------------------------------------
        # PRODUCTION
        # ----------------------------------------------------

        production = c.execute("""
            SELECT COALESCE(SUM(buckets),0)
            FROM production_data
            WHERE production_date BETWEEN ? AND ?
        """, (
            selected_start,
            selected_end
        )).fetchone()[0]

        # ----------------------------------------------------
        # DRILLING
        # ----------------------------------------------------

        drilling = c.execute("""
            SELECT
                COALESCE(SUM(drilled_holes),0),
                COALESCE(SUM(charged_holes),0),
                COALESCE(SUM(nonels_used),0),
                COALESCE(SUM(buckets_used),0),
                COALESCE(SUM(fuel_used),0),
                COALESCE(SUM(total_length),0),
                COALESCE(SUM(bits_used),0),
                COALESCE(SUM(cortex_wire_used),0),
                COALESCE(SUM(fuse_used),0)
            FROM drilling_data
            WHERE report_date BETWEEN ? AND ?
        """, (
            selected_start,
            selected_end
        )).fetchone()

        (
            drilled,
            charged,
            nonels,
            drilling_buckets,
            fuel,
            length,
            bits,
            cortex,
            fuse
        ) = drilling

        # ----------------------------------------------------
        # COST
        # ----------------------------------------------------

        total_cost = c.execute("""
            SELECT COALESCE(SUM(amount),0)
            FROM costs
            WHERE cost_date BETWEEN ? AND ?
        """, (
            selected_start,
            selected_end
        )).fetchone()[0]

        # ----------------------------------------------------
        # SHIFTS
        # ----------------------------------------------------

        total_shifts = c.execute("""
            SELECT COUNT(id)
            FROM shift_data
            WHERE shift_date BETWEEN ? AND ?
        """, (
            selected_start,
            selected_end
        )).fetchone()[0]

        # ----------------------------------------------------
        # DAY/NIGHT
        # ----------------------------------------------------

        day = c.execute("""
            SELECT
                COALESCE(SUM(drilled_holes),0),
                COALESCE(SUM(charged_holes),0),
                COALESCE(SUM(nonels_used),0),
                COALESCE(SUM(buckets_used),0),
                COALESCE(SUM(fuel_used),0),
                COALESCE(SUM(total_length),0),
                COALESCE(SUM(bits_used),0),
                COALESCE(SUM(cortex_wire_used),0),
                COALESCE(SUM(fuse_used),0)
            FROM drilling_data
            WHERE report_date BETWEEN ? AND ?
              AND shift='Day Shift'
        """, (
            selected_start,
            selected_end
        )).fetchone()

        night = c.execute("""
            SELECT
                COALESCE(SUM(drilled_holes),0),
                COALESCE(SUM(charged_holes),0),
                COALESCE(SUM(nonels_used),0),
                COALESCE(SUM(buckets_used),0),
                COALESCE(SUM(fuel_used),0),
                COALESCE(SUM(total_length),0),
                COALESCE(SUM(bits_used),0),
                COALESCE(SUM(cortex_wire_used),0),
                COALESCE(SUM(fuse_used),0)
            FROM drilling_data
            WHERE report_date BETWEEN ? AND ?
              AND shift='Night Shift'
        """, (
            selected_start,
            selected_end
        )).fetchone()

        c.close()

        content = f"""

        <h1>
            📆 NEERIKA MINE — WEEKLY MANAGEMENT REPORT
        </h1>

        <div class="filter no-print">

            <form
                method="GET"
                action="/weekly_report">

                <label>
                    Anzia Tarehe
                </label>

                <input
                    type="date"
                    name="start_date"
                    value="{esc(selected_start)}"
                    required>

                <button type="submit">
                    🔎 Onyesha Ripoti
                </button>

            </form>

        </div>


        <div class="no-print">

            <button
                class="btn-download"
                onclick="exportReportPDF(
                    'report-area',
                    'Weekly_Report_{selected_start}_{selected_end}'
                )">
                📥 Pakua PDF
            </button>

            <button
                class="btn-print"
                onclick="window.print()">
                🖨️ Print Report
            </button>

        </div>


        <div
            id="report-area"
            class="card"
            style="margin-top:15px;">

            <div class="report-header">

                <div>

                    <h1>
                        NEERIKA MINE
                    </h1>

                    <p>
                        Weekly Production,
                        Drilling & Operations Report
                    </p>

                </div>

                <div>

                    <strong>
                        Kipindi:
                    </strong>

                    <br>

                    {esc(selected_start)}
                    hadi
                    {esc(selected_end)}

                </div>

            </div>


            <h2>
                📊 Weekly Summary
            </h2>


            <div class="card-grid">

                <div class="card green">
                    <h3>Production</h3>
                    <div class="number">
                        {production:,}
                    </div>
                    <div class="small">
                        BUCKETS
                    </div>
                </div>


                <div class="card blue">
                    <h3>Drilled Holes</h3>
                    <div class="number">
                        {drilled:,}
                    </div>
                    <div class="small">
                        HOLES
                    </div>
                </div>


                <div class="card red">
                    <h3>Charged Holes</h3>
                    <div class="number">
                        {charged:,}
                    </div>
                    <div class="small">
                        HOLES
                    </div>
                </div>


                <div class="card purple">
                    <h3>Nonels</h3>
                    <div class="number">
                        {nonels:,}
                    </div>
                    <div class="small">
                        PCS
                    </div>
                </div>


                <div class="card gold">
                    <h3>Bits Used</h3>
                    <div class="number">
                        {bits:,}
                    </div>
                    <div class="small">
                        PCS
                    </div>
                </div>


                <div class="card blue">
                    <h3>Cortex Wire</h3>
                    <div class="number">
                        {cortex:,.2f}
                    </div>
                    <div class="small">
                        METERS
                    </div>
                </div>


                <div class="card red">
                    <h3>Fuse</h3>
                    <div class="number">
                        {fuse:,}
                    </div>
                    <div class="small">
                        PCS
                    </div>
                </div>


                <div class="card green">
                    <h3>Drilling Buckets</h3>
                    <div class="number">
                        {drilling_buckets:,}
                    </div>
                    <div class="small">
                        BUCKETS
                    </div>
                </div>


                <div class="card gold">
                    <h3>Fuel</h3>
                    <div class="number">
                        {fuel:,.2f}
                    </div>
                    <div class="small">
                        PCS
                    </div>
                </div>


                <div class="card purple">
                    <h3>Drilling Length</h3>
                    <div class="number">
                        {length:,.2f}
                    </div>
                    <div class="small">
                        FT
                    </div>
                </div>


                <div class="card red">
                    <h3>Total Cost</h3>
                    <div class="number">
                        TSh {total_cost:,.2f}
                    </div>
                </div>


                <div class="card orange">
                    <h3>Shifts</h3>
                    <div class="number">
                        {total_shifts:,}
                    </div>
                </div>

            </div>


            <h2>
                ☀️ Day Shift
            </h2>

            <div class="card-grid">

                <div class="card">
                    <h3>Drilled</h3>
                    <div class="number">{day[0]:,}</div>
                    <div class="small">HOLES</div>
                </div>

                <div class="card">
                    <h3>Charged</h3>
                    <div class="number">{day[1]:,}</div>
                    <div class="small">HOLES</div>
                </div>

                <div class="card">
                    <h3>Nonels</h3>
                    <div class="number">{day[2]:,}</div>
                    <div class="small">PCS</div>
                </div>

                <div class="card">
                    <h3>Bits</h3>
                    <div class="number">{day[6]:,}</div>
                    <div class="small">PCS</div>
                </div>

                <div class="card">
                    <h3>Cortex Wire</h3>
                    <div class="number">{day[7]:,.2f}</div>
                    <div class="small">METERS</div>
                </div>

                <div class="card">
                    <h3>Fuse</h3>
                    <div class="number">{day[8]:,}</div>
                    <div class="small">PCS</div>
                </div>

            </div>


            <h2>
                🌙 Night Shift
            </h2>

            <div class="card-grid">

                <div class="card">
                    <h3>Drilled</h3>
                    <div class="number">{night[0]:,}</div>
                    <div class="small">HOLES</div>
                </div>

                <div class="card">
                    <h3>Charged</h3>
                    <div class="number">{night[1]:,}</div>
                    <div class="small">HOLES</div>
                </div>

                <div class="card">
                    <h3>Nonels</h3>
                    <div class="number">{night[2]:,}</div>
                    <div class="small">PCS</div>
                </div>

                <div class="card">
                    <h3>Bits</h3>
                    <div class="number">{night[6]:,}</div>
                    <div class="small">PCS</div>
                </div>

                <div class="card">
                    <h3>Cortex Wire</h3>
                    <div class="number">{night[7]:,.2f}</div>
                    <div class="small">METERS</div>
                </div>

                <div class="card">
                    <h3>Fuse</h3>
                    <div class="number">{night[8]:,}</div>
                    <div class="small">PCS</div>
                </div>

            </div>


            <div class="total-box">

                <h2 style="color:white;">
                    📊 Day + Night Daily/Weekly Total
                </h2>

                <p>
                    Drilled:
                    <strong>{drilled:,}</strong>
                </p>

                <p>
                    Charged:
                    <strong>{charged:,}</strong>
                </p>

                <p>
                    Nonels:
                    <strong>{nonels:,}</strong>
                </p>

                <p>
                    Bits:
                    <strong>{bits:,} pcs</strong>
                </p>

                <p>
                    Cortex Wire:
                    <strong>{cortex:,.2f} meters</strong>
                </p>

                <p>
                    Fuse:
                    <strong>{fuse:,} pcs</strong>
                </p>

                <p>
                    Drilling Length:
                    <strong>{length:,.2f} ft</strong>
                </p>

            </div>


            <p
                style="
                margin-top:30px;
                text-align:right;
                color:#6b7280;
                font-size:12px;">
                Report Generated By:
                {esc(u[1])}
                ({esc(u[2])})
                |
                Date:
                {today_string()}
            </p>

        </div>


        <br>

        <a
            href="/dashboard"
            class="no-print">
            ← Rudi Dashboard
        </a>

        """

        self.send_html(
            page(
                "Weekly Management Report",
                content,
                u
            )
        )


    # ========================================================
    # MONTHLY MANAGEMENT REPORT
    # ========================================================

    def monthly_report(self, u):

        selected_month = self.q().get(
            "month",
            [date.today().strftime("%Y-%m")]
        )[0]

        try:

            month_date = datetime.strptime(
                selected_month,
                "%Y-%m"
            )

        except Exception:

            selected_month = date.today().strftime("%Y-%m")

            month_date = datetime.strptime(
                selected_month,
                "%Y-%m"
            )

        month_label = month_date.strftime(
            "%B %Y"
        )

        c = get_db()

        # ----------------------------------------------------
        # MONTHLY PRODUCTION
        # ----------------------------------------------------

        production = c.execute("""
            SELECT COALESCE(SUM(buckets),0)
            FROM production_data
            WHERE LEFT(production_date,7)=?
        """, (
            selected_month,
        )).fetchone()[0]

        # ----------------------------------------------------
        # MONTHLY DRILLING
        # ----------------------------------------------------

        drilling = c.execute("""
            SELECT
                COALESCE(SUM(drilled_holes),0),
                COALESCE(SUM(charged_holes),0),
                COALESCE(SUM(nonels_used),0),
                COALESCE(SUM(buckets_used),0),
                COALESCE(SUM(fuel_used),0),
                COALESCE(SUM(total_length),0),
                COALESCE(SUM(bits_used),0),
                COALESCE(SUM(cortex_wire_used),0),
                COALESCE(SUM(fuse_used),0)
            FROM drilling_data
            WHERE LEFT(report_date,7)=?
        """, (
            selected_month,
        )).fetchone()

        (
            drilled,
            charged,
            nonels,
            drilling_buckets,
            fuel,
            length,
            bits,
            cortex,
            fuse
        ) = drilling

        # ----------------------------------------------------
        # MONTHLY COST
        # ----------------------------------------------------

        total_cost = c.execute("""
            SELECT COALESCE(SUM(amount),0)
            FROM costs
            WHERE LEFT(cost_date,7)=?
        """, (
            selected_month,
        )).fetchone()[0]

        # ----------------------------------------------------
        # MONTHLY SHIFTS
        # ----------------------------------------------------

        total_shifts = c.execute("""
            SELECT COUNT(id)
            FROM shift_data
            WHERE LEFT(shift_date,7)=?
        """, (
            selected_month,
        )).fetchone()[0]

        # ----------------------------------------------------
        # DAY SHIFT
        # ----------------------------------------------------

        day = c.execute("""
            SELECT
                COALESCE(SUM(drilled_holes),0),
                COALESCE(SUM(charged_holes),0),
                COALESCE(SUM(nonels_used),0),
                COALESCE(SUM(buckets_used),0),
                COALESCE(SUM(fuel_used),0),
                COALESCE(SUM(total_length),0),
                COALESCE(SUM(bits_used),0),
                COALESCE(SUM(cortex_wire_used),0),
                COALESCE(SUM(fuse_used),0)
            FROM drilling_data
            WHERE LEFT(report_date,7)=?
              AND shift='Day Shift'
        """, (
            selected_month,
        )).fetchone()

        # ----------------------------------------------------
        # NIGHT SHIFT
        # ----------------------------------------------------

        night = c.execute("""
            SELECT
                COALESCE(SUM(drilled_holes),0),
                COALESCE(SUM(charged_holes),0),
                COALESCE(SUM(nonels_used),0),
                COALESCE(SUM(buckets_used),0),
                COALESCE(SUM(fuel_used),0),
                COALESCE(SUM(total_length),0),
                COALESCE(SUM(bits_used),0),
                COALESCE(SUM(cortex_wire_used),0),
                COALESCE(SUM(fuse_used),0)
            FROM drilling_data
            WHERE LEFT(report_date,7)=?
              AND shift='Night Shift'
        """, (
            selected_month,
        )).fetchone()

        c.close()

        content = f"""

        <h1>
            📅 NEERIKA MINE —
            MONTHLY MANAGEMENT REPORT
        </h1>


        <div class="filter no-print">

            <form
                method="GET"
                action="/monthly_report">

                <label>
                    Chagua Mwezi
                </label>

                <input
                    type="month"
                    name="month"
                    value="{esc(selected_month)}"
                    required>

                <button type="submit">
                    🔎 Onyesha Ripoti
                </button>

            </form>

        </div>


        <div class="no-print">

            <button
                class="btn-download"
                onclick="exportReportPDF(
                    'report-area',
                    'Monthly_Report_{selected_month}'
                )">
                📥 Pakua PDF
            </button>

            <button
                class="btn-print"
                onclick="window.print()">
                🖨️ Print Report
            </button>

        </div>


        <div
            id="report-area"
            class="card"
            style="margin-top:15px;">

            <div class="report-header">

                <div>

                    <h1>
                        NEERIKA MINE
                    </h1>

                    <p>
                        Monthly Operations,
                        Production & Drilling Report
                    </p>

                </div>

                <div>

                    <strong>
                        Mwezi:
                    </strong>

                    <br>

                    {esc(month_label)}

                </div>

            </div>


            <h2>
                📊 Monthly Summary
            </h2>


            <div class="card-grid">

                <div class="card green">
                    <h3>Production</h3>
                    <div class="number">
                        {production:,}
                    </div>
                    <div class="small">
                        BUCKETS
                    </div>
                </div>


                <div class="card blue">
                    <h3>Drilled Holes</h3>
                    <div class="number">
                        {drilled:,}
                    </div>
                    <div class="small">
                        HOLES
                    </div>
                </div>


                <div class="card red">
                    <h3>Charged Holes</h3>
                    <div class="number">
                        {charged:,}
                    </div>
                    <div class="small">
                        HOLES
                    </div>
                </div>


                <div class="card purple">
                    <h3>Nonels</h3>
                    <div class="number">
                        {nonels:,}
                    </div>
                    <div class="small">
                        PCS
                    </div>
                </div>


                <div class="card gold">
                    <h3>Bits Used</h3>
                    <div class="number">
                        {bits:,}
                    </div>
                    <div class="small">
                        PCS
                    </div>
                </div>


                <div class="card blue">
                    <h3>Cortex Wire</h3>
                    <div class="number">
                        {cortex:,.2f}
                    </div>
                    <div class="small">
                        METERS
                    </div>
                </div>


                <div class="card red">
                    <h3>Fuse</h3>
                    <div class="number">
                        {fuse:,}
                    </div>
                    <div class="small">
                        PCS
                    </div>
                </div>


                <div class="card green">
                    <h3>Buckets</h3>
                    <div class="number">
                        {drilling_buckets:,}
                    </div>
                    <div class="small">
                        BUCKETS
                    </div>
                </div>


                <div class="card gold">
                    <h3>Fuel</h3>
                    <div class="number">
                        {fuel:,.2f}
                    </div>
                    <div class="small">
                        PCS
                    </div>
                </div>


                <div class="card purple">
                    <h3>Drilling Length</h3>
                    <div class="number">
                        {length:,.2f}
                    </div>
                    <div class="small">
                        FT
                    </div>
                </div>


                <div class="card red">
                    <h3>Total Cost</h3>
                    <div class="number">
                        TSh {total_cost:,.2f}
                    </div>
                </div>


                <div class="card orange">
                    <h3>Shifts</h3>
                    <div class="number">
                        {total_shifts:,}
                    </div>
                </div>

            </div>


            <h2>
                ☀️ Day Shift — Monthly
            </h2>

            <div class="card-grid">

                <div class="card">
                    <h3>Drilled</h3>
                    <div class="number">{day[0]:,}</div>
                    <div class="small">HOLES</div>
                </div>

                <div class="card">
                    <h3>Charged</h3>
                    <div class="number">{day[1]:,}</div>
                    <div class="small">HOLES</div>
                </div>

                <div class="card">
                    <h3>Nonels</h3>
                    <div class="number">{day[2]:,}</div>
                    <div class="small">PCS</div>
                </div>

                <div class="card">
                    <h3>Bits</h3>
                    <div class="number">{day[6]:,}</div>
                    <div class="small">PCS</div>
                </div>

                <div class="card">
                    <h3>Cortex Wire</h3>
                    <div class="number">{day[7]:,.2f}</div>
                    <div class="small">METERS</div>
                </div>

                <div class="card">
                    <h3>Fuse</h3>
                    <div class="number">{day[8]:,}</div>
                    <div class="small">PCS</div>
                </div>

            </div>


            <h2>
                🌙 Night Shift — Monthly
            </h2>

            <div class="card-grid">

                <div class="card">
                    <h3>Drilled</h3>
                    <div class="number">{night[0]:,}</div>
                    <div class="small">HOLES</div>
                </div>

                <div class="card">
                    <h3>Charged</h3>
                    <div class="number">{night[1]:,}</div>
                    <div class="small">HOLES</div>
                </div>

                <div class="card">
                    <h3>Nonels</h3>
                    <div class="number">{night[2]:,}</div>
                    <div class="small">PCS</div>
                </div>

                <div class="card">
                    <h3>Bits</h3>
                    <div class="number">{night[6]:,}</div>
                    <div class="small">PCS</div>
                </div>

                <div class="card">
                    <h3>Cortex Wire</h3>
                    <div class="number">{night[7]:,.2f}</div>
                    <div class="small">METERS</div>
                </div>

                <div class="card">
                    <h3>Fuse</h3>
                    <div class="number">{night[8]:,}</div>
                    <div class="small">PCS</div>
                </div>

            </div>


            <div class="total-box">

                <h2 style="color:white;">
                    📊 Day + Night Monthly Total
                </h2>

                <p>
                    Drilled:
                    <strong>{drilled:,} holes</strong>
                </p>

                <p>
                    Charged:
                    <strong>{charged:,} holes</strong>
                </p>

                <p>
                    Nonels:
                    <strong>{nonels:,} pcs</strong>
                </p>

                <p>
                    Bits:
                    <strong>{bits:,} pcs</strong>
                </p>

                <p>
                    Cortex Wire:
                    <strong>{cortex:,.2f} meters</strong>
                </p>

                <p>
                    Fuse:
                    <strong>{fuse:,} pcs</strong>
                </p>

                <p>
                    Drilling Length:
                    <strong>{length:,.2f} ft</strong>
                </p>

                <p>
                    Drilling Buckets:
                    <strong>{drilling_buckets:,}</strong>
                </p>

            </div>


            <p
                style="
                margin-top:30px;
                text-align:right;
                color:#6b7280;
                font-size:12px;">

                Report Generated By:
                {esc(u[1])}
                ({esc(u[2])})

                |

                Date:
                {today_string()}

            </p>

        </div>


        <br>

        <a
            href="/dashboard"
            class="no-print">
            ← Rudi Dashboard
        </a>

        """

        self.send_html(
            page(
                "Monthly Management Report",
                content,
                u
            )
        )


        # ========================================================
    # ADD DRILLING
    # ========================================================

    def add_drilling_form(self, u):

        today = today_string()

        content = f"""

        <h1>
            🕳️ Daily Drilling Report
        </h1>

        <div class="card">

        <form
            method="POST"
            action="/add_drilling">

            <label>
                📅 Tarehe
            </label>

            <input
                type="date"
                name="report_date"
                value="{today}"
                required>


            <label>
                🔄 Shift
            </label>

            <select
                name="shift"
                required>

                <option value="Day Shift">
                    ☀️ Day Shift
                </option>

                <option value="Night Shift">
                    🌙 Night Shift
                </option>

            </select>


            <label>
                🪨 Bits Used — pcs
                <small>(Optional)</small>
            </label>

            <input
                type="number"
                name="bits_used"
                min="0"
                step="1"
                placeholder="Acha wazi kama hakuna data">


            <label>
                🧵 Cortex Wire Used — meters
                <small>(Optional)</small>
            </label>

            <input
                type="number"
                name="cortex_wire_used"
                min="0"
                step="0.01"
                placeholder="Acha wazi kama hakuna data">


            <label>
                🔥 Fuse Used — pcs
                <small>(Optional)</small>
            </label>

            <input
                type="number"
                name="fuse_used"
                min="0"
                step="1"
                placeholder="Acha wazi kama hakuna data">


            <label>
                🕳️ Drilled Holes
            </label>

            <input
                type="number"
                name="drilled_holes"
                min="0"
                step="1"
                required>


            <label>
                💥 Charged Holes
            </label>

            <input
                type="number"
                name="charged_holes"
                min="0"
                step="1"
                required>


            <label>
                💣 Nonels
            </label>

            <input
                type="number"
                name="nonels_used"
                min="0"
                step="1"
                required>


            <label>
                🪣 Buckets
            </label>

            <input
                type="number"
                name="buckets_used"
                min="0"
                step="1"
                required>


            <label>
                ⛽ Fuel — PCS
            </label>

            <input
                type="number"
                name="fuel_used"
                min="0"
                step="0.01"
                required>


            <label>
                📏 Drilling Length — FT
            </label>

            <input
                type="number"
                name="total_length"
                min="0"
                step="0.01"
                required>


            <button type="submit">
                💾 Hifadhi Drilling Report
            </button>

        </form>

        </div>


        <br>

        <a href="/dashboard">
            ← Rudi Dashboard
        </a>

        """

        self.send_html(
            page(
                "Add Drilling",
                content,
                u
            )
        )


    # ========================================================
    # SAVE DRILLING
    # ========================================================

    def save_drilling(self, data, u):

        try:

            report_date = data.get(
                "report_date",
                [""]
            )[0]

            shift = data.get(
                "shift",
                ["Day Shift"]
            )[0]

            if not valid_date(report_date):
                raise ValueError("Invalid date")

            if shift not in [
                "Day Shift",
                "Night Shift"
            ]:
                raise ValueError("Invalid shift")

            drilled_holes = safe_int(
                data.get("drilled_holes", ["0"])[0]
            )

            charged_holes = safe_int(
                data.get("charged_holes", ["0"])[0]
            )

            nonels_used = safe_int(
                data.get("nonels_used", ["0"])[0]
            )

            buckets_used = safe_int(
                data.get("buckets_used", ["0"])[0]
            )

            fuel_used = safe_float(
                data.get("fuel_used", ["0"])[0]
            )

            total_length = safe_float(
                data.get("total_length", ["0"])[0]
            )

            bits_used = safe_int(
                data.get("bits_used", ["0"])[0]
            )

            cortex_wire_used = safe_float(
                data.get("cortex_wire_used", ["0"])[0]
            )

            fuse_used = safe_int(
                data.get("fuse_used", ["0"])[0]
            )

            # ------------------------------------------------
            # BASIC VALIDATION
            # ------------------------------------------------

            if charged_holes > drilled_holes:
                raise ValueError(
                    "Charged holes cannot be greater than drilled holes."
                )

            # ------------------------------------------------
            # SAVE TO DATABASE
            # ------------------------------------------------

            c = get_db()

            try:

                c.execute("""
                    INSERT INTO drilling_data(
                        report_date,
                        drilled_holes,
                        charged_holes,
                        nonels_used,
                        buckets_used,
                        fuel_used,
                        total_length,
                        username,
                        shift,
                        bits_used,
                        cortex_wire_used,
                        fuse_used
                    )
                    VALUES(
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                    )
                """, (
                    report_date,
                    drilled_holes,
                    charged_holes,
                    nonels_used,
                    buckets_used,
                    fuel_used,
                    total_length,
                    u[1],
                    shift,
                    bits_used,
                    cortex_wire_used,
                    fuse_used
                ))

                c.commit()

            except Exception:
                c.rollback()
                raise

            finally:
                c.close()

            # ------------------------------------------------
            # SUCCESS
            # ------------------------------------------------

            content = f"""
            <div class="card">

                <div class="success">
                    ✅ Drilling report imehifadhiwa
                    kikamilifu.
                </div>

                <h2>
                    Drilling Report
                </h2>

                <p>
                    <b>Tarehe:</b>
                    {esc(report_date)}
                </p>

                <p>
                    <b>Shift:</b>
                    {esc(shift)}
                </p>

                <p>
                    <b>Drilled Holes:</b>
                    {drilled_holes:,}
                </p>

                <p>
                    <b>Charged Holes:</b>
                    {charged_holes:,}
                </p>

                <p>
                    <b>Nonels:</b>
                    {nonels_used:,}
                </p>

                <p>
                    <b>Bits:</b>
                    {bits_used:,}
                </p>

                <p>
                    <b>Cortex Wire:</b>
                    {cortex_wire_used:,.2f} m
                </p>

                <p>
                    <b>Fuse:</b>
                    {fuse_used:,} pcs
                </p>

                <p>
                    <b>Buckets:</b>
                    {buckets_used:,}
                </p>

                <p>
                    <b>Fuel:</b>
                    {fuel_used:,.2f}
                </p>

                <p>
                    <b>Drilling Length:</b>
                    {total_length:,.2f} ft
                </p>

                <p>
                    <b>Aliyeweka:</b>
                    {esc(u[1])}
                </p>

                <br>

                <a
                    href="/dashboard"
                    class="edit-btn">
                    📊 Rudi Dashboard
                </a>

                <a
                    href="/add_drilling"
                    class="edit-btn">
                    ➕ Weka Report Nyingine
                </a>

            </div>
            """

            self.send_html(
                page(
                    "Drilling Saved",
                    content,
                    u
                )
            )

        except Exception as e:

            content = f"""
            <div class="card">

                <h1>
                    ❌ Error Saving Drilling
                </h1>

                <div class="alert">
                    {esc(str(e))}
                </div>

                <br>

                <a
                    href="/add_drilling"
                    class="edit-btn">
                    ← Rudi kwenye Drilling Form
                </a>

            </div>
            """

            self.send_html(
                page(
                    "Drilling Error",
                    content,
                    u
                ),
                400
            )
                # ========================================================
    # EDIT DRILLING FORM
    # ========================================================

    def edit_drilling_form(self, u):

        record_id = self.gid()

        if not record_id or not record_id.isdigit():
            self.redirect("/dashboard")
            return

        c = get_db()

        try:
            row = c.execute("""
                SELECT
                    id,
                    report_date,
                    shift,
                    drilled_holes,
                    charged_holes,
                    nonels_used,
                    buckets_used,
                    fuel_used,
                    total_length,
                    bits_used,
                    cortex_wire_used,
                    fuse_used
                FROM drilling_data
                WHERE id = ?
            """, (int(record_id),)).fetchone()

        finally:
            c.close()

        if not row:
            self.send_html(
                page(
                    "Drilling Not Found",
                    """
                    <div class="card">
                        <h1>❌ Drilling Report Haipo</h1>

                        <div class="alert">
                            Report uliyoomba haikupatikana.
                        </div>

                        <a href="/dashboard">
                            ← Rudi Dashboard
                        </a>
                    </div>
                    """,
                    u
                ),
                404
            )
            return

        content = f"""

        <h1>
            ✏️ Edit Drilling Report
        </h1>

        <div class="card">

            <form
                method="POST"
                action="/edit_drilling?id={row[0]}">

                <label>
                    📅 Tarehe
                </label>

                <input
                    type="date"
                    name="report_date"
                    value="{esc(row[1])}"
                    required>


                <label>
                    🔄 Shift
                </label>

                <select
                    name="shift"
                    required>

                    <option
                        value="Day Shift"
                        {"selected" if row[2] == "Day Shift" else ""}>
                        ☀️ Day Shift
                    </option>

                    <option
                        value="Night Shift"
                        {"selected" if row[2] == "Night Shift" else ""}>
                        🌙 Night Shift
                    </option>

                </select>


                <label>
    🪨 Bits Used — pcs
    <small>(Optional)</small>
</label>

<input
    type="number"
    name="bits_used"
    min="0"
    step="1"
    value="{row[9] if row[9] is not None else ''}"
    placeholder="Acha wazi kama hakuna data">


                <label>
    🧵 Cortex Wire Used — meters
    <small>(Optional)</small>
</label>

<input
    type="number"
    name="cortex_wire_used"
    min="0"
    step="0.01"
    value="{row[10] if row[10] is not None else ''}"
    placeholder="Acha wazi kama hakuna data">


                <label>
    🔥 Fuse Used — pcs
    <small>(Optional)</small>
</label>

<input
    type="number"
    name="fuse_used"
    min="0"
    step="1"
    value="{row[11] if row[11] is not None else ''}"
    placeholder="Acha wazi kama hakuna data">


                <label>
                    🕳️ Drilled Holes
                </label>

                <input
                    type="number"
                    name="drilled_holes"
                    min="0"
                    step="1"
                    value="{row[3] or 0}"
                    required>


                <label>
                    💥 Charged Holes
                </label>

                <input
                    type="number"
                    name="charged_holes"
                    min="0"
                    step="1"
                    value="{row[4] or 0}"
                    required>


                <label>
                    💣 Nonels
                </label>

                <input
                    type="number"
                    name="nonels_used"
                    min="0"
                    step="1"
                    value="{row[5] or 0}"
                    required>


                <label>
                    🪣 Buckets
                </label>

                <input
                    type="number"
                    name="buckets_used"
                    min="0"
                    step="1"
                    value="{row[6] or 0}"
                    required>


                <label>
                    ⛽ Fuel
                </label>

                <input
                    type="number"
                    name="fuel_used"
                    min="0"
                    step="0.01"
                    value="{row[7] or 0}"
                    required>


                <label>
                    📏 Drilling Length — FT
                </label>

                <input
                    type="number"
                    name="total_length"
                    min="0"
                    step="0.01"
                    value="{row[8] or 0}"
                    required>


                <button type="submit">
                    💾 Update Drilling Report
                </button>

            </form>

        </div>

        <br>

        <a href="/dashboard">
            ← Rudi Dashboard
        </a>

        """

        self.send_html(
            page(
                "Edit Drilling",
                content,
                u
            )
        )


    # ========================================================
    # UPDATE DRILLING
    # ========================================================

    def update_drilling(self, data, u):

        record_id = self.gid()

        if not record_id or not record_id.isdigit():

            self.send_html(
                page(
                    "Invalid ID",
                    """
                    <div class="alert">
                        Drilling ID si sahihi.
                    </div>
                    """,
                    u
                ),
                400
            )

            return

        try:

            report_date = data.get(
                "report_date",
                [""]
            )[0]

            shift = data.get(
                "shift",
                ["Day Shift"]
            )[0]

            if not valid_date(report_date):
                raise ValueError("Invalid date")

            if shift not in [
                "Day Shift",
                "Night Shift"
            ]:
                raise ValueError("Invalid shift")

            drilled_holes = safe_int(
                data.get(
                    "drilled_holes",
                    ["0"]
                )[0]
            )

            charged_holes = safe_int(
                data.get(
                    "charged_holes",
                    ["0"]
                )[0]
            )

            nonels_used = safe_int(
                data.get(
                    "nonels_used",
                    ["0"]
                )[0]
            )

            buckets_used = safe_int(
                data.get(
                    "buckets_used",
                    ["0"]
                )[0]
            )

            fuel_used = safe_float(
                data.get(
                    "fuel_used",
                    ["0"]
                )[0]
            )

            total_length = safe_float(
                data.get(
                    "total_length",
                    ["0"]
                )[0]
            )

            bits_used = safe_int(
                data.get(
                    "bits_used",
                    ["0"]
                )[0]
            )

            cortex_wire_used = safe_float(
                data.get(
                    "cortex_wire_used",
                    ["0"]
                )[0]
            )

            fuse_used = safe_int(
                data.get(
                    "fuse_used",
                    ["0"]
                )[0]
            )

            if charged_holes > drilled_holes:
                raise ValueError(
                    "Charged holes cannot be greater than drilled holes."
                )

            c = get_db()

            try:

                existing = c.execute("""
                    SELECT id
                    FROM drilling_data
                    WHERE id = ?
                """, (
                    int(record_id),
                )).fetchone()

                if not existing:
                    raise ValueError(
                        "Drilling report haipatikani."
                    )

                c.execute("""
                    UPDATE drilling_data
                    SET
                        report_date = ?,
                        shift = ?,
                        drilled_holes = ?,
                        charged_holes = ?,
                        nonels_used = ?,
                        buckets_used = ?,
                        fuel_used = ?,
                        total_length = ?,
                        bits_used = ?,
                        cortex_wire_used = ?,
                        fuse_used = ?
                    WHERE id = ?
                """, (
                    report_date,
                    shift,
                    drilled_holes,
                    charged_holes,
                    nonels_used,
                    buckets_used,
                    fuel_used,
                    total_length,
                    bits_used,
                    cortex_wire_used,
                    fuse_used,
                    int(record_id)
                ))

                c.commit()

            except Exception:
                c.rollback()
                raise

            finally:
                c.close()

            self.redirect("/dashboard")

        except Exception as e:

            content = f"""
            <div class="card">

                <h1>
                    ❌ Error Updating Drilling
                </h1>

                <div class="alert">
                    {esc(str(e))}
                </div>

                <br>

                <a
                    href="/dashboard"
                    class="edit-btn">
                    ← Rudi Dashboard
                </a>

            </div>
            """

            self.send_html(
                page(
                    "Drilling Update Error",
                    content,
                    u
                ),
                400
            )


    # ========================================================
    # DELETE DRILLING
    # ========================================================

    def delete_drilling(self, u):

        record_id = self.gid()

        if not record_id or not record_id.isdigit():

            self.send_html(
                page(
                    "Invalid ID",
                    """
                    <div class="alert">
                        Drilling ID si sahihi.
                    </div>
                    """,
                    u
                ),
                400
            )

            return

        c = get_db()

        try:

            existing = c.execute("""
                SELECT id
                FROM drilling_data
                WHERE id = ?
            """, (
                int(record_id),
            )).fetchone()

            if not existing:

                self.send_html(
                    page(
                        "Not Found",
                        """
                        <div class="card">

                            <h1>
                                ❌ Report Haipo
                            </h1>

                            <div class="alert">
                                Drilling report hiyo
                                haikupatikana.
                            </div>

                            <a href="/dashboard">
                                ← Rudi Dashboard
                            </a>

                        </div>
                        """,
                        u
                    ),
                    404
                )

                return

            c.execute("""
                DELETE FROM drilling_data
                WHERE id = ?
            """, (
                int(record_id),
            ))

            c.commit()

        except Exception:

            c.rollback()
            raise

        finally:

            c.close()

        self.redirect("/dashboard")
            # ========================================================
    # ADD PRODUCTION FORM
    # ========================================================

    def add_production_form(self, u):

        today = today_string()

        content = f"""

        <h1>
            🪣 Daily Production Report
        </h1>

        <div class="card">

            <form
                method="POST"
                action="/add_production">

                <label>
                    📅 Tarehe
                </label>

                <input
                    type="date"
                    name="production_date"
                    value="{today}"
                    required>


                <label>
                    🪨 Material Type
                </label>

                <select
                    name="material_type"
                    required>

                    <option value="Ore">
                        Ore
                    </option>

                    <option value="Waste">
                        Waste
                    </option>

                    <option value="Mineralized Material">
                        Mineralized Material
                    </option>

                    <option value="Other">
                        Other
                    </option>

                </select>


                <label>
                    🪣 Number of Buckets
                </label>

                <input
                    type="number"
                    name="buckets"
                    min="0"
                    step="1"
                    required>


                <label>
                    📍 Location / Working Area
                </label>

                <input
                    type="text"
                    name="location"
                    maxlength="200"
                    required>


                <label>
                    📝 Remarks
                </label>

                <textarea
                    name="remarks"
                    maxlength="1000"
                    placeholder="Andika maelezo ya production..."></textarea>


                <button type="submit">
                    💾 Hifadhi Production Report
                </button>

            </form>

        </div>

        <br>

        <a href="/dashboard">
            ← Rudi Dashboard
        </a>

        """

        self.send_html(
            page(
                "Add Production",
                content,
                u
            )
        )


    # ========================================================
    # SAVE PRODUCTION
    # ========================================================

    def save_production(self, data, u):

        try:

            production_date = data.get(
                "production_date",
                [""]
            )[0].strip()

            material_type = data.get(
                "material_type",
                [""]
            )[0].strip()

            buckets = safe_int(
                data.get(
                    "buckets",
                    ["0"]
                )[0]
            )

            location = data.get(
                "location",
                [""]
            )[0].strip()

            remarks = data.get(
                "remarks",
                [""]
            )[0].strip()

            if not valid_date(production_date):
                raise ValueError(
                    "Production date si sahihi."
                )

            if not material_type:
                raise ValueError(
                    "Material type inahitajika."
                )

            if buckets < 0:
                raise ValueError(
                    "Buckets haiwezi kuwa chini ya zero."
                )

            if not location:
                raise ValueError(
                    "Location inahitajika."
                )

            if len(material_type) > 100:
                raise ValueError(
                    "Material type ni ndefu sana."
                )

            if len(location) > 200:
                raise ValueError(
                    "Location ni ndefu sana."
                )

            if len(remarks) > 1000:
                raise ValueError(
                    "Remarks ni ndefu sana."
                )

            c = get_db()

            try:

                c.execute("""
                    INSERT INTO production_data(
                        production_date,
                        material_type,
                        buckets,
                        location,
                        remarks,
                        username
                    )
                    VALUES(
                        ?, ?, ?, ?, ?, ?
                    )
                """, (
                    production_date,
                    material_type,
                    buckets,
                    location,
                    remarks,
                    u[1]
                ))

                c.commit()

            except Exception:
                c.rollback()
                raise

            finally:
                c.close()

            content = f"""

            <div class="card">

                <div class="success">
                    ✅ Production report imehifadhiwa
                    kikamilifu.
                </div>

                <h2>
                    🪣 Production Report
                </h2>

                <p>
                    <b>Tarehe:</b>
                    {esc(production_date)}
                </p>

                <p>
                    <b>Material:</b>
                    {esc(material_type)}
                </p>

                <p>
                    <b>Buckets:</b>
                    {buckets:,}
                </p>

                <p>
                    <b>Location:</b>
                    {esc(location)}
                </p>

                <p>
                    <b>Remarks:</b>
                    {esc(remarks or "-")}
                </p>

                <p>
                    <b>Aliyeweka:</b>
                    {esc(u[1])}
                </p>

                <br>

                <a
                    href="/dashboard"
                    class="edit-btn">
                    📊 Rudi Dashboard
                </a>

                <a
                    href="/add_production"
                    class="edit-btn">
                    ➕ Weka Production Nyingine
                </a>

            </div>

            """

            self.send_html(
                page(
                    "Production Saved",
                    content,
                    u
                )
            )

        except Exception as e:

            content = f"""

            <div class="card">

                <h1>
                    ❌ Error Saving Production
                </h1>

                <div class="alert">
                    {esc(str(e))}
                </div>

                <br>

                <a
                    href="/add_production"
                    class="edit-btn">
                    ← Rudi kwenye Production Form
                </a>

            </div>

            """

            self.send_html(
                page(
                    "Production Error",
                    content,
                    u
                ),
                400
            )


    # ========================================================
    # EDIT PRODUCTION FORM
    # ========================================================

    def edit_production_form(self, u):

        record_id = self.gid()

        if not record_id or not record_id.isdigit():

            self.redirect("/dashboard")
            return

        c = get_db()

        try:

            row = c.execute("""
                SELECT
                    id,
                    production_date,
                    material_type,
                    buckets,
                    location,
                    remarks
                FROM production_data
                WHERE id = ?
            """, (
                int(record_id),
            )).fetchone()

        finally:

            c.close()

        if not row:

            self.send_html(
                page(
                    "Production Not Found",
                    """
                    <div class="card">

                        <h1>
                            ❌ Production Report Haipo
                        </h1>

                        <div class="alert">
                            Report uliyoomba
                            haikupatikana.
                        </div>

                        <a href="/dashboard">
                            ← Rudi Dashboard
                        </a>

                    </div>
                    """,
                    u
                ),
                404
            )

            return

        content = f"""

        <h1>
            ✏️ Edit Production Report
        </h1>

        <div class="card">

            <form
                method="POST"
                action="/edit_production?id={row[0]}">

                <label>
                    📅 Tarehe
                </label>

                <input
                    type="date"
                    name="production_date"
                    value="{esc(row[1])}"
                    required>


                <label>
                    🪨 Material Type
                </label>

                <select
                    name="material_type"
                    required>

                    <option
                        value="Ore"
                        {"selected" if row[2] == "Ore" else ""}>
                        Ore
                    </option>

                    <option
                        value="Waste"
                        {"selected" if row[2] == "Waste" else ""}>
                        Waste
                    </option>

                    <option
                        value="Mineralized Material"
                        {"selected" if row[2] == "Mineralized Material" else ""}>
                        Mineralized Material
                    </option>

                    <option
                        value="Other"
                        {"selected" if row[2] == "Other" else ""}>
                        Other
                    </option>

                </select>


                <label>
                    🪣 Number of Buckets
                </label>

                <input
                    type="number"
                    name="buckets"
                    min="0"
                    step="1"
                    value="{row[3] or 0}"
                    required>


                <label>
                    📍 Location / Working Area
                </label>

                <input
                    type="text"
                    name="location"
                    maxlength="200"
                    value="{esc(row[4])}"
                    required>


                <label>
                    📝 Remarks
                </label>

                <textarea
                    name="remarks"
                    maxlength="1000">{esc(row[5] or "")}</textarea>


                <button type="submit">
                    💾 Update Production Report
                </button>

            </form>

        </div>

        <br>

        <a href="/dashboard">
            ← Rudi Dashboard
        </a>

        """

        self.send_html(
            page(
                "Edit Production",
                content,
                u
            )
        )


    # ========================================================
    # UPDATE PRODUCTION
    # ========================================================

    def update_production(self, data, u):

        record_id = self.gid()

        if not record_id or not record_id.isdigit():

            self.send_html(
                page(
                    "Invalid ID",
                    """
                    <div class="alert">
                        Production ID si sahihi.
                    </div>
                    """,
                    u
                ),
                400
            )

            return

        try:

            production_date = data.get(
                "production_date",
                [""]
            )[0].strip()

            material_type = data.get(
                "material_type",
                [""]
            )[0].strip()

            buckets = safe_int(
                data.get(
                    "buckets",
                    ["0"]
                )[0]
            )

            location = data.get(
                "location",
                [""]
            )[0].strip()

            remarks = data.get(
                "remarks",
                [""]
            )[0].strip()

            if not valid_date(production_date):
                raise ValueError(
                    "Production date si sahihi."
                )

            if not material_type:
                raise ValueError(
                    "Material type inahitajika."
                )

            if not location:
                raise ValueError(
                    "Location inahitajika."
                )

            if len(material_type) > 100:
                raise ValueError(
                    "Material type ni ndefu sana."
                )

            if len(location) > 200:
                raise ValueError(
                    "Location ni ndefu sana."
                )

            if len(remarks) > 1000:
                raise ValueError(
                    "Remarks ni ndefu sana."
                )

            c = get_db()

            try:

                existing = c.execute("""
                    SELECT id
                    FROM production_data
                    WHERE id = ?
                """, (
                    int(record_id),
                )).fetchone()

                if not existing:
                    raise ValueError(
                        "Production report haipatikani."
                    )

                c.execute("""
                    UPDATE production_data
                    SET
                        production_date = ?,
                        material_type = ?,
                        buckets = ?,
                        location = ?,
                        remarks = ?
                    WHERE id = ?
                """, (
                    production_date,
                    material_type,
                    buckets,
                    location,
                    remarks,
                    int(record_id)
                ))

                c.commit()

            except Exception:
                c.rollback()
                raise

            finally:
                c.close()

            self.redirect("/dashboard")

        except Exception as e:

            content = f"""

            <div class="card">

                <h1>
                    ❌ Error Updating Production
                </h1>

                <div class="alert">
                    {esc(str(e))}
                </div>

                <br>

                <a
                    href="/dashboard"
                    class="edit-btn">
                    ← Rudi Dashboard
                </a>

            </div>

            """

            self.send_html(
                page(
                    "Production Update Error",
                    content,
                    u
                ),
                400
            )


    # ========================================================
    # DELETE PRODUCTION
    # ========================================================

    def delete_production(self, u):

        record_id = self.gid()

        if not record_id or not record_id.isdigit():

            self.send_html(
                page(
                    "Invalid ID",
                    """
                    <div class="alert">
                        Production ID si sahihi.
                    </div>
                    """,
                    u
                ),
                400
            )

            return

        c = get_db()

        try:

            existing = c.execute("""
                SELECT id
                FROM production_data
                WHERE id = ?
            """, (
                int(record_id),
            )).fetchone()

            if not existing:

                self.send_html(
                    page(
                        "Not Found",
                        """
                        <div class="card">

                            <h1>
                                ❌ Production Report Haipo
                            </h1>

                            <div class="alert">
                                Production report hiyo
                                haikupatikana.
                            </div>

                            <a href="/dashboard">
                                ← Rudi Dashboard
                            </a>

                        </div>
                        """,
                        u
                    ),
                    404
                )

                return

            c.execute("""
                DELETE FROM production_data
                WHERE id = ?
            """, (
                int(record_id),
            ))

            c.commit()

        except Exception:

            c.rollback()
            raise

        finally:

            c.close()

        self.redirect("/dashboard")
            # ========================================================
    # ADD SHIFT FORM
    # ========================================================

    def add_shift_form(self, u):

        today = today_string()

        leaders = get_shift_leaders()

        content = f"""

        <h1>
            👷 Daily Shift Report
        </h1>

        <div class="card">

            <form
                method="POST"
                action="/add_shift">

                <label>
                    📅 Tarehe
                </label>

                <input
                    type="date"
                    name="shift_date"
                    value="{today}"
                    required>


                <label>
                    🔄 Shift
                </label>

                <select
                    name="shift_name"
                    required>

                    <option value="Shift A">
                        Shift A
                    </option>

                    <option value="Shift B">
                        Shift B
                    </option>

                    <option value="Shift C">
                        Shift C
                    </option>

                </select>


                <label>
                    👷 Shift Leader
                </label>

                <select
                    name="leader"
                    required>

                    <option
                        value="{esc(leaders.get('Shift A', ''))}">
                        {esc(leaders.get('Shift A', ''))}
                    </option>

                    <option
                        value="{esc(leaders.get('Shift B', ''))}">
                        {esc(leaders.get('Shift B', ''))}
                    </option>

                    <option
                        value="{esc(leaders.get('Shift C', ''))}">
                        {esc(leaders.get('Shift C', ''))}
                    </option>

                </select>


                <label>
                    👷 Drillers
                </label>

                <textarea
                    name="drillers"
                    maxlength="1000"
                    placeholder="Majina ya drillers..."></textarea>


                <label>
                    💥 Blasters
                </label>

                <textarea
                    name="blasters"
                    maxlength="1000"
                    placeholder="Majina ya blasters..."></textarea>


                <label>
                    🕐 Start Time
                </label>

                <input
                    type="time"
                    name="start_time"
                    required>


                <label>
                    🕐 End Time
                </label>

                <input
                    type="time"
                    name="end_time"
                    required>


                <label>
                    📍 Location / Working Area
                </label>

                <input
                    type="text"
                    name="location"
                    maxlength="200"
                    required>


                <label>
                    📝 Remarks
                </label>

                <textarea
                    name="remarks"
                    maxlength="1000"
                    placeholder="Maelezo ya shift..."></textarea>


                <button type="submit">
                    💾 Hifadhi Shift Report
                </button>

            </form>

        </div>

        <br>

        <a href="/dashboard">
            ← Rudi Dashboard
        </a>

        """

        self.send_html(
            page(
                "Add Shift",
                content,
                u
            )
        )


    # ========================================================
    # SAVE SHIFT
    # ========================================================

    def save_shift(self, data, u):

        try:

            shift_date = data.get(
                "shift_date",
                [""]
            )[0].strip()

            shift_name = data.get(
                "shift_name",
                [""]
            )[0].strip()

            leader = data.get(
                "leader",
                [""]
            )[0].strip()

            drillers = data.get(
                "drillers",
                [""]
            )[0].strip()

            blasters = data.get(
                "blasters",
                [""]
            )[0].strip()

            start_time = data.get(
                "start_time",
                [""]
            )[0].strip()

            end_time = data.get(
                "end_time",
                [""]
            )[0].strip()

            location = data.get(
                "location",
                [""]
            )[0].strip()

            remarks = data.get(
                "remarks",
                [""]
            )[0].strip()


            # ------------------------------------------------
            # VALIDATION
            # ------------------------------------------------

            if not valid_date(shift_date):
                raise ValueError(
                    "Shift date si sahihi."
                )

            if shift_name not in [
                "Shift A",
                "Shift B",
                "Shift C"
            ]:
                raise ValueError(
                    "Shift name si sahihi."
                )

            if not leader:
                raise ValueError(
                    "Shift leader anahitajika."
                )

            if not drillers:
                raise ValueError(
                    "Drillers wanahitajika."
                )

            if not blasters:
                raise ValueError(
                    "Blasters wanahitajika."
                )

            if not start_time:
                raise ValueError(
                    "Start time inahitajika."
                )

            if not end_time:
                raise ValueError(
                    "End time inahitajika."
                )

            if not location:
                raise ValueError(
                    "Location inahitajika."
                )

            if len(leader) > 200:
                raise ValueError(
                    "Leader name ni ndefu sana."
                )

            if len(drillers) > 1000:
                raise ValueError(
                    "Drillers field ni ndefu sana."
                )

            if len(blasters) > 1000:
                raise ValueError(
                    "Blasters field ni ndefu sana."
                )

            if len(location) > 200:
                raise ValueError(
                    "Location ni ndefu sana."
                )

            if len(remarks) > 1000:
                raise ValueError(
                    "Remarks ni ndefu sana."
                )


            # ------------------------------------------------
            # SAVE
            # ------------------------------------------------

            c = get_db()

            try:

                c.execute("""
                    INSERT INTO shift_data(
                        shift_date,
                        shift_name,
                        leader,
                        drillers,
                        blasters,
                        start_time,
                        end_time,
                        location,
                        remarks,
                        username
                    )
                    VALUES(
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                    )
                """, (
                    shift_date,
                    shift_name,
                    leader,
                    drillers,
                    blasters,
                    start_time,
                    end_time,
                    location,
                    remarks,
                    u[1]
                ))

                c.commit()

            except Exception:
                c.rollback()
                raise

            finally:
                c.close()


            # ------------------------------------------------
            # SUCCESS
            # ------------------------------------------------

            content = f"""

            <div class="card">

                <div class="success">
                    ✅ Shift report imehifadhiwa
                    kikamilifu.
                </div>

                <h2>
                    👷 Shift Report
                </h2>

                <p>
                    <b>Tarehe:</b>
                    {esc(shift_date)}
                </p>

                <p>
                    <b>Shift:</b>
                    {esc(shift_name)}
                </p>

                <p>
                    <b>Leader:</b>
                    {esc(leader)}
                </p>

                <p>
                    <b>Drillers:</b>
                    {esc(drillers)}
                </p>

                <p>
                    <b>Blasters:</b>
                    {esc(blasters)}
                </p>

                <p>
                    <b>Muda:</b>
                    {esc(start_time)}
                    -
                    {esc(end_time)}
                </p>

                <p>
                    <b>Location:</b>
                    {esc(location)}
                </p>

                <p>
                    <b>Remarks:</b>
                    {esc(remarks or "-")}
                </p>

                <p>
                    <b>Aliyeweka:</b>
                    {esc(u[1])}
                </p>

                <br>

                <a
                    href="/dashboard"
                    class="edit-btn">
                    📊 Rudi Dashboard
                </a>

                <a
                    href="/add_shift"
                    class="edit-btn">
                    ➕ Weka Shift Nyingine
                </a>

            </div>

            """

            self.send_html(
                page(
                    "Shift Saved",
                    content,
                    u
                )
            )


        except Exception as e:

            content = f"""

            <div class="card">

                <h1>
                    ❌ Error Saving Shift
                </h1>

                <div class="alert">
                    {esc(str(e))}
                </div>

                <br>

                <a
                    href="/add_shift"
                    class="edit-btn">
                    ← Rudi kwenye Shift Form
                </a>

            </div>

            """

            self.send_html(
                page(
                    "Shift Error",
                    content,
                    u
                ),
                400
            )


    # ========================================================
    # EDIT SHIFT FORM
    # ========================================================

    def edit_shift_form(self, u):

        record_id = self.gid()

        if not record_id or not record_id.isdigit():

            self.redirect("/dashboard")
            return

        c = get_db()

        try:

            row = c.execute("""
                SELECT
                    id,
                    shift_date,
                    shift_name,
                    leader,
                    drillers,
                    blasters,
                    start_time,
                    end_time,
                    location,
                    remarks
                FROM shift_data
                WHERE id = ?
            """, (
                int(record_id),
            )).fetchone()

        finally:

            c.close()

        if not row:

            self.send_html(
                page(
                    "Shift Not Found",
                    """
                    <div class="card">

                        <h1>
                            ❌ Shift Report Haipo
                        </h1>

                        <div class="alert">
                            Shift report uliyoomba
                            haikupatikana.
                        </div>

                        <a href="/dashboard">
                            ← Rudi Dashboard
                        </a>

                    </div>
                    """,
                    u
                ),
                404
            )

            return

        leaders = get_shift_leaders()

        content = f"""

        <h1>
            ✏️ Edit Shift Report
        </h1>

        <div class="card">

            <form
                method="POST"
                action="/edit_shift?id={row[0]}">

                <label>
                    📅 Tarehe
                </label>

                <input
                    type="date"
                    name="shift_date"
                    value="{esc(row[1])}"
                    required>


                <label>
                    🔄 Shift
                </label>

                <select
                    name="shift_name"
                    required>

                    <option
                        value="Shift A"
                        {"selected" if row[2] == "Shift A" else ""}>
                        Shift A
                    </option>

                    <option
                        value="Shift B"
                        {"selected" if row[2] == "Shift B" else ""}>
                        Shift B
                    </option>

                    <option
                        value="Shift C"
                        {"selected" if row[2] == "Shift C" else ""}>
                        Shift C
                    </option>

                </select>


                <label>
                    👷 Shift Leader
                </label>

                <select
                    name="leader"
                    required>
        """

        for shift_name, leader_name in leaders.items():

            content += f"""
                    <option
                        value="{esc(leader_name)}"
                        {"selected" if row[3] == leader_name else ""}>
                        {esc(shift_name)}
                        —
                        {esc(leader_name)}
                    </option>
            """

        content += f"""

                </select>


                <label>
                    👷 Drillers
                </label>

                <textarea
                    name="drillers"
                    maxlength="1000"
                    required>{esc(row[4])}</textarea>


                <label>
                    💥 Blasters
                </label>

                <textarea
                    name="blasters"
                    maxlength="1000"
                    required>{esc(row[5])}</textarea>


                <label>
                    🕐 Start Time
                </label>

                <input
                    type="time"
                    name="start_time"
                    value="{esc(row[6])}"
                    required>


                <label>
                    🕐 End Time
                </label>

                <input
                    type="time"
                    name="end_time"
                    value="{esc(row[7])}"
                    required>


                <label>
                    📍 Location / Working Area
                </label>

                <input
                    type="text"
                    name="location"
                    maxlength="200"
                    value="{esc(row[8])}"
                    required>


                <label>
                    📝 Remarks
                </label>

                <textarea
                    name="remarks"
                    maxlength="1000">{esc(row[9] or "")}</textarea>


                <button type="submit">
                    💾 Update Shift Report
                </button>

            </form>

        </div>

        <br>

        <a href="/dashboard">
            ← Rudi Dashboard
        </a>

        """

        self.send_html(
            page(
                "Edit Shift",
                content,
                u
            )
        )


    # ========================================================
    # UPDATE SHIFT
    # ========================================================

    def update_shift(self, data, u):

        record_id = self.gid()

        if not record_id or not record_id.isdigit():

            self.send_html(
                page(
                    "Invalid ID",
                    """
                    <div class="alert">
                        Shift ID si sahihi.
                    </div>
                    """,
                    u
                ),
                400
            )

            return

        try:

            shift_date = data.get(
                "shift_date",
                [""]
            )[0].strip()

            shift_name = data.get(
                "shift_name",
                [""]
            )[0].strip()

            leader = data.get(
                "leader",
                [""]
            )[0].strip()

            drillers = data.get(
                "drillers",
                [""]
            )[0].strip()

            blasters = data.get(
                "blasters",
                [""]
            )[0].strip()

            start_time = data.get(
                "start_time",
                [""]
            )[0].strip()

            end_time = data.get(
                "end_time",
                [""]
            )[0].strip()

            location = data.get(
                "location",
                [""]
            )[0].strip()

            remarks = data.get(
                "remarks",
                [""]
            )[0].strip()


            if not valid_date(shift_date):
                raise ValueError(
                    "Shift date si sahihi."
                )

            if shift_name not in [
                "Shift A",
                "Shift B",
                "Shift C"
            ]:
                raise ValueError(
                    "Shift name si sahihi."
                )

            if not leader:
                raise ValueError(
                    "Shift leader anahitajika."
                )

            if not drillers:
                raise ValueError(
                    "Drillers wanahitajika."
                )

            if not blasters:
                raise ValueError(
                    "Blasters wanahitajika."
                )

            if not start_time:
                raise ValueError(
                    "Start time inahitajika."
                )

            if not end_time:
                raise ValueError(
                    "End time inahitajika."
                )

            if not location:
                raise ValueError(
                    "Location inahitajika."
                )

            if len(leader) > 200:
                raise ValueError(
                    "Leader name ni ndefu sana."
                )

            if len(drillers) > 1000:
                raise ValueError(
                    "Drillers field ni ndefu sana."
                )

            if len(blasters) > 1000:
                raise ValueError(
                    "Blasters field ni ndefu sana."
                )

            if len(location) > 200:
                raise ValueError(
                    "Location ni ndefu sana."
                )

            if len(remarks) > 1000:
                raise ValueError(
                    "Remarks ni ndefu sana."
                )


            c = get_db()

            try:

                existing = c.execute("""
                    SELECT id
                    FROM shift_data
                    WHERE id = ?
                """, (
                    int(record_id),
                )).fetchone()

                if not existing:
                    raise ValueError(
                        "Shift report haipatikani."
                    )

                c.execute("""
                    UPDATE shift_data
                    SET
                        shift_date = ?,
                        shift_name = ?,
                        leader = ?,
                        drillers = ?,
                        blasters = ?,
                        start_time = ?,
                        end_time = ?,
                        location = ?,
                        remarks = ?
                    WHERE id = ?
                """, (
                    shift_date,
                    shift_name,
                    leader,
                    drillers,
                    blasters,
                    start_time,
                    end_time,
                    location,
                    remarks,
                    int(record_id)
                ))

                c.commit()

            except Exception:

                c.rollback()
                raise

            finally:

                c.close()

            self.redirect("/dashboard")


        except Exception as e:

            content = f"""

            <div class="card">

                <h1>
                    ❌ Error Updating Shift
                </h1>

                <div class="alert">
                    {esc(str(e))}
                </div>

                <br>

                <a
                    href="/dashboard"
                    class="edit-btn">
                    ← Rudi Dashboard
                </a>

            </div>

            """

            self.send_html(
                page(
                    "Shift Update Error",
                    content,
                    u
                ),
                400
            )


    # ========================================================
    # DELETE SHIFT
    # ========================================================

    def delete_shift(self, u):

        record_id = self.gid()

        if not record_id or not record_id.isdigit():

            self.send_html(
                page(
                    "Invalid ID",
                    """
                    <div class="alert">
                        Shift ID si sahihi.
                    </div>
                    """,
                    u
                ),
                400
            )

            return

        c = get_db()

        try:

            existing = c.execute("""
                SELECT id
                FROM shift_data
                WHERE id = ?
            """, (
                int(record_id),
            )).fetchone()

            if not existing:

                self.send_html(
                    page(
                        "Not Found",
                        """
                        <div class="card">

                            <h1>
                                ❌ Shift Report Haipo
                            </h1>

                            <div class="alert">
                                Shift report hiyo
                                haikupatikana.
                            </div>

                            <a href="/dashboard">
                                ← Rudi Dashboard
                            </a>

                        </div>
                        """,
                        u
                    ),
                    404
                )

                return

            c.execute("""
                DELETE FROM shift_data
                WHERE id = ?
            """, (
                int(record_id),
            ))

            c.commit()

        except Exception:

            c.rollback()
            raise

        finally:

            c.close()

        self.redirect("/dashboard")
            # ========================================================
    # SHIFT LEADERS MANAGEMENT
    # ========================================================

    def shift_leaders_form(self, u):

        leaders = get_shift_leaders()

        content = """

        <h1>
            ⚙️ Shift Leaders Management
        </h1>

        <div class="card">

            <p>
                Hapa Manager anaweza kubadilisha
                viongozi wa Shift A, B na C.
            </p>

            <form
                method="POST"
                action="/shift_leaders">

        """

        content += f"""
                <label>
                    👷 Shift A Leader
                </label>

                <input
                    type="text"
                    name="shift_a"
                    value="{esc(leaders.get('Shift A', ''))}"
                    maxlength="200"
                    required>


                <label>
                    👷 Shift B Leader
                </label>

                <input
                    type="text"
                    name="shift_b"
                    value="{esc(leaders.get('Shift B', ''))}"
                    maxlength="200"
                    required>


                <label>
                    👷 Shift C Leader
                </label>

                <input
                    type="text"
                    name="shift_c"
                    value="{esc(leaders.get('Shift C', ''))}"
                    maxlength="200"
                    required>


                <button type="submit">
                    💾 Hifadhi Shift Leaders
                </button>

            </form>

        </div>


        <h2>
            👷 Current Shift Leaders
        </h2>

        <div class="card-grid">

            <div class="card blue">
                <h3>Shift A</h3>

                <div class="number"
                     style="font-size:20px;">
                    {esc(leaders.get('Shift A', '-'))}
                </div>
            </div>


            <div class="card purple">
                <h3>Shift B</h3>

                <div class="number"
                     style="font-size:20px;">
                    {esc(leaders.get('Shift B', '-'))}
                </div>
            </div>


            <div class="card green">
                <h3>Shift C</h3>

                <div class="number"
                     style="font-size:20px;">
                    {esc(leaders.get('Shift C', '-'))}
                </div>
            </div>

        </div>


        <br>

        <a href="/dashboard">
            ← Rudi Dashboard
        </a>

        """

        self.send_html(
            page(
                "Shift Leaders",
                content,
                u
            )
        )


    # ========================================================
    # SAVE SHIFT LEADERS
    # ========================================================

    def save_shift_leaders(self, data, u):

        # ----------------------------------------------------
        # SECURITY
        # ----------------------------------------------------

        if u[2] != "Manager":

            self.deny(u)
            return


        try:

            shift_a = data.get(
                "shift_a",
                [""]
            )[0].strip()

            shift_b = data.get(
                "shift_b",
                [""]
            )[0].strip()

            shift_c = data.get(
                "shift_c",
                [""]
            )[0].strip()


            # ------------------------------------------------
            # VALIDATION
            # ------------------------------------------------

            if not shift_a:
                raise ValueError(
                    "Shift A leader anahitajika."
                )

            if not shift_b:
                raise ValueError(
                    "Shift B leader anahitajika."
                )

            if not shift_c:
                raise ValueError(
                    "Shift C leader anahitajika."
                )


            if len(shift_a) > 200:
                raise ValueError(
                    "Shift A leader name ni ndefu sana."
                )

            if len(shift_b) > 200:
                raise ValueError(
                    "Shift B leader name ni ndefu sana."
                )

            if len(shift_c) > 200:
                raise ValueError(
                    "Shift C leader name ni ndefu sana."
                )


            c = get_db()

            try:

                # --------------------------------------------
                # SHIFT A
                # --------------------------------------------

                c.execute("""
                    INSERT INTO shift_leaders(
                        shift_name,
                        leader
                    )
                    VALUES(?, ?)
                    ON CONFLICT (shift_name)
                    DO UPDATE SET
                        leader = EXCLUDED.leader
                """, (
                    "Shift A",
                    shift_a
                ))


                # --------------------------------------------
                # SHIFT B
                # --------------------------------------------

                c.execute("""
                    INSERT INTO shift_leaders(
                        shift_name,
                        leader
                    )
                    VALUES(?, ?)
                    ON CONFLICT (shift_name)
                    DO UPDATE SET
                        leader = EXCLUDED.leader
                """, (
                    "Shift B",
                    shift_b
                ))


                # --------------------------------------------
                # SHIFT C
                # --------------------------------------------

                c.execute("""
                    INSERT INTO shift_leaders(
                        shift_name,
                        leader
                    )
                    VALUES(?, ?)
                    ON CONFLICT (shift_name)
                    DO UPDATE SET
                        leader = EXCLUDED.leader
                """, (
                    "Shift C",
                    shift_c
                ))


                c.commit()

            except Exception:

                c.rollback()
                raise

            finally:

                c.close()


            # ------------------------------------------------
            # SUCCESS
            # ------------------------------------------------

            content = """

            <div class="card">

                <div class="success">
                    ✅ Shift Leaders wamebadilishwa
                    kikamilifu.
                </div>

                <h2>
                    👷 Shift Leaders Updated
                </h2>

                <p>
                    Mabadiliko yatahifadhiwa kwenye
                    PostgreSQL / Supabase.
                </p>

                <br>

                <a
                    href="/shift_leaders"
                    class="edit-btn">
                    ⚙️ Angalia Shift Leaders
                </a>

                <a
                    href="/dashboard"
                    class="edit-btn">
                    📊 Rudi Dashboard
                </a>

            </div>

            """

            self.send_html(
                page(
                    "Shift Leaders Updated",
                    content,
                    u
                )
            )


        except Exception as e:

            content = f"""

            <div class="card">

                <h1>
                    ❌ Error Updating Shift Leaders
                </h1>

                <div class="alert">
                    {esc(str(e))}
                </div>

                <br>

                <a
                    href="/shift_leaders"
                    class="edit-btn">
                    ← Rudi Shift Leaders
                </a>

            </div>

            """

            self.send_html(
                page(
                    "Shift Leaders Error",
                    content,
                    u
                ),
                400
            )
                # ========================================================
    # ADD COST FORM
    # ========================================================

    def add_cost_form(self, u):

        today = today_string()

        content = f"""

        <h1>
            💰 Daily Cost / Expense Report
        </h1>

        <div class="card">

            <form
                method="POST"
                action="/add_cost">

                <label>
                    📅 Tarehe
                </label>

                <input
                    type="date"
                    name="cost_date"
                    value="{today}"
                    required>


                <label>
                    🏷️ Aina ya Gharama
                </label>

                <select
                    name="cost_type"
                    required>

                    <option value="Gharama za Vifaa">
                        Gharama za Vifaa
                    </option>

                    <option value="Gharama Nyinginezo">
                        Gharama Nyinginezo
                    </option>

                </select>


                <label>
                    🔧 Kifaa / Item
                </label>

                <input
                    type="text"
                    name="item_name"
                    maxlength="200"
                    placeholder="Mfano: Drill bit, fuel, rope..."
                    required>


                <label>
                    🔢 Quantity
                </label>

                <input
                    type="number"
                    name="quantity"
                    min="0"
                    step="0.01"
                    value="1"
                    required>


                <label>
                    💵 Amount — TSh
                </label>

                <input
                    type="number"
                    name="amount"
                    min="0"
                    step="0.01"
                    required>


                <label>
                    📝 Remarks
                </label>

                <textarea
                    name="remarks"
                    maxlength="1000"
                    placeholder="Maelezo ya gharama..."></textarea>


                <button type="submit">
                    💾 Hifadhi Gharama
                </button>

            </form>

        </div>

        <br>

        <a href="/dashboard">
            ← Rudi Dashboard
        </a>

        """

        self.send_html(
            page(
                "Add Cost",
                content,
                u
            )
        )


    # ========================================================
    # SAVE COST
    # ========================================================

    def save_cost(self, data, u):

        try:

            cost_date = data.get(
                "cost_date",
                [""]
            )[0].strip()

            cost_type = data.get(
                "cost_type",
                [""]
            )[0].strip()

            item_name = data.get(
                "item_name",
                [""]
            )[0].strip()

            quantity = safe_float(
                data.get(
                    "quantity",
                    ["1"]
                )[0],
                default=1.0
            )

            amount = safe_float(
                data.get(
                    "amount",
                    ["0"]
                )[0]
            )

            remarks = data.get(
                "remarks",
                [""]
            )[0].strip()


            # ------------------------------------------------
            # VALIDATION
            # ------------------------------------------------

            if not valid_date(cost_date):
                raise ValueError(
                    "Cost date si sahihi."
                )

            if cost_type not in [
                "Gharama za Vifaa",
                "Gharama Nyinginezo"
            ]:
                raise ValueError(
                    "Aina ya gharama si sahihi."
                )

            if not item_name:
                raise ValueError(
                    "Item name inahitajika."
                )

            if quantity < 0:
                raise ValueError(
                    "Quantity haiwezi kuwa chini ya zero."
                )

            if amount < 0:
                raise ValueError(
                    "Amount haiwezi kuwa chini ya zero."
                )

            if len(item_name) > 200:
                raise ValueError(
                    "Item name ni ndefu sana."
                )

            if len(remarks) > 1000:
                raise ValueError(
                    "Remarks ni ndefu sana."
                )


            # ------------------------------------------------
            # SAVE
            # ------------------------------------------------

            c = get_db()

            try:

                c.execute("""
                    INSERT INTO costs(
                        cost_date,
                        cost_type,
                        item_name,
                        quantity,
                        amount,
                        remarks,
                        username
                    )
                    VALUES(
                        ?, ?, ?, ?, ?, ?, ?
                    )
                """, (
                    cost_date,
                    cost_type,
                    item_name,
                    quantity,
                    amount,
                    remarks,
                    u[1]
                ))

                c.commit()

            except Exception:

                c.rollback()
                raise

            finally:

                c.close()


            # ------------------------------------------------
            # SUCCESS
            # ------------------------------------------------

            content = f"""

            <div class="card">

                <div class="success">
                    ✅ Gharama imehifadhiwa
                    kikamilifu.
                </div>

                <h2>
                    💰 Cost Report
                </h2>

                <p>
                    <b>Tarehe:</b>
                    {esc(cost_date)}
                </p>

                <p>
                    <b>Aina:</b>
                    {esc(cost_type)}
                </p>

                <p>
                    <b>Item:</b>
                    {esc(item_name)}
                </p>

                <p>
                    <b>Quantity:</b>
                    {quantity:,.2f}
                </p>

                <p>
                    <b>Amount:</b>
                    TSh {amount:,.2f}
                </p>

                <p>
                    <b>Remarks:</b>
                    {esc(remarks or "-")}
                </p>

                <p>
                    <b>Aliyeweka:</b>
                    {esc(u[1])}
                </p>

                <br>

                <a
                    href="/dashboard"
                    class="edit-btn">
                    📊 Rudi Dashboard
                </a>

                <a
                    href="/add_cost"
                    class="edit-btn">
                    ➕ Weka Gharama Nyingine
                </a>

            </div>

            """

            self.send_html(
                page(
                    "Cost Saved",
                    content,
                    u
                )
            )


        except Exception as e:

            content = f"""

            <div class="card">

                <h1>
                    ❌ Error Saving Cost
                </h1>

                <div class="alert">
                    {esc(str(e))}
                </div>

                <br>

                <a
                    href="/add_cost"
                    class="edit-btn">
                    ← Rudi kwenye Cost Form
                </a>

            </div>

            """

            self.send_html(
                page(
                    "Cost Error",
                    content,
                    u
                ),
                400
            )


    # ========================================================
    # EDIT COST FORM
    # ========================================================

    def edit_cost_form(self, u):

        record_id = self.gid()

        if not record_id or not record_id.isdigit():

            self.redirect("/dashboard")
            return

        c = get_db()

        try:

            row = c.execute("""
                SELECT
                    id,
                    cost_date,
                    cost_type,
                    item_name,
                    quantity,
                    amount,
                    remarks
                FROM costs
                WHERE id = ?
            """, (
                int(record_id),
            )).fetchone()

        finally:

            c.close()

        if not row:

            self.send_html(
                page(
                    "Cost Not Found",
                    """
                    <div class="card">

                        <h1>
                            ❌ Cost Record Haipo
                        </h1>

                        <div class="alert">
                            Gharama uliyoomba
                            haikupatikana.
                        </div>

                        <a href="/dashboard">
                            ← Rudi Dashboard
                        </a>

                    </div>
                    """,
                    u
                ),
                404
            )

            return


        content = f"""

        <h1>
            ✏️ Edit Cost / Expense
        </h1>

        <div class="card">

            <form
                method="POST"
                action="/edit_cost?id={row[0]}">

                <label>
                    📅 Tarehe
                </label>

                <input
                    type="date"
                    name="cost_date"
                    value="{esc(row[1])}"
                    required>


                <label>
                    🏷️ Aina ya Gharama
                </label>

                <select
                    name="cost_type"
                    required>

                    <option
                        value="Gharama za Vifaa"
                        {"selected" if row[2] == "Gharama za Vifaa" else ""}>
                        Gharama za Vifaa
                    </option>

                    <option
                        value="Gharama Nyinginezo"
                        {"selected" if row[2] == "Gharama Nyinginezo" else ""}>
                        Gharama Nyinginezo
                    </option>

                </select>


                <label>
                    🔧 Kifaa / Item
                </label>

                <input
                    type="text"
                    name="item_name"
                    maxlength="200"
                    value="{esc(row[3])}"
                    required>


                <label>
                    🔢 Quantity
                </label>

                <input
                    type="number"
                    name="quantity"
                    min="0"
                    step="0.01"
                    value="{row[4]:.2f}"
                    required>


                <label>
                    💵 Amount — TSh
                </label>

                <input
                    type="number"
                    name="amount"
                    min="0"
                    step="0.01"
                    value="{row[5]:.2f}"
                    required>


                <label>
                    📝 Remarks
                </label>

                <textarea
                    name="remarks"
                    maxlength="1000">{esc(row[6] or "")}</textarea>


                <button type="submit">
                    💾 Update Gharama
                </button>

            </form>

        </div>

        <br>

        <a href="/dashboard">
            ← Rudi Dashboard
        </a>

        """

        self.send_html(
            page(
                "Edit Cost",
                content,
                u
            )
        )


    # ========================================================
    # UPDATE COST
    # ========================================================

    def update_cost(self, data, u):

        record_id = self.gid()

        if not record_id or not record_id.isdigit():

            self.send_html(
                page(
                    "Invalid ID",
                    """
                    <div class="alert">
                        Cost ID si sahihi.
                    </div>
                    """,
                    u
                ),
                400
            )

            return


        try:

            cost_date = data.get(
                "cost_date",
                [""]
            )[0].strip()

            cost_type = data.get(
                "cost_type",
                [""]
            )[0].strip()

            item_name = data.get(
                "item_name",
                [""]
            )[0].strip()

            quantity = safe_float(
                data.get(
                    "quantity",
                    ["1"]
                )[0],
                default=1.0
            )

            amount = safe_float(
                data.get(
                    "amount",
                    ["0"]
                )[0]
            )

            remarks = data.get(
                "remarks",
                [""]
            )[0].strip()


            if not valid_date(cost_date):
                raise ValueError(
                    "Cost date si sahihi."
                )

            if cost_type not in [
                "Gharama za Vifaa",
                "Gharama Nyinginezo"
            ]:
                raise ValueError(
                    "Aina ya gharama si sahihi."
                )

            if not item_name:
                raise ValueError(
                    "Item name inahitajika."
                )

            if quantity < 0:
                raise ValueError(
                    "Quantity haiwezi kuwa chini ya zero."
                )

            if amount < 0:
                raise ValueError(
                    "Amount haiwezi kuwa chini ya zero."
                )

            if len(item_name) > 200:
                raise ValueError(
                    "Item name ni ndefu sana."
                )

            if len(remarks) > 1000:
                raise ValueError(
                    "Remarks ni ndefu sana."
                )


            c = get_db()

            try:

                existing = c.execute("""
                    SELECT id
                    FROM costs
                    WHERE id = ?
                """, (
                    int(record_id),
                )).fetchone()

                if not existing:
                    raise ValueError(
                        "Cost record haipatikani."
                    )


                c.execute("""
                    UPDATE costs
                    SET
                        cost_date = ?,
                        cost_type = ?,
                        item_name = ?,
                        quantity = ?,
                        amount = ?,
                        remarks = ?
                    WHERE id = ?
                """, (
                    cost_date,
                    cost_type,
                    item_name,
                    quantity,
                    amount,
                    remarks,
                    int(record_id)
                ))

                c.commit()

            except Exception:

                c.rollback()
                raise

            finally:

                c.close()


            self.redirect("/dashboard")


        except Exception as e:

            content = f"""

            <div class="card">

                <h1>
                    ❌ Error Updating Cost
                </h1>

                <div class="alert">
                    {esc(str(e))}
                </div>

                <br>

                <a
                    href="/dashboard"
                    class="edit-btn">
                    ← Rudi Dashboard
                </a>

            </div>

            """

            self.send_html(
                page(
                    "Cost Update Error",
                    content,
                    u
                ),
                400
            )


    # ========================================================
    # DELETE COST
    # ========================================================

    def delete_cost(self, u):

        record_id = self.gid()

        if not record_id or not record_id.isdigit():

            self.send_html(
                page(
                    "Invalid ID",
                    """
                    <div class="alert">
                        Cost ID si sahihi.
                    </div>
                    """,
                    u
                ),
                400
            )

            return


        c = get_db()

        try:

            existing = c.execute("""
                SELECT id
                FROM costs
                WHERE id = ?
            """, (
                int(record_id),
            )).fetchone()

            if not existing:

                self.send_html(
                    page(
                        "Not Found",
                        """
                        <div class="card">

                            <h1>
                                ❌ Cost Record Haipo
                            </h1>

                            <div class="alert">
                                Gharama hiyo
                                haikupatikana.
                            </div>

                            <a href="/dashboard">
                                ← Rudi Dashboard
                            </a>

                        </div>
                        """,
                        u
                    ),
                    404
                )

                return


            c.execute("""
                DELETE FROM costs
                WHERE id = ?
            """, (
                int(record_id),
            ))

            c.commit()

        except Exception:

            c.rollback()
            raise

        finally:

            c.close()


        self.redirect("/dashboard")
            # ========================================================
    # LOGOUT
    # ========================================================

    def logout(self):

        cookie = self.headers.get(
            "Cookie",
            ""
        )

        session_id = None

        for part in cookie.split(";"):

            part = part.strip()

            if part.startswith("session_id="):

                session_id = part.split(
                    "=",
                    1
                )[1]

        if session_id:

            c = get_db()

            try:

                c.execute("""
                    DELETE FROM sessions
                    WHERE session_id = ?
                """, (
                    session_id,
                ))

                c.commit()

            except Exception:

                c.rollback()

            finally:

                c.close()


        self.send_response(302)

        self.send_header(
            "Location",
            "/login"
        )

        self.send_header(
            "Set-Cookie",
            "session_id=; "
            "HttpOnly; "
            "SameSite=Lax; "
            "Path=/; "
            "Max-Age=0"
        )

        self.end_headers()


# ============================================================
# SERVER STARTUP
# ============================================================

def main():

    port = int(
        os.environ.get(
            "PORT",
            "8080"
        )
    )

    print(
        "NEERIKA MINE: initializing database...",
        flush=True
    )

    try:

        init_db()

        print(
            "NEERIKA MINE: database initialized",
            flush=True
        )

    except Exception as e:

        print(
            "NEERIKA MINE: DATABASE ERROR:",
            str(e),
            flush=True
        )

        raise


    server = ThreadingHTTPServer(
        (
            HOST,
            port
        ),
        MyWebsite
    )

    print(
        "NEERIKA MINE: server started on port "
        + str(port),
        flush=True
    )


    try:

        server.serve_forever()

    except KeyboardInterrupt:

        print(
            "NEERIKA MINE: server stopped",
            flush=True
        )

    finally:

        server.server_close()


# ============================================================
# APPLICATION ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
