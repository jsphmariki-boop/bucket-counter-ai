from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
import os, hashlib, secrets
import psycopg2
from urllib.parse import parse_qs
from datetime import date, datetime, timedelta

DATABASE_URL = os.environ.get("DATABASE_URL")


# =========================================================
# DATABASE CONNECTION
# =========================================================

class PGCursor:
    def __init__(self, cursor):
        self.cursor = cursor

    def execute(self, sql, params=None):
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
            raise RuntimeError("DATABASE_URL haijawekwa kwenye Render.")
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


def hash_password(p):
    return hashlib.sha256(p.encode()).hexdigest()


# =========================================================
# DATABASE INITIALIZATION + SAFE MIGRATION
# =========================================================

def init_db():
    c = get_db()
    x = c.cursor()

    # USERS
    x.execute('''
        CREATE TABLE IF NOT EXISTS users(
            id SERIAL PRIMARY KEY,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            role TEXT NOT NULL
        )
    ''')

    # SESSIONS
    x.execute('''
        CREATE TABLE IF NOT EXISTS sessions(
            session_id TEXT PRIMARY KEY,
            user_id INTEGER NOT NULL
        )
    ''')

    # DRILLING
    x.execute('''
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
    ''')

    # =====================================================
    # SAFE MIGRATION FOR NEW DRILLING FIELDS
    # =====================================================

    # Day / Night shift
    x.execute('''
        ALTER TABLE drilling_data
        ADD COLUMN IF NOT EXISTS shift TEXT NOT NULL DEFAULT 'Day'
    ''')

    # Bits in pieces
    x.execute('''
        ALTER TABLE drilling_data
        ADD COLUMN IF NOT EXISTS bits_used INTEGER NOT NULL DEFAULT 0
    ''')

    # Cortex wire in meters
    x.execute('''
        ALTER TABLE drilling_data
        ADD COLUMN IF NOT EXISTS cortex_wire_used REAL NOT NULL DEFAULT 0
    ''')

    # Fuse in pieces
    x.execute('''
        ALTER TABLE drilling_data
        ADD COLUMN IF NOT EXISTS fuses_used INTEGER NOT NULL DEFAULT 0
    ''')

    # PRODUCTION
    x.execute('''
        CREATE TABLE IF NOT EXISTS production_data(
            id SERIAL PRIMARY KEY,
            production_date TEXT NOT NULL,
            material_type TEXT NOT NULL,
            buckets INTEGER NOT NULL,
            location TEXT NOT NULL,
            remarks TEXT,
            username TEXT NOT NULL
        )
    ''')

    # SHIFT DATA
    x.execute('''
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
    ''')

    # SHIFT LEADERS
    x.execute('''
        CREATE TABLE IF NOT EXISTS shift_leaders(
            id SERIAL PRIMARY KEY,
            shift_name TEXT UNIQUE NOT NULL,
            leader TEXT NOT NULL
        )
    ''')

    # COSTS
    x.execute('''
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
    ''')

    leaders_list = [
        ("Shift A", "Omary Mashamba"),
        ("Shift B", "Masumbuko"),
        ("Shift C", "Mwamnyange")
    ]

    for shift_name, leader in leaders_list:
        x.execute('''
            INSERT INTO shift_leaders(shift_name, leader)
            VALUES(?, ?)
            ON CONFLICT (shift_name) DO NOTHING
        ''', (shift_name, leader))

    users = [
        ("Joseph", "boyo", "Geologist"),
        ("manager", "boyo", "Manager"),
        ("director", "boyo", "Director")
    ]

    for username, password, role in users:
        x.execute('''
            INSERT INTO users(username, password, role)
            VALUES(?, ?, ?)
            ON CONFLICT (username) DO NOTHING
        ''', (
            username,
            hash_password(password),
            role
        ))

    c.commit()
    c.close()


# =========================================================
# SHIFT LEADERS
# =========================================================

def leaders():
    c = get_db()
    r = c.execute(
        'SELECT shift_name,leader FROM shift_leaders ORDER BY id'
    ).fetchall()
    c.close()

    d = {a: b for a, b in r}

    d.setdefault('Shift A', 'Omary Mashamba')
    d.setdefault('Shift B', 'Masumbuko')
    d.setdefault('Shift C', 'Mwamnyange')

    return d


# =========================================================
# PAGE TEMPLATE
# =========================================================

def page(title, content, user=None):

    nav = ''

    if user:
        nav = f'''
        <div class="topbar">
            <div class="brand">NEERIKA MINE</div>

            <div class="user-info">
                <span>{user[1]}</span>

                <span class="role">
                    {user[2]}
                </span>

                <select
                    id="lang-selector"
                    onchange="changeLanguage(this.value)"
                    style="
                        padding:4px 8px;
                        border-radius:6px;
                        background:#374151;
                        color:white;
                        border:1px solid #4b5563;
                        cursor:pointer;
                    "
                >
                    <option value="sw">🇹🇿 Swahili</option>
                    <option value="en">🇬🇧 English</option>
                </select>

                <a
                    href="/logout"
                    class="logout"
                    data-sw="Kutoka"
                    data-en="Logout"
                >
                    Logout
                </a>
            </div>
        </div>
        '''

    return f'''
<!DOCTYPE html>
<html lang="sw">

<head>

<meta charset="UTF-8">

<meta
    name="viewport"
    content="width=device-width,initial-scale=1.0"
>

<title>{title} - NEERIKA MINE</title>

<script src="https://cdnjs.cloudflare.com/ajax/libs/html2pdf.js/0.10.1/html2pdf.bundle.min.js"></script>

<style>

*{{
    box-sizing:border-box
}}

body{{
    margin:0;
    font-family:Arial,sans-serif;
    background:#f3f4f6;
    color:#111827
}}

.topbar{{
    background:#111827;
    color:white;
    padding:16px 25px;
    display:flex;
    justify-content:space-between;
    align-items:center;
    flex-wrap:wrap
}}

.brand{{
    font-size:22px;
    font-weight:bold;
    color:#fbbf24
}}

.user-info{{
    display:flex;
    gap:12px;
    align-items:center;
    flex-wrap:wrap
}}

.role{{
    background:#374151;
    padding:5px 10px;
    border-radius:6px;
    font-size:13px
}}

.logout{{
    color:white;
    text-decoration:none;
    background:#dc2626;
    padding:7px 12px;
    border-radius:6px
}}

.container{{
    max-width:1300px;
    margin:auto;
    padding:25px
}}

h2{{
    margin-top:30px
}}

.card-grid{{
    display:grid;
    grid-template-columns:
        repeat(auto-fit,minmax(210px,1fr));
    gap:18px;
    margin:20px 0
}}

.card,.chart,.filter{{
    background:white;
    padding:20px;
    border-radius:12px;
    box-shadow:0 3px 10px rgba(0,0,0,.08)
}}

.card h3{{
    margin-top:0;
    font-size:15px;
    color:#4b5563
}}

.number{{
    font-size:27px;
    font-weight:bold
}}

.gold{{
    border-left:5px solid #fbbf24
}}

.blue{{
    border-left:5px solid #2563eb
}}

.green{{
    border-left:5px solid #16a34a
}}

.red{{
    border-left:5px solid #dc2626
}}

.purple{{
    border-left:5px solid #7c3aed
}}

.orange{{
    border-left:5px solid #f97316
}}

.actions{{
    display:grid;
    grid-template-columns:
        repeat(auto-fit,minmax(210px,1fr));
    gap:15px;
    margin:20px 0
}}

.action{{
    display:block;
    padding:18px;
    background:#111827;
    color:white;
    text-decoration:none;
    border-radius:10px;
    text-align:center;
    font-weight:bold
}}

form{{
    background:white;
    padding:25px;
    border-radius:12px;
    box-shadow:0 3px 10px rgba(0,0,0,.08)
}}

label{{
    display:block;
    margin-top:15px;
    margin-bottom:6px;
    font-weight:bold
}}

input,select,textarea{{
    width:100%;
    padding:11px;
    border:1px solid #d1d5db;
    border-radius:7px;
    font-size:15px
}}

textarea{{
    min-height:90px
}}

button,.btn-download{{
    margin-top:20px;
    background:#f59e0b;
    border:0;
    padding:12px 20px;
    border-radius:7px;
    font-weight:bold;
    cursor:pointer;
    color:#111827;
    display:inline-block;
    text-decoration:none
}}

.btn-download{{
    background:#2563eb;
    color:white;
    margin-right:10px
}}

.btn-print{{
    background:#16a34a;
    color:white
}}

.table-container{{
    overflow-x:auto;
    background:white;
    border-radius:10px;
    padding:10px
}}

table{{
    width:100%;
    border-collapse:collapse;
    min-width:1000px
}}

th,td{{
    padding:11px;
    border-bottom:1px solid #e5e7eb;
    text-align:left
}}

th{{
    background:#111827;
    color:white
}}

.edit-btn,.delete-btn{{
    color:white;
    padding:7px 10px;
    border-radius:5px;
    text-decoration:none;
    display:inline-block;
    margin:2px
}}

.edit-btn{{
    background:#2563eb
}}

.delete-btn{{
    background:#dc2626
}}

.alert{{
    background:#fee2e2;
    color:#991b1b;
    padding:14px;
    border-radius:8px;
    margin:10px 0
}}

.success{{
    background:#dcfce7;
    color:#166534;
    padding:14px;
    border-radius:8px;
    margin:10px 0
}}

.warning{{
    background:#fef3c7;
    color:#92400e;
    padding:14px;
    border-radius:8px;
    margin:10px 0
}}

.shift-card{{
    background:white;
    padding:18px;
    border-radius:10px;
    border-left:5px solid #2563eb;
    margin-bottom:12px
}}

.small{{
    color:#6b7280;
    font-size:13px
}}

.footer{{
    margin-top:40px;
    padding:25px;
    background:#111827;
    color:white;
    text-align:center
}}

.login-box{{
    max-width:420px;
    margin:70px auto
}}

.filter-row{{
    display:grid;
    grid-template-columns:1fr auto;
    gap:12px;
    align-items:end
}}

.filter-row button{{
    margin-top:0
}}

.bar-row{{
    display:grid;
    grid-template-columns:80px 1fr 100px;
    gap:10px;
    align-items:center;
    margin:12px 0
}}

.bar-bg{{
    background:#e5e7eb;
    height:25px;
    border-radius:20px;
    overflow:hidden
}}

.bar{{
    height:25px;
    border-radius:20px
}}

.prod{{
    background:#16a34a
}}

.drill{{
    background:#2563eb
}}

.cost{{
    background:#f59e0b
}}

.status{{
    display:inline-block;
    padding:12px 15px;
    border-radius:8px;
    font-weight:bold
}}

.good{{
    background:#dcfce7;
    color:#166534
}}

.attention{{
    background:#fef3c7;
    color:#92400e
}}

.danger{{
    background:#fee2e2;
    color:#991b1b
}}

.shift-badge{{
    display:inline-block;
    padding:5px 9px;
    border-radius:5px;
    font-size:12px;
    font-weight:bold;
    color:white
}}

.day{{
    background:#2563eb
}}

.night{{
    background:#7c3aed
}}

.report-table{{
    width:100%;
    border-collapse:collapse;
    margin-top:15px
}}

.report-table th,
.report-table td{{
    border:1px solid #d1d5db;
    padding:9px
}}

.report-table th{{
    background:#111827;
    color:white
}}

@media print {{

    .topbar,
    .filter,
    .no-print,
    button,
    .btn-download,
    .btn-print,
    .actions{{
        display:none !important
    }}

    body{{
        background:white;
        color:black
    }}

    .container{{
        max-width:100%;
        padding:0
    }}

    .card,
    .chart{{
        box-shadow:none;
        border:1px solid #ddd;
        page-break-inside:avoid
    }}
}}

</style>

</head>

<body>

{nav}

<div class="container">

{content}

</div>

<div class="footer">

<strong>NEERIKA MINE</strong>

<br>

<span
    data-sw="Geology & Mining Services"
    data-en="Geology & Mining Services"
>
Geology & Mining Services
</span>

<br>

© 2026 Haki Zote Zimehifadhiwa / All Rights Reserved

</div>

<script>

function changeLanguage(lang) {{
    localStorage.setItem('neerika_lang', lang);
    applyLanguage(lang);
}}

function applyLanguage(lang) {{

    document
        .querySelectorAll('[data-sw]')
        .forEach(el => {{

            if (el.hasAttribute('data-' + lang)) {{
                el.textContent =
                    el.getAttribute('data-' + lang);
            }}

        }});

    const sel =
        document.getElementById('lang-selector');

    if (sel) {{
        sel.value = lang;
    }}
}}

function exportReportPDF(elementId, fileName) {{

    const element =
        document.getElementById(elementId);

    const opt = {{

        margin:
            [0.5, 0.5, 0.5, 0.5],

        filename:
            fileName + '.pdf',

        image: {{
            type: 'jpeg',
            quality: 0.98
        }},

        html2canvas: {{
            scale: 2,
            useCORS: true
        }},

        jsPDF: {{
            unit: 'in',
            format: 'letter',
            orientation: 'portrait'
        }}

    }};

    html2pdf()
        .set(opt)
        .from(element)
        .save();
}}

document.addEventListener(
    'DOMContentLoaded',
    () => {{

        const savedLang =
            localStorage.getItem(
                'neerika_lang'
            ) || 'sw';

        applyLanguage(savedLang);

    }}
);

</script>

</body>

</html>
'''


# =========================================================
# MAIN WEBSITE
# =========================================================

class MyWebsite(BaseHTTPRequestHandler):

    # -----------------------------------------------------
    # USER SESSION
    # -----------------------------------------------------

    def user(self):

        ck = self.headers.get('Cookie', '')
        sid = None

        for p in ck.split(';'):

            if p.strip().startswith('session_id='):
                sid = p.strip().split('=', 1)[1]

        if not sid:
            return None

        c = get_db()

        u = c.execute('''
            SELECT
                users.id,
                users.username,
                users.role
            FROM sessions
            JOIN users
                ON users.id=sessions.user_id
            WHERE sessions.session_id=?
        ''', (sid,)).fetchone()

        c.close()

        return u

    # -----------------------------------------------------
    # SEND HTML
    # -----------------------------------------------------

    def send_html(self, h, status=200):

        b = h.encode()

        self.send_response(status)

        self.send_header(
            'Content-Type',
            'text/html; charset=utf-8'
        )

        self.send_header(
            'Content-Length',
            str(len(b))
        )

        self.end_headers()

        self.wfile.write(b)

    # -----------------------------------------------------

    def redirect(self, x):

        self.send_response(302)

        self.send_header(
            'Location',
            x
        )

        self.end_headers()

    # -----------------------------------------------------

    def q(self):

        return parse_qs(
            self.path.split('?', 1)[1]
            if '?' in self.path
            else ''
        )

    # -----------------------------------------------------

    def gid(self):

        return self.q().get(
            'id',
            [None]
        )[0]

    # -----------------------------------------------------

    def selected_date(self):

        d = self.q().get(
            'date',
            [str(date.today())]
        )[0]

        try:

            datetime.strptime(
                d,
                '%Y-%m-%d'
            )

            return d

        except:

            return str(date.today())

    # -----------------------------------------------------

    def deny(self, u):

        self.send_html(
            page(
                'Access Denied',
                '''
                <div class="alert"
                    data-sw="Huna ruhusa ya kufanya operation hii."
                    data-en="You do not have permission to perform this operation.">
                    Huna ruhusa ya kufanya operation hii.
                </div>

                <a
                    href="/dashboard"
                    data-sw="← Rudi Dashboard"
                    data-en="← Back to Dashboard"
                >
                    ← Rudi Dashboard
                </a>
                ''',
                u
            ),
            403
        )

    # =====================================================
    # GET
    # =====================================================

    def do_GET(self):

        p = self.path.split('?')[0]

        # LOGIN
        if p == '/login':

            self.send_html(
                page(
                    'Login',
                    '''
                    <div class="login-box">

                        <div class="card">

                            <h1>NEERIKA MINE</h1>

                            <p
                                data-sw="Mining Management System"
                                data-en="Mining Management System"
                            >
                                Mining Management System
                            </p>

                            <form
                                method="POST"
                                action="/login"
                            >

                                <label
                                    data-sw="Username"
                                    data-en="Username"
                                >
                                    Username
                                </label>

                                <input
                                    name="username"
                                    required
                                >

                                <label
                                    data-sw="Password"
                                    data-en="Password"
                                >
                                    Password
                                </label>

                                <input
                                    type="password"
                                    name="password"
                                    required
                                >

                                <button
                                    data-sw="Login"
                                    data-en="Login"
                                >
                                    Login
                                </button>

                            </form>

                        </div>

                    </div>
                    '''
                )
            )

            return

        u = self.user()

        if not u:

            self.redirect('/login')

            return

        # -------------------------------------------------
        # ROUTES
        # -------------------------------------------------

        routes = {

            '/add_drilling':
                ('GeologistManager',
                 self.add_drilling_form),

            '/edit_drilling':
                ('GeologistManager',
                 self.edit_drilling_form),

            '/delete_drilling':
                ('GeologistManager',
                 self.delete_drilling),

            '/add_production':
                ('GeologistManager',
                 self.add_production_form),

            '/edit_production':
                ('GeologistManager',
                 self.edit_production_form),

            '/delete_production':
                ('GeologistManager',
                 self.delete_production),

            '/add_shift':
                ('GeologistManager',
                 self.add_shift_form),

            '/edit_shift':
                ('GeologistManager',
                 self.edit_shift_form),

            '/delete_shift':
                ('GeologistManager',
                 self.delete_shift),

            '/add_cost':
                ('GeologistManager',
                 self.add_cost_form),

            '/edit_cost':
                ('GeologistManager',
                 self.edit_cost_form),

            '/delete_cost':
                ('GeologistManager',
                 self.delete_cost)
        }

        if p == '/dashboard':

            self.dashboard(u)

            return

        if p == '/shift_leaders':

            if u[2] not in [
                'Manager',
                'Director',
                'Geologist'
            ]:

                self.deny(u)

            else:

                self.shift_leaders_form(u)

            return

        if p == '/weekly_report':

            if u[2] not in [
                'Manager',
                'Director'
            ]:

                self.deny(u)

            else:

                self.weekly_report(u)

            return

        if p == '/monthly_report':

            if u[2] not in [
                'Manager',
                'Director'
            ]:

                self.deny(u)

            else:

                self.monthly_report(u)

            return

        if p == '/logout':

            self.logout()

            return

        if p in routes:

            roles, fn = routes[p]

            allowed = (
                ['Geologist', 'Manager']
                if roles == 'GeologistManager'
                else [roles]
            )

            if u[2] not in allowed:

                self.deny(u)

            else:

                fn(u)

            return

        self.redirect('/dashboard')

    # =====================================================
    # POST
    # =====================================================

    def do_POST(self):

        p = self.path.split('?')[0]

        n = int(
            self.headers.get(
                'Content-Length',
                0
            )
        )

        d = parse_qs(
            self.rfile.read(n).decode()
        )

        # LOGIN
        if p == '/login':

            c = get_db()

            u = c.execute(
                '''
                SELECT
                    id,
                    username,
                    role
                FROM users
                WHERE
                    LOWER(username)=LOWER(?)
                    AND password=?
                ''',
                (
                    d.get(
                        'username',
                        ['']
                    )[0].strip(),

                    hash_password(
                        d.get(
                            'password',
                            ['']
                        )[0]
                    )
                )
            ).fetchone()

            if not u:

                c.close()

                self.send_html(
                    page(
                        'Login Error',
                        '''
                        <div class="alert"
                            data-sw="Username au password sio sahihi."
                            data-en="Invalid username or password.">
                            Username au password sio sahihi.
                        </div>
                        '''
                    )
                )

                return

            sid = secrets.token_hex(32)

            c.execute(
                'INSERT INTO sessions VALUES(?,?)',
                (
                    sid,
                    u[0]
                )
            )

            c.commit()

            c.close()

            self.send_response(302)

            self.send_header(
                'Location',
                '/dashboard'
            )

            self.send_header(
                'Set-Cookie',
                f'session_id={sid}; HttpOnly; SameSite=Lax; Path=/'
            )

            self.end_headers()

            return

        u = self.user()

        if not u:

            self.redirect('/login')

            return

        handlers = {

            '/add_drilling':
                ('GeologistManager',
                 self.save_drilling),

            '/edit_drilling':
                ('GeologistManager',
                 self.update_drilling),

            '/add_production':
                ('GeologistManager',
                 self.save_production),

            '/edit_production':
                ('GeologistManager',
                 self.update_production),

            '/add_shift':
                ('GeologistManager',
                 self.save_shift),

            '/edit_shift':
                ('GeologistManager',
                 self.update_shift),

            '/shift_leaders':
                ('Manager',
                 self.save_shift_leaders),

            '/add_cost':
                ('GeologistManager',
                 self.save_cost),

            '/edit_cost':
                ('GeologistManager',
                 self.update_cost)
        }

        if p in handlers:

            roles, fn = handlers[p]

            allowed = (
                ['Geologist', 'Manager']
                if roles == 'GeologistManager'
                else [roles]
            )

            if u[2] not in allowed:

                self.deny(u)

                return

            fn(d, u)

    # =====================================================
    # DASHBOARD
    # =====================================================

    def dashboard(self, u):

        sd = self.selected_date()

        c = get_db()
        x = c.cursor()

        # -------------------------------------------------
        # TOTAL DRILLING FOR SELECTED DATE
        # -------------------------------------------------

        x.execute('''
            SELECT
                COALESCE(SUM(drilled_holes),0),
                COALESCE(SUM(charged_holes),0),
                COALESCE(SUM(nonels_used),0),
                COALESCE(SUM(buckets_used),0),
                COALESCE(SUM(fuel_used),0),
                COALESCE(SUM(total_length),0),
                COALESCE(SUM(bits_used),0),
                COALESCE(SUM(cortex_wire_used),0),
                COALESCE(SUM(fuses_used),0)
            FROM drilling_data
            WHERE report_date=?
        ''', (sd,))

        (
            drilled,
            charged,
            nonels,
            old_buckets,
            fuel,
            length,
            bits,
            cortex,
            fuses
        ) = x.fetchone()

        # -------------------------------------------------
        # DAY SHIFT TOTAL
        # -------------------------------------------------

        x.execute('''
            SELECT
                COALESCE(SUM(drilled_holes),0),
                COALESCE(SUM(charged_holes),0),
                COALESCE(SUM(nonels_used),0),
                COALESCE(SUM(bits_used),0),
                COALESCE(SUM(cortex_wire_used),0),
                COALESCE(SUM(fuses_used),0),
                COALESCE(SUM(fuel_used),0),
                COALESCE(SUM(total_length),0)
            FROM drilling_data
            WHERE report_date=?
            AND shift='Day'
        ''', (sd,))

        (
            day_drilled,
            day_charged,
            day_nonels,
            day_bits,
            day_cortex,
            day_fuses,
            day_fuel,
            day_length
        ) = x.fetchone()

        # -------------------------------------------------
        # NIGHT SHIFT TOTAL
        # -------------------------------------------------

        x.execute('''
            SELECT
                COALESCE(SUM(drilled_holes),0),
                COALESCE(SUM(charged_holes),0),
                COALESCE(SUM(nonels_used),0),
                COALESCE(SUM(bits_used),0),
                COALESCE(SUM(cortex_wire_used),0),
                COALESCE(SUM(fuses_used),0),
                COALESCE(SUM(fuel_used),0),
                COALESCE(SUM(total_length),0)
            FROM drilling_data
            WHERE report_date=?
            AND shift='Night'
        ''', (sd,))

        (
            night_drilled,
            night_charged,
            night_nonels,
            night_bits,
            night_cortex,
            night_fuses,
            night_fuel,
            night_length
        ) = x.fetchone()

        # -------------------------------------------------
        # PRODUCTION
        # -------------------------------------------------

        x.execute('''
            SELECT COALESCE(SUM(buckets),0)
            FROM production_data
            WHERE production_date=?
        ''', (sd,))

        prod = x.fetchone()[0]

        x.execute('''
            SELECT COALESCE(SUM(buckets),0)
            FROM production_data
        ''')

        cumprod = x.fetchone()[0]

        # -------------------------------------------------
        # COST
        # -------------------------------------------------

        x.execute('''
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
            WHERE cost_date=?
        ''', (sd,))

        eqcost, othercost, totalcost = x.fetchone()

        x.execute('''
            SELECT COALESCE(SUM(amount),0)
            FROM costs
        ''')

        cumcost = x.fetchone()[0]

        # -------------------------------------------------
        # CUMULATIVE DRILLING
        # -------------------------------------------------

        x.execute('''
            SELECT
                COALESCE(SUM(drilled_holes),0),
                COALESCE(SUM(charged_holes),0),
                COALESCE(SUM(nonels_used),0),
                COALESCE(SUM(fuel_used),0),
                COALESCE(SUM(total_length),0)
            FROM drilling_data
        ''')

        cd, cc, cn, cf, cl = x.fetchone()

        # -------------------------------------------------
        # SHIFTS
        # -------------------------------------------------

        x.execute('''
            SELECT
                id,
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
            WHERE shift_date=?
            ORDER BY id DESC
        ''', (sd,))

        shifts = x.fetchall()

        # -------------------------------------------------
        # 7 DAY TREND
        # -------------------------------------------------

        trend = []

        dt = datetime.strptime(
            sd,
            '%Y-%m-%d'
        ).date()

        for i in range(6, -1, -1):

            d = str(
                dt - timedelta(days=i)
            )

            x.execute('''
                SELECT COALESCE(SUM(buckets),0)
                FROM production_data
                WHERE production_date=?
            ''', (d,))

            pp = x.fetchone()[0]

            x.execute('''
                SELECT
                    COALESCE(SUM(drilled_holes),0),
                    COALESCE(SUM(charged_holes),0),
                    COALESCE(SUM(total_length),0)
                FROM drilling_data
                WHERE report_date=?
            ''', (d,))

            rr = x.fetchone()

            x.execute('''
                SELECT COALESCE(SUM(amount),0)
                FROM costs
                WHERE cost_date=?
            ''', (d,))

            co = x.fetchone()[0]

            trend.append(
                (
                    d[5:],
                    pp,
                    rr[0],
                    rr[1],
                    rr[2],
                    co
                )
            )

        # -------------------------------------------------
        # HISTORY
        # -------------------------------------------------

        ph = x.execute('''
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
        ''').fetchall()

        dh = x.execute('''
            SELECT
                id,
                report_date,
                shift,
                drilled_holes,
                charged_holes,
                nonels_used,
                bits_used,
                cortex_wire_used,
                fuses_used,
                buckets_used,
                fuel_used,
                total_length,
                username
            FROM drilling_data
            ORDER BY id DESC
            LIMIT 30
        ''').fetchall()

        ch = x.execute('''
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
        ''').fetchall()

        sh = x.execute('''
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
        ''').fetchall()

        c.close()

        # -------------------------------------------------
        # CHARTS
        # -------------------------------------------------

        maxp = max(
            [r[1] for r in trend] + [1]
        )

        maxd = max(
            [r[2] for r in trend] + [1]
        )

        maxc = max(
            [r[5] for r in trend] + [1]
        )

        prodchart = ''.join(
            f'''
            <div class="bar-row">
                <span>{r[0]}</span>

                <div class="bar-bg">
                    <div
                        class="bar prod"
                        style="width:{int(r[1]/maxp*100)}%"
                    ></div>
                </div>

                <strong>{r[1]:,}</strong>
            </div>
            '''
            for r in trend
        )

        drillchart = ''.join(
            f'''
            <div class="bar-row">
                <span>{r[0]}</span>

                <div class="bar-bg">
                    <div
                        class="bar drill"
                        style="width:{int(r[2]/maxd*100)}%"
                    ></div>
                </div>

                <strong>{r[2]:,}</strong>
            </div>
            '''
            for r in trend
        )

        costchart = ''.join(
            f'''
            <div class="bar-row">
                <span>{r[0]}</span>

                <div class="bar-bg">
                    <div
                        class="bar cost"
                        style="width:{int(r[5]/maxc*100)}%"
                    ></div>
                </div>

                <strong>
                    TSh {r[5]:,.0f}
                </strong>
            </div>
            '''
            for r in trend
        )

        # -------------------------------------------------
        # STATUS
        # -------------------------------------------------

        status = (
            'GOOD PERFORMANCE'
            if prod > 0 and drilled > 0
            else (
                'NEEDS ATTENTION'
                if prod > 0 or drilled > 0
                else 'NO ACTIVITY RECORDED'
            )
        )

        sc = (
            'good'
            if status.startswith('GOOD')
            else (
                'attention'
                if status.startswith('NEEDS')
                else 'danger'
            )
        )

        alerts = []

        if charged > drilled:

            alerts.append(
                f'🚨 Charged holes ({charged:,}) '
                f'ni nyingi kuliko drilled holes '
                f'({drilled:,}).'
            )

        if prod == 0:

            alerts.append(
                f'⚠️ Hakuna production iliyorekodiwa '
                f'tarehe {sd}.'
            )

        if drilled == 0:

            alerts.append(
                f'⚠️ Hakuna drilling report '
                f'iliyorekodiwa tarehe {sd}.'
            )

        if not shifts:

            alerts.append(
                f'⚠️ Hakuna shift iliyorekodiwa '
                f'tarehe {sd}.'
            )

        if prod > 0 and totalcost / prod > 10000:

            alerts.append(
                f'⚠️ Cost per bucket ni '
                f'TSh {totalcost/prod:,.2f}.'
            )

        # =================================================
        # PAGE CONTENT
        # =================================================

        content = f'''
        <h1>📊 NEERIKA MINE Dashboard</h1>

        <p>
            <span
                data-sw="Karibu"
                data-en="Welcome"
            >
                Karibu
            </span>

            <strong>{u[1]}</strong>
            — {u[2]}
        </p>

        <div class="filter">

            <form
                method="GET"
                action="/dashboard"
            >

                <div class="filter-row">

                    <div>

                        <label
                            style="margin-top:0"
                            data-sw="📅 Chagua Tarehe"
                            data-en="📅 Select Date"
                        >
                            📅 Chagua Tarehe
                        </label>

                        <input
                            type="date"
                            name="date"
                            value="{sd}"
                            required
                        >

                    </div>

                    <button
                        data-sw="🔎 Angalia"
                        data-en="🔎 Filter"
                    >
                        🔎 Angalia
                    </button>

                </div>

            </form>

        </div>
        '''

        # -------------------------------------------------
        # QUICK ACTIONS
        # -------------------------------------------------

        if u[2] in ['Geologist', 'Manager']:

            content += '''
            <h2
                data-sw="⚡ Quick Actions (Jaza Report)"
                data-en="⚡ Quick Actions (Data Entry)"
            >
                ⚡ Quick Actions (Jaza Report)
            </h2>

            <div class="actions">

                <a
                    class="action"
                    href="/add_drilling"
                    data-sw="🕳️ Weka Drilling Report"
                    data-en="🕳️ Add Drilling Report"
                >
                    🕳️ Weka Drilling Report
                </a>

                <a
                    class="action"
                    href="/add_production"
                    data-sw="🪣 Weka Production"
                    data-en="🪣 Add Production"
                >
                    🪣 Weka Production
                </a>

                <a
                    class="action"
                    href="/add_shift"
                    data-sw="👷 Weka Shift"
                    data-en="👷 Add Shift"
                >
                    👷 Weka Shift
                </a>

                <a
                    class="action"
                    href="/add_cost"
                    data-sw="💰 Weka Gharama"
                    data-en="💰 Add Cost"
                >
                    💰 Weka Gharama
                </a>

            </div>
            '''

        # -------------------------------------------------
        # MANAGEMENT
        # -------------------------------------------------

        if u[2] in ['Manager', 'Director']:

            content += '''
            <h2
                data-sw="⚙️ Management Controls"
                data-en="⚙️ Management Controls"
            >
                ⚙️ Management Controls
            </h2>

            <div class="actions">

                <a
                    class="action"
                    href="/shift_leaders"
                    data-sw="⚙️ Edit Shift Leaders"
                    data-en="⚙️ Edit Shift Leaders"
                >
                    ⚙️ Edit Shift Leaders
                </a>

                <a
                    class="action"
                    href="/weekly_report"
                    data-sw="📆 Weekly Report"
                    data-en="📆 Weekly Report"
                >
                    📆 Weekly Report
                </a>

                <a
                    class="action"
                    href="/monthly_report"
                    data-sw="📅 Monthly Report"
                    data-en="📅 Monthly Report"
                >
                    📅 Monthly Report
                </a>

            </div>
            '''

        # -------------------------------------------------
        # PERFORMANCE STATUS
        # -------------------------------------------------

        content += f'''
        <h2
            data-sw="🚦 Performance Status"
            data-en="🚦 Performance Status"
        >
            🚦 Performance Status
        </h2>

        <div class="card">

            <span class="status {sc}">
                {status}
            </span>

            <p class="small">
                Tarehe {sd}
                |
                Production {prod:,} buckets
                |
                Drilled {drilled:,} holes
                |
                Cost TSh {totalcost:,.2f}
            </p>

        </div>


        <h2
            data-sw="🪣 Production Performance"
            data-en="🪣 Production Performance"
        >
            🪣 Production Performance
        </h2>

        <div class="card-grid">

            <div class="card green">
                <h3>Material Produced</h3>
                <div class="number">{prod:,}</div>
                <div class="small">BUCKETS</div>
            </div>

            <div class="card blue">
                <h3>Cumulative Production</h3>
                <div class="number">{cumprod:,}</div>
                <div class="small">BUCKETS</div>
            </div>

            <div class="card gold">
                <h3>7-Day Production</h3>
                <div class="number">
                    {sum(r[1] for r in trend):,}
                </div>
                <div class="small">BUCKETS</div>
            </div>

        </div>

        <div class="chart">

            <h3>
                📈 Production Trend — 7 Days
            </h3>

            {prodchart}

        </div>


        <h2>
            🕳️ Drilling Performance
        </h2>

        <div class="card-grid">

            <div class="card blue">
                <h3>Drilled</h3>
                <div class="number">
                    {drilled:,}
                </div>
                <div class="small">HOLES</div>
            </div>

            <div class="card red">
                <h3>Charged</h3>
                <div class="number">
                    {charged:,}
                </div>
                <div class="small">HOLES</div>
            </div>

            <div class="card purple">
                <h3>Nonels</h3>
                <div class="number">
                    {nonels:,}
                </div>
                <div class="small">PCS</div>
            </div>

            <div class="card orange">
                <h3>Bits Used</h3>
                <div class="number">
                    {bits:,}
                </div>
                <div class="small">PCS</div>
            </div>

            <div class="card purple">
                <h3>Cortex Wire</h3>
                <div class="number">
                    {cortex:,.2f}
                </div>
                <div class="small">METERS</div>
            </div>

            <div class="card red">
                <h3>Fuse Used</h3>
                <div class="number">
                    {fuses:,}
                </div>
                <div class="small">PCS</div>
            </div>

            <div class="card blue">
                <h3>Drilling Length</h3>
                <div class="number">
                    {length:,.2f}
                </div>
                <div class="small">FT</div>
            </div>

            <div class="card gold">
                <h3>Fuel</h3>
                <div class="number">
                    {fuel:,.2f}
                </div>
                <div class="small">L</div>
            </div>

        </div>


        <h2>
            ☀️ Day Shift / 🌙 Night Shift
        </h2>

        <div class="card-grid">

            <div class="card blue">

                <h3>☀️ Day — Drilled</h3>

                <div class="number">
                    {day_drilled:,}
                </div>

                <div class="small">
                    HOLES
                </div>

            </div>

            <div class="card purple">

                <h3>🌙 Night — Drilled</h3>

                <div class="number">
                    {night_drilled:,}
                </div>

                <div class="small">
                    HOLES
                </div>

            </div>

            <div class="card orange">

                <h3>☀️ Day — Bits</h3>

                <div class="number">
                    {day_bits:,}
                </div>

                <div class="small">
                    PCS
                </div>

            </div>

            <div class="card orange">

                <h3>🌙 Night — Bits</h3>

                <div class="number">
                    {night_bits:,}
                </div>

                <div class="small">
                    PCS
                </div>

            </div>

            <div class="card purple">

                <h3>☀️ Day — Cortex</h3>

                <div class="number">
                    {day_cortex:,.2f}
                </div>

                <div class="small">
                    METERS
                </div>

            </div>

            <div class="card purple">

                <h3>🌙 Night — Cortex</h3>

                <div class="number">
                    {night_cortex:,.2f}
                </div>

                <div class="small">
                    METERS
                </div>

            </div>

            <div class="card red">

                <h3>☀️ Day — Fuse</h3>

                <div class="number">
                    {day_fuses:,}
                </div>

                <div class="small">
                    PCS
                </div>

            </div>

            <div class="card red">

                <h3>🌙 Night — Fuse</h3>

                <div class="number">
                    {night_fuses:,}
                </div>

                <div class="small">
                    PCS
                </div>

            </div>

        </div>


        <div class="chart">

            <h3>
                🕳️ Drilling Performance Trend — 7 Days
            </h3>

            {drillchart}

        </div>


        <h2>
            💰 Cost Analysis
        </h2>

        <div class="card-grid">

            <div class="card gold">
                <h3>Equipment Cost</h3>
                <div class="number">
                    TSh {eqcost:,.2f}
                </div>
            </div>

            <div class="card red">
                <h3>Other Cost</h3>
                <div class="number">
                    TSh {othercost:,.2f}
                </div>
            </div>

            <div class="card blue">
                <h3>Total Cost</h3>
                <div class="number">
                    TSh {totalcost:,.2f}
                </div>
            </div>

            <div class="card purple">
                <h3>Cumulative Cost</h3>
                <div class="number">
                    TSh {cumcost:,.2f}
                </div>
            </div>

        </div>

        <div class="chart">

            <h3>
                💰 Cost Trend — 7 Days
            </h3>

            {costchart}

        </div>


        <h2>
            🚨 Issues / Alerts
        </h2>
        '''

        if alerts:

            content += ''.join(
                f'<div class="alert">{a}</div>'
                for a in alerts
            )

        else:

            content += '''
            <div class="success">
                ✅ Hakuna issue kubwa
                iliyogunduliwa kwenye data
                ya tarehe hii.
            </div>
            '''

        # -------------------------------------------------
        # SHIFTS
        # -------------------------------------------------

        content += '''
        <h2>
            👷 Shift ya Tarehe Iliyochaguliwa
        </h2>
        '''

        if shifts:

            for r in shifts:

                b = (
                    f'''
                    <a
                        href="/edit_shift?id={r[0]}"
                        class="edit-btn"
                    >
                        ✏️ Edit
                    </a>

                    <a
                        href="/delete_shift?id={r[0]}"
                        class="delete-btn"
                        onclick="return confirm('Una uhakika unataka kufuta shift hii?')"
                    >
                        🗑️ Delete
                    </a>
                    '''
                    if u[2] in [
                        'Geologist',
                        'Manager'
                    ]
                    else ''
                )

                content += f'''
                <div class="shift-card">

                    <h3>
                        {r[1]} — {r[2]}
                    </h3>

                    <p>
                        <b>Drillers:</b>
                        {r[3]}
                    </p>

                    <p>
                        <b>Blasters:</b>
                        {r[4]}
                    </p>

                    <p>
                        <b>Muda:</b>
                        {r[5]} - {r[6]}
                    </p>

                    <p>
                        <b>Location:</b>
                        {r[7]}
                    </p>

                    <p>
                        <b>Remarks:</b>
                        {r[8] or "-"}
                    </p>

                    {b}

                </div>
                '''

        else:

            content += '''
            <div class="card">
                Hakuna shift iliyorekodiwa
                kwenye tarehe hii.
            </div>
            '''

        # =================================================
        # PRODUCTION HISTORY
        # =================================================

        content += '''
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
            <th>Maelezo</th>
            <th>Aliyeweka</th>
            <th>Action</th>
        </tr>
        '''

        for r in ph:

            b = (
                f'''
                <a
                    href="/edit_production?id={r[0]}"
                    class="edit-btn"
                >
                    ✏️ Edit
                </a>

                <a
                    href="/delete_production?id={r[0]}"
                    class="delete-btn"
                    onclick="return confirm('Una uhakika unataka kufuta production hii?')"
                >
                    🗑️ Delete
                </a>
                '''
                if u[2] in [
                    'Geologist',
                    'Manager'
                ]
                else ''
            )

            content += f'''
            <tr>

                <td>{r[1]}</td>
                <td>{r[2]}</td>
                <td>{r[3]:,}</td>
                <td>{r[4]}</td>
                <td>{r[5] or "-"}</td>
                <td>{r[6]}</td>
                <td>{b}</td>

            </tr>
            '''

        content += '''
        </table>

        </div>
        '''

        # =================================================
        # DRILLING HISTORY
        # =================================================

        content += '''
        <h2>
            📋 Drilling History
        </h2>

        <div class="table-container">

        <table>

        <tr>

            <th>Tarehe</th>
            <th>Shift</th>
            <th>Drilled</th>
            <th>Charged</th>
            <th>Nonels</th>
            <th>Bits</th>
            <th>Cortex Wire</th>
            <th>Fuse</th>
            <th>Buckets</th>
            <th>Fuel</th>
            <th>Length FT</th>
            <th>Aliyeweka</th>
            <th>Action</th>

        </tr>
        '''

        for r in dh:

            shift_class = (
                'day'
                if r[2] == 'Day'
                else 'night'
            )

            b = (
                f'''
                <a
                    href="/edit_drilling?id={r[0]}"
                    class="edit-btn"
                >
                    ✏️ Edit
                </a>

                <a
                    href="/delete_drilling?id={r[0]}"
                    class="delete-btn"
                    onclick="return confirm('Una uhakika unataka kufuta report hii?')"
                >
                    🗑️ Delete
                </a>
                '''
                if u[2] in [
                    'Geologist',
                    'Manager'
                ]
                else ''
            )

            content += f'''
            <tr>

                <td>{r[1]}</td>

                <td>
                    <span class="shift-badge {shift_class}">
                        {r[2]}
                    </span>
                </td>

                <td>{r[3]:,}</td>

                <td>{r[4]:,}</td>

                <td>{r[5]:,}</td>

                <td>
                    {r[6]:,} PCS
                </td>

                <td>
                    {r[7]:,.2f} M
                </td>

                <td>
                    {r[8]:,} PCS
                </td>

                <td>
                    {r[9]:,}
                </td>

                <td>
                    {r[10]:,.2f} L
                </td>

                <td>
                    {r[11]:,.2f}
                </td>

                <td>{r[12]}</td>

                <td>{b}</td>

            </tr>
            '''

        content += '''
        </table>

        </div>
        '''

                # =================================================
        # COST HISTORY
        # =================================================

        content += '''
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
        '''

        for r in ch:

            b = (
                f'''
                <a
                    href="/edit_cost?id={r[0]}"
                    class="edit-btn"
                >
                    ✏️ Edit
                </a>

                <a
                    href="/delete_cost?id={r[0]}"
                    class="delete-btn"
                    onclick="return confirm('Una uhakika unataka kufuta gharama hii?')"
                >
                    🗑️ Delete
                </a>
                '''
                if u[2] in [
                    'Geologist',
                    'Manager'
                ]
                else ''
            )

            content += f'''
            <tr>

                <td>{r[1]}</td>

                <td>
                    {r[2]}
                </td>

                <td>
                    {r[3]}
                </td>

                <td>
                    {r[4]:,.2f}
                </td>

                <td>
                    TSh {r[5]:,.2f}
                </td>

                <td>
                    {r[6] or "-"}
                </td>

                <td>
                    {r[7]}
                </td>

                <td>
                    {b}
                </td>

            </tr>
            '''

        content += '''
        </table>

        </div>
        '''

        # =================================================
        # FINAL DASHBOARD RESPONSE
        # =================================================

        content += '''
        <div class="no-print"
             style="margin-top:30px;text-align:center;">

            <button
                class="btn-print"
                onclick="window.print()"
            >
                🖨️ Print Dashboard
            </button>

            <button
                class="btn-download"
                onclick="exportReportPDF(
                    'dashboard-report',
                    'NEERIKA-MINE-Dashboard'
                )"
            >
                📄 Download PDF
            </button>

        </div>
        '''

        final_content = f'''
        <div id="dashboard-report">

            {content}

        </div>
        '''

        self.send_html(
            page(
                'Dashboard',
                final_content,
                u
            )
        )
        # =========================================================
# START NEERIKA MINE
# =========================================================

if __name__ == '__main__':
    init_db()

    port = int(os.environ.get('PORT', '10000'))

    server = ThreadingHTTPServer(
        ('0.0.0.0', port),
        MyWebsite
    )

    print(f'NEERIKA MINE running on port {port}', flush=True)

    server.serve_forever()
