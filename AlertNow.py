from flask import Flask, request, jsonify, render_template, redirect, url_for, session, send_file
from flask_socketio import SocketIO
import logging
import ast
import os
import json
import sqlite3
import joblib
import cv2
import numpy as np
from collections import Counter
from datetime import datetime
from alert_data import alerts
from collections import deque
import logging
import pytz

# Assuming these files are in the same directory as app.py
from BarangayDashboard import get_barangay_stats, get_latest_alert
from CDRRMODashboard import get_cdrrmo_stats
from PNPDashboard import get_pnp_stats
from alert_data import alerts

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Replace with a strong, secret key
socketio = SocketIO(app, cors_allowed_origins="*")
logging.basicConfig(level=logging.DEBUG)


data_dir = os.path.join(os.path.dirname(__file__), 'data')
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
    logging.info(f"Created data directory at {data_dir}")

# Load barangay_coords from coords.txt (assuming it's in an 'assets' folder)
try:
    with open(os.path.join('assets', 'coords.txt'), 'r') as f:
        barangay_coords = ast.literal_eval(f.read())
except FileNotFoundError:
    logging.error("coords.txt not found in assets directory. Using empty dict.")
    barangay_coords = {}
except Exception as e:
    logging.error(f"Error loading coords.txt: {e}. Using empty dict.")
    barangay_coords = {}

# Example municipality coordinates (expand this dictionary with all assigned municipalities)
municipality_coords = {
    "San Pablo City": {"lat": 14.0642, "lon": 121.3233},
    "Quezon Province": {"lat": 13.9347, "lon": 121.9473},
    # Add more municipalities here based on sign-up data
    # Example: "Davao City": {"lat": 7.0731, "lon": 125.6125}
}

# Load Google Maps API key from environment variable or use a default
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY', 'AIzaSyBSXRZPDX1x1d91Ck-pskiwGA8Y2-5gDVs')

@socketio.on('responded')
def handle_responded(data):
    timestamp = data.get('timestamp')
    lat = data.get('lat')
    lon = data.get('lon')
    barangay = data.get('barangay')
    emergency_type = data.get('emergency_type')
    app.logger.debug(f"Received response for alert at {timestamp} - Lat: {lat}, Lon: {lon}, Barangay: {barangay}, Type: {emergency_type}")
    # Add logic to update alert status or notify other clients if needed
    socketio.emit('alert_responded', {
        'timestamp': timestamp,
        'lat': lat,
        'lon': lon,
        'barangay': barangay,
        'emergency_type': emergency_type
    })

# Load ML model
try:
    dt_classifier = joblib.load('decision_tree_model.pkl')
    logging.info("decision_tree_model.pkl loaded successfully.")
except FileNotFoundError:
    logging.error("decision_tree_model.pkl not found. ML prediction will not work.")
    dt_classifier = None
except Exception as e:
    logging.error(f"Error loading decision_tree_model.pkl: {e}")
    dt_classifier = None

def get_db_connection():
    db_path = os.path.join(os.path.dirname(__file__), 'data', 'users_web.db')
    app.logger.debug(f"Database path: {db_path}")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn

def get_db_connection():
    db_path = os.path.join('/data', 'users_web.db')
    if not os.path.exists(db_path):
        db_path = os.path.join(os.path.dirname(__file__), 'data', 'users_web.db')
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn

@app.route('/export_users', methods=['GET'])
def export_users():
    if session.get('role') != 'admin':  # Restrict access
        return "Unauthorized", 403
    conn = get_db_connection()
    users = conn.execute('SELECT * FROM users').fetchall()
    conn.close()
    return jsonify([dict(user) for user in users])

@app.route('/download_db', methods=['GET'])
def download_db():
    db_path = os.path.join('/data', 'users_web.db')
    if not os.path.exists(db_path):
        db_path = os.path.join(os.path.dirname(__file__), 'data', 'users_web.db')
    if not os.path.exists(db_path):
        return "Database file not found", 404
    app.logger.debug(f"Serving database from: {db_path}")
    return send_file(db_path, as_attachment=True, download_name='users_web.db')


def construct_unique_id(role, barangay=None, contact_no=None, assigned_municipality=None):
    """Constructs a unique identifier based on role and relevant details."""
    if role == 'barangay':
        return f"{barangay}_{contact_no}"
    else:  # cdrrmo or pnp
        return f"{role}_{assigned_municipality}_{contact_no}"

# --- Routes ---
@app.route('/')
def home():
    app.logger.debug("Rendering SignUpType.html")
    return render_template('SignUpType.html')

@app.route('/signup_barangay', methods=['GET', 'POST'])
def signup_barangay():
    app.logger.debug("Accessing /signup_barangay with method: %s", request.method)
    if request.method == 'POST':
        app.logger.debug(f"Form data received: {request.form}")
        barangay = request.form['barangay']
        assigned_municipality = request.form['municipality']
        province = request.form['province']
        contact_no = request.form['contact_no']
        password = request.form['password']
        unique_id = construct_unique_id('barangay', barangay=barangay, contact_no=contact_no)
        
        conn = get_db_connection()
        try:
            app.logger.debug("Attempting to insert user data into database")
            cursor = conn.execute('''
                INSERT INTO users (barangay, role, contact_no, assigned_municipality, province, password)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (barangay, 'barangay', contact_no, assigned_municipality, province, password))
            app.logger.debug(f"Rows affected by INSERT: {cursor.rowcount}")
            conn.commit()
            # Verify insertion
            user = conn.execute('SELECT * FROM users WHERE barangay = ? AND contact_no = ?', 
                               (barangay, contact_no)).fetchone()
            if user:
                app.logger.debug(f"User found in database after insertion: {dict(user)}")
            else:
                app.logger.error("User not found in database after insertion")
            app.logger.debug("User data inserted successfully: %s", unique_id)
            return redirect(url_for('login'))
        except sqlite3.IntegrityError as e:
            app.logger.error("IntegrityError: %s", e)
            return "User already exists", 400
        except Exception as e:
            app.logger.error(f"Exception during signup: {e}", exc_info=True)
            return f"Signup failed: {e}", 500
        finally:
            conn.close()
    return render_template('SignUpPage.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    app.logger.debug("Accessing /login with method: %s", request.method)
    if request.method == 'POST':
        barangay = request.form['barangay']
        contact_no = request.form['contact_no']
        password = request.form['password']
        unique_id = construct_unique_id('barangay', barangay=barangay, contact_no=contact_no)
        
        conn = get_db_connection()
        user = conn.execute('''
            SELECT * FROM users WHERE barangay = ? AND contact_no = ? AND password = ?
        ''', (barangay, contact_no, password)).fetchone()
        conn.close()
        
        if user:
            session['unique_id'] = unique_id
            session['role'] = user['role']
            if user['role'] == 'barangay':
                app.logger.debug(f"Web login successful for barangay: {unique_id}")
                return redirect(url_for('barangay_dashboard'))
            else:
                app.logger.warning(f"Web login for /login attempted by non-barangay role: {unique_id} ({user['role']})")
                return "Unauthorized role for this login page", 403
        app.logger.warning(f"Web login failed for unique_id: {unique_id}")
        return "Invalid credentials", 401
    return render_template('LogInPage.html')

@app.route('/api/login', methods=['POST'])
def api_login():
    data = request.get_json()
    barangay = data.get('barangay')
    contact_no = data.get('contact_no')
    password = data.get('password')
    unique_id = construct_unique_id('barangay', barangay=barangay, contact_no=contact_no)
    
    conn = get_db_connection()
    user = conn.execute('''
        SELECT * FROM users WHERE barangay = ? AND contact_no = ? AND password = ?
    ''', (barangay, contact_no, password)).fetchone()
    conn.close()
    
    if user:
        app.logger.debug(f"API login successful for user: {unique_id} with role: {user['role']}")
        return jsonify({'status': 'success', 'role': user['role']})
    app.logger.warning(f"API login failed for unique_id: {unique_id}")
    return jsonify({'error': 'Invalid credentials'}), 401

@app.route('/signup_cdrrmo_pnp', methods=['GET', 'POST'])
def signup_cdrrmo_pnp():
    if request.method == 'POST':
        role = request.form['role'].lower()
        assigned_municipality = request.form['municipality']
        contact_no = request.form['contact_no']
        password = request.form['password']
        unique_id = construct_unique_id(role, assigned_municipality, contact_no)
        
        conn = get_db_connection()
        try:
            # Check for unique contact_no
            existing_user = conn.execute('SELECT * FROM users WHERE contact_no = ?', (contact_no,)).fetchone()
            if existing_user:
                app.logger.error("Signup failed: Contact number %s already exists", contact_no)
                return "Contact number already exists", 400
            
            # Insert user data without province
            conn.execute('''
                INSERT INTO users (role, contact_no, assigned_municipality, password)
                VALUES (?, ?, ?, ?)
            ''', (role, contact_no, assigned_municipality, password))
            conn.commit()
            app.logger.debug("User signed up successfully: %s", unique_id)
            return redirect(url_for('login_cdrrmo_pnp'))
        except sqlite3.IntegrityError as e:
            app.logger.error("IntegrityError during signup: %s", e)
            return "User already exists", 400
        except Exception as e:
            app.logger.error(f"Signup failed for {unique_id}: {e}", exc_info=True)
            return f"Signup failed: {e}", 500
        finally:
            conn.close()
    return render_template('CDRRMOPNPUp.html')

# --- Web App Navigation Routes ---
@app.route('/go_to_login_page', methods=['GET'])
def go_to_login_page():
    app.logger.debug("Redirecting to /login (Barangay/Resident web login)")
    return redirect(url_for('login'))

@app.route('/go_to_signup_type', methods=['GET'])
def go_to_signup_type():
    app.logger.debug("Redirecting to / (web signup type selection)")
    return redirect(url_for('home'))

@app.route('/chooese_login_type', methods=['GET'])
def chooese_login_type():
    app.logger.debug("Rendering LoginType.html (web)")
    return render_template('LoginType.html')

@app.route('/go_to_cdrrmopnpin', methods=['GET'])
def go_to_cdrrmopnpin():
    app.logger.debug("Redirecting to /login_cdrrmo_pnp (CDRRMO/PNP web login)")
    return redirect(url_for('login_cdrrmo_pnp'))

@app.route('/signup_muna', methods=['GET'])
def signup_muna():
    app.logger.debug("Redirecting to /signup_cdrrmo_pnp (CDRRMO/PNP web signup)")
    return redirect(url_for('signup_cdrrmo_pnp'))

@app.route('/signup_na', methods=['GET'])
def signup_na():
    app.logger.debug("Redirecting to /signup_barangay (Barangay/Resident web signup)")
    return redirect(url_for('signup_barangay'))

@app.route('/login_cdrrmo_pnp', methods=['GET', 'POST'])
def login_cdrrmo_pnp():
    app.logger.debug("Accessing /login_cdrrmo_pnp with method: %s", request.method)
    if request.method == 'POST':
        assigned_municipality = request.form['municipality']
        contact_no = request.form['contact_no']
        password = request.form['password']
        unique_id_cdrrmo = construct_unique_id('cdrrmo', assigned_municipality=assigned_municipality, contact_no=contact_no)
        unique_id_pnp = construct_unique_id('pnp', assigned_municipality=assigned_municipality, contact_no=contact_no)
        
        conn = get_db_connection()
        user = conn.execute('''
            SELECT * FROM users WHERE role = ? AND contact_no = ? AND password = ? AND assigned_municipality = ?
        ''', ('cdrrmo', contact_no, password, assigned_municipality)).fetchone()
        if not user:
            user = conn.execute('''
                SELECT * FROM users WHERE role = ? AND contact_no = ? AND password = ? AND assigned_municipality = ?
            ''', ('pnp', contact_no, password, assigned_municipality)).fetchone()
        conn.close()
        
        if user:
            session['unique_id'] = construct_unique_id(user['role'], assigned_municipality=assigned_municipality, contact_no=contact_no)
            session['role'] = user['role']
            app.logger.debug(f"Web login successful for user: {session['unique_id']} ({user['role']})")
            if user['role'] == 'cdrrmo':
                return redirect(url_for('cdrrmo_dashboard'))
            elif user['role'] == 'pnp':
                return redirect(url_for('pnp_dashboard'))
        app.logger.warning(f"Web login failed for assigned_municipality: {assigned_municipality}, contact: {contact_no}")
        return "Invalid credentials", 401
    return render_template('CDRRMOPNPIn.html')

@app.route('/logout')
def logout():
    role = session.pop('role', None)
    session.clear()
    app.logger.debug(f"User logged out. Redirecting from role: {role}")
    if role == 'barangay':
        return redirect(url_for('login'))
    else:
        return redirect(url_for('login_cdrrmo_pnp'))

def load_coords():
    coords_path = os.path.join(app.root_path, 'assets', 'coords.txt')
    alerts = []
    try:
        with open(coords_path, 'r') as f:
            for line in f:
                if line.strip():
                    parts = line.strip().split(',')
                    if len(parts) == 4:
                        barangay, municipality, message, timestamp = parts
                        alerts.append({
                            "barangay": barangay.strip(),
                            "municipality": municipality.strip(),
                            "message": message.strip(),
                            "timestamp": timestamp.strip()
                        })
    except FileNotFoundError:
        print("Warning: coords.txt not found, using empty alerts.")
    except Exception as e:
        print(f"Error loading coords.txt: {e}")
    return alerts

alerts = deque(maxlen=100)

@app.route('/send_alert', methods=['POST'])
def send_alert():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        lat = data.get('lat')
        lon = data.get('lon')
        emergency_type = data.get('emergency_type', 'General')
        image = data.get('image')
        user_role = data.get('user_role', 'unknown')
        image_upload_time = data.get('imageUploadTime', datetime.now().isoformat())

        # Check image expiration
        if image:
            upload_time = datetime.fromisoformat(image_upload_time)
            if (datetime.now() - upload_time).total_seconds() > 30 * 60:
                image = None  # Expire image if older than 30 minutes
                emergency_type = 'Not Specified'

        alert = {
            'lat': lat,
            'lon': lon,
            'emergency_type': emergency_type,
            'image': image,
            'role': user_role,
            'house_no': data.get('house_no', 'N/A'),
            'street_no': data.get('street_no', 'N/A'),
            'barangay': data.get('barangay', 'N/A'),
            'timestamp': datetime.now(pytz.timezone('Asia/Manila')).isoformat(),
            'imageUploadTime': image_upload_time
        }
        alerts.append(alert)
        socketio.emit('new_alert', alert)
        return jsonify({'status': 'success', 'message': 'Alert sent'}), 200
    except Exception as e:
        app.logger.error(f"Error processing send_alert: {e}", exc_info=True)
        return jsonify({'error': 'Internal server error'}), 500

# New /api/stats endpoint
@app.route('/api/stats')
def get_stats():
    total = len(alerts)
    critical = len([a for a in alerts if a.get('emergency_type', '').lower() == 'critical'])
    return jsonify({'total': total, 'critical': critical})

# Updated /api/distribution endpoint
@app.route('/api/distribution')
def get_distribution():
    role = request.args.get('role', 'all')
    if role == 'barangay':
        filtered_alerts = [a for a in alerts if a.get('role') == 'barangay' or a.get('barangay')]
    elif role == 'cdrrmo':
        filtered_alerts = [a for a in alerts if a.get('role') == 'cdrrmo' or a.get('assigned_municipality')]
    elif role == 'pnp':
        filtered_alerts = [a for a in alerts if a.get('role') == 'pnp' or a.get('assigned_municipality')]
    else:
        filtered_alerts = alerts
    types = [a.get('emergency_type', 'unknown') for a in filtered_alerts]
    return jsonify(dict(Counter(types)))

@app.route('/add_alert', methods=['POST'])
def add_alert():
    data = request.form
    new_alert = {
        "barangay": data['barangay'],
        "municipality": data['municipality'],
        "message": data['message'],
        "timestamp": data['timestamp']
    }
    alerts.append(new_alert)
    return jsonify({"status": "success", "alert": new_alert})

@app.route('/export_alerts')
def export_alerts():
    with open('alerts.json', 'w') as f:
        json.dump(alerts, f, indent=4)
    return jsonify({"status": "success", "file": "alerts.json"})

@app.route('/api/predict_image', methods=['POST'])
def predict_image():
    if dt_classifier is None:
        return jsonify({'error': 'Model not loaded'}), 500
    data = request.get_json()
    base64_image = data.get('image')
    if not base64_image:
        return jsonify({'error': 'No image provided'}), 400
    
    try:
        import base64
        img_data = base64.b64decode(base64_image)
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            return jsonify({'error': 'Failed to decode image'}), 400

        img = cv2.resize(img, (64, 64))
        features = img.flatten().reshape(1, -1)
        prediction = dt_classifier.predict(features)[0]
        app.logger.debug(f"Image predicted as: {prediction}")
        return jsonify({'emergency_type': prediction})
    except Exception as e:
        app.logger.error(f"Image prediction failed: {e}", exc_info=True)
        return jsonify({'error': 'Prediction failed'}), 500

@app.route('/barangay_dashboard')
def barangay_dashboard():
    unique_id = session.get('unique_id')
    conn = get_db_connection()
    user = conn.execute('''
        SELECT * FROM users WHERE barangay = ? AND contact_no = ?
    ''', (unique_id.split('_')[0], unique_id.split('_')[1])).fetchone()
    conn.close()
    
    if not unique_id or not user or user['role'] != 'barangay':
        app.logger.warning("Unauthorized access to barangay_dashboard. Session: %s, User: %s", session, user)
        return redirect(url_for('login'))
    
    barangay = user['barangay']
    assigned_municipality = user['assigned_municipality'] or 'San Pablo City'
    latest_alert = get_latest_alert()
    stats = get_barangay_stats()
    coords = barangay_coords.get(assigned_municipality, {}).get(barangay, {'lat': 14.5995, 'lon': 120.9842})
    
    try:
        lat_coord = float(coords.get('lat', 14.5995))
        lon_coord = float(coords.get('lon', 120.9842))
    except (ValueError, TypeError):
        app.logger.error(f"Invalid coordinates for {barangay} in {assigned_municipality}, using defaults")
        lat_coord = 14.5995
        lon_coord = 120.9842

    app.logger.debug(f"Rendering BarangayDashboard for {barangay} in {assigned_municipality} with coords: lat={lat_coord}, lon={lon_coord}")
    return render_template('BarangayDashboard.html', 
                           latest_alert=latest_alert, 
                           stats=stats, 
                           barangay=barangay, 
                           lat_coord=lat_coord, 
                           lon_coord=lon_coord, 
                           google_api_key=GOOGLE_API_KEY)

@app.route('/cdrrmo_dashboard')
def cdrrmo_dashboard():
    unique_id = session.get('unique_id')
    conn = get_db_connection()
    user = conn.execute('''
        SELECT * FROM users WHERE role = ? AND contact_no = ? AND assigned_municipality = ?
    ''', ('cdrrmo', unique_id.split('_')[2], unique_id.split('_')[1])).fetchone()
    conn.close()
    
    if not unique_id or not user or user['role'] != 'cdrrmo':
        app.logger.warning("Unauthorized access to cdrrmo_dashboard. Session: %s, User: %s", session, user)
        return redirect(url_for('login_cdrrmo_pnp'))
    
    assigned_municipality = user['assigned_municipality']
    if not assigned_municipality:
        app.logger.error(f"No municipality assigned for user {unique_id}")
        assigned_municipality = "San Pablo City"  # Fallback municipality
    stats = get_cdrrmo_stats()
    coords = municipality_coords.get(assigned_municipality, {'lat': 14.5995, 'lon': 120.9842})
    
    try:
        lat_coord = float(coords.get('lat', 14.5995))
        lon_coord = float(coords.get('lon', 120.9842))
    except (ValueError, TypeError):
        app.logger.error(f"Invalid coordinates for {assigned_municipality}, using defaults")
        lat_coord = 14.5995
        lon_coord = 120.9842

    app.logger.debug(f"Rendering CDRRMODashboard for {assigned_municipality} with coords: lat={lat_coord}, lon={lon_coord}")
    return render_template('CDRRMODashboard.html', 
                           stats=stats, 
                           municipality=assigned_municipality, 
                           lat_coord=lat_coord, 
                           lon_coord=lon_coord, 
                           google_api_key=GOOGLE_API_KEY)

@app.route('/pnp_dashboard')
def pnp_dashboard():
    unique_id = session.get('unique_id')
    conn = get_db_connection()
    user = conn.execute('''
        SELECT * FROM users WHERE role = ? AND contact_no = ? AND assigned_municipality = ?
    ''', ('pnp', unique_id.split('_')[2], unique_id.split('_')[1])).fetchone()
    conn.close()
    
    if not unique_id or not user or user['role'] != 'pnp':
        app.logger.warning("Unauthorized access to pnp_dashboard. Session: %s, User: %s", session, user)
        return redirect(url_for('login_cdrrmo_pnp'))
    
    assigned_municipality = user['assigned_municipality']
    if not assigned_municipality:
        app.logger.error(f"No municipality assigned for user {unique_id}")
        assigned_municipality = "San Pablo City"  # Fallback municipality
    stats = get_pnp_stats()
    coords = municipality_coords.get(assigned_municipality, {'lat': 14.5995, 'lon': 120.9842})
    
    try:
        lat_coord = float(coords.get('lat', 14.5995))
        lon_coord = float(coords.get('lon', 120.9842))
    except (ValueError, TypeError):
        app.logger.error(f"Invalid coordinates for {assigned_municipality}, using defaults")
        lat_coord = 14.5995
        lon_coord = 120.9842

    app.logger.debug(f"Rendering PNPDashboard for {assigned_municipality} with coords: lat={lat_coord}, lon={lon_coord}")
    return render_template('PNPDashboard.html', 
                           stats=stats, 
                           municipality=assigned_municipality, 
                           lat_coord=lat_coord, 
                           lon_coord=lon_coord, 
                           google_api_key=GOOGLE_API_KEY)

if __name__ == '__main__':
    db_path = os.path.join(os.path.dirname(__file__), 'data', 'users_web.db')
    try:
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS users (
                barangay TEXT,
                role TEXT NOT NULL,
                contact_no TEXT,
                assigned_municipality TEXT,
                province TEXT,
                password TEXT NOT NULL,
                PRIMARY KEY (barangay, contact_no)
            )
        ''')
        conn.commit()
        conn.close()
        logging.info("Database 'users_web.db' initialized successfully or already exists.")
    except Exception as e:
        logging.error(f"Failed to initialize database: {e}", exc_info=True)

    port = int(os.environ.get('PORT', 5000))
    socketio.run(app, host="0.0.0.0", port=port, debug=True)
