from flask import Flask, request, jsonify, render_template, redirect, url_for, session, send_file
from flask_socketio import SocketIO, emit, join_room
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
import pytz
import pandas as pd
import uuid
from models import lr_road, lr_fire  # Updated import
import BarangayDashboard
import BFPDashboard
import CDRRMODashboard
import PNPDashboard
# Import dashboard and analytics functions
from BarangayDashboard import get_barangay_stats, get_latest_alert
from CDRRMODashboard import get_cdrrmo_stats, get_latest_alert
from PNPDashboard import get_pnp_stats, get_latest_alert
from BFPDashboard import get_bfp_stats, get_latest_alert
from BarangayAnalytics import (
    get_barangay_trends, get_barangay_distribution, get_barangay_causes,
    
)
from CDRRMOAnalytics import (
    get_cdrrmo_trends, get_cdrrmo_distribution, get_cdrrmo_causes,
    
)
from PNPAnalytics import (
    get_pnp_trends, get_pnp_distribution, get_pnp_causes,
    
)
from BFPAnalytics import (
    get_bfp_trends, get_bfp_distribution, get_bfp_causes,
    
)

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load ML models
lr_road = None
lr_fire = None
try:
    lr_road = joblib.load('training/Road Models/lr_road_accident.pkl')
    logger.info("lr_road_accident.pkl loaded successfully.")
except FileNotFoundError:
    logger.error("lr_road_accident.pkl not found.")
except Exception as e:
    logger.error(f"Error loading lr_road_accident.pkl: {e}")
try:
    lr_fire = joblib.load('training/Fire Models/lr_fire_incident.pkl')
    logger.info("lr_fire_incident.pkl loaded successfully.")
except FileNotFoundError:
    logger.error("lr_fire_incident.pkl not found.")
except Exception as e:
    logger.error(f"Error loading lr_fire_incident.pkl: {e}")

# Load datasets
road_accident_df = pd.DataFrame()
try:
    road_accident_df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'dataset', 'road_accident.csv'))
    logger.info("Successfully loaded road_accident.csv")
except FileNotFoundError:
    logger.error("road_accident.csv not found in dataset directory")
except Exception as e:
    logger.error(f"Error loading road_accident.csv: {e}")

# Load decision tree model
model_path = os.path.join(os.path.dirname(__file__), 'training', 'decision_tree_model.pkl')
try:
    dt_classifier = joblib.load(model_path)
    logger.info("decision_tree_model.pkl loaded successfully.")
except FileNotFoundError:
    logger.error(f"{model_path} not found. ML prediction will not work.")
    dt_classifier = None
except Exception as e:
    logger.error(f"Error loading {model_path}: {e}")
    dt_classifier = None

# Load fire incident models
fire_models_path = os.path.join(os.path.dirname(__file__), 'training', 'Fire Models')
try:
    
    rf_fire = joblib.load(os.path.join(fire_models_path, 'rf_fire_incident.pkl'))
    svm_fire = joblib.load(os.path.join(fire_models_path, 'svm_fire_incident.pkl'))
    xgb_fire = joblib.load(os.path.join(fire_models_path, 'xgb_fire_incident.pkl'))
    logger.info("Fire incident models loaded successfully.")
except Exception as e:
    logger.error(f"Error loading fire incident models: {e}")
    lr_fire = rf_fire = svm_fire = xgb_fire = None

# Load road accident models
road_models_path = os.path.join(os.path.dirname(__file__), 'training', 'Road Models')
try:
    
    rf_road = joblib.load(os.path.join(road_models_path, 'rf_road_accident.pkl'))
    svm_road = joblib.load(os.path.join(road_models_path, 'svm_road_accident.pkl'))
    xgb_road = joblib.load(os.path.join(road_models_path, 'xgb_road_accident.pkl'))
    logger.info("Road accident models loaded successfully.")
except Exception as e:
    logger.error(f"Error loading road accident models: {e}")
    lr_road = rf_road = svm_road = xgb_road = None

app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'your-secret-key-here')
socketio = SocketIO(app, async_mode='threading', cors_allowed_origins="*", max_http_buffer_size=10000000)

# Initialize alerts list
alerts = []

# Function to classify images
def classify_image(base64_image):
    if dt_classifier is None:
        logger.error("Decision tree classifier not loaded")
        return 'unknown'
    try:
        import base64
        img_data = base64.b64decode(base64_image)
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        if img is None:
            logger.error("Failed to decode image")
            return 'unknown'
        img = cv2.resize(img, (64, 64))
        features = img.flatten().reshape(1, -1)
        prediction = dt_classifier.predict(features)[0]
        logger.debug(f"Image classified as: {prediction}")
        return prediction
    except Exception as e:
        logger.error(f"Image classification failed: {e}")
        return 'unknown'


@socketio.on('connect')
def handle_connect():
    logger.info('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    logger.info('Client disconnected')

@socketio.on('register_role')
def handle_register_role(data):
    role = data.get('role')
    if role == 'barangay':
        barangay = data.get('barangay')
        if barangay:
            join_room(f"barangay_{barangay}")
            logger.info(f"Client registered to room: barangay_{barangay}")
    elif role in ['cdrrmo', 'pnp', 'bfp']:
        municipality = data.get('municipality', 'default')
        join_room(f"{role}_{municipality}")
        logger.info(f"Client registered to room: {role}_{municipality}")

@socketio.on('alert')
def handle_alert(data):
    barangay = data.get('barangay')
    if barangay:
        data['timestamp'] = datetime.now().isoformat()
        emit('new_alert', data, room=f"barangay_{barangay}")
        logger.info(f"New alert emitted to barangay_{barangay}: {data}")
    else:
        logger.warning("No barangay specified in alert data")

@socketio.on('forward_alert')
def handle_forward_alert(data):
    alert = data.get('alert', {})
    targets = data.get('targets', [])
    emergency_barangay = alert.get('emergency_barangay', alert.get('barangay', 'Unknown'))
    resident_barangay = alert.get('resident_barangay', alert.get('barangay', 'Unknown'))
    alert['emergency_barangay'] = emergency_barangay
    alert['resident_barangay'] = resident_barangay
    
    if not alert.get('image'):
        logger.info("Alert without image not forwarded to CDRRMO, PNP, or BFP")
        return
    
    for target in targets:
        if target in ['bfp', 'cdrrmo', 'pnp']:
            municipality = alert.get('municipality', 'default')
            emit('forward_alert', alert, room=f"{target}_{municipality}")
            logger.info(f"Alert forwarded to {target}_{municipality}: {alert}")
        else:
            logger.warning(f"Invalid target for forward_alert: {target}")

@socketio.on('update_map')
def handle_update_map(data):
    barangay = data.get('barangay')
    lat = data.get('lat')
    lon = data.get('lon')
    map_data = {'lat': lat, 'lon': lon}
    if barangay:
        emit('map_update', map_data, room=f"barangay_{barangay}")
        logger.info(f"Map update emitted to barangay_{barangay}: {map_data}")
    for role in ['cdrrmo', 'pnp', 'bfp']:
        municipality = data.get('municipality', 'default')
        emit('map_update', map_data, room=f"{role}_{municipality}")
        logger.info(f"Map update emitted to {role}_{municipality}: {map_data}")

# SocketIO event handler for handling responses
@socketio.on('response_submitted')
def handle_response(data):
    try:
        alert_id = data.get('alert_id')
        if alert_id:
            global alerts
            alerts = [a for a in alerts if a.get('alert_id') != alert_id]
            emit('alert_removed', {'alert_id': alert_id}, broadcast=True)
            logger.info(f"Alert {alert_id} removed due to response")
    except Exception as e:
        logger.error(f"Error handling response: {e}")

GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY', 'AIzaSyBSXRZPDX1x1d91Ck-pskiwGA8Y2-5gDVs')
barangay_coords = {}
try:
    with open(os.path.join(os.path.dirname(__file__), 'assets', 'coords.txt'), 'r') as f:
        barangay_coords = ast.literal_eval(f.read())
except FileNotFoundError:
    logger.error("coords.txt not found in assets directory. Using empty dict.")
except Exception as e:
    logger.error(f"Error loading coords.txt: {e}. Using empty dict.")

municipality_coords = {
    "San Pablo City": {"lat": 14.0642, "lon": 121.3233},
    "Quezon Province": {"lat": 13.9347, "lon": 121.9473},
}

def get_db_connection():
    db_path = os.path.join('/database', 'users_web.db')
    if not os.path.exists(db_path):
        db_path = os.path.join(os.path.dirname(__file__), 'database', 'users_web.db')
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn

@app.route('/export_users', methods=['GET'])
def export_users():
    if session.get('role') != 'admin':
        return "Unauthorized", 403
    conn = get_db_connection()
    users = conn.execute('SELECT * FROM users').fetchall()
    conn.close()
    return jsonify([dict(user) for user in users])

@app.route('/download_db', methods=['GET'])
def download_db():
    db_path = os.path.join('/database', 'users_web.db')
    if not os.path.exists(db_path):
        db_path = os.path.join(os.path.dirname(__file__), 'database', 'users_web.db')
    if not os.path.exists(db_path):
        return "Database file not found", 404
    logger.debug(f"Serving database from: {db_path}")
    return send_file(db_path, as_attachment=True, download_name='users_web.db')

def construct_unique_id(role, barangay=None, assigned_municipality=None, contact_no=None):
    if role == 'barangay':
        return f"{barangay}_{contact_no}"
    return f"{role}_{assigned_municipality}_{contact_no}"

@app.route('/')
def home():
    logger.debug("Rendering SignUpType.html")
    return render_template('SignUpType.html')

@app.route('/signup_barangay', methods=['GET', 'POST'])
def signup_barangay():
    if request.method == 'POST':
        barangay = request.form['barangay']
        assigned_municipality = request.form['municipality']
        province = request.form['province']
        contact_no = request.form['contact_no']
        password = request.form['password']
        unique_id = construct_unique_id('barangay', barangay=barangay, contact_no=contact_no)
        
        conn = get_db_connection()
        try:
            existing_user = conn.execute('SELECT * FROM users WHERE contact_no = ?', (contact_no,)).fetchone()
            if existing_user:
                logger.error("Signup failed: Contact number %s already exists", contact_no)
                return "Contact number already exists", 400
            
            conn.execute('''
                INSERT INTO users (barangay, role, contact_no, assigned_municipality, province, password)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (barangay, 'barangay', contact_no, assigned_municipality, province, password))
            conn.commit()
            logger.debug("User signed up successfully: %s", unique_id)
            return redirect(url_for('login'))
        except sqlite3.IntegrityError as e:
            logger.error("IntegrityError during signup: %s", e)
            return "User already exists", 400
        except Exception as e:
            logger.error(f"Signup failed for {unique_id}: {e}")
            return f"Signup failed: {e}", 500
        finally:
            conn.close()
    return render_template('SignUpPage.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    logger.debug("Accessing /login with method: %s", request.method)
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
            role = user['role']
            if role == 'barangay':
                return redirect(url_for('barangay_dashboard', barangay=user['barangay']))
            elif role == 'bfp':
                return redirect(url_for('bfp_dashboard', municipality=user['assigned_municipality']))
            elif role == 'cdrrmo':
                return redirect(url_for('cdrrmo_dashboard', municipality=user['assigned_municipality']))
            elif role == 'pnp':
                return redirect(url_for('pnp_dashboard', municipality=user['assigned_municipality']))
        return "Invalid credentials", 401
    return render_template('LoginPage.html')

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
        logger.debug(f"API login successful for user: {unique_id} with role: {user['role']}")
        return jsonify({'status': 'success', 'role': user['role']})
    logger.warning(f"API login failed for unique_id: {unique_id}")
    return jsonify({'error': 'Invalid credentials'}), 401

@app.route('/signup_cdrrmo_pnp_bfp', methods=['GET', 'POST'])
def signup_cdrrmo_pnp_bfp():
    if request.method == 'POST':
        role = request.form['role'].lower()
        assigned_municipality = request.form['municipality']
        contact_no = request.form['contact_no']
        password = request.form['password']
        unique_id = construct_unique_id(role, assigned_municipality=assigned_municipality, contact_no=contact_no)
        
        conn = get_db_connection()
        try:
            existing_user = conn.execute('SELECT * FROM users WHERE contact_no = ?', (contact_no,)).fetchone()
            if existing_user:
                logger.error("Signup failed: Contact number %s already exists", contact_no)
                return "Contact number already exists", 400
            
            conn.execute('''
                INSERT INTO users (role, contact_no, assigned_municipality, password)
                VALUES (?, ?, ?, ?)
            ''', (role, contact_no, assigned_municipality, password))
            conn.commit()
            logger.debug("User signed up successfully: %s", unique_id)
            return redirect(url_for('login_cdrrmo_pnp_bfp'))
        except sqlite3.IntegrityError as e:
            logger.error("IntegrityError during signup: %s", e)
            return "User already exists", 400
        except Exception as e:
            logger.error(f"Signup failed for {unique_id}: {e}")
            return f"Signup failed: {e}", 500
        finally:
            conn.close()
    return render_template('CDRRMOPNPBFPUp.html')

@app.route('/login_cdrrmo_pnp_bfp', methods=['GET', 'POST'])
def login_cdrrmo_pnp_bfp():
    logger.debug("Accessing /login_cdrrmo_pnp_bfp with method: %s", request.method)
    if request.method == 'POST':
        role = request.form['role'].lower()
        if 'role' not in request.form:
            logger.error("Role field is missing in the form data")
            return "Role is required", 400
        assigned_municipality = request.form['municipality']
        contact_no = request.form['contact_no']
        password = request.form['password']
        
        if role not in ['cdrrmo', 'pnp', 'bfp']:
            logger.error(f"Invalid role provided: {role}")
            return "Invalid role", 400
        
        logger.debug(f"Login attempt: role={role}, municipality={assigned_municipality}, contact_no={contact_no}")
        
        conn = get_db_connection()
        user = conn.execute('''
            SELECT * FROM users WHERE role = ? AND contact_no = ? AND password = ? AND assigned_municipality = ?
        ''', (role, contact_no, password, assigned_municipality)).fetchone()
        conn.close()
        
        if user:
            unique_id = construct_unique_id(user['role'], assigned_municipality=assigned_municipality, contact_no=contact_no)
            session['unique_id'] = unique_id
            session['role'] = user['role']
            logger.debug(f"Web login successful for user: {unique_id} ({user['role']})")
            if user['role'] == 'cdrrmo':
                return redirect(url_for('cdrrmo_dashboard'))
            elif user['role'] == 'pnp':
                return redirect(url_for('pnp_dashboard'))
            elif user['role'] == 'bfp':
                return redirect(url_for('bfp_dashboard'))
        logger.warning(f"Web login failed for assigned_municipality: {assigned_municipality}, contact: {contact_no}, role: {role}")
        return "Invalid credentials", 401
    return render_template('CDRRMOPNPBFPIn.html')

@app.route('/api/stats')
def api_stats():
    role = request.args.get('role', 'barangay')
    if role == 'barangay':
        from BarangayDashboard import get_barangay_stats
        stats = get_barangay_stats()
    elif role == 'bfp':
        from BFPDashboard import get_bfp_stats
        stats = get_bfp_stats()
    elif role == 'cdrrmo':
        from CDRRMODashboard import get_cdrrmo_stats
        stats = get_cdrrmo_stats()
    elif role == 'pnp':
        from PNPDashboard import get_pnp_stats
        stats = get_pnp_stats()
    else:
        stats = Counter()
    return jsonify({'total': stats.total(), 'critical': stats.get('critical', 0)})

@app.route('/api/distribution')
def api_distribution():
    role = request.args.get('role', 'barangay')
    if role == 'barangay':
        from BarangayDashboard import get_barangay_stats
        distribution = get_barangay_stats()
    elif role == 'bfp':
        from BFPDashboard import get_bfp_stats
        distribution = get_bfp_stats()
    elif role == 'cdrrmo':
        from CDRRMODashboard import get_cdrrmo_stats
        distribution = get_cdrrmo_stats()
    elif role == 'pnp':
        from PNPDashboard import get_pnp_stats
        distribution = get_pnp_stats()
    else:
        distribution = Counter()
    return jsonify(dict(distribution))


@app.route('/go_to_login_page', methods=['GET'])
def go_to_login_page():
    logger.debug("Redirecting to /login")
    return redirect(url_for('login'))

@app.route('/go_to_signup_type', methods=['GET'])
def go_to_signup_type():
    logger.debug("Redirecting to /")
    return redirect(url_for('home'))

@app.route('/choose_login_type', methods=['GET'])
def choose_login_type():
    logger.debug("Rendering LoginType.html")
    return render_template('LoginType.html')

@app.route('/go_to_cdrrmopnpbfpin', methods=['GET'])
def go_to_cdrrmopnpbfpin():
    logger.debug("Redirecting to /login_cdrrmo_pnp_bfp")
    return redirect(url_for('login_cdrrmo_pnp_bfp'))

@app.route('/signup_muna', methods=['GET'])
def signup_muna():
    logger.debug("Redirecting to /signup_cdrrmo_pnp_bfp")
    return redirect(url_for('signup_cdrrmo_pnp_bfp'))

@app.route('/signup_na', methods=['GET'])
def signup_na():
    logger.debug("Redirecting to /signup_barangay")
    return redirect(url_for('signup_barangay'))

@app.route('/logout')
def logout():
    role = session.pop('role', None)
    session.clear()
    logger.debug(f"User logged out. Redirecting from role: {role}")
    if role == 'barangay':
        return redirect(url_for('login'))
    else:
        return redirect(url_for('login_cdrrmo_pnp_bfp'))

def load_coords():
    coords_path = os.path.join(app.root_path, 'assets', 'coords.txt')
    alerts_data = []
    try:
        with open(coords_path, 'r') as f:
            for line in f:
                if line.strip():
                    parts = line.strip().split(',')
                    if len(parts) == 4:
                        barangay, municipality, message, timestamp = parts
                        alerts_data.append({
                            "barangay": barangay.strip(),
                            "municipality": municipality.strip(),
                            "message": message.strip(),
                            "timestamp": timestamp.strip()
                        })
    except FileNotFoundError:
        logger.warning("coords.txt not found, using empty alerts.")
    except Exception as e:
        logger.error(f"Error loading coords.txt: {e}")
    return alerts_data

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

        if image:
            upload_time = datetime.fromisoformat(image_upload_time)
            if (datetime.now() - upload_time).total_seconds() > 30 * 60:
                image = None
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
            'imageUploadTime': image_upload_time,
            'alert_id': str(uuid.uuid4()),
            'user_barangay': data.get('barangay', 'Unknown')
        }
        
        if alert['image']:
            prediction = classify_image(alert['image'])
            if prediction not in ['road_accident', 'fire_incident']:
                alert['image'] = None

        alerts.append(alert)
        socketio.emit('new_alert', alert)
        return jsonify({'status': 'success', 'message': 'Alert sent'}), 200
    except Exception as e:
        logger.error(f"Error processing send_alert: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/stats')
def get_stats():
    try:
        total = len(alerts)
        critical = len([a for a in alerts if a.get('emergency_type', '').lower() == 'critical'])
        return jsonify({'total': total, 'critical': critical})
    except Exception as e:
        logger.error(f"Error in get_stats: {e}")
        return jsonify({'error': 'Failed to retrieve stats'}), 500

@app.route('/api/distribution')
def get_distribution():
    try:
        role = request.args.get('role', 'all')
        if role == 'barangay':
            filtered_alerts = [a for a in alerts if a.get('role') == 'barangay' or a.get('barangay')]
        elif role == 'cdrrmo':
            filtered_alerts = [a for a in alerts if a.get('role') == 'cdrrmo' or a.get('assigned_municipality')]
        elif role == 'pnp':
            filtered_alerts = [a for a in alerts if a.get('role') == 'pnp' or a.get('assigned_municipality')]
        elif role == 'bfp':
            filtered_alerts = [a for a in alerts if a.get('role') == 'bfp' or a.get('assigned_municipality')]
        else:
            filtered_alerts = alerts
        types = [a.get('emergency_type', 'unknown') for a in filtered_alerts]
        return jsonify(dict(Counter(types)))
    except Exception as e:
        logger.error(f"Error in get_distribution: {e}")
        return jsonify({'error': 'Failed to retrieve distribution'}), 500

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

@app.route('/api/analytics')
def get_analytics():
    try:
        role = request.args.get('role', 'all')
        if role == 'barangay':
            trends = get_barangay_trends()
            distribution = get_barangay_distribution()
            causes = get_barangay_causes()
        elif role == 'cdrrmo':
            trends = get_cdrrmo_trends()
            distribution = get_cdrrmo_distribution()
            causes = get_cdrrmo_causes()
        elif role == 'pnp':
            trends = get_pnp_trends()
            distribution = get_pnp_distribution()
            causes = get_pnp_causes()
        elif role == 'bfp':
            trends = get_bfp_trends()
            distribution = get_bfp_distribution()
            causes = get_bfp_causes()
        else:
            return jsonify({'error': 'Invalid role'}), 400
        return jsonify({'trends': trends, 'distribution': distribution, 'causes': causes})
    except Exception as e:
        logger.error(f"Error in get_analytics: {e}")
        return jsonify({'error': 'Failed to retrieve analytics'}), 500

@app.route('/api/predict_image', methods=['POST'])
def predict_image():
    if dt_classifier is None:
        logger.error("Machine learning model not loaded")
        return jsonify({'error': 'Model not loaded'}), 500
    data = request.get_json()
    base64_image = data.get('image')
    if not base64_image:
        logger.error("No image provided in predict_image")
        return jsonify({'error': 'No image provided'}), 400
    
    try:
        import base64
        img_data = base64.b64decode(base64_image)
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            logger.error("Failed to decode image")
            return jsonify({'error': 'Failed to decode image'}), 400

        img = cv2.resize(img, (64, 64))
        features = img.flatten().reshape(1, -1)
        prediction = dt_classifier.predict(features)[0]
        logger.debug(f"Image predicted as: {prediction}")
        return jsonify({'emergency_type': prediction})
    except Exception as e:
        logger.error(f"Image prediction failed: {e}")
        return jsonify({'error': 'Prediction failed'}), 500

@app.route('/barangay_dashboard')
def barangay_dashboard():
    if 'username' not in session:
        return redirect(url_for('login'))
    conn = get_db_connection()
    user = conn.execute('SELECT * FROM users WHERE username = ?', (session['username'],)).fetchone()
    conn.close()
    if user and user['role'] == 'barangay':
        stats = BarangayDashboard.get_barangay_stats()
        latest_alert = BarangayDashboard.get_latest_alert()
        lat = latest_alert.get('lat', 14.0549) if latest_alert else 14.0549
        lon = latest_alert.get('lon', 121.3013) if latest_alert else 121.3013
        return render_template('BarangayDashboard.html', barangay=user['barangay'], stats=stats, lat_coord=lat, lon_coord=lon)
    return "Unauthorized", 403

@app.route('/cdrrmo_dashboard')
def cdrrmo_dashboard():
    if 'username' not in session:
        return redirect(url_for('login'))
    conn = get_db_connection()
    user = conn.execute('SELECT * FROM users WHERE contact_no = ?', (session['username'],)).fetchone()
    conn.close()
    if user and user['role'] == 'cdrrmo':
        stats = CDRRMODashboard.get_cdrrmo_stats()
        latest_alert = CDRRMODashboard.get_latest_alert()
        lat = latest_alert.get('lat', 14.0549) if latest_alert else 14.0549
        lon = latest_alert.get('lon', 121.3013) if latest_alert else 121.3013
        return render_template('CDRRMODashboard.html', municipality=user['assigned_municipality'], stats=stats, lat_coord=lat, lon_coord=lon)
    return "Unauthorized", 403

@app.route('/bfp_dashboard')
def bfp_dashboard():
    if 'username' not in session:
        return redirect(url_for('login'))
    conn = get_db_connection()
    user = conn.execute('SELECT * FROM users WHERE contact_no = ?', (session['username'],)).fetchone()
    conn.close()
    if user and user['role'] == 'bfp':
        stats = BFPDashboard.get_bfp_stats()
        latest_alert = BFPDashboard.get_latest_alert()
        lat = latest_alert.get('lat', 14.0549) if latest_alert else 14.0549
        lon = latest_alert.get('lon', 121.3013) if latest_alert else 121.3013
        return render_template('BFPDashboard.html', municipality=user['assigned_municipality'], stats=stats, lat_coord=lat, lon_coord=lon)
    return "Unauthorized", 403

@app.route('/pnp_dashboard')
def pnp_dashboard():
    if 'username' not in session:
        return redirect(url_for('login'))
    conn = get_db_connection()
    user = conn.execute('SELECT * FROM users WHERE contact_no = ?', (session['username'],)).fetchone()
    conn.close()
    if user and user['role'] == 'pnp':
        stats = PNPDashboard.get_pnp_stats()
        latest_alert = PNPDashboard.get_latest_alert()
        lat = latest_alert.get('lat', 14.0549) if latest_alert else 14.0549
        lon = latest_alert.get('lon', 121.3013) if latest_alert else 121.3013
        return render_template('PNPDashboard.html', municipality=user['assigned_municipality'], stats=stats, lat_coord=lat, lon_coord=lon)
    return "Unauthorized", 403

@app.route('/barangay/analytics')
def barangay_analytics():
    if 'role' not in session or session['role'] != 'barangay':
        logger.warning("Unauthorized access to barangay_analytics")
        return redirect(url_for('login'))
    unique_id = session.get('unique_id')
    conn = get_db_connection()
    user = conn.execute('SELECT barangay FROM users WHERE barangay = ? AND contact_no = ?',
                        (unique_id.split('_')[0], unique_id.split('_')[1])).fetchone()
    conn.close()
    barangay = user['barangay'] if user else "Unknown"
    current_datetime = datetime.now(pytz.timezone('Asia/Manila')).strftime('%a/%m/%d/%y %H:%M:%S')
    return render_template('BarangayAnalytics.html', barangay=barangay, current_datetime=current_datetime)

@app.route('/api/barangay_analytics_data')
def barangay_analytics_data():
    time_filter = request.args.get('time', 'weekly')
    trends = get_barangay_trends(time_filter)
    distribution = get_barangay_distribution(time_filter)
    causes_data = get_barangay_causes(time_filter)
    weather = {'Sunny': 10, 'Rainy': 5, 'Foggy': 2}
    road_conditions = {'Dry': 12, 'Wet': 4, 'Icy': 1}
    vehicle_types = {'Car': 8, 'Motorcycle': 6, 'Truck': 3}
    driver_age = {'18-25': 5, '26-35': 7, '36-50': 3, '51+': 2}
    driver_gender = {'Male': 12, 'Female': 5}
    accident_type = {'Collision': 10, 'Rollover': 4, 'Pedestrian': 3}
    injuries = [5, 3, 2, 1] * (len(trends['labels']) // 4 + 1)
    fatalities = [1, 0, 1, 0] * (len(trends['labels']) // 4 + 1)
    return jsonify({
        'trends': trends,
        'distribution': distribution,
        'causes': causes_data['road'],
        'weather': weather,
        'road_conditions': road_conditions,
        'vehicle_types': vehicle_types,
        'driver_age': driver_age,
        'driver_gender': driver_gender,
        'accident_type': accident_type,
        'injuries': injuries[:len(trends['labels'])],
        'fatalities': fatalities[:len(trends['labels'])]
    })

@app.route('/cdrrmo/analytics')
def cdrrmo_analytics():
    if 'role' not in session or session['role'] != 'cdrrmo':
        logger.warning("Unauthorized access to cdrrmo_analytics")
        return redirect(url_for('login_cdrrmo_pnp_bfp'))
    unique_id = session.get('unique_id')
    conn = get_db_connection()
    user = conn.execute('SELECT assigned_municipality FROM users WHERE role = ? AND contact_no = ? AND assigned_municipality = ?',
                        ('cdrrmo', unique_id.split('_')[2], unique_id.split('_')[1])).fetchone()
    conn.close()
    municipality = user['assigned_municipality'] if user else "Unknown"
    current_datetime = datetime.now(pytz.timezone('Asia/Manila')).strftime('%a/%m/%d/%y %H:%M:%S')
    barangays = ["Barangay 1", "Barangay 2", "Barangay 3"]  # Placeholder, replace with actual query
    return render_template('CDRRMOAnalytics.html', municipality=municipality, current_datetime=current_datetime, barangays=barangays)

@app.route('/api/cdrrmo_analytics_data', methods=['GET'])
def get_cdrrmo_analytics_data():
    time_filter = request.args.get('time', 'weekly')
    trends = get_barangay_trends(time_filter)
    distribution = get_barangay_distribution(time_filter)
    causes_data = get_barangay_causes(time_filter)
    weather = {'Sunny': 10, 'Rainy': 5, 'Foggy': 2}
    road_conditions = {'Dry': 12, 'Wet': 4, 'Icy': 1}
    vehicle_types = {'Car': 8, 'Motorcycle': 6, 'Truck': 3}
    driver_age = {'18-25': 5, '26-35': 7, '36-50': 3, '51+': 2}
    driver_gender = {'Male': 12, 'Female': 5}
    accident_type = {'Collision': 10, 'Rollover': 4, 'Pedestrian': 3}
    injuries = [5, 3, 2, 1] * (len(trends['labels']) // 4 + 1)
    fatalities = [1, 0, 1, 0] * (len(trends['labels']) // 4 + 1)
    return jsonify({
        'trends': trends,
        'distribution': distribution,
        'causes': causes_data['road'],
        'weather': weather,
        'road_conditions': road_conditions,
        'vehicle_types': vehicle_types,
        'driver_age': driver_age,
        'driver_gender': driver_gender,
        'accident_type': accident_type,
        'injuries': injuries[:len(trends['labels'])],
        'fatalities': fatalities[:len(trends['labels'])]
    })

@app.route('/pnp/analytics')
def pnp_analytics():
    if 'role' not in session or session['role'] != 'pnp':
        logger.warning("Unauthorized access to pnp_analytics")
        return redirect(url_for('login_cdrrmo_pnp_bfp'))
    unique_id = session.get('unique_id')
    conn = get_db_connection()
    user = conn.execute('SELECT assigned_municipality FROM users WHERE role = ? AND contact_no = ? AND assigned_municipality = ?',
                        ('pnp', unique_id.split('_')[2], unique_id.split('_')[1])).fetchone()
    conn.close()
    municipality = user['assigned_municipality'] if user else "Unknown"
    current_datetime = datetime.now(pytz.timezone('Asia/Manila')).strftime('%a/%m/%d/%y %H:%M:%S')
    barangays = ["Barangay 1", "Barangay 2", "Barangay 3"]  # Placeholder, replace with actual query
    return render_template('PNPAnalytics.html', municipality=municipality, current_datetime=current_datetime, barangays=barangays)

@app.route('/api/pnp_analytics_data', methods=['GET'])
def get_pnp_analytics_data():
    time_filter = request.args.get('time', 'weekly')
    trends = get_barangay_trends(time_filter)
    distribution = get_barangay_distribution(time_filter)
    causes_data = get_barangay_causes(time_filter)
    weather = {'Sunny': 10, 'Rainy': 5, 'Foggy': 2}
    road_conditions = {'Dry': 12, 'Wet': 4, 'Icy': 1}
    vehicle_types = {'Car': 8, 'Motorcycle': 6, 'Truck': 3}
    driver_age = {'18-25': 5, '26-35': 7, '36-50': 3, '51+': 2}
    driver_gender = {'Male': 12, 'Female': 5}
    accident_type = {'Collision': 10, 'Rollover': 4, 'Pedestrian': 3}
    injuries = [5, 3, 2, 1] * (len(trends['labels']) // 4 + 1)
    fatalities = [1, 0, 1, 0] * (len(trends['labels']) // 4 + 1)
    return jsonify({
        'trends': trends,
        'distribution': distribution,
        'causes': causes_data['road'],
        'weather': weather,
        'road_conditions': road_conditions,
        'vehicle_types': vehicle_types,
        'driver_age': driver_age,
        'driver_gender': driver_gender,
        'accident_type': accident_type,
        'injuries': injuries[:len(trends['labels'])],
        'fatalities': fatalities[:len(trends['labels'])]
    })

@app.route('/bfp/analytics')
def bfp_analytics():
    if 'role' not in session or session['role'] != 'bfp':
        logger.warning("Unauthorized access to bfp_analytics")
        return redirect(url_for('login_cdrrmo_pnp_bfp'))
    unique_id = session.get('unique_id')
    conn = get_db_connection()
    user = conn.execute('SELECT assigned_municipality FROM users WHERE role = ? AND contact_no = ? AND assigned_municipality = ?',
                        ('bfp', unique_id.split('_')[2], unique_id.split('_')[1])).fetchone()
    conn.close()
    municipality = user['assigned_municipality'] if user else "Unknown"
    current_datetime = datetime.now(pytz.timezone('Asia/Manila')).strftime('%a/%m/%d/%y %H:%M:%S')
    barangays = ["Barangay 1", "Barangay 2", "Barangay 3"]  # Placeholder, replace with actual query
    return render_template('BFPAnalytics.html', municipality=municipality, current_datetime=current_datetime, barangays=barangays)

@app.route('/api/bfp_analytics_data', methods=['GET'])
def get_bfp_analytics_data():
    try:
        time_filter = request.args.get('time', 'weekly')
        barangay = request.args.get('barangay', '')
        trends = get_bfp_trends(time_filter, barangay)
        distribution = get_bfp_distribution(time_filter, barangay)
        causes = get_bfp_causes(time_filter, barangay)
        
        return jsonify({
            'trends': trends,
            'distribution': distribution,
            'causes': causes
        })
    except Exception as e:
        logger.error(f"Error in get_bfp_analytics_data: {e}")
        return jsonify({'error': 'Failed to retrieve analytics data'}), 500

def get_latest_alert():
    try:
        if alerts:
            return alerts[-1]
        return None
    except Exception as e:
        logger.error(f"Error in get_latest_alert: {e}")
        return None

def get_barangay_stats():
    try:
        types = [a.get('emergency_type', 'unknown') for a in alerts if a.get('role') == 'barangay' or a.get('barangay')]
        return Counter(types)
    except Exception as e:
        logger.error(f"Error in get_barangay_stats: {e}")
        return Counter()

def get_cdrrmo_stats():
    try:
        types = [a.get('emergency_type', 'unknown') for a in alerts if a.get('role') == 'cdrrmo' or a.get('assigned_municipality')]
        return Counter(types)
    except Exception as e:
        logger.error(f"Error in get_cdrrmo_stats: {e}")
        return Counter()

def get_pnp_stats():
    try:
        types = [a.get('emergency_type', 'unknown') for a in alerts if a.get('role') == 'pnp' or a.get('assigned_municipality')]
        return Counter(types)
    except Exception as e:
        logger.error(f"Error in get_pnp_stats: {e}")
        return Counter()

def get_bfp_stats():
    try:
        types = [a.get('emergency_type', 'unknown') for a in alerts if a.get('role') == 'bfp' or a.get('assigned_municipality')]
        return Counter(types)
    except Exception as e:
        logger.error(f"Error in get_bfp_stats: {e}")
        return Counter()

if __name__ == '__main__':
    db_path = os.path.join(os.path.dirname(__file__), 'database', 'users_web.db')
    try:
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS users (
                barangay TEXT,
                role TEXT NOT NULL,
                contact_no TEXT UNIQUE NOT NULL,
                assigned_municipality TEXT,
                province TEXT,
                password TEXT NOT NULL
            )
        ''')
        conn.commit()
        conn.close()
        logger.info("Database 'users_web.db' initialized successfully or already exists.")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")

    port = int(os.environ.get('PORT', 5000))
    socketio.run(app, host="0.0.0.0", port=port, debug=True, allow_unsafe_werkzeug=True)
