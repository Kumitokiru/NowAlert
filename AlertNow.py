from flask import Flask, request, jsonify, render_template, redirect, url_for, session, send_file
from flask_socketio import SocketIO
import logging
import ast
import os
import json
import psycopg
from psycopg.rows import dict_row
from urllib.parse import urlparse
import joblib
import cv2
import numpy as np
from collections import Counter
from datetime import datetime
from alert_data import alerts
from collections import deque
import pytz
import pandas as pd
import psycopg.errors as psycopg_errors

# Import dashboard stats functions
from BarangayDashboard import get_barangay_stats, get_latest_alert
from CDRRMODashboard import get_cdrrmo_stats
from PNPDashboard import get_pnp_stats
from BFPDashboard import get_bfp_stats

# Import analytics functions
from BarangayAnalytics import get_barangay_trends, get_barangay_distribution, get_barangay_causes
from CDRRMOAnalytics import get_cdrrmo_trends, get_cdrrmo_distribution, get_cdrrmo_causes
from PNPAnalytics import get_pnp_trends, get_pnp_distribution, get_pnp_causes
from BFPAnalytics import get_bfp_trends, get_bfp_distribution, get_bfp_causes

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Replace with a strong, secret key
socketio = SocketIO(app, cors_allowed_origins="*")
logging.basicConfig(level=logging.DEBUG)

# Ensure data directory exists
data_dir = os.path.join(os.path.dirname(__file__), 'data')
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
    logging.info(f"Created data directory at {data_dir}")

# Parse the PostgreSQL database URL
DATABASE_URL = "postgresql://root:6E2X2PMWnJqcxzWQyn7OUGMHh02xoF6L@dpg-d1a1hrngi27c73f0s3i0-a/android_users"
parsed_url = urlparse(DATABASE_URL)
DB_HOST = parsed_url.hostname
DB_NAME = parsed_url.path.lstrip('/')
DB_USER = parsed_url.username
DB_PASSWORD = parsed_url.password

# Database connection function
def get_db_connection():
    try:
        conn = psycopg.connect(
            host=DB_HOST,
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            row_factory=dict_row
        )
        return conn
    except psycopg.Error as e:
        logging.error(f"Failed to connect to database: {e}")
        raise

# Load barangay coordinates
try:
    with open(os.path.join('assets', 'coords.txt'), 'r') as f:
        barangay_coords = ast.literal_eval(f.read())
except FileNotFoundError:
    logging.error("coords.txt not found in assets directory. Using empty dict.")
    barangay_coords = {}
except Exception as e:
    logging.error(f"Error loading coords.txt: {e}. Using empty dict.")
    barangay_coords = {}

# Municipality coordinates
municipality_coords = {
    "San Pablo City": {"lat": 14.0642, "lon": 121.3233},
    "Quezon Province": {"lat": 13.9347, "lon": 121.9473},
    # Add more as needed
}

# Google Maps API key
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY', 'AIzaSyBSXRZPDX1x1d91Ck-pskiwGA8Y2-5gDVs')

# SocketIO event for alert response
@socketio.on('responded')
def handle_responded(data):
    timestamp = data.get('timestamp')
    for alert in alerts:
        if alert['timestamp'] == timestamp:
            alert['responded'] = True
            break
    socketio.emit('alert_responded', {
        'timestamp': timestamp,
        'lat': data.get('lat'),
        'lon': data.get('lon'),
        'barangay': data.get('barangay'),
        'emergency_type': data.get('emergency_type')
    })

# Load ML models
try:
    dt_classifier = joblib.load('training/decision_tree_model.pkl')
    logging.info("decision_tree_model.pkl loaded successfully.")
except FileNotFoundError:
    logging.error("decision_tree_model.pkl not found. ML prediction will not work.")
    dt_classifier = None
except Exception as e:
    logging.error(f"Error loading decision_tree_model.pkl: {e}")
    dt_classifier = None

# Load fire incident models
try:
    lr_fire = joblib.load('training/Fire Models/lr_fire_incident.pkl')
    rf_fire = joblib.load('training/Fire Models/rf_fire_incident.pkl')
    svm_fire = joblib.load('training/Fire Models/svm_fire_incident.pkl')
    xgb_fire = joblib.load('training/Fire Models/xgb_fire_incident.pkl')
    logging.info("Fire incident models loaded successfully.")
except Exception as e:
    logging.error(f"Error loading fire incident models: {e}")
    lr_fire = rf_fire = svm_fire = xgb_fire = None

# Load road accident models
try:
    lr_road = joblib.load('training/Road Models/lr_road_accident.pkl')
    rf_road = joblib.load('training/Road Models/rf_road_accident.pkl')
    svm_road = joblib.load('training/Road Models/svm_road_accident.pkl')
    xgb_road = joblib.load('training/Road Models/xgb_road_accident.pkl')
    logging.info("Road accident models loaded successfully.")
except Exception as e:
    logging.error(f"Error loading road accident models: {e}")
    lr_road = rf_road = svm_road = xgb_road = None

# Utility routes
@app.route('/export_users', methods=['GET'])
def export_users():
    if session.get('role') != 'admin':
        return "Unauthorized", 403
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM users')
    users = cursor.fetchall()
    conn.close()
    return jsonify([dict(user) for user in users])

@app.route('/download_db', methods=['GET'])
def download_db():
    return "Database download not supported for PostgreSQL", 403

# Unique ID constructor
def construct_unique_id(role, barangay=None, contact_no=None, assigned_municipality=None):
    if role == 'barangay':
        return f"{barangay}_{contact_no}"
    else:  # cdrrmo, pnp, bfp
        return f"{role}_{assigned_municipality}_{contact_no}"

# Core routes
@app.route('/')
def home():
    app.logger.debug("Rendering SignUpType.html")
    return render_template('SignUpType.html')

@app.route('/signup_barangay', methods=['GET', 'POST'])
def signup_barangay():
    app.logger.debug("Accessing /signup_barangay with method: %s", request.method)
    if request.method == 'POST':
        barangay = request.form['barangay']
        assigned_municipality = request.form['municipality']
        province = request.form['province']
        contact_no = request.form['contact_no']
        password = request.form['password']
        unique_id = construct_unique_id('barangay', barangay=barangay, contact_no=contact_no)
        
        conn = get_db_connection()
        cursor = conn.cursor()
        try:
            cursor.execute('''
                INSERT INTO users (barangay, role, contact_no, assigned_municipality, province, password)
                VALUES (%s, %s, %s, %s, %s, %s)
            ''', (barangay, 'barangay', contact_no, assigned_municipality, province, password))
            conn.commit()
            app.logger.debug(f"User data inserted successfully: {unique_id}")
            return redirect(url_for('login'))
        except psycopg_errors.UniqueViolation as e:
            app.logger.error("UniqueViolation: %s", e)
            return "User already exists", 400
        except psycopg.Error as e:
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
        cursor = conn.cursor()
        cursor.execute('''
            SELECT * FROM users WHERE barangay = %s AND contact_no = %s AND password = %s
        ''', (barangay, contact_no, password))
        user = cursor.fetchone()
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
    cursor = conn.cursor()
    cursor.execute('''
        SELECT * FROM users WHERE barangay = %s AND contact_no = %s AND password = %s
    ''', (barangay, contact_no, password))
    user = cursor.fetchone()
    conn.close()
    
    if user:
        app.logger.debug(f"API login successful for user: {unique_id} with role: {user['role']}")
        return jsonify({'status': 'success', 'role': user['role']})
    app.logger.warning(f"API login failed for unique_id: {unique_id}")
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
        cursor = conn.cursor()
        try:
            cursor.execute('SELECT * FROM users WHERE contact_no = %s', (contact_no,))
            existing_user = cursor.fetchone()
            if existing_user:
                app.logger.error("Signup failed: Contact number %s already exists", contact_no)
                return "Contact number already exists", 400
            
            cursor.execute('''
                INSERT INTO users (role, contact_no, assigned_municipality, password)
                VALUES (%s, %s, %s, %s)
            ''', (role, contact_no, assigned_municipality, password))
            conn.commit()
            app.logger.debug("User signed up successfully: %s", unique_id)
            return redirect(url_for('login_cdrrmo_pnp_bfp'))
        except psycopg_errors.UniqueViolation as e:
            app.logger.error("UniqueViolation during signup: %s", e)
            return "User already exists", 400
        except psycopg.Error as e:
            app.logger.error(f"Signup failed for {unique_id}: {e}", exc_info=True)
            return f"Signup failed: {e}", 500
        finally:
            conn.close()
    return render_template('CDRRMOPNPBFPUp.html')

@app.route('/login_cdrrmo_pnp_bfp', methods=['GET', 'POST'])
def login_cdrrmo_pnp_bfp():
    app.logger.debug("Accessing /login_cdrrmo_pnp_bfp with method: %s", request.method)
    if request.method == 'POST':
        assigned_municipality = request.form['municipality']
        contact_no = request.form['contact_no']
        password = request.form['password']
        role = request.form['role'].lower()
        
        if role not in ['cdrrmo', 'pnp', 'bfp']:
            app.logger.error(f"Invalid role provided: {role}")
            return "Invalid role", 400
        
        app.logger.debug(f"Login attempt: role={role}, municipality={assigned_municipality}, contact_no={contact_no}")
        
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('''
            SELECT * FROM users WHERE role = %s AND contact_no = %s AND password = %s AND assigned_municipality = %s
        ''', (role, contact_no, password, assigned_municipality))
        user = cursor.fetchone()
        conn.close()
        
        if user:
            unique_id = construct_unique_id(user['role'], assigned_municipality=assigned_municipality, contact_no=contact_no)
            session['unique_id'] = unique_id
            session['role'] = user['role']
            app.logger.debug(f"Web login successful for user: {unique_id} ({user['role']})")
            if user['role'] == 'cdrrmo':
                return redirect(url_for('cdrrmo_dashboard'))
            elif user['role'] == 'pnp':
                return redirect(url_for('pnp_dashboard'))
            elif user['role'] == 'bfp':
                return redirect(url_for('bfp_dashboard'))
        app.logger.warning(f"Web login failed for assigned_municipality: {assigned_municipality}, contact: {contact_no}, role: {role}")
        return "Invalid credentials", 401
    return render_template('CDRRMOPNPBFPIn.html')

# Navigation routes
@app.route('/go_to_login_page', methods=['GET'])
def go_to_login_page():
    app.logger.debug("Redirecting to /login")
    return redirect(url_for('login'))

@app.route('/go_to_signup_type', methods=['GET'])
def go_to_signup_type():
    app.logger.debug("Redirecting to /")
    return redirect(url_for('home'))

@app.route('/choose_login_type', methods=['GET'])
def chooese_login_type():
    app.logger.debug("Rendering LoginType.html")
    return render_template('LoginType.html')

@app.route('/go_to_cdrrmopnpbfpin', methods=['GET'])
def go_to_cdrrmopnpbfpin():
    app.logger.debug("Redirecting to /login_cdrrmo_pnp_bfp")
    return redirect(url_for('login_cdrrmo_pnp_bfp'))

@app.route('/signup_muna', methods=['GET'])
def signup_muna():
    app.logger.debug("Redirecting to /signup_cdrrmo_pnp_bfp")
    return redirect(url_for('signup_cdrrmo_pnp_bfp'))

@app.route('/signup_na', methods=['GET'])
def signup_na():
    app.logger.debug("Redirecting to /signup_barangay")
    return redirect(url_for('signup_barangay'))

@app.route('/logout')
def logout():
    role = session.pop('role', None)
    session.clear()
    app.logger.debug(f"User logged out. Redirecting from role: {role}")
    if role == 'barangay':
        return redirect(url_for('login'))
    else:
        return redirect(url_for('login_cdrrmo_pnp_bfp'))

# Alert handling
alerts = deque(maxlen=100)

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
        image_upload_time = data.get('imageUploadTime', datetime.now(pytz.utc).isoformat())
        upload_time = datetime.fromisoformat(image_upload_time.replace('Z', '+00:00'))
        if (datetime.now(pytz.utc) - upload_time).total_seconds() > 30 * 60:
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
            'responded': False
        }
        alerts.append(alert)
        socketio.emit('new_alert', alert)
        return jsonify({'status': 'success', 'message': 'Alert sent'}), 200
    except Exception as e:
        app.logger.error(f"Error processing send_alert: {e}", exc_info=True)
        return jsonify({'error': 'Internal server error'}), 500

# API endpoints
@app.route('/api/stats')
def get_stats():
    total = len(alerts)
    critical = len([a for a in alerts if a.get('emergency_type', '').lower() == 'critical'])
    return jsonify({'total': total, 'critical': critical})

@app.route('/api/distribution')
def get_distribution():
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

@app.route('/api/analytics')
def get_analytics():
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

# Dashboard routes
@app.route('/barangay_dashboard')
def barangay_dashboard():
    unique_id = session.get('unique_id')
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT * FROM users WHERE barangay = %s AND contact_no = %s
    ''', (unique_id.split('_')[0], unique_id.split('_')[1]))
    user = cursor.fetchone()
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

    app.logger.debug(f"Rendering BarangayDashboard for {barangay} in {assigned_municipality}")
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
    cursor = conn.cursor()
    cursor.execute('''
        SELECT * FROM users WHERE role = %s AND contact_no = %s AND assigned_municipality = %s
    ''', ('cdrrmo', unique_id.split('_')[2], unique_id.split('_')[1]))
    user = cursor.fetchone()
    conn.close()
    
    if not unique_id or not user or user['role'] != 'cdrrmo':
        app.logger.warning("Unauthorized access to cdrrmo_dashboard. Session: %s, User: %s", session, user)
        return redirect(url_for('login_cdrrmo_pnp_bfp'))
    
    assigned_municipality = user['assigned_municipality'] or "San Pablo City"
    stats = get_cdrrmo_stats()
    coords = municipality_coords.get(assigned_municipality, {'lat': 14.5995, 'lon': 120.9842})
    
    try:
        lat_coord = float(coords.get('lat', 14.5995))
        lon_coord = float(coords.get('lon', 120.9842))
    except (ValueError, TypeError):
        app.logger.error(f"Invalid coordinates for {assigned_municipality}, using defaults")
        lat_coord = 14.5995
        lon_coord = 120.9842

    app.logger.debug(f"Rendering CDRRMODashboard for {assigned_municipality}")
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
    cursor = conn.cursor()
    cursor.execute('''
        SELECT * FROM users WHERE role = %s AND contact_no = %s AND assigned_municipality = %s
    ''', ('pnp', unique_id.split('_')[2], unique_id.split('_')[1]))
    user = cursor.fetchone()
    conn.close()
    
    if not unique_id or not user or user['role'] != 'pnp':
        app.logger.warning("Unauthorized access to pnp_dashboard. Session: %s, User: %s", session, user)
        return redirect(url_for('login_cdrrmo_pnp_bfp'))
    
    assigned_municipality = user['assigned_municipality'] or "San Pablo City"
    stats = get_pnp_stats()
    coords = municipality_coords.get(assigned_municipality, {'lat': 14.5995, 'lon': 120.9842})
    
    try:
        lat_coord = float(coords.get('lat', 14.5995))
        lon_coord = float(coords.get('lon', 120.9842))
    except (ValueError, TypeError):
        app.logger.error(f"Invalid coordinates for {assigned_municipality}, using defaults")
        lat_coord = 14.5995
        lon_coord = 120.9842

    app.logger.debug(f"Rendering PNPDashboard for {assigned_municipality}")
    return render_template('PNPDashboard.html', 
                           stats=stats, 
                           municipality=assigned_municipality, 
                           lat_coord=lat_coord, 
                           lon_coord=lon_coord, 
                           google_api_key=GOOGLE_API_KEY)

@app.route('/bfp_dashboard')
def bfp_dashboard():
    unique_id = session.get('unique_id')
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT * FROM users WHERE role = %s AND contact_no = %s AND assigned_municipality = %s
    ''', ('bfp', unique_id.split('_')[2], unique_id.split('_')[1]))
    user = cursor.fetchone()
    conn.close()
    
    if not unique_id or not user or user['role'] != 'bfp':
        app.logger.warning("Unauthorized access to bfp_dashboard. Session: %s, User: %s", session, user)
        return redirect(url_for('login_cdrrmo_pnp_bfp'))
    
    assigned_municipality = user['assigned_municipality'] or "San Pablo City"
    stats = get_bfp_stats()
    coords = municipality_coords.get(assigned_municipality, {'lat': 14.5995, 'lon': 120.9842})
    
    try:
        lat_coord = float(coords.get('lat', 14.5995))
        lon_coord = float(coords.get('lon', 120.9842))
    except (ValueError, TypeError):
        app.logger.error(f"Invalid coordinates for {assigned_municipality}, using defaults")
        lat_coord = 14.5995
        lon_coord = 120.9842

    app.logger.debug(f"Rendering BFPDashboard for {assigned_municipality}")
    return render_template('BFPDashboard.html', 
                           stats=stats, 
                           municipality=assigned_municipality, 
                           lat_coord=lat_coord, 
                           lon_coord=lon_coord, 
                           google_api_key=GOOGLE_API_KEY)

# Analytics routes
@app.route('/barangay_analytics', methods=['GET'])
def barangay_analytics():
    if 'role' not in session or session['role'] != 'barangay':
        return redirect(url_for('login'))
    trends = get_barangay_trends()
    distribution = get_barangay_distribution()
    causes = get_barangay_causes()
    return render_template('BarangayAnalytics.html', trends=trends, distribution=distribution, causes=causes)

@app.route('/cdrrmo_analytics', methods=['GET'])
def cdrrmo_analytics():
teki    if 'role' not in session or session['role'] != 'cdrrmo':
        return redirect(url_for('login_cdrrmo_pnp_bfp'))
    trends = get_cdrrmo_trends()
    distribution = get_cdrrmo_distribution()
    causes = get_cdrrmo_causes()
    return render_template('CDRRMOAnalytics.html', trends=trends, distribution=distribution, causes=causes)

@app.route('/pnp_analytics', methods=['GET'])
def pnp_analytics():
    if 'role' not in session or session['role'] != 'pnp':
        return redirect(url_for('login_cdrrmo_pnp_bfp'))
    trends = get_pnp_trends()
    distribution = get_pnp_distribution()
    causes = get_pnp_causes()
    return render_template('PNPAnalytics.html', trends=trends, distribution=distribution, causes=causes)

@app.route('/bfp_analytics', methods=['GET'])
def bfp_analytics():
    if 'role' not in session or session['role'] != 'bfp':
        return redirect(url_for('login_cdrrmo_pnp_bfp'))
    trends = get_bfp_trends()
    distribution = get_bfp_distribution()
    causes = get_bfp_causes()
    return render_template('BFPAnalytics.html', trends=trends, distribution=distribution, causes=causes)

if __name__ == '__main__':
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                barangay TEXT NOT NULL,
                role TEXT NOT NULL,
                contact_no TEXT UNIQUE NOT NULL,
                assigned_municipality TEXT,
                province TEXT,
                password TEXT NOT NULL,
                PRIMARY KEY (barangay, contact_no)
            )
        ''')
        conn.commit()
        conn.close()
        logging.info("Database 'android_users' table initialized successfully or already exists.")
    except psycopg.Error as e:
        logging.error(f"Failed to initialize database: {e}", exc_info=True)

    port = int(os.environ.get('PORT', 5000))
    socketio.run(app, host="0.0.0.0", port=port, debug=True)
