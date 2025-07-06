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
from collections import Counter, deque
from datetime import datetime
import pytz
import pickle
import pandas as pd

from BarangayDashboard import get_barangay_stats, get_latest_alert
from CDRRMODashboard import get_cdrrmo_stats
from PNPDashboard import get_pnp_stats
from BFPDashboard import get_bfp_stats

# Import analytics functions
from BarangayAnalytics import (
    get_barangay_trends,
    get_barangay_distribution,
    get_barangay_causes,
    get_barangay_weather_impact,
    get_barangay_road_conditions,
    get_barangay_vehicle_types,
    get_barangay_driver_age,
    get_barangay_driver_gender,
    get_barangay_accident_type,
    get_barangay_injuries,
    get_barangay_fatalities
)

from CDRRMOAnalytics import (
    get_cdrrmo_trends,
    get_cdrrmo_distribution,
    get_cdrrmo_causes,
    get_cdrrmo_weather_impact,
    get_cdrrmo_road_conditions,
    get_cdrrmo_vehicle_types,
    get_cdrrmo_driver_age,
    get_cdrrmo_driver_gender,
    get_cdrrmo_accident_type,
    get_cdrrmo_injuries,
    get_cdrrmo_fatalities
)

from PNPAnalytics import (
    get_pnp_trends,
    get_pnp_distribution,
    get_pnp_causes,
    get_pnp_weather_impact,
    get_pnp_road_conditions,
    get_pnp_vehicle_types,
    get_pnp_driver_age,
    get_pnp_driver_gender,
    get_pnp_accident_type,
    get_pnp_injuries,
    get_pnp_fatalities
)

from BFPAnalytics import (
    get_bfp_trends,
    get_bfp_distribution,
    get_bfp_causes,
    get_bfp_weather_impact,
    get_bfp_property_types,
    get_bfp_fire_severity,
    get_bfp_casualty_count,
    get_bfp_response_time,
    get_bfp_fire_duration
)

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load datasets
road_accident_df = pd.DataFrame()
try:
    road_accident_df = pd.read_csv('dataset/road_accident.csv')
    logger.info("Successfully loaded road_accident.csv")
except FileNotFoundError:
    logger.error("road_accident.csv not found in dataset directory")
except Exception as e:
    logger.error(f"Error loading road_accident.csv: {e}")

fire_incident_df = pd.DataFrame()
try:
    fire_incident_df = pd.read_csv('dataset/fire_incident.csv')
    logger.info("Successfully loaded fire_incident.csv")
except FileNotFoundError:
    logger.error("fire_incident.csv not found in dataset directory")
except Exception as e:
    logger.error(f"Error loading fire_incident.csv: {e}")

app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'your-secret-key-here')
socketio = SocketIO(app, cors_allowed_origins="*")

# Initialize alerts deque
alerts = deque(maxlen=100)

# SocketIO event for alert response
@socketio.on('responded')
def handle_responded(data):
    timestamp = data.get('timestamp')
    for alert in alerts:
        if alert.get('timestamp') == timestamp:
            alert['responded'] = True
            break
    socketio.emit('alert_responded', {
        'timestamp': timestamp,
        'lat': data.get('lat'),
        'lon': data.get('lon'),
        'barangay': data.get('barangay'),
        'emergency_type': data.get('emergency_type')
    })

# Load machine learning models with fallbacks
dt_classifier = None
try:
    dt_classifier = joblib.load('training/decision_tree_model.pkl')
    logging.info("decision_tree_model.pkl loaded successfully.")
except FileNotFoundError:
    logging.error("decision_tree_model.pkl not found. ML prediction will not work.")
except Exception as e:
    logging.error(f"Error loading decision_tree_model.pkl: {e}")

lr_fire = rf_fire = svm_fire = xgb_fire = None
try:
    lr_fire = joblib.load('training/Fire Models/lr_fire_incident.pkl')
    rf_fire = joblib.load('training/Fire Models/rf_fire_incident.pkl')
    svm_fire = joblib.load('training/Fire Models/svm_fire_incident.pkl')
    xgb_fire = joblib.load('training/Fire Models/xgb_fire_incident.pkl')
    logging.info("Fire incident models loaded successfully.")
except Exception as e:
    logging.error(f"Error loading fire incident models: {e}")

lr_road = rf_road = svm_road = xgb_road = None
try:
    lr_road = joblib.load('training/Road Models/lr_road_accident.pkl')
    rf_road = joblib.load('training/Road Models/rf_road_accident.pkl')
    svm_road = joblib.load('training/Road Models/svm_road_accident.pkl')
    xgb_road = joblib.load('training/Road Models/xgb_road_accident.pkl')
    logging.info("Road accident models loaded successfully.")
except Exception as e:
    logging.error(f"Error loading road accident models: {e}")

GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY', 'your-google-api-key-here')
barangay_coords = {}
try:
    with open(os.path.join('assets', 'coords.txt'), 'r') as f:
        barangay_coords = ast.literal_eval(f.read())
except FileNotFoundError:
    logging.error("coords.txt not found in assets directory. Using empty dict.")
except Exception as e:
    logging.error(f"Error loading coords.txt: {e}. Using empty dict.")

# Municipality coordinates
municipality_coords = {
    "San Pablo City": {"lat": 14.0642, "lon": 121.3233},
    "Quezon Province": {"lat": 13.9347, "lon": 121.9473},
}

def get_db_connection():
    db_path = os.path.join(os.path.dirname(__file__), 'database', 'users_web.db')
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn

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
            logger.error(f"Signup failed for {unique_id}: {e}", exc_info=True)
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
            session['unique_id'] = unique_id
            session['role'] = user['role']
            logger.debug(f"Web login successful for barangay: {unique_id}")
            return redirect(url_for('barangay_dashboard'))
        logger.warning(f"Web login failed for unique_id: {unique_id}")
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
            logger.error(f"Signup failed for {unique_id}: {e}", exc_info=True)
            return f"Signup failed: {e}", 500
        finally:
            conn.close()
    return render_template('CDRRMOPNPBFPUp.html')

@app.route('/login_cdrrmo_pnp_bfp', methods=['GET', 'POST'])
def login_cdrrmo_pnp_bfp():
    logger.debug("Accessing /login_cdrrmo_pnp_bfp with method: %s", request.method)
    if request.method == 'POST':
        role = request.form['role'].lower()
        assigned_municipality = request.form['municipality']
        contact_no = request.form['contact_no']
        password = request.form['password']
        
        if role not in ['cdrrmo', 'pnp', 'bfp']:
            logger.error(f"Invalid role provided: {role}")
            return "Invalid role", 400
        
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

@app.route('/send_alert', methods=['POST'])
def send_alert():
    try:
        data = request.get_json()
        if not data:
            logger.error("No data provided in send_alert")
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
            'timestamp': datetime.now(pytz.timezone('America/Los_Angeles')).isoformat(),
            'imageUploadTime': image_upload_time,
            'responded': False
        }
        alerts.append(alert)
        socketio.emit('new_alert', alert)
        logger.debug("Alert sent successfully")
        return jsonify({'status': 'success', 'message': 'Alert sent'}), 200
    except Exception as e:
        logger.error(f"Error processing send_alert: {e}", exc_info=True)
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/stats')
def get_stats():
    try:
        total = len(alerts)
        critical = len([a for a in alerts if a.get('emergency_type', '').lower() == 'critical'])
        return jsonify({'total': total, 'critical': critical})
    except Exception as e:
        logger.error(f"Error in get_stats: {e}", exc_info=True)
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
        logger.error(f"Error in get_distribution: {e}", exc_info=True)
        return jsonify({'error': 'Failed to retrieve distribution'}), 500

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
        logger.error(f"Error in get_analytics: {e}", exc_info=True)
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
        logger.error(f"Image prediction failed: {e}", exc_info=True)
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
        logger.warning("Unauthorized access to barangay_dashboard. Session: %s, User: %s", session, user)
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
        logger.error(f"Invalid coordinates for {barangay} in {assigned_municipality}, using defaults")
        lat_coord = 14.5995
        lon_coord = 120.9842

    logger.debug(f"Rendering BarangayDashboard for {barangay} in {assigned_municipality}")
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
        logger.warning("Unauthorized access to cdrrmo_dashboard. Session: %s, User: %s", session, user)
        return redirect(url_for('login_cdrrmo_pnp_bfp'))
    
    assigned_municipality = user['assigned_municipality'] or "San Pablo City"
    stats = get_cdrrmo_stats()
    coords = municipality_coords.get(assigned_municipality, {'lat': 14.5995, 'lon': 120.9842})
    
    try:
        lat_coord = float(coords.get('lat', 14.5995))
        lon_coord = float(coords.get('lon', 120.9842))
    except (ValueError, TypeError):
        logger.error(f"Invalid coordinates for {assigned_municipality}, using defaults")
        lat_coord = 14.5995
        lon_coord = 120.9842

    logger.debug(f"Rendering CDRRMODashboard for {assigned_municipality}")
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
        logger.warning("Unauthorized access to pnp_dashboard. Session: %s, User: %s", session, user)
        return redirect(url_for('login_cdrrmo_pnp_bfp'))
    
    assigned_municipality = user['assigned_municipality'] or "San Pablo City"
    stats = get_pnp_stats()
    coords = municipality_coords.get(assigned_municipality, {'lat': 14.5995, 'lon': 120.9842})
    
    try:
        lat_coord = float(coords.get('lat', 14.5995))
        lon_coord = float(coords.get('lon', 120.9842))
    except (ValueError, TypeError):
        logger.error(f"Invalid coordinates for {assigned_municipality}, using defaults")
        lat_coord = 14.5995
        lon_coord = 120.9842

    logger.debug(f"Rendering PNPDashboard for {assigned_municipality}")
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
    user = conn.execute('''
        SELECT * FROM users WHERE role = ? AND contact_no = ? AND assigned_municipality = ?
    ''', ('bfp', unique_id.split('_')[2], unique_id.split('_')[1])).fetchone()
    conn.close()
    
    if not unique_id or not user or user['role'] != 'bfp':
        logger.warning("Unauthorized access to bfp_dashboard. Session: %s, User: %s", session, user)
        return redirect(url_for('login_cdrrmo_pnp_bfp'))
    
    assigned_municipality = user['assigned_municipality'] or "San Pablo City"
    stats = get_bfp_stats()
    coords = municipality_coords.get(assigned_municipality, {'lat': 14.5995, 'lon': 120.9842})
    
    try:
        lat_coord = float(coords.get('lat', 14.5995))
        lon_coord = float(coords.get('lon', 120.9842))
    except (ValueError, TypeError):
        logger.error(f"Invalid coordinates for {assigned_municipality}, using defaults")
        lat_coord = 14.5995
        lon_coord = 120.9842

    logger.debug(f"Rendering BFPDashboard for {assigned_municipality}")
    return render_template('BFPDashboard.html', 
                           stats=stats, 
                           municipality=assigned_municipality, 
                           lat_coord=lat_coord,
                           lon_coord=lon_coord,
                           google_api_key=GOOGLE_API_KEY)

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
    current_datetime = datetime.now(pytz.timezone('America/Los_Angeles')).strftime('%a/%m/%d/%y %H:%M:%S')
    return render_template('BarangayAnalytics.html', barangay=barangay, current_datetime=current_datetime)

@app.route('/api/barangay_analytics_data', methods=['GET'])
def get_barangay_analytics_data():
    try:
        time_filter = request.args.get('time', 'weekly')
        trends = get_barangay_trends(time_filter)
        distribution = get_barangay_distribution(time_filter)
        causes = get_barangay_causes(time_filter)
        weather = get_barangay_weather_impact(time_filter)
        road_conditions = get_barangay_road_conditions(time_filter)
        vehicle_types = get_barangay_vehicle_types(time_filter)
        driver_age = get_barangay_driver_age(time_filter)
        driver_gender = get_barangay_driver_gender(time_filter)
        accident_type = get_barangay_accident_type(time_filter)
        injuries = get_barangay_injuries(time_filter)
        fatalities = get_barangay_fatalities(time_filter)
        return jsonify({
            'trends': trends,
            'distribution': distribution,
            'causes': causes,
            'weather': weather,
            'road_conditions': road_conditions,
            'vehicle_types': vehicle_types,
            'driver_age': driver_age,
            'driver_gender': driver_gender,
            'accident_type': accident_type,
            'injuries': injuries,
            'fatalities': fatalities
        })
    except Exception as e:
        logger.error(f"Error in get_barangay_analytics_data: {e}", exc_info=True)
        return jsonify({'error': 'Failed to retrieve analytics data'}), 500

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
    current_datetime = datetime.now(pytz.timezone('America/Los_Angeles')).strftime('%a/%m/%d/%y %H:%M:%S')
    barangays = ["Barangay 1", "Barangay 2", "Barangay 3"]  # Replace with actual database query
    return render_template('CDRRMOAnalytics.html', municipality=municipality, current_datetime=current_datetime, barangays=barangays)

@app.route('/api/cdrrmo_analytics_data', methods=['GET'])
def get_cdrrmo_analytics_data():
    try:
        time_filter = request.args.get('time', 'weekly')
        barangay = request.args.get('barangay', '')
        trends = get_cdrrmo_trends(time_filter, barangay)
        distribution = get_cdrrmo_distribution(time_filter, barangay)
        causes = get_cdrrmo_causes(time_filter, barangay)
        weather = get_cdrrmo_weather_impact(time_filter, barangay)
        road_conditions = get_cdrrmo_road_conditions(time_filter, barangay)
        vehicle_types = get_cdrrmo_vehicle_types(time_filter, barangay)
        driver_age = get_cdrrmo_driver_age(time_filter, barangay)
        driver_gender = get_cdrrmo_driver_gender(time_filter, barangay)
        accident_type = get_cdrrmo_accident_type(time_filter, barangay)
        injuries = get_cdrrmo_injuries(time_filter, barangay)
        fatalities = get_cdrrmo_fatalities(time_filter, barangay)
        return jsonify({
            'trends': trends,
            'distribution': distribution,
            'causes': causes,
            'weather': weather,
            'road_conditions': road_conditions,
            'vehicle_types': vehicle_types,
            'driver_age': driver_age,
            'driver_gender': driver_gender,
            'accident_type': accident_type,
            'injuries': injuries,
            'fatalities': fatalities
        })
    except Exception as e:
        logger.error(f"Error in get_cdrrmo_analytics_data: {e}", exc_info=True)
        return jsonify({'error': 'Failed to retrieve analytics data'}), 500

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
    current_datetime = datetime.now(pytz.timezone('America/Los_Angeles')).strftime('%a/%m/%d/%y %H:%M:%S')
    barangays = ["Barangay 1", "Barangay 2", "Barangay 3"]  # Replace with actual database query
    return render_template('PNPAnalytics.html', municipality=municipality, current_datetime=current_datetime, barangays=barangays)

@app.route('/api/pnp_analytics_data', methods=['GET'])
def get_pnp_analytics_data():
    try:
        time_filter = request.args.get('time', 'weekly')
        trends = get_pnp_trends(time_filter)
        distribution = get_pnp_distribution(time_filter)
        causes = get_pnp_causes(time_filter)
        weather = get_pnp_weather_impact(time_filter)
        road_conditions = get_pnp_road_conditions(time_filter)
        vehicle_types = get_pnp_vehicle_types(time_filter)
        driver_age = get_pnp_driver_age(time_filter)
        driver_gender = get_pnp_driver_gender(time_filter)
        accident_type = get_pnp_accident_type(time_filter)
        injuries = get_pnp_injuries(time_filter)
        fatalities = get_pnp_fatalities(time_filter)
        return jsonify({
            'trends': trends,
            'distribution': distribution,
            'causes': causes,
            'weather': weather,
            'road_conditions': road_conditions,
            'vehicle_types': vehicle_types,
            'driver_age': driver_age,
            'driver_gender': driver_gender,
            'accident_type': accident_type,
            'injuries': injuries,
            'fatalities': fatalities
        })
    except Exception as e:
        logger.error(f"Error in get_pnp_analytics_data: {e}", exc_info=True)
        return jsonify({'error': 'Failed to retrieve analytics data'}), 500

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
    current_datetime = datetime.now(pytz.timezone('America/Los_Angeles')).strftime('%a/%m/%d/%y %H:%M:%S')
    barangays = ["Barangay 1", "Barangay 2", "Barangay 3"]  # Replace with actual database query
    return render_template('BFPAnalytics.html', municipality=municipality, current_datetime=current_datetime, barangays=barangays)

@app.route('/api/bfp_analytics_data', methods=['GET'])
def get_bfp_analytics_data():
    try:
        time_filter = request.args.get('time', 'weekly')
        barangay = request.args.get('barangay', '')
        trends = get_bfp_trends(time_filter, barangay)
        distribution = get_bfp_distribution(time_filter, barangay)
        causes = get_bfp_causes(time_filter, barangay)
        weather = get_bfp_weather_impact(time_filter, barangay)
        property_types = get_bfp_property_types(time_filter, barangay)
        fire_severity = get_bfp_fire_severity(time_filter, barangay)
        casualty_count = get_bfp_casualty_count(time_filter, barangay)
        response_time = get_bfp_response_time(time_filter, barangay)
        fire_duration = get_bfp_fire_duration(time_filter, barangay)
        return jsonify({
            'trends': trends,
            'distribution': distribution,
            'causes': causes,
            'weather': weather,
            'property_types': property_types,
            'fire_severity': fire_severity,
            'casualty_count': casualty_count,
            'response_time': response_time,
            'fire_duration': fire_duration
        })
    except Exception as e:
        logger.error(f"Error in get_bfp_analytics_data: {e}", exc_info=True)
        return jsonify({'error': 'Failed to retrieve analytics data'}), 500

def get_latest_alert():
    try:
        if alerts:
            return list(alerts)[-1]
        return None
    except Exception as e:
        logger.error(f"Error in get_latest_alert: {e}", exc_info=True)
        return None

def get_barangay_stats():
    try:
        types = [a.get('emergency_type', 'unknown') for a in alerts if a.get('role') == 'barangay' or a.get('barangay')]
        return Counter(types)
    except Exception as e:
        logger.error(f"Error in get_barangay_stats: {e}", exc_info=True)
        return Counter()

def get_cdrrmo_stats():
    try:
        types = [a.get('emergency_type', 'unknown') for a in alerts if a.get('role') == 'cdrrmo' or a.get('assigned_municipality')]
        return Counter(types)
    except Exception as e:
        logger.error(f"Error in get_cdrrmo_stats: {e}", exc_info=True)
        return Counter()

def get_pnp_stats():
    try:
        types = [a.get('emergency_type', 'unknown') for a in alerts if a.get('role') == 'pnp' or a.get('assigned_municipality')]
        return Counter(types)
    except Exception as e:
        logger.error(f"Error in get_pnp_stats: {e}", exc_info=True)
        return Counter()

def get_bfp_stats():
    try:
        types = [a.get('emergency_type', 'unknown') for a in alerts if a.get('role') == 'bfp' or a.get('assigned_municipality')]
        return Counter(types)
    except Exception as e:
        logger.error(f"Error in get_bfp_stats: {e}", exc_info=True)
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
        logger.error(f"Failed to initialize database: {e}", exc_info=True)

    port = int(os.environ.get('PORT', 5000))
    socketio.run(app, host="0.0.0.0", port=port, debug=True, allow_unsafe_werkzeug=True)