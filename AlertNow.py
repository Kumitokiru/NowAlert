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
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='gevent')
logging.basicConfig(level=logging.DEBUG)

# Ensure data directory exists
data_dir = os.path.join(os.path.dirname(__file__), 'data')
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
    logging.info(f"Created data directory at {data_dir}")

# Parse the internal database URL
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
        try:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO users (barangay, role, contact_no, assigned_municipality, province, password)
                VALUES (%s, %s, %s, %s, %s, %s)
            ''', (barangay, 'barangay', contact_no, assigned_municipality, province, password))
            conn.commit()
            app.logger.debug(f"User data inserted successfully: {unique_id}")
            return redirect(url_for('login'))
        except psycopg.errors.UniqueViolation as e:
            app.logger.error("IntegrityError: %s", e)
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

# Placeholder for barangay dashboard (add your implementation as needed)
@app.route('/barangay_dashboard')
def barangay_dashboard():
    if session.get('role') != 'barangay':
        return "Unauthorized", 403
    return render_template('BarangayDashboard.html')

if __name__ == '__main__':
    try:
        conn = get_db_connection()
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
        logging.info("Database 'android_users' table initialized successfully or already exists.")
    except psycopg.Error as e:
        logging.error(f"Failed to initialize database: {e}", exc_info=True)

    port = int(os.environ.get('PORT', 5000))
    socketio.run(app, host="0.0.0.0", port=port, debug=True, allow_unsafe_werkzeug=True)
