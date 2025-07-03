from flask import Blueprint, request, redirect, url_for, render_template
from AlertNow import app  # Ensure you import the app instance
import sqlite3
import os

signup_bp = Blueprint('signup', __name__)



def get_db_connection():
    db_path = os.getenv('DB_PATH', os.path.join(os.path.dirname(__file__), 'data', 'users_web.db'))
    if not os.path.exists(db_path):
        if not os.path.exists(os.path.dirname(db_path)):
            os.makedirs(os.path.dirname(db_path))
        open(db_path, 'a').close()
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn

def get_connection_to_db():
    if os.getenv('RENDER') == 'true':  # Render sets this environment variable
        db_path = '/data/users_web.db'
    else:
        db_path = os.path.join(os.path.dirname(__file__), 'data', 'users_web.db')
    app.logger.debug(f"Database path: {db_path}")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn

@signup_bp.route('/signup_barangay', methods=['GET', 'POST'])
def signup_barangay():
    if request.method == 'POST':
        barangay = request.form['barangay']
        assigned_municipality = request.form['municipality']
        province = request.form['province']
        contact_no = request.form['contact_no']
        password = request.form['password']
        username = f"{barangay}_{contact_no}"

        conn = get_db_connection()
        try:
            conn.execute('''
                INSERT INTO users (barangay, role, contact_no, assigned_municipality, province, password)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (barangay, 'barangay', contact_no, assigned_municipality, province, password))
            conn.commit()
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            return "User already exists", 400
        except Exception as e:
            app.logger.error(f"Signup failed for {barangay}: {e}", exc_info=True)  # Use current_app
            return f"Signup failed: {e}", 500
        finally:
            conn.close()
    return render_template('SignUpPage.html')

@signup_bp.route('/signup_na', methods=['GET'])
def signup_na():
    return render_template('SignUpPage.html')