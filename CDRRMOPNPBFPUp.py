from flask import request, redirect, url_for, render_template
import sqlite3
import os
from AlertNow import app  # Import the Flask app instance from AlertNow.py

def get_db_connection():
    db_path = os.path.join(os.path.dirname(__file__), 'data', 'users_web.db')
    if not os.path.exists(os.path.dirname(db_path)):
        os.makedirs(os.path.dirname(db_path))
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn

def construct_unique_id(role, assigned_municipality, contact_no):
    """Constructs a unique identifier for CDRRMO or PNP users."""
    return f"{role}_{assigned_municipality}_{contact_no}"

def signup_cdrmo_pnp():
    if request.method == 'POST':
        role = request.form['role'].lower()  # Ensure role is lowercase (cdrrmo or pnp)
        assigned_municipality = request.form['municipality']
        contact_no = request.form['contact_no']
        password = request.form['password']
        unique_id = construct_unique_id(role, assigned_municipality, contact_no)
        
        conn = get_db_connection()
        try:
            # Check if contact_no already exists to ensure uniqueness
            existing_user = conn.execute('SELECT * FROM users WHERE contact_no = ?', (contact_no,)).fetchone()
            if existing_user:
                app.logger.error("Signup failed: Contact number %s already exists", contact_no)
                return "Contact number already exists", 400
            
            # Insert user data with barangay as NULL for CDRRMO/PNP
            conn.execute('''
                INSERT INTO users (barangay, role, contact_no, assigned_municipality, province, password)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (None, role, contact_no, assigned_municipality, None, password))
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

def signup_muna():
    return render_template('CDRRMOPNPUp.html')