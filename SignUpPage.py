from flask import Blueprint, request, redirect, url_for, render_template
from AlertNow import app, get_db_connection, construct_unique_id
import psycopg2

signup_bp = Blueprint('signup', __name__)

@signup_bp.route('/signup_barangay', methods=['GET', 'POST'])
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
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO users (barangay, role, contact_no, assigned_municipality, province, password)
                VALUES (%s, %s, %s, %s, %s, %s)
            ''', (barangay, 'barangay', contact_no, assigned_municipality, province, password))
            conn.commit()
            app.logger.debug(f"User data inserted successfully: {unique_id}")
            return redirect(url_for('login'))
        except psycopg2.IntegrityError as e:
            app.logger.error("IntegrityError: %s", e)
            return "User already exists", 400
        except Exception as e:
            app.logger.error(f"Exception during signup: {e}", exc_info=True)
            return f"Signup failed: {e}", 500
        finally:
            conn.close()
    return render_template('SignUpPage.html')

@signup_bp.route('/signup_na', methods=['GET'])
def signup_na():
    return redirect(url_for('signup.signup_barangay'))
