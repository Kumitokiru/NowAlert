from flask import request, redirect, url_for, render_template
from AlertNow import app, get_db_connection, construct_unique_id
import psycopg2

def signup_cdrrmo_pnp_bfp():
    if request.method == 'POST':
        role = request.form['role'].lower()
        assigned_municipality = request.form['municipality']
        contact_no = request.form['contact_no']
        password = request.form['password']
        unique_id = construct_unique_id(role, assigned_municipality=assigned_municipality, contact_no=contact_no)
        
        conn = get_db_connection()
        try:
            cursor = conn.cursor()
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
        except psycopg2.IntegrityError as e:
            app.logger.error("IntegrityError during signup: %s", e)
            return "User already exists", 400
        except Exception as e:
            app.logger.error(f"Signup failed for {unique_id}: {e}", exc_info=True)
            return f"Signup failed: {e}", 500
        finally:
            conn.close()
    return render_template('CDRRMOPNPBFPUp.html')

def signup_muna():
    return redirect(url_for('signup_cdrrmo_pnp_bfp'))
