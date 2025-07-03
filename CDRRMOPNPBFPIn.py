from flask import request, redirect, url_for, render_template
import requests

def login_cdrmo_pnp():
    if request.method == 'POST':
        municipality = request.form['municipality']
        contact_no = request.form['contact_no']
        password = request.form['password']
        payload = {
            'municipality': municipality,
            'contact_no': contact_no,
            'password': password
        }
        response = requests.post('https://your-server.com/login_cdrmo_pnp', json=payload)
        if response.status_code == 200:
            data = response.json()
            role = data.get('role')
            if role == 'cdrmo':
                return redirect(url_for('cdrrmo_dashboard'))
            elif role == 'pnp':
                return redirect(url_for('pnp_dashboard'))
        return "Invalid credentials", 401
    return render_template('CDRRMOPNPIn.html')

def chooese_login_type():
    return render_template('LoginType.html')

def go_to_cdrrmopnpbfpin():
    return render_template('CDRRMOPNPBFPIn.html')
