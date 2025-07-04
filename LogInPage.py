from flask import request, redirect, url_for, render_template
import requests

def login_page():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        payload = {'username': username, 'password': password}
        response = requests.post('https://your-server.com/login', json=payload)
        if response.status_code == 200:
            data = response.json()
            role = data.get('role')
            if role == 'official':
                return redirect(url_for('barangay_dashboard'))
            elif role == 'cdrmo':
                return redirect(url_for('cdrrmo_dashboard'))
            elif role == 'pnp':
                return redirect(url_for('pnp_dashboard'))
            elif role == 'bfp':
                return redirect(url_for('bfp_dashboard'))
        return "Invalid credentials", 401
    return render_template('LogInPage.html')

def choose_login_type():
    return render_template('LoginType.html')
