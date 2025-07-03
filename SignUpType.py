from flask import render_template

def signup_type():
    return render_template('SignUpType.html')

def login_type():
    return render_template('LoginType.html')