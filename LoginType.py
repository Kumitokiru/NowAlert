from flask import render_template

def login_type():
    return render_template('LoginType.html')

def go_to_signup_type():
    return render_template('SignUpType.html')