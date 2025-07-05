from flask import Flask, redirect, request, session, url_for
import requests
import os
from urllib.parse import urlencode

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Google Fit API credentials (replace with your own)
GOOGLE_CLIENT_ID = os.getenv('GOOGLE_CLIENT_ID', 'YOUR_GOOGLE_CLIENT_ID')
GOOGLE_CLIENT_SECRET = os.getenv('GOOGLE_CLIENT_SECRET', 'YOUR_GOOGLE_CLIENT_SECRET')
GOOGLE_REDIRECT_URI = 'http://localhost:5000/google_fit_callback'

# Scopes for Google Fit (activity, heart rate, etc.)
GOOGLE_FIT_SCOPES = [
    'https://www.googleapis.com/auth/fitness.activity.read',
    'https://www.googleapis.com/auth/fitness.heart_rate.read',
    'https://www.googleapis.com/auth/fitness.body.read',
]

@app.route('/')
def home():
    return '<h2>Welcome! <a href="/google_fit_login">Connect Google Fit</a></h2>'

@app.route('/google_fit_login')
def google_fit_login():
    params = {
        'client_id': GOOGLE_CLIENT_ID,
        'redirect_uri': GOOGLE_REDIRECT_URI,
        'response_type': 'code',
        'scope': ' '.join(GOOGLE_FIT_SCOPES),
        'access_type': 'offline',
        'prompt': 'consent',
    }
    auth_url = 'https://accounts.google.com/o/oauth2/v2/auth?' + urlencode(params)
    return redirect(auth_url)

@app.route('/google_fit_callback')
def google_fit_callback():
    code = request.args.get('code')
    if not code:
        return 'No code provided.'
    # Exchange code for tokens
    token_url = 'https://oauth2.googleapis.com/token'
    data = {
        'code': code,
        'client_id': GOOGLE_CLIENT_ID,
        'client_secret': GOOGLE_CLIENT_SECRET,
        'redirect_uri': GOOGLE_REDIRECT_URI,
        'grant_type': 'authorization_code',
    }
    response = requests.post(token_url, data=data)
    if response.status_code != 200:
        return f'Error exchanging code: {response.text}'
    token_data = response.json()
    session['google_access_token'] = token_data['access_token']
    session['google_refresh_token'] = token_data.get('refresh_token')
    return redirect(url_for('google_fit_activities'))

@app.route('/google_fit_activities')
def google_fit_activities():
    access_token = session.get('google_access_token')
    if not access_token:
        return redirect(url_for('google_fit_login'))
    # Example: Fetch step count data
    headers = {'Authorization': f'Bearer {access_token}'}
    dataset = '0-{}'.format(int((requests.get('https://www.googleapis.com/fitness/v1/users/me/dataSources').elapsed.total_seconds())*1e9))
    url = 'https://www.googleapis.com/fitness/v1/users/me/dataset:aggregate'
    # Example request for steps (see Google Fit API docs for more data types)
    body = {
        "aggregateBy": [{
            "dataTypeName": "com.google.step_count.delta",
            "dataSourceId": "derived:com.google.step_count.delta:com.google.android.gms:estimated_steps"
        }],
        "bucketByTime": { "durationMillis": 86400000 },
        "startTimeMillis": 0,
        "endTimeMillis": int((requests.get('https://www.googleapis.com/fitness/v1/users/me/dataSources').elapsed.total_seconds())*1000)
    }
    res = requests.post(url, headers=headers, json=body)
    if res.status_code != 200:
        return f'Error fetching Google Fit data: {res.text}'
    data = res.json()
    return f'<pre>{data}</pre>'

if __name__ == '__main__':
    app.run(debug=True) 