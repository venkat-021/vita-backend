from flask import Flask, redirect, request, session, url_for
import requests
from urllib.parse import urlencode
import os
from dotenv import load_dotenv
import sqlite3

# Load environment variables
load_dotenv()
load_dotenv(dotenv_path='.env_google')

app = Flask(__name__)
app.secret_key = os.urandom(24)  # For session management

# Strava API credentials
STRAVA_CLIENT_ID = os.getenv('STRAVA_CLIENT_ID')
STRAVA_CLIENT_SECRET = os.getenv('STRAVA_CLIENT_SECRET')
REDIRECT_URI = 'http://localhost:5000/exchange_token'

print("STRAVA_CLIENT_ID:", STRAVA_CLIENT_ID)
print("STRAVA_CLIENT_SECRET:", STRAVA_CLIENT_SECRET)

@app.route('/')
def index():
    return '''
    <h1>Welcome to Health System Strava Integration</h1>
    <a href="/strava_login">Login with Strava</a>
    <p>
        <b>Tip:</b> On the Strava login page, you can log in with your Strava email and password,
        <br>
        <b>or</b> click <b>"Sign in with Google"</b> if your Strava account is linked to Google.
    </p>
    '''

@app.route('/strava_login')
def strava_login():
    params = {
        'client_id': STRAVA_CLIENT_ID,
        'redirect_uri': REDIRECT_URI,
        'response_type': 'code',
        'scope': 'read,activity:read',
        'approval_prompt': 'force'
    }
    
    auth_url = f"https://www.strava.com/oauth/authorize?{urlencode(params)}"
    return redirect(auth_url)

@app.route('/exchange_token')
def exchange_token():
    if 'error' in request.args:
        return f"Error: {request.args['error']}"
    
    code = request.args.get('code')
    if not code:
        return "No code provided"
    
    # Exchange code for token
    token_url = "https://www.strava.com/oauth/token"
    data = {
        'client_id': STRAVA_CLIENT_ID,
        'client_secret': STRAVA_CLIENT_SECRET,
        'code': code,
        'grant_type': 'authorization_code'
    }
    
    response = requests.post(token_url, data=data)
    if response.status_code != 200:
        return f"Error getting token: {response.text}"
    
    token_data = response.json()
    session['access_token'] = token_data['access_token']
    session['refresh_token'] = token_data['refresh_token']
    
    return redirect(url_for('activities'))

@app.route('/activities')
def activities():
    if 'access_token' not in session:
        return redirect(url_for('strava_login'))
    
    headers = {'Authorization': f"Bearer {session['access_token']}"}
    activities_url = "https://www.strava.com/api/v3/athlete/activities"
    
    response = requests.get(activities_url, headers=headers)
    if response.status_code != 200:
        return f"Error fetching activities: {response.text}"
    
    activities = response.json()
    
    # Store activities in the database
    store_activities(activities)
    
    # Format activities for display
    html = "<h1>Your Recent Activities</h1>"
    for activity in activities[:5]:  # Show last 5 activities
        html += f"""
        <div style='margin: 20px; padding: 10px; border: 1px solid #ccc;'>
            <h3>{activity['name']}</h3>
            <p>Type: {activity['type']}</p>
            <p>Distance: {activity['distance']/1000:.2f} km</p>
            <p>Duration: {activity['moving_time']/60:.2f} minutes</p>
            <p>Calories: {activity.get('calories', 'N/A')}</p>
        </div>
        """
    
    return html

@app.route('/refresh_token')
def refresh_token():
    if 'refresh_token' not in session:
        return redirect(url_for('strava_login'))
    
    data = {
        'client_id': STRAVA_CLIENT_ID,
        'client_secret': STRAVA_CLIENT_SECRET,
        'grant_type': 'refresh_token',
        'refresh_token': session['refresh_token']
    }
    
    response = requests.post("https://www.strava.com/oauth/token", data=data)
    if response.status_code != 200:
        return f"Error refreshing token: {response.text}"
    
    token_data = response.json()
    session['access_token'] = token_data['access_token']
    session['refresh_token'] = token_data['refresh_token']
    
    return redirect(url_for('activities'))

def store_activities(activities):
    conn = sqlite3.connect('health_analysis.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS strava_activities (
            id INTEGER PRIMARY KEY,
            name TEXT,
            type TEXT,
            distance REAL,
            moving_time INTEGER,
            calories REAL
        )
    ''')
    for activity in activities:
        c.execute('''
            INSERT INTO strava_activities (name, type, distance, moving_time, calories)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            activity['name'],
            activity['type'],
            activity['distance'],
            activity['moving_time'],
            activity.get('calories')
        ))
    conn.commit()
    conn.close()

def get_all_activities():
    conn = sqlite3.connect('health_analysis.db')
    c = conn.cursor()
    c.execute('SELECT * FROM strava_activities')
    activities = c.fetchall()
    conn.close()
    return activities

@app.route('/all_activities')
def all_activities():
    activities = get_all_activities()
    html = "<h1>All Stored Strava Activities</h1>"
    for activity in activities:
        html += f"""
        <div style='margin: 20px; padding: 10px; border: 1px solid #ccc;'>
            <h3>{activity[1]}</h3>
            <p>Type: {activity[2]}</p>
            <p>Distance: {activity[3]/1000:.2f} km</p>
            <p>Duration: {activity[4]/60:.2f} minutes</p>
            <p>Calories: {activity[5]}</p>
        </div>
        """
    return html

def get_activity_details(activity_id, access_token):
    url = f"https://www.strava.com/api/v3/activities/{activity_id}"
    headers = {"Authorization": f"Bearer {access_token}"}
    res = requests.get(url, headers=headers)
    return res.json()

@app.route('/google_fit_login')
def google_fit_login():
    ...

if __name__ == '__main__':
    app.run(debug=True) 