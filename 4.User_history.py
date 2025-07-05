import sqlite3
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from matplotlib.gridspec import GridSpec

def get_db_connection():
    """Create a database connection"""
    try:
        conn = sqlite3.connect("health_analysis.db")
        conn.row_factory = sqlite3.Row
        return conn
    except sqlite3.Error as e:
        print(f"Database connection error: {e}")
        raise

def create_history_table():
    """Create the user_measurement_history table if it doesn't exist"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_measurement_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                date_recorded DATE NOT NULL,
                weight_kg REAL NOT NULL,
                height_cm REAL NOT NULL,
                waist_cm REAL NOT NULL,
                hip_cm REAL NOT NULL,
                arm_length_cm REAL NOT NULL,
                leg_length_cm REAL NOT NULL,
                arm_circumference_cm REAL NOT NULL,
                bmi REAL NOT NULL,
                whr REAL NOT NULL,
                whtr REAL NOT NULL,
                bfp REAL NOT NULL,
                lbm REAL NOT NULL,
                smm REAL NOT NULL,
                fat_mass REAL NOT NULL,
                bmr REAL NOT NULL,
                bsa REAL NOT NULL,
                ibw REAL NOT NULL,
                bai REAL NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        conn.commit()
    except sqlite3.Error as e:
        print(f"Error creating history table: {e}")
        raise
    finally:
        conn.close()

def save_measurement_history(user_id, measurements):
    """Save current measurements to history"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Calculate all metrics
        height_m = measurements['height_cm'] / 100
        bmi = measurements['weight_kg'] / (height_m ** 2)
        whr = measurements['waist_cm'] / measurements['hip_cm']
        whtr = measurements['waist_cm'] / measurements['height_cm']
        
        # Calculate age for BFP
        from BMI_calc import calculate_age, calculate_bfp, calculate_lbm, calculate_smm
        from BMI_calc import calculate_bmr, calculate_bsa, calculate_ibw, calculate_bai
        
        user_data = get_user_data(user_id)
        age = calculate_age(user_data['date_of_birth'])
        gender = user_data['gender']
        
        bfp = calculate_bfp(bmi, age, gender)
        lbm = calculate_lbm(measurements['weight_kg'], measurements['height_cm'], gender)
        smm = calculate_smm(lbm, gender)
        fat_mass = measurements['weight_kg'] - lbm
        bmr = calculate_bmr(measurements['weight_kg'], measurements['height_cm'], age, gender)
        bsa = calculate_bsa(measurements['weight_kg'], measurements['height_cm'])
        ibw = calculate_ibw(measurements['height_cm'], gender)
        bai = calculate_bai(measurements['hip_cm'], height_m)
        
        cursor.execute('''
            INSERT INTO user_measurement_history (
                user_id, date_recorded, weight_kg, height_cm, waist_cm, hip_cm,
                arm_length_cm, leg_length_cm, arm_circumference_cm,
                bmi, whr, whtr, bfp, lbm, smm, fat_mass,
                bmr, bsa, ibw, bai
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            user_id, datetime.now().date(), measurements['weight_kg'],
            measurements['height_cm'], measurements['waist_cm'],
            measurements['hip_cm'], measurements['arm_length_cm'],
            measurements['leg_length_cm'], measurements['arm_circumference_cm'],
            bmi, whr, whtr, bfp, lbm, smm, fat_mass,
            bmr, bsa, ibw, bai
        ))
        
        conn.commit()
    except sqlite3.Error as e:
        print(f"Error saving measurement history: {e}")
        raise
    except KeyError as e:
        print(f"Missing required measurement: {e}")
        raise
    finally:
        conn.close()

def get_user_data(user_id):
    """Get user data from the database"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT u.username, p.*
            FROM users u
            JOIN user_profiles p ON u.id = p.user_id
            WHERE u.id = ?
        ''', (user_id,))
        
        user_data = cursor.fetchone()
        if not user_data:
            raise Exception("User data not found")
            
        return user_data
    except Exception as e:
        print(f"Error fetching user data: {e}")
        raise
    finally:
        conn.close()

def get_measurement_history(user_id):
    """Get all measurement history for a user"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # First check if the table exists
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='user_measurement_history'
        """)
        if not cursor.fetchone():
            print("❌ Error: user_measurement_history table does not exist!")
            print("Please run the script with a user who has measurement data.")
            return None
        
        # Then check if there's data for this user
        cursor.execute('''
            SELECT COUNT(*) 
            FROM user_measurement_history 
            WHERE user_id = ?
        ''', (user_id,))
        
        count = cursor.fetchone()[0]
        if count == 0:
            print(f"❌ No measurement history found for user ID: {user_id}")
            print("Please add some measurements first.")
            return None
            
        # If we have data, fetch it with proper date formatting
        cursor.execute('''
            SELECT 
                id,
                user_id,
                strftime('%Y-%m-%d', date_recorded) as date_recorded,
                weight_kg,
                height_cm,
                waist_cm,
                hip_cm,
                arm_length_cm,
                leg_length_cm,
                arm_circumference_cm,
                bmi,
                whr,
                whtr,
                bfp,
                lbm,
                smm,
                fat_mass,
                bmr,
                bsa,
                ibw,
                bai
            FROM user_measurement_history
            WHERE user_id = ?
            ORDER BY date_recorded ASC
        ''', (user_id,))
        
        history = cursor.fetchall()
        print(f"✅ Found {len(history)} measurements for user ID: {user_id}")
        
        # Debug: Print the first record to check date format
        if history:
            first_record = dict(history[0])
            print("\nDebug - First record date format:")
            print(f"Date recorded: {first_record['date_recorded']}")
            print(f"Type: {type(first_record['date_recorded'])}")
        
        return history
    except sqlite3.Error as e:
        print(f"❌ Database error: {e}")
        raise
    finally:
        conn.close()

def create_progress_charts(user_id):
    """Create progress charts for the user"""
    try:
        history = get_measurement_history(user_id)
        if not history:
            print("No measurement history found for this user.")
            return
        
        # Convert history to pandas DataFrame and handle dates properly
        df = pd.DataFrame([dict(row) for row in history])
        
        # Debug: Print DataFrame info
        print("\nDebug - DataFrame Info:")
        print(df.info())
        print("\nDebug - First few rows:")
        print(df.head())
        
        # Convert date_recorded to datetime
        df['date_recorded'] = pd.to_datetime(df['date_recorded'])
        
        # Create directory for user's charts
        user_dir = f"user_{user_id}_charts"
        os.makedirs(user_dir, exist_ok=True)
        
        # Set style - using a built-in matplotlib style instead of seaborn
        plt.style.use('ggplot')
        
        # Format dates for x-axis
        date_format = '%Y-%m-%d'
        df['date_formatted'] = df['date_recorded'].dt.strftime(date_format)
        
        # 1. Weight and BMI Progress (Bar Graph)
        plt.figure(figsize=(15, 8))
        gs = GridSpec(2, 2)
        
        # Weight Bar Graph
        ax1 = plt.subplot(gs[0, 0])
        sns.barplot(x='date_formatted', y='weight_kg', data=df, ax=ax1, color='skyblue')
        ax1.set_title('Weight Progress (Bar Graph)')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Weight (kg)')
        plt.xticks(rotation=45)
        
        # BMI Bar Graph
        ax2 = plt.subplot(gs[0, 1])
        sns.barplot(x='date_formatted', y='bmi', data=df, ax=ax2, color='lightgreen')
        ax2.set_title('BMI Progress (Bar Graph)')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('BMI')
        plt.xticks(rotation=45)
        
        # Weight Line Graph
        ax3 = plt.subplot(gs[1, 0])
        sns.lineplot(x='date_formatted', y='weight_kg', data=df, marker='o', ax=ax3, color='blue')
        ax3.set_title('Weight Progress (Line Graph)')
        ax3.set_xlabel('Date')
        ax3.set_ylabel('Weight (kg)')
        plt.xticks(rotation=45)
        
        # BMI Line Graph
        ax4 = plt.subplot(gs[1, 1])
        sns.lineplot(x='date_formatted', y='bmi', data=df, marker='o', ax=ax4, color='green')
        ax4.set_title('BMI Progress (Line Graph)')
        ax4.set_xlabel('Date')
        ax4.set_ylabel('BMI')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(f'{user_dir}/weight_bmi_progress.png')
        plt.close()
        
        # 2. Body Composition Progress (Stacked Bar Graph)
        plt.figure(figsize=(15, 8))
        gs = GridSpec(2, 2)
        
        # Stacked Bar Graph
        ax1 = plt.subplot(gs[0, :])
        df_melted = pd.melt(df, id_vars=['date_formatted'], 
                           value_vars=['lbm', 'fat_mass', 'smm'],
                           var_name='Component', value_name='Mass')
        sns.barplot(x='date_formatted', y='Mass', hue='Component', data=df_melted, ax=ax1)
        ax1.set_title('Body Composition Progress (Stacked Bar)')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Mass (kg)')
        plt.xticks(rotation=45)
        
        # Pie Chart for Latest Measurements
        ax2 = plt.subplot(gs[1, 0])
        latest = df.iloc[-1]
        components = [latest['lbm'], latest['fat_mass'], latest['smm']]
        labels = ['Lean Body Mass', 'Fat Mass', 'Skeletal Muscle Mass']
        colors = ['lightgreen', 'lightcoral', 'skyblue']
        ax2.pie(components, labels=labels, autopct='%1.1f%%', colors=colors)
        ax2.set_title('Latest Body Composition')
        
        # Line Graph
        ax3 = plt.subplot(gs[1, 1])
        sns.lineplot(x='date_formatted', y='lbm', data=df, label='Lean Body Mass', marker='o', ax=ax3, color='green')
        sns.lineplot(x='date_formatted', y='fat_mass', data=df, label='Fat Mass', marker='o', ax=ax3, color='red')
        sns.lineplot(x='date_formatted', y='smm', data=df, label='Skeletal Muscle Mass', marker='o', ax=ax3, color='blue')
        ax3.set_title('Body Composition Progress (Line Graph)')
        ax3.set_xlabel('Date')
        ax3.set_ylabel('Mass (kg)')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(f'{user_dir}/body_composition_progress.png')
        plt.close()
        
        # 3. Health Indicators Progress (Multiple Plots)
        plt.figure(figsize=(15, 10))
        gs = GridSpec(2, 2)
        
        # WHR and WHtR Bar Graph
        ax1 = plt.subplot(gs[0, 0])
        df_melted = pd.melt(df, id_vars=['date_formatted'], 
                           value_vars=['whr', 'whtr'],
                           var_name='Ratio', value_name='Value')
        sns.barplot(x='date_formatted', y='Value', hue='Ratio', data=df_melted, ax=ax1)
        ax1.set_title('WHR and WHtR Progress')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Ratio')
        plt.xticks(rotation=45)
        
        # Body Fat Percentage Progress
        ax2 = plt.subplot(gs[0, 1])
        sns.barplot(x='date_formatted', y='bfp', data=df, ax=ax2, color='lightcoral')
        ax2.set_title('Body Fat Percentage Progress')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Body Fat %')
        plt.xticks(rotation=45)
        
        # BMR Progress
        ax3 = plt.subplot(gs[1, 0])
        sns.barplot(x='date_formatted', y='bmr', data=df, ax=ax3, color='lightgreen')
        ax3.set_title('BMR Progress')
        ax3.set_xlabel('Date')
        ax3.set_ylabel('BMR (kcal)')
        plt.xticks(rotation=45)
        
        # BSA Progress
        ax4 = plt.subplot(gs[1, 1])
        sns.barplot(x='date_formatted', y='bsa', data=df, ax=ax4, color='skyblue')
        ax4.set_title('BSA Progress')
        ax4.set_xlabel('Date')
        ax4.set_ylabel('BSA (m²)')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(f'{user_dir}/health_indicators_progress.png')
        plt.close()
        
        # 4. Trend Analysis
        plt.figure(figsize=(15, 10))
        gs = GridSpec(2, 2)
        
        # Weight Trend
        ax1 = plt.subplot(gs[0, 0])
        sns.regplot(x=range(len(df)), y='weight_kg', data=df, ax=ax1, color='blue')
        ax1.set_title('Weight Trend Analysis')
        ax1.set_xlabel('Measurement Number')
        ax1.set_ylabel('Weight (kg)')
        
        # BMI Trend
        ax2 = plt.subplot(gs[0, 1])
        sns.regplot(x=range(len(df)), y='bmi', data=df, ax=ax2, color='green')
        ax2.set_title('BMI Trend Analysis')
        ax2.set_xlabel('Measurement Number')
        ax2.set_ylabel('BMI')
        
        # Body Fat % Trend
        ax3 = plt.subplot(gs[1, 0])
        sns.regplot(x=range(len(df)), y='bfp', data=df, ax=ax3, color='red')
        ax3.set_title('Body Fat % Trend Analysis')
        ax3.set_xlabel('Measurement Number')
        ax3.set_ylabel('Body Fat %')
        
        # BMR Trend
        ax4 = plt.subplot(gs[1, 1])
        sns.regplot(x=range(len(df)), y='bmr', data=df, ax=ax4, color='purple')
        ax4.set_title('BMR Trend Analysis')
        ax4.set_xlabel('Measurement Number')
        ax4.set_ylabel('BMR (kcal)')
        
        plt.tight_layout()
        plt.savefig(f'{user_dir}/trend_analysis.png')
        plt.close()
        
        print(f"\n📊 Progress charts have been saved in the '{user_dir}' directory:")
        print(f"- {user_dir}/weight_bmi_progress.png")
        print(f"- {user_dir}/body_composition_progress.png")
        print(f"- {user_dir}/health_indicators_progress.png")
        print(f"- {user_dir}/trend_analysis.png")
        
    except Exception as e:
        print(f"Error creating progress charts: {e}")
        raise

def generate_progress_report(user_id):
    """Generate a comprehensive progress report"""
    try:
        history = get_measurement_history(user_id)
        if not history:
            print("No measurement history found for this user.")
            return
        
        # Get user data
        user_data = get_user_data(user_id)
        
        # Convert history to pandas DataFrame
        df = pd.DataFrame([dict(row) for row in history])
        df['date_recorded'] = pd.to_datetime(df['date_recorded'])
        
        # Calculate changes
        first_record = df.iloc[0]
        latest_record = df.iloc[-1]
        
        print(f"\n📈 Progress Report for {user_data['username']}")
        print("\n--- Overall Changes ---")
        print(f"Time Period: {first_record['date_recorded']} to {latest_record['date_recorded']}")
        print(f"Duration: {(latest_record['date_recorded'] - first_record['date_recorded']).days} days")
        
        print("\n--- Weight and Body Composition Changes ---")
        print(f"Weight: {first_record['weight_kg']:.1f} kg → {latest_record['weight_kg']:.1f} kg")
        print(f"BMI: {first_record['bmi']:.1f} → {latest_record['bmi']:.1f}")
        print(f"Body Fat %: {first_record['bfp']:.1f}% → {latest_record['bfp']:.1f}%")
        print(f"Lean Body Mass: {first_record['lbm']:.1f} kg → {latest_record['lbm']:.1f} kg")
        print(f"Skeletal Muscle Mass: {first_record['smm']:.1f} kg → {latest_record['smm']:.1f} kg")
        
        print("\n--- Health Indicators Changes ---")
        print(f"WHR: {first_record['whr']:.2f} → {latest_record['whr']:.2f}")
        print(f"WHtR: {first_record['whtr']:.2f} → {latest_record['whtr']:.2f}")
        print(f"BMR: {first_record['bmr']:.0f} kcal → {latest_record['bmr']:.0f} kcal")
        
        # Calculate trends
        print("\n--- Trends ---")
        weight_trend = np.polyfit(range(len(df)), df['weight_kg'], 1)[0]
        bmi_trend = np.polyfit(range(len(df)), df['bmi'], 1)[0]
        bfp_trend = np.polyfit(range(len(df)), df['bfp'], 1)[0]
        
        print(f"Weight Trend: {weight_trend:.2f} kg per measurement")
        print(f"BMI Trend: {bmi_trend:.2f} per measurement")
        print(f"Body Fat % Trend: {bfp_trend:.2f}% per measurement")
        
        # Create progress charts
        create_progress_charts(user_id)
        
    except Exception as e:
        print(f"Error generating progress report: {e}")
        raise

if __name__ == "__main__":
    try:
        # Create history table if it doesn't exist
        create_history_table()
        
        # Load current user ID
        from BMI_calc import load_current_user
        current_user_id = load_current_user()
        
        if current_user_id is None:
            print("❌ Please log in first to view your progress.")
            print("Run 'python Login_page.py' to log in.")
        else:
            print(f"✅ User ID: {current_user_id} loaded successfully")
            # Generate progress report
            generate_progress_report(current_user_id)
            
    except Exception as e:
        print(f"❌ An error occurred: {str(e)}")
        print("\nTroubleshooting steps:")
        print("1. Make sure you're logged in")
        print("2. Check if you have any measurement data")
        print("3. Verify the database connection")
        print("4. Ensure all required packages are installed:")
        print("   pip install matplotlib seaborn pandas numpy") 