# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import sqlite3
import os
from datetime import datetime

# Import from existing project modules
try:
    from BMI_calc import load_current_user, get_user_data
except ImportError:
    print("❌ Error: BMI_calc module not found. Some features may not work properly.")
    # Fallback functions if import fails
    def load_current_user():
        return None
    def get_user_data(user_id):
        return None

# Database connection helper
def get_db_connection():
    """Create a database connection"""
    try:
        conn = sqlite3.connect("health_analysis.db")
        conn.row_factory = sqlite3.Row
        return conn
    except sqlite3.Error as e:
        print(f"Database connection error: {e}")
        return None

# Function to get user health data from database
def get_user_health_data(user_id):
    """Retrieve health metrics for diabetes prediction from database"""
    try:
        conn = get_db_connection()
        if not conn:
            return None
        
        cursor = conn.cursor()
        
        # Query to get the most recent health metrics
        cursor.execute('''
            SELECT 
                u.gender,
                u.age,
                m.glucose_level,
                m.blood_pressure,
                m.insulin_level,
                u.weight_kg,
                u.height_cm,
                m.diabetes_pedigree
            FROM users u
            LEFT JOIN health_metrics m ON u.id = m.user_id
            WHERE u.id = ?
            ORDER BY m.date_recorded DESC
            LIMIT 1
        ''', (user_id,))
        
        result = cursor.fetchone()
        conn.close()
        
        if not result:
            return {}
            
        # Calculate BMI if height and weight are available
        bmi = None
        if result['height_cm'] and result['weight_kg']:
            height_m = result['height_cm'] / 100
            bmi = result['weight_kg'] / (height_m * height_m)
        
        # Return dictionary of available health data
        health_data = {
            'age': result['age'],
            'glucose': result['glucose_level'],
            'blood_pressure': result['blood_pressure'],
            'insulin': result['insulin_level'],
            'bmi': bmi,
            'diabetes_pedigree': result['diabetes_pedigree'],
            'pregnancies': None  # This would need a separate table or field
        }
        
        # Filter out None values
        return {k: v for k, v in health_data.items() if v is not None}
        
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return {}
    except Exception as e:
        print(f"Error retrieving health data: {e}")
        return {}

# Function to save prediction result to database
def save_prediction_result(user_id, prediction, confidence):
    """Save diabetes prediction result to database"""
    try:
        conn = get_db_connection()
        if not conn:
            return False
            
        cursor = conn.cursor()
        
        # Create table if not exists
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS diabetes_predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                prediction INTEGER NOT NULL,
                confidence REAL NOT NULL,
                date_recorded TIMESTAMP NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        # Insert prediction
        cursor.execute('''
            INSERT INTO diabetes_predictions
            (user_id, prediction, confidence, date_recorded)
            VALUES (?, ?, ?, ?)
        ''', (user_id, int(prediction), confidence, datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        
        conn.commit()
        conn.close()
        return True
        
    except sqlite3.Error as e:
        print(f"Database error when saving prediction: {e}")
        return False
    except Exception as e:
        print(f"Error saving prediction: {e}")
        return False

# Main diabetes prediction function
def predict_diabetes():
    # Load dataset
    url = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
    df = pd.read_csv(url)

    # Remove 'SkinThickness' from dataset
    df.drop(columns=['SkinThickness'], inplace=True)

    # Replace 0s with np.nan for specific columns (excluding 'Pregnancies' and now 'SkinThickness' which is removed)
    columns_to_replace = ['Glucose', 'BloodPressure', 'Insulin', 'BMI']
    df[columns_to_replace] = df[columns_to_replace].replace(0, np.nan)

    # Fill NaN with median values
    df.fillna(df.median(numeric_only=True), inplace=True)

    # Features and target
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)
    print("✅ Model Accuracy:", accuracy_score(y_test, y_pred))
    print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))

    # Confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap='Blues')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    # Get current user if logged in
    user_id = load_current_user()
    
    # Initialize dictionary for user data
    user_data_dict = {}
    
    # If user is logged in, try to get data from database
    if user_id:
        print(f"\n🔍 Welcome User #{user_id}! Checking your health records...")
        user_data_dict = get_user_health_data(user_id)
        if user_data_dict:
            print("✅ Found your health data in our records!")
        else:
            print("⚠️ No health data found in our records. Please enter all values manually.")
    else:
        print("\n⚠️ You're not logged in. To use your stored health data, please log in first.")
        print("Run 'python Login_page.py' to log in.")

    # ------------------------------
    # 🔍 User Input for Prediction
    # ------------------------------
    print("\n🔍 Enter your health details to check Diabetes risk:\n")
    try:
        # Get Pregnancies (always ask as it's likely not in database)
        Pregnancies = user_data_dict.get('pregnancies')
        if Pregnancies is None:
            Pregnancies = int(input("Enter number of Pregnancies: "))
        else:
            print(f"Using Pregnancies from database: {Pregnancies}")
            
        # Get Glucose level
        Glucose = user_data_dict.get('glucose')
        if Glucose is None:
            Glucose = float(input("Enter Glucose level (mg/dL): "))
        else:
            print(f"Using Glucose level from database: {Glucose} mg/dL")
            
        # Get Blood Pressure
        BloodPressure = user_data_dict.get('blood_pressure')
        if BloodPressure is None:
            BloodPressure = float(input("Enter Blood Pressure (mm Hg): "))
        else:
            print(f"Using Blood Pressure from database: {BloodPressure} mm Hg")
            
        # Get Insulin level
        Insulin = user_data_dict.get('insulin')
        if Insulin is None:
            Insulin = float(input("Enter Insulin level (mu U/ml): "))
        else:
            print(f"Using Insulin level from database: {Insulin} mu U/ml")
            
        # Get BMI
        BMI = user_data_dict.get('bmi')
        if BMI is None:
            BMI = float(input("Enter BMI (e.g., 28.5): "))
        else:
            print(f"Using BMI from database: {BMI}")
            
        # Get Diabetes Pedigree Function
        DiabetesPedigreeFunction = user_data_dict.get('diabetes_pedigree')
        if DiabetesPedigreeFunction is None:
            DiabetesPedigreeFunction = float(input("Enter Diabetes Pedigree Function (e.g., 0.45): "))
        else:
            print(f"Using Diabetes Pedigree Function from database: {DiabetesPedigreeFunction}")
            
        # Get Age
        Age = user_data_dict.get('age')
        if Age is None:
            Age = int(input("Enter Age (in years): "))
        else:
            print(f"Using Age from database: {Age} years")

        # Create and scale input
        user_data = np.array([[Pregnancies, Glucose, BloodPressure, Insulin,
                              BMI, DiabetesPedigreeFunction, Age]])
        user_scaled = scaler.transform(user_data)

        # Predict
        prediction = model.predict(user_scaled)
        prediction_proba = model.predict_proba(user_scaled)

        result = "🛑 Diabetes Detected (Positive)" if prediction[0] == 1 else "✅ No Diabetes Detected (Negative)"
        confidence = prediction_proba[0][prediction[0]] * 100
        print(f"\n🔎 Prediction Result: {result}")
        print(f"📈 Confidence: {confidence:.2f}%")
        
        # Save prediction to database if user is logged in
        if user_id:
            if save_prediction_result(user_id, prediction[0], confidence):
                print("✅ Prediction saved to your health records.")
            else:
                print("⚠️ Could not save prediction to your health records.")

    except Exception as e:
        print("❌ Error: Please enter valid numeric inputs only.")
        print(str(e))

    # ------------------------------
    # 🔬 Feature Importance Plot
    # ------------------------------
    importances = model.feature_importances_
    feature_names = X.columns

    plt.figure(figsize=(10,6))
    sns.barplot(x=importances, y=feature_names)
    plt.title("Feature Importance in Diabetes Prediction")
    plt.xlabel("Importance Score")
    plt.ylabel("Features")
    plt.tight_layout()
    plt.show()

# Run the prediction if script is executed directly
if __name__ == "__main__":
    predict_diabetes()
