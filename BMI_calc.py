import sqlite3
from datetime import datetime
import json
import os
import sys
import importlib.util
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Setup paths for all imports with numeric prefixes
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Function to import modules with numeric prefixes
def import_module_with_numeric_prefix(module_name, numeric_prefix=None):
    try:
        # First try direct import
        if numeric_prefix is None:
            return importlib.import_module(module_name)
        else:
            # Try with numeric prefix using importlib
            file_path = os.path.join(current_dir, f'{numeric_prefix}.{module_name}.py')
            if not os.path.exists(file_path):
                # Try without the .py extension (might be directory)
                file_path = os.path.join(current_dir, f'{numeric_prefix}.{module_name}')
                if not os.path.exists(file_path):
                    raise ImportError(f"Could not find {module_name} module with prefix {numeric_prefix}")
            
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            if spec is None:
                raise ImportError(f"Could not load {file_path}")
            
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            return module
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return None

# Define a flag to indicate if we're being imported by another module
BEING_IMPORTED = __name__ != "__main__"

# Import User_history module only if we're the main script (avoid circular imports)
User_history = None
if not BEING_IMPORTED:
    try:
        User_history = import_module_with_numeric_prefix("User_history")
        if User_history is None:
            User_history = import_module_with_numeric_prefix("User_history", "4")
        
        if User_history is None:
            print("❌ Warning: User_history module not found. History features will be disabled.")
    except Exception as e:
        print(f"❌ Error importing User_history module: {e}")
        print("History features will be disabled.")

# Import suggestion module
try:
    # Try direct import first
    try:
        from suggestion_connection import generate_health_advice, print_health_advice
    except ImportError:
        # Try with numeric prefix
        suggestion_connection = import_module_with_numeric_prefix("suggestion_connection", "99")
        if suggestion_connection:
            generate_health_advice = suggestion_connection.generate_health_advice
            print_health_advice = suggestion_connection.print_health_advice
        else:
            raise ImportError("Could not import suggestion_connection module")
except Exception as e:
    print(f"❌ Warning: Suggestion module not found: {e}")
    print("Health advice features will be disabled.")
    
    # Create dummy functions as fallbacks
    def generate_health_advice(*args, **kwargs):
        return "Unable to generate health advice: suggestion module not found."
    
    def print_health_advice(*args, **kwargs):
        print("Unable to provide health advice: suggestion module not found.")

# Set the style for all plots
plt.style.use('default')
sns.set_theme()

def get_db_connection():
    """Create a database connection"""
    try:
        conn = sqlite3.connect("health_analysis.db")
        conn.row_factory = sqlite3.Row
        return conn
    except sqlite3.Error as e:
        print(f"Database connection error: {e}")
        raise

def load_current_user():
    """Load current user ID from file"""
    try:
        if os.path.exists('current_user.json'):
            with open('current_user.json', 'r') as f:
                data = json.load(f)
                return data.get('user_id')
    except Exception as e:
        print(f"Error loading current user: {e}")
    return None

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

def calculate_age(date_of_birth):
    """Calculate age from date of birth"""
    today = datetime.now()
    dob = datetime.strptime(date_of_birth, '%Y-%m-%d')
    age = today.year - dob.year
    if today.month < dob.month or (today.month == dob.month and today.day < dob.day):
        age -= 1
    return age

def calculate_bmi(weight_kg, height_m):
    """Calculate Body Mass Index (BMI)"""
    return weight_kg / (height_m ** 2)

def calculate_whr(waist_cm, hip_cm):
    """Calculate Waist-to-Hip Ratio (WHR)"""
    return waist_cm / hip_cm

def calculate_whtr(waist_cm, height_cm):
    """Calculate Waist-to-Height Ratio (WHtR)"""
    return waist_cm / height_cm

def calculate_bfp(bmi, age, gender):
    """Calculate Body Fat Percentage (BFP) using BMI formula"""
    if gender.upper() == 'M':
        return (1.20 * bmi) + (0.23 * age) - 16.2
    else:
        return (1.20 * bmi) + (0.23 * age) - 5.4

def calculate_lbm(weight_kg, height_cm, gender):
    """Calculate Lean Body Mass (LBM)"""
    if gender.upper() == 'M':
        return (0.407 * weight_kg) + (0.267 * height_cm) - 19.2
    else:
        return (0.252 * weight_kg) + (0.473 * height_cm) - 48.3

def calculate_smm(lbm, gender):
    """Calculate Skeletal Muscle Mass (SMM)"""
    if gender.upper() == 'M':
        return 0.53 * lbm
    else:
        return 0.45 * lbm

def calculate_proportions(arm_length, leg_length, height_cm, arm_circumference):
    """Calculate various body proportions"""
    return {
        'arm_to_leg': arm_length / leg_length,
        'arm_to_height': arm_length / height_cm,
        'leg_to_height': leg_length / height_cm,
        'arm_circ_to_length': arm_circumference / arm_length,
        'arm_circ_to_height': arm_circumference / height_cm
    }

def check_metabolic_risks(bmi, whr, whtr, bfp, waist_cm, gender):
    """Check for various metabolic risk factors"""
    risks = []
    
    # BMI Risk
    if bmi >= 30:
        risks.append("Obesity Risk")
    
    # WHR Risk
    if gender.upper() == 'M' and whr > 0.90:
        risks.append("Abdominal Obesity (Male)")
    elif gender.upper() == 'F' and whr > 0.85:
        risks.append("Abdominal Obesity (Female)")
    
    # WHtR Risk
    if whtr > 0.5:
        risks.append("Cardiovascular Risk")
    
    # BFP Risk
    if gender.upper() == 'M' and bfp > 25:
        risks.append("High Fat Risk (Male)")
    elif gender.upper() == 'F' and bfp > 32:
        risks.append("High Fat Risk (Female)")
    
    # Waist Circumference Risk
    if gender.upper() == 'M' and waist_cm > 102:
        risks.append("Metabolic Syndrome Indicator (Male)")
    elif gender.upper() == 'F' and waist_cm > 88:
        risks.append("Metabolic Syndrome Indicator (Female)")
    
    return risks

def validate_measurements(weight_kg, height_cm, waist_cm, hip_cm, arm_length_cm, leg_length_cm, arm_circumference_cm):
    """Validate measurement inputs"""
    if not all(isinstance(x, (int, float)) for x in [weight_kg, height_cm, waist_cm, hip_cm, arm_length_cm, leg_length_cm, arm_circumference_cm]):
        raise ValueError("All measurements must be numeric values")
    
    if any(x <= 0 for x in [weight_kg, height_cm, waist_cm, hip_cm, arm_length_cm, leg_length_cm, arm_circumference_cm]):
        raise ValueError("All measurements must be positive values")
    
    # Realistic range checks
    if not (20 <= weight_kg <= 300):  # kg
        raise ValueError("Weight must be between 20 and 300 kg")
    if not (100 <= height_cm <= 250):  # cm
        raise ValueError("Height must be between 100 and 250 cm")
    if not (30 <= waist_cm <= 200):  # cm
        raise ValueError("Waist circumference must be between 30 and 200 cm")
    if not (40 <= hip_cm <= 200):  # cm
        raise ValueError("Hip circumference must be between 40 and 200 cm")
    if not (20 <= arm_length_cm <= 100):  # cm
        raise ValueError("Arm length must be between 20 and 100 cm")
    if not (30 <= leg_length_cm <= 120):  # cm
        raise ValueError("Leg length must be between 30 and 120 cm")
    if not (10 <= arm_circumference_cm <= 50):  # cm
        raise ValueError("Arm circumference must be between 10 and 50 cm")

def calculate_bmr(weight_kg, height_cm, age, gender):
    """Calculate Basal Metabolic Rate using Mifflin-St Jeor Equation"""
    if gender.upper() == 'M':
        return (10 * weight_kg) + (6.25 * height_cm) - (5 * age) + 5
    else:
        return (10 * weight_kg) + (6.25 * height_cm) - (5 * age) - 161

def calculate_bsa(weight_kg, height_cm):
    """Calculate Body Surface Area using Mosteller formula"""
    return ((weight_kg * height_cm) / 3600) ** 0.5

def calculate_ibw(height_cm, gender):
    """Calculate Ideal Body Weight using Devine formula"""
    if gender.upper() == 'M':
        return 50 + 2.3 * ((height_cm - 152.4) / 2.54)
    else:
        return 45.5 + 2.3 * ((height_cm - 152.4) / 2.54)

def calculate_bai(hip_cm, height_m):
    """Calculate Body Adiposity Index"""
    return (hip_cm / (height_m ** 1.5)) - 18

def create_body_composition_chart(bmi, bfp, lbm, fat_mass, smm):
    """Create a pie chart showing body composition distribution"""
    plt.figure(figsize=(10, 6))
    labels = ['Lean Body Mass', 'Fat Mass', 'Skeletal Muscle Mass']
    sizes = [lbm, fat_mass, smm]
    colors = ['#2ecc71', '#e74c3c', '#3498db']
    
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    plt.axis('equal')
    plt.title('Body Composition Distribution', pad=20, fontsize=14)
    plt.savefig('body_composition.png')
    plt.close()

def create_bmi_gauge(bmi):
    """Create a gauge chart for BMI"""
    # Define BMI categories and colors
    categories = ['Underweight', 'Normal', 'Overweight', 'Obese']
    colors = ['#3498db', '#2ecc71', '#f1c40f', '#e74c3c']
    ranges = [0, 18.5, 25, 30, 40]
    
    # Create figure and polar axis
    fig, ax = plt.subplots(figsize=(10, 6), subplot_kw=dict(polar=True))
    
    # Set the angle range
    angles = np.linspace(0, np.pi, len(ranges)-1)
    
    # Plot the gauge segments
    for i in range(len(ranges)-1):
        ax.bar(angles[i], ranges[i+1]-ranges[i], width=0.5, bottom=ranges[i], color=colors[i])
    
    # Add BMI value in the center
    ax.text(0, 0, f'BMI: {bmi:.1f}', ha='center', va='center', fontsize=20)
    
    # Customize the plot
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_rticks([])  # Remove radial ticks
    ax.set_thetagrids([])  # Remove angular ticks
    
    plt.title('BMI Gauge', pad=20, fontsize=14)
    plt.savefig('bmi_gauge.png')
    plt.close()

def create_health_indicators_chart(whr, whtr, bfp):
    """Create a radar chart for health indicators"""
    # Data for radar chart
    categories = ['WHR', 'WHtR', 'BFP']
    values = [whr, whtr, bfp/100]  # Normalize BFP to 0-1 range
    
    # Number of variables
    N = len(categories)
    
    # Compute angle for each axis
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    # Add the first value again to close the loop
    values += values[:1]
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6), subplot_kw=dict(polar=True))
    
    # Plot data
    ax.plot(angles, values, linewidth=2, linestyle='solid', color='#3498db')
    ax.fill(angles, values, color='#3498db', alpha=0.4)
    
    # Set category labels
    plt.xticks(angles[:-1], categories)
    
    # Set y-axis limits
    ax.set_ylim(0, 1)
    
    plt.title('Health Indicators Radar Chart', pad=20, fontsize=14)
    plt.savefig('health_indicators.png')
    plt.close()

def create_proportions_chart(proportions):
    """Create a bar chart for body proportions"""
    plt.figure(figsize=(12, 6))
    
    # Extract values and labels
    labels = list(proportions.keys())
    values = list(proportions.values())
    
    # Create bar chart
    bars = plt.bar(labels, values, color=sns.color_palette("husl", len(labels)))
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom')
    
    plt.title('Body Proportions Analysis', pad=20, fontsize=14)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('body_proportions.png')
    plt.close()

def create_history_table():
    """Create the user measurement history table if it doesn't exist"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_measurement_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                date_recorded TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                weight_kg REAL,
                height_cm REAL,
                waist_cm REAL,
                hip_cm REAL,
                arm_length_cm REAL,
                leg_length_cm REAL,
                arm_circumference_cm REAL,
                bmi REAL,
                whr REAL,
                whtr REAL,
                bfp REAL,
                lbm REAL,
                smm REAL,
                fat_mass REAL,
                bmr REAL,
                bsa REAL,
                ibw REAL,
                bai REAL,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        ''')
        
        conn.commit()
    except Exception as e:
        print(f"Error creating history table: {e}")
        raise
    finally:
        conn.close()

def get_body_composition_summary(user_id):
    """Generate a complete body composition summary using user data"""
    try:
        # Create history table if it doesn't exist
        create_history_table()
        
        # Get user data
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
            
        # Convert Row object to dictionary
        user_data = dict(user_data)
            
        # Verify all required measurements are present
        required_fields = [
            'height', 'weight', 'waist', 'hip',
            'arm_length', 'leg_length', 'arm_circumference',
            'date_of_birth', 'gender'
        ]
        
        missing_fields = [field for field in required_fields if field not in user_data or not user_data[field]]
        if missing_fields:
            raise Exception(f"Missing required measurements: {', '.join(missing_fields)}")
            
        # Calculate age
        age = calculate_age(user_data['date_of_birth'])
        
        # Calculate all metrics
        height_m = user_data['height'] / 100
        bmi = calculate_bmi(user_data['weight'], height_m)
        whr = calculate_whr(user_data['waist'], user_data['hip'])
        whtr = calculate_whtr(user_data['waist'], user_data['height'])
        bfp = calculate_bfp(bmi, age, user_data['gender'])
        lbm = calculate_lbm(user_data['weight'], user_data['height'], user_data['gender'])
        smm = calculate_smm(lbm, user_data['gender'])
        fat_mass = user_data['weight'] - lbm
        bmr = calculate_bmr(user_data['weight'], user_data['height'], age, user_data['gender'])
        bsa = calculate_bsa(user_data['weight'], user_data['height'])
        ibw = calculate_ibw(user_data['height'], user_data['gender'])
        bai = calculate_bai(user_data['hip'], height_m)
        
        # Calculate proportions
        proportions = calculate_proportions(
            user_data['arm_length'],
            user_data['leg_length'],
            user_data['height'],
            user_data['arm_circumference']
        )
        
        # Check metabolic risks
        risks = check_metabolic_risks(
            bmi, whr, whtr, bfp, user_data['waist'],
            user_data['gender']
        )
        
        # Save measurements to history
        try:
            # Try different ways to import User_history
            try:
                from User_history import save_measurement_history
            except ImportError:
                # Try with numeric prefix
                import importlib.util
                history_file = os.path.join(os.path.dirname(__file__), '4.User_history.py')
                if os.path.exists(history_file):
                    spec = importlib.util.spec_from_file_location('User_history', history_file)
                    history_module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(history_module)
                    save_measurement_history = history_module.save_measurement_history
                else:
                    print("⚠️ Warning: Unable to save measurement history - module not found")
                    # Create a dummy function
                    def save_measurement_history(user_id, measurements):
                        print("⚠️ History saving disabled: User_history module not found")
        except Exception as e:
            print(f"⚠️ Warning: Unable to save measurement history: {e}")
            # Create a dummy function
            def save_measurement_history(user_id, measurements):
                print("⚠️ History saving disabled due to error")
        measurements = {
            'weight_kg': user_data['weight'],
            'height_cm': user_data['height'],
            'waist_cm': user_data['waist'],
            'hip_cm': user_data['hip'],
            'arm_length_cm': user_data['arm_length'],
            'leg_length_cm': user_data['leg_length'],
            'arm_circumference_cm': user_data['arm_circumference'],
            'bmi': bmi,
            'whr': whr,
            'whtr': whtr,
            'bfp': bfp,
            'lbm': lbm,
            'smm': smm,
            'fat_mass': fat_mass,
            'bmr': bmr,
            'bsa': bsa,
            'ibw': ibw,
            'bai': bai
        }
        save_measurement_history(user_id, measurements)
        
        # Print summary
        print(f"\n📊 Body Composition Analysis for {user_data['username']}")
        print("\n--- Basic Measurements ---")
        print(f"Height: {user_data['height']:.1f} cm")
        print(f"Weight: {user_data['weight']:.1f} kg")
        print(f"Age: {age} years")
        print(f"Gender: {user_data['gender']}")
        
        print("\n--- Body Composition Metrics ---")
        print(f"BMI: {bmi:.2f}")
        print(f"Body Fat Percentage: {bfp:.1f}%")
        print(f"Fat Mass: {fat_mass:.1f} kg")
        print(f"Lean Body Mass: {lbm:.1f} kg")
        print(f"Skeletal Muscle Mass: {smm:.1f} kg")
        
        print("\n--- Health Indicators ---")
        print(f"Waist-to-Hip Ratio (WHR): {whr:.2f}")
        print(f"Waist-to-Height Ratio (WHtR): {whtr:.2f}")
        print(f"Basal Metabolic Rate (BMR): {bmr:.0f} kcal/day")
        print(f"Body Surface Area (BSA): {bsa:.2f} m²")
        print(f"Ideal Body Weight (IBW): {ibw:.1f} kg")
        print(f"Body Adiposity Index (BAI): {bai:.1f}")
        
        print("\n--- Body Proportions ---")
        print(f"Arm-to-Leg Ratio: {proportions['arm_to_leg']:.2f}")
        print(f"Arm-to-Height Ratio: {proportions['arm_to_height']:.2f}")
        print(f"Leg-to-Height Ratio: {proportions['leg_to_height']:.2f}")
        print(f"Arm Circumference-to-Height Ratio: {proportions['arm_circ_to_height']:.2f}")
        
        print("\n--- Metabolic Risk Assessment ---")
        for risk in risks:
            print(f"⚠️ {risk}")
        
        # Generate health advice
        try:
            # Try different ways to import suggestion functions
            try:
                from suggestion import generate_health_advice, print_health_advice
            except ImportError:
                try:
                    # Try with suggestion_connection module
                    from suggestion_connection import generate_health_advice, print_health_advice
                except ImportError:
                    # Try with numeric prefixes
                    import importlib.util
                    
                    # Try suggestion module first
                    suggestion_file = os.path.join(os.path.dirname(__file__), '3.suggestion.py')
                    if os.path.exists(suggestion_file):
                        spec = importlib.util.spec_from_file_location('suggestion', suggestion_file)
                        suggestion_module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(suggestion_module)
                        generate_health_advice = suggestion_module.generate_health_advice
                        print_health_advice = suggestion_module.print_health_advice
                    else:
                        # Try suggestion_connection module
                        connection_file = os.path.join(os.path.dirname(__file__), '99.suggestion_connection.py')
                        if os.path.exists(connection_file):
                            spec = importlib.util.spec_from_file_location('suggestion_connection', connection_file)
                            connection_module = importlib.util.module_from_spec(spec)
                            spec.loader.exec_module(connection_module)
                            generate_health_advice = connection_module.generate_health_advice
                            print_health_advice = connection_module.print_health_advice
                        else:
                            print("⚠️ Warning: Health advice generation disabled - modules not found")
                            # Create dummy functions
                            def generate_health_advice(*args, **kwargs):
                                return "Health advice generation disabled due to missing modules."
                            def print_health_advice(*args, **kwargs):
                                print("⚠️ Health advice disabled: Required modules not found")
        except Exception as e:
            print(f"⚠️ Warning: Health advice generation error: {e}")
            # Create dummy functions
            def generate_health_advice(*args, **kwargs):
                return "Health advice generation disabled due to an error."
            def print_health_advice(*args, **kwargs):
                print("⚠️ Health advice disabled due to an error")
        advice = generate_health_advice(
            bmi, whr, whtr, bfp, user_data['waist'],
            user_data['gender'], proportions['arm_to_leg']
        )
        print_health_advice(advice)
        
    except Exception as e:
        print(f"Error generating body composition summary: {e}")
        raise
    finally:
        conn.close()

def update_user_profile(user_id):
    """Update user profile with missing measurements"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Get current user data
        cursor.execute('''
            SELECT u.username, p.*
            FROM users u
            JOIN user_profiles p ON u.id = p.user_id
            WHERE u.id = ?
        ''', (user_id,))
        
        user_data = cursor.fetchone()
        if not user_data:
            raise Exception("User data not found")
            
        # Convert Row object to dictionary
        user_data = dict(user_data)
            
        print(f"\n📝 Updating profile for {user_data['username']}")
        print("Please enter your measurements (press Enter to keep current value):")
        
        # Get current values or defaults
        current_values = {
            'height': user_data.get('height', ''),
            'weight': user_data.get('weight', ''),
            'waist': user_data.get('waist', ''),
            'hip': user_data.get('hip', ''),
            'arm_length': user_data.get('arm_length', ''),
            'leg_length': user_data.get('leg_length', ''),
            'arm_circumference': user_data.get('arm_circumference', ''),
            'date_of_birth': user_data.get('date_of_birth', ''),
            'gender': user_data.get('gender', '')
        }
        
        # Get new measurements
        new_values = {}
        for field, current in current_values.items():
            if not current:
                while True:
                    try:
                        if field == 'gender':
                            value = input(f"Enter your gender (M/F): ").upper()
                            if value not in ['M', 'F']:
                                print("Please enter 'M' for male or 'F' for female")
                                continue
                        elif field == 'date_of_birth':
                            value = input(f"Enter your date of birth (YYYY-MM-DD): ")
                            # Validate date format
                            datetime.strptime(value, '%Y-%m-%d')
                        else:
                            value = float(input(f"Enter your {field.replace('_', ' ')}: "))
                            if value <= 0:
                                print("Please enter a positive value")
                                continue
                        new_values[field] = value
                        break
                    except ValueError as e:
                        print(f"Invalid input. Please try again. ({str(e)})")
            else:
                print(f"Current {field.replace('_', ' ')}: {current}")
                value = input(f"Enter new value (press Enter to keep current): ")
                if value.strip():
                    try:
                        if field == 'gender':
                            value = value.upper()
                            if value not in ['M', 'F']:
                                print("Please enter 'M' for male or 'F' for female")
                                continue
                        elif field == 'date_of_birth':
                            # Validate date format
                            datetime.strptime(value, '%Y-%m-%d')
                        else:
                            value = float(value)
                            if value <= 0:
                                print("Please enter a positive value")
                                continue
                        new_values[field] = value
                    except ValueError as e:
                        print(f"Invalid input. Please try again. ({str(e)})")
        
        # Update database with new values
        if new_values:
            update_fields = []
            update_values = []
            for field, value in new_values.items():
                update_fields.append(f"{field} = ?")
                update_values.append(value)
            
            update_values.append(user_id)
            cursor.execute(f'''
                UPDATE user_profiles
                SET {', '.join(update_fields)}
                WHERE user_id = ?
            ''', update_values)
            
            conn.commit()
            print("\n✅ Profile updated successfully!")
        else:
            print("\nℹ️ No changes made to profile.")
            
    except Exception as e:
        print(f"Error updating profile: {e}")
        raise
    finally:
        conn.close()

if __name__ == "__main__":
    try:
        # Load current user ID from file
        current_user_id = load_current_user()
        
        if current_user_id is None:
            print("❌ Please log in first to calculate your body composition.")
            print("Run 'python Login_page.py' to log in.")
        else:
            try:
                get_body_composition_summary(current_user_id)
            except Exception as e:
                if "Missing required measurements" in str(e):
                    print("\n❌ Your profile is missing required measurements.")
                    update_user_profile(current_user_id)
                    print("\n🔄 Running body composition analysis with updated profile...")
                    get_body_composition_summary(current_user_id)
                else:
                    raise
    except Exception as e:
        print(f"❌ An error occurred: {str(e)}")
