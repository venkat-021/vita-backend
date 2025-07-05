import sqlite3
import hashlib
import re
import logging
from datetime import datetime
import getpass
import os
import secrets
import string
import json

# Import Diabetes prediction module
try:
    # Using the correct filename with number prefix
    import sys
    import os
    # Add the current directory to path if not already there
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.append(current_dir)
    
    # Import the diabetes module and its authentication functions
    import importlib.util
    spec = importlib.util.spec_from_file_location("diabetes_module", "11.Diabetes_prediction.py")
    diabetes_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(diabetes_module)
    run_diabetes_prediction = diabetes_module.run_diabetes_prediction
    get_auth_token = diabetes_module.get_auth_token
    
    # Function to create auth file for direct access
    def create_diabetes_auth_file():
        """Create an authentication file for direct diabetes prediction access"""
        try:
            auth_token = get_auth_token()
            auth_file = os.path.join(current_dir, '.diabetes_auth')
            with open(auth_file, 'w') as f:
                f.write(auth_token)
            return True
        except Exception as e:
            print(f"Error creating auth file: {e}")
            return False
except ImportError:
    print("❌ Warning: Diabetes prediction module not found.")
    # Create dummy functions if the module is not available
    def run_diabetes_prediction(auth_token=None):
        print("❌ Diabetes prediction module not available.")
        return False
    def get_auth_token():
        return None
    def create_diabetes_auth_file():
        return False

# Set up logging
logging.basicConfig(
    filename='health_system.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Global variable for current user
current_user_id = None

def save_current_user(user_id):
    """Save current user ID to a file"""
    try:
        with open('current_user.json', 'w') as f:
            json.dump({'user_id': user_id}, f)
    except Exception as e:
        logging.error(f"Error saving current user: {e}")

def load_current_user():
    """Load current user ID from file"""
    try:
        if os.path.exists('current_user.json'):
            with open('current_user.json', 'r') as f:
                data = json.load(f)
                return data.get('user_id')
    except Exception as e:
        logging.error(f"Error loading current user: {e}")
    return None

# Connect to the database
def get_db_connection():
    try:
        conn = sqlite3.connect("health_analysis.db")
        conn.row_factory = sqlite3.Row
        return conn
    except sqlite3.Error as e:
        logging.error(f"Database connection error: {e}")
        raise

def check_and_add_columns():
    """Check and add missing columns to the database"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Check if profile_completed column exists
        cursor.execute("PRAGMA table_info(users)")
        columns = [column[1] for column in cursor.fetchall()]
        
        if 'profile_completed' not in columns:
            cursor.execute('''
                ALTER TABLE users 
                ADD COLUMN profile_completed BOOLEAN DEFAULT 0
            ''')
            logging.info("Added profile_completed column to users table")
        
        # Check if reset_token and reset_token_expiry columns exist
        if 'reset_token' not in columns:
            cursor.execute('''
                ALTER TABLE users 
                ADD COLUMN reset_token TEXT
            ''')
            logging.info("Added reset_token column to users table")
            
        if 'reset_token_expiry' not in columns:
            cursor.execute('''
                ALTER TABLE users 
                ADD COLUMN reset_token_expiry TIMESTAMP
            ''')
            logging.info("Added reset_token_expiry column to users table")
        
        # Check if leg_length column exists in user_profiles
        cursor.execute("PRAGMA table_info(user_profiles)")
        profile_columns = [column[1] for column in cursor.fetchall()]
        
        if 'leg_length' not in profile_columns:
            cursor.execute('''
                ALTER TABLE user_profiles 
                ADD COLUMN leg_length REAL
            ''')
            logging.info("Added leg_length column to user_profiles table")
        
        conn.commit()
    except Exception as e:
        logging.error(f"Error updating database schema: {e}")
        raise
    finally:
        conn.close()

# Create necessary tables
def create_tables():
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Users table with additional fields
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL,
        email TEXT UNIQUE,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        last_login TIMESTAMP,
        is_active BOOLEAN DEFAULT 1,
        reset_token TEXT,
        reset_token_expiry TIMESTAMP,
        profile_completed BOOLEAN DEFAULT 0
    )
    ''')
    
    # User profiles table with required measurements
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS user_profiles (
        user_id INTEGER PRIMARY KEY,
        date_of_birth DATE NOT NULL,
        gender TEXT NOT NULL,
        height REAL NOT NULL,
        weight REAL NOT NULL,
        arm_length REAL NOT NULL,
        arm_circumference REAL NOT NULL,
        hip REAL NOT NULL,
        waist REAL NOT NULL,
        leg_length REAL NOT NULL,
        last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users (id)
    )
    ''')
    
    conn.commit()
    conn.close()

def hash_password(password):
    """Hash a password using SHA-256"""
    return hashlib.sha256(password.encode()).hexdigest()

def validate_password(password):
    """Validate password strength"""
    if len(password) < 8:
        return False, "Password must be at least 8 characters long"
    if not re.search(r"[A-Z]", password):
        return False, "Password must contain at least one uppercase letter"
    if not re.search(r"[a-z]", password):
        return False, "Password must contain at least one lowercase letter"
    if not re.search(r"\d", password):
        return False, "Password must contain at least one number"
    if not re.search(r"[!@#$%^&*(),.?\":{}|<>]", password):
        return False, "Password must contain at least one special character"
    return True, "Password is strong"

def validate_email(email):
    """Validate email format"""
    pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
    return bool(re.match(pattern, email))

def generate_reset_token():
    """Generate a secure random token for password reset"""
    return ''.join(secrets.choice(string.ascii_letters + string.digits) for _ in range(32))

def check_username_exists(username):
    """Check if username already exists"""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM users WHERE username = ?", (username,))
    exists = cursor.fetchone() is not None
    conn.close()
    return exists

def register_user():
    global current_user_id
    while True:
        try:
            username = input("👤 Create a username: ").strip()
            if len(username) < 3:
                print("⚠️ Username must be at least 3 characters long")
                continue
                
            if check_username_exists(username):
                print("⚠️ This username is already taken. Please choose another one.")
                continue

            email = input("📧 Enter your email: ").strip()
            if not validate_email(email):
                print("⚠️ Invalid email format")
                continue
                
            while True:
                password = getpass.getpass("🔑 Create a password: ")
                is_valid, message = validate_password(password)
                if not is_valid:
                    print(f"⚠️ {message}")
                    continue
                confirm_password = getpass.getpass("🔑 Confirm password: ")
                if password != confirm_password:
                    print("⚠️ Passwords do not match")
                    continue
                break

            conn = get_db_connection()
            cursor = conn.cursor()
            
            hashed_password = hash_password(password)
            cursor.execute(
                "INSERT INTO users (username, password, email, profile_completed) VALUES (?, ?, ?, ?)",
                (username, hashed_password, email, 0)
            )
            
            current_user_id = cursor.lastrowid
            conn.commit()
            logging.info(f"New user registered: {username}")
            print("✅ Registration successful!")
            
            # Immediately prompt for profile completion
            print("\n📝 Please complete your profile to continue.")
            if update_profile():
                cursor.execute(
                    "UPDATE users SET profile_completed = 1 WHERE id = ?",
                    (current_user_id,)
                )
                conn.commit()
                print("✅ Profile completed successfully!")
                return True
            else:
                print("❌ Profile update failed. Please try again.")
                return False
            
        except sqlite3.IntegrityError:
            print("⚠️ Username or email already exists. Try different ones.")
            logging.warning(f"Registration failed - duplicate username/email: {username}")
        except Exception as e:
            print(f"❌ An error occurred: {str(e)}")
            logging.error(f"Registration error: {str(e)}")
        finally:
            conn.close()

def login_user():
    global current_user_id
    max_attempts = 3
    attempts = 0

    while attempts < max_attempts:
        try:
            username = input("👤 Username: ").strip()
            password = getpass.getpass("🔑 Password: ")
            
            conn = get_db_connection()
            cursor = conn.cursor()
            
            hashed_password = hash_password(password)
            cursor.execute(
                "SELECT * FROM users WHERE username=? AND password=? AND is_active=1",
                (username, hashed_password)
            )
            user = cursor.fetchone()
            
            if user:
                current_user_id = user['id']
                save_current_user(current_user_id)  # Save user ID to file
                # Update last login
                cursor.execute(
                    "UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE id = ?",
                    (current_user_id,)
                )
                conn.commit()
                
                logging.info(f"User logged in: {username}")
                print(f"✅ Welcome back, {username}!")
                return True
            else:
                attempts += 1
                remaining = max_attempts - attempts
                print(f"❌ Login failed. Incorrect username or password. {remaining} attempts remaining.")
                
                if attempts == max_attempts:
                    print("\n🔑 Forgot your password?")
                    reset_choice = input("Would you like to reset your password? (yes/no): ").lower()
                    if reset_choice == 'yes':
                        if forgot_password():
                            print("✅ Please login with your new password.")
                            return login_user()  # Retry login with new password
                        else:
                            print("❌ Password reset failed. Please try again later.")
                
                logging.warning(f"Failed login attempt for user: {username}")
                
        except Exception as e:
            print(f"❌ An error occurred: {str(e)}")
            logging.error(f"Login error: {str(e)}")
        finally:
            conn.close()
    
    print("❌ Too many failed attempts. Please try again later.")
    return False

def logout():
    """Logout the current user"""
    global current_user_id
    current_user_id = None
    try:
        if os.path.exists('current_user.json'):
            os.remove('current_user.json')
    except Exception as e:
        logging.error(f"Error during logout: {e}")
    print("👋 Logged out successfully!")

def forgot_password():
    try:
        email = input("\n📧 Enter your registered email: ").strip()
        
        if not validate_email(email):
            print("⚠️ Invalid email format")
            return False
            
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Check if email exists
        cursor.execute("SELECT id FROM users WHERE email = ?", (email,))
        user = cursor.fetchone()
        
        if not user:
            print("❌ No account found with this email address.")
            return False
            
        # Generate reset token
        reset_token = generate_reset_token()
        expiry = datetime.now().timestamp() + 3600  # Token valid for 1 hour
        
        # Store reset token
        cursor.execute(
            "UPDATE users SET reset_token = ?, reset_token_expiry = ? WHERE email = ?",
            (reset_token, expiry, email)
        )
        conn.commit()
        
        # In a real application, you would send this token via email
        # For demonstration, we'll just show it
        print(f"\n🔑 Your password reset token is: {reset_token}")
        print("⚠️ In a real application, this would be sent to your email.")
        print("⚠️ This token will expire in 1 hour.")
        
        # Verify token and set new password
        token = input("\nEnter the reset token: ").strip()
        
        cursor.execute(
            "SELECT id FROM users WHERE email = ? AND reset_token = ? AND reset_token_expiry > ?",
            (email, token, datetime.now().timestamp())
        )
        user = cursor.fetchone()
        
        if not user:
            print("❌ Invalid or expired token.")
            return False
            
        # Set new password
        while True:
            new_password = getpass.getpass("🔑 Enter new password: ")
            is_valid, message = validate_password(new_password)
            if not is_valid:
                print(f"⚠️ {message}")
                continue
            confirm_password = getpass.getpass("🔑 Confirm new password: ")
            if new_password != confirm_password:
                print("⚠️ Passwords do not match")
                continue
            break
            
        # Update password and clear reset token
        hashed_password = hash_password(new_password)
        cursor.execute(
            "UPDATE users SET password = ?, reset_token = NULL, reset_token_expiry = NULL WHERE email = ?",
            (hashed_password, email)
        )
        conn.commit()
        
        print("✅ Password has been reset successfully!")
        logging.info(f"Password reset successful for email: {email}")
        return True
        
    except Exception as e:
        print(f"❌ An error occurred: {str(e)}")
        logging.error(f"Password reset error: {str(e)}")
        return False
    finally:
        conn.close()

def update_profile():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        print("\n📝 Required Profile Information")
        print("Please enter your measurements:")

        # Get date of birth
        while True:
            date_of_birth = input("Date of Birth (YYYY-MM-DD): ")
            try:
                datetime.strptime(date_of_birth, '%Y-%m-%d')
                break
            except ValueError:
                print("⚠️ Invalid date format. Please use YYYY-MM-DD")

        # Get gender
        while True:
            gender = input("Gender (M/F/Other): ").upper()
            if gender in ['M', 'F', 'OTHER']:
                break
            print("⚠️ Please enter M, F, or Other")

        # Get measurements
        print("\n📏 Enter your measurements (in cm):")
        try:
            height = float(input("Height (cm): "))
            weight = float(input("Weight (kg): "))
            arm_length = float(input("Arm Length (cm): "))
            arm_circ = float(input("Arm Circumference (cm): "))
            hip = float(input("Hip (cm): "))
            waist = float(input("Waist (cm): "))
            leg_length = float(input("Leg Length (cm): "))
        except ValueError:
            print("⚠️ Please enter valid numbers for measurements")
            return False

        # Insert or update profile
        cursor.execute('''
            INSERT INTO user_profiles 
            (user_id, date_of_birth, gender, height, weight, arm_length, 
             arm_circumference, hip, waist, leg_length)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
            date_of_birth = excluded.date_of_birth,
            gender = excluded.gender,
            height = excluded.height,
            weight = excluded.weight,
            arm_length = excluded.arm_length,
            arm_circumference = excluded.arm_circumference,
            hip = excluded.hip,
            waist = excluded.waist,
            leg_length = excluded.leg_length,
            last_updated = CURRENT_TIMESTAMP
        ''', (current_user_id, date_of_birth, gender, height, weight, 
              arm_length, arm_circ, hip, waist, leg_length))

        # Calculate metrics for history
        height_m = height / 100
        bmi = weight / (height_m ** 2)
        whr = waist / hip
        whtr = waist / height
        
        # Calculate age for BFP
        from BMI_calc import calculate_age, calculate_bfp, calculate_lbm, calculate_smm
        from BMI_calc import calculate_bmr, calculate_bsa, calculate_ibw, calculate_bai
        
        age = calculate_age(date_of_birth)
        bfp = calculate_bfp(bmi, age, gender)
        lbm = calculate_lbm(weight, height, gender)
        smm = calculate_smm(lbm, gender)
        fat_mass = weight - lbm
        bmr = calculate_bmr(weight, height, age, gender)
        bsa = calculate_bsa(weight, height)
        ibw = calculate_ibw(height, gender)
        bai = calculate_bai(hip, height_m)

        # Save to measurement history
        cursor.execute('''
            INSERT INTO user_measurement_history (
                user_id, date_recorded, weight_kg, height_cm, waist_cm, hip_cm,
                arm_length_cm, leg_length_cm, arm_circumference_cm,
                bmi, whr, whtr, bfp, lbm, smm, fat_mass,
                bmr, bsa, ibw, bai
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            current_user_id, datetime.now().date(), weight, height, waist,
            hip, arm_length, leg_length, arm_circ, bmi, whr, whtr,
            bfp, lbm, smm, fat_mass, bmr, bsa, ibw, bai
        ))

        conn.commit()
        print("✅ Profile updated successfully!")
        logging.info(f"Profile updated for user ID: {current_user_id}")
        return True

    except Exception as e:
        print(f"❌ An error occurred: {str(e)}")
        logging.error(f"Profile update error: {str(e)}")
        return False
    finally:
        conn.close()

def view_profile():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute('''
            SELECT u.username, u.email, p.*
            FROM users u
            JOIN user_profiles p ON u.id = p.user_id
            WHERE u.id = ?
        ''', (current_user_id,))

        profile = cursor.fetchone()

        if profile:
            print("\n📋 Your Profile")
            print(f"Username: {profile['username']}")
            print(f"Email: {profile['email']}")
            print(f"Date of Birth: {profile['date_of_birth']}")
            print(f"Gender: {profile['gender']}")
            print("\n📏 Body Measurements:")
            print(f"Height: {profile['height']} cm")
            print(f"Weight: {profile['weight']} kg")
            print(f"Arm Length: {profile['arm_length']} cm")
            print(f"Arm Circumference: {profile['arm_circumference']} cm")
            print(f"Leg Length: {profile['leg_length']} cm")
            print(f"Hip: {profile['hip']} cm")
            print(f"Waist: {profile['waist']} cm")
            print(f"Last Updated: {profile['last_updated']}")
        else:
            print("❌ Profile not found")

    except Exception as e:
        print(f"❌ An error occurred: {str(e)}")
        logging.error(f"Profile view error: {str(e)}")
    finally:
        conn.close()

def profile_menu():
    while True:
        print("\n👤 Profile Management")
        print("1. View Profile")
        print("2. Update Profile")
        print("3. Continue with Current Profile")
        print("4. Logout")

        choice = input("Choose an option (1-4): ")

        if choice == '1':
            view_profile()
        elif choice == '2':
            if update_profile():
                print("✅ Profile updated successfully!")
        elif choice == '3':
            print("✅ Continuing with current profile...")
            # Set up auth token for direct access to diabetes prediction
            # Import the necessary function from the Diabetes_prediction module
            try:
                import importlib.util
                diabetes_file = os.path.join(os.path.dirname(__file__), '11.Diabetes_prediction.py')
                spec = importlib.util.spec_from_file_location('diabetes_prediction', diabetes_file)
                diabetes_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(diabetes_module)
                
                # Generate a new auth token in the Diabetes_prediction module
                diabetes_module.generate_auth_token()
                
                # Inform user they can now run diabetes prediction
                print("✅ Authentication set up successfully!")
                print("ℹ️ You can now run diabetes risk prediction with:")
                print("python 11.Diabetes_prediction.py")
            except Exception as e:
                print(f"❌ Error setting up diabetes prediction: {e}")
                print("ℹ️ You can try running diabetes risk prediction with:")
                print("python 11.Diabetes_prediction.py")
            return True
        elif choice == '4':
            logout()
            return False
        else:
            print("❌ Invalid choice. Please enter 1-4.")

def main_menu():
    while True:
        print("\n🏥 Health Analysis System")
        print("1. Register")
        print("2. Login")
        print("3. Exit")
        
        choice = input("Choose an option (1-3): ")
        
        if choice == '1':
            if register_user():
                if profile_menu():
                    print("\n✅ You can now proceed with your health analysis.")
                    break
        elif choice == '2':
            if login_user():
                if profile_menu():
                    print("\n✅ You can now proceed with your health analysis.")
                    break
        elif choice == '3':
            print("👋 Thank you for using Health Analysis System!")
            break
        else:
            print("❌ Invalid choice. Please enter 1-3.")

if __name__ == "__main__":
    try:
        create_tables()
        check_and_add_columns()
        main_menu()
    except Exception as e:
        logging.error(f"System error: {str(e)}")
        print("❌ A system error occurred. Please check the logs.")

