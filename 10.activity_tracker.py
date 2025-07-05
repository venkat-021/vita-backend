import sqlite3
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from BMI_calc import load_current_user, get_user_data

class ActivityTracker:
    def __init__(self):
        self.conn = self.get_db_connection()
        self.create_activity_tables()
        
    def get_db_connection(self):
        """Create a database connection"""
        try:
            conn = sqlite3.connect("health_analysis.db")
            conn.row_factory = sqlite3.Row
            return conn
        except sqlite3.Error as e:
            print(f"Database connection error: {e}")
            raise

    def create_activity_tables(self):
        """Create necessary tables for activity tracking"""
        try:
            cursor = self.conn.cursor()
            
            # Create activities table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS activities (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    met_value REAL NOT NULL,
                    category TEXT NOT NULL
                )
            ''')
            
            # Create user_activities table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_activities (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    activity_id INTEGER NOT NULL,
                    date_recorded DATE NOT NULL,
                    duration_minutes INTEGER NOT NULL,
                    calories_burned REAL NOT NULL,
                    notes TEXT,
                    FOREIGN KEY (user_id) REFERENCES users (id),
                    FOREIGN KEY (activity_id) REFERENCES activities (id)
                )
            ''')
            
            # Insert default activities if not exists
            default_activities = [
                ('Walking', 3.5, 'Cardio'),
                ('Running', 9.8, 'Cardio'),
                ('Cycling', 7.5, 'Cardio'),
                ('Swimming', 6.0, 'Cardio'),
                ('Yoga', 3.0, 'Flexibility'),
                ('Weight Training', 5.0, 'Strength'),
                ('HIIT', 8.0, 'Cardio'),
                ('Dancing', 4.5, 'Cardio'),
                ('Pilates', 3.5, 'Flexibility'),
                ('Basketball', 6.5, 'Sports'),
                ('Tennis', 7.0, 'Sports'),
                ('Golf', 3.5, 'Sports'),
                ('Hiking', 5.5, 'Cardio'),
                ('Stretching', 2.5, 'Flexibility'),
                ('Meditation', 1.5, 'Mindfulness')
            ]
            
            cursor.executemany('''
                INSERT OR IGNORE INTO activities (name, met_value, category)
                VALUES (?, ?, ?)
            ''', default_activities)
            
            self.conn.commit()
        except sqlite3.Error as e:
            print(f"Error creating activity tables: {e}")
            raise

    def log_activity(self, user_id, activity_name, duration_minutes, notes=""):
        """Log a new activity for the user"""
        try:
            cursor = self.conn.cursor()
            
            # Get activity details
            cursor.execute('SELECT id, met_value FROM activities WHERE name = ?', (activity_name,))
            activity = cursor.fetchone()
            
            if not activity:
                print(f"Activity '{activity_name}' not found in database.")
                return False
            
            # Get user's weight for calorie calculation
            user_data = get_user_data(user_id)
            weight_kg = user_data['weight_kg']
            
            # Calculate calories burned using MET formula
            # Calories = MET × Weight (kg) × Duration (hours)
            calories_burned = activity['met_value'] * weight_kg * (duration_minutes / 60)
            
            # Insert activity log
            cursor.execute('''
                INSERT INTO user_activities 
                (user_id, activity_id, date_recorded, duration_minutes, calories_burned, notes)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (user_id, activity['id'], datetime.now().date(), duration_minutes, 
                 calories_burned, notes))
            
            self.conn.commit()
            print(f"✅ Activity logged successfully! Calories burned: {calories_burned:.1f}")
            return True
            
        except sqlite3.Error as e:
            print(f"Error logging activity: {e}")
            return False

    def get_activity_history(self, user_id, days=30):
        """Get user's activity history for the specified number of days"""
        try:
            cursor = self.conn.cursor()
            
            cursor.execute('''
                SELECT 
                    a.name as activity_name,
                    a.category,
                    ua.date_recorded,
                    ua.duration_minutes,
                    ua.calories_burned,
                    ua.notes
                FROM user_activities ua
                JOIN activities a ON ua.activity_id = a.id
                WHERE ua.user_id = ?
                AND ua.date_recorded >= date('now', ?)
                ORDER BY ua.date_recorded DESC
            ''', (user_id, f'-{days} days'))
            
            return cursor.fetchall()
            
        except sqlite3.Error as e:
            print(f"Error fetching activity history: {e}")
            return []

    def generate_activity_report(self, user_id, days=30):
        """Generate a comprehensive activity report with visualizations"""
        try:
            history = self.get_activity_history(user_id, days)
            if not history:
                print("No activity history found for the specified period.")
                return
            
            # Convert to DataFrame
            df = pd.DataFrame([dict(row) for row in history])
            
            # Create directory for user's activity charts
            user_dir = f"user_{user_id}_activity_charts"
            os.makedirs(user_dir, exist_ok=True)
            
            # Set style
            plt.style.use('ggplot')
            
            # 1. Daily Activity Summary
            plt.figure(figsize=(15, 8))
            daily_summary = df.groupby('date_recorded').agg({
                'duration_minutes': 'sum',
                'calories_burned': 'sum'
            }).reset_index()
            
            gs = GridSpec(2, 1)
            
            # Duration plot
            ax1 = plt.subplot(gs[0])
            sns.barplot(x='date_recorded', y='duration_minutes', data=daily_summary, color='skyblue')
            ax1.set_title('Daily Activity Duration')
            ax1.set_xlabel('Date')
            ax1.set_ylabel('Duration (minutes)')
            plt.xticks(rotation=45)
            
            # Calories plot
            ax2 = plt.subplot(gs[1])
            sns.barplot(x='date_recorded', y='calories_burned', data=daily_summary, color='lightgreen')
            ax2.set_title('Daily Calories Burned')
            ax2.set_xlabel('Date')
            ax2.set_ylabel('Calories')
            plt.xticks(rotation=45)
            
            plt.tight_layout()
            plt.savefig(f'{user_dir}/daily_activity_summary.png')
            plt.close()
            
            # 2. Activity Type Distribution
            plt.figure(figsize=(12, 6))
            activity_summary = df.groupby('activity_name').agg({
                'duration_minutes': 'sum',
                'calories_burned': 'sum'
            }).reset_index()
            
            # Pie chart for duration distribution
            plt.subplot(1, 2, 1)
            plt.pie(activity_summary['duration_minutes'], 
                   labels=activity_summary['activity_name'],
                   autopct='%1.1f%%')
            plt.title('Activity Duration Distribution')
            
            # Pie chart for calories distribution
            plt.subplot(1, 2, 2)
            plt.pie(activity_summary['calories_burned'],
                   labels=activity_summary['activity_name'],
                   autopct='%1.1f%%')
            plt.title('Calories Burned Distribution')
            
            plt.tight_layout()
            plt.savefig(f'{user_dir}/activity_distribution.png')
            plt.close()
            
            # 3. Category Analysis
            plt.figure(figsize=(12, 6))
            category_summary = df.groupby('category').agg({
                'duration_minutes': 'sum',
                'calories_burned': 'sum'
            }).reset_index()
            
            # Bar plot for category duration
            plt.subplot(1, 2, 1)
            sns.barplot(x='category', y='duration_minutes', data=category_summary, color='skyblue')
            plt.title('Activity Duration by Category')
            plt.xticks(rotation=45)
            
            # Bar plot for category calories
            plt.subplot(1, 2, 2)
            sns.barplot(x='category', y='calories_burned', data=category_summary, color='lightgreen')
            plt.title('Calories Burned by Category')
            plt.xticks(rotation=45)
            
            plt.tight_layout()
            plt.savefig(f'{user_dir}/category_analysis.png')
            plt.close()
            
            # Print summary statistics
            print("\n📊 Activity Report Summary:")
            print(f"Total Activities: {len(df)}")
            print(f"Total Duration: {df['duration_minutes'].sum():.1f} minutes")
            print(f"Total Calories Burned: {df['calories_burned'].sum():.1f}")
            print(f"Average Daily Duration: {df.groupby('date_recorded')['duration_minutes'].sum().mean():.1f} minutes")
            print(f"Average Daily Calories: {df.groupby('date_recorded')['calories_burned'].sum().mean():.1f}")
            
            print(f"\n📈 Activity charts have been saved in the '{user_dir}' directory:")
            print(f"- {user_dir}/daily_activity_summary.png")
            print(f"- {user_dir}/activity_distribution.png")
            print(f"- {user_dir}/category_analysis.png")
            
        except Exception as e:
            print(f"Error generating activity report: {e}")
            raise

def main():
    try:
        # Initialize activity tracker
        tracker = ActivityTracker()
        
        # Get current user
        user_id = load_current_user()
        if user_id is None:
            print("❌ Please log in first to track activities.")
            print("Run 'python Login_page.py' to log in.")
            return
        
        while True:
            print("\n=== Activity Tracking Menu ===")
            print("1. Log New Activity")
            print("2. View Activity History")
            print("3. Generate Activity Report")
            print("4. Exit")
            
            choice = input("\nEnter your choice (1-4): ")
            
            if choice == '1':
                print("\nAvailable Activities:")
                cursor = tracker.conn.cursor()
                cursor.execute('SELECT name, category FROM activities ORDER BY category, name')
                activities = cursor.fetchall()
                
                for activity in activities:
                    print(f"- {activity['name']} ({activity['category']})")
                
                activity_name = input("\nEnter activity name: ")
                duration = int(input("Enter duration in minutes: "))
                notes = input("Enter any notes (optional): ")
                
                tracker.log_activity(user_id, activity_name, duration, notes)
                
            elif choice == '2':
                history = tracker.get_activity_history(user_id)
                if history:
                    print("\nRecent Activities:")
                    for activity in history:
                        print(f"\nDate: {activity['date_recorded']}")
                        print(f"Activity: {activity['activity_name']}")
                        print(f"Duration: {activity['duration_minutes']} minutes")
                        print(f"Calories Burned: {activity['calories_burned']:.1f}")
                        if activity['notes']:
                            print(f"Notes: {activity['notes']}")
                else:
                    print("No activity history found.")
                    
            elif choice == '3':
                days = int(input("Enter number of days for report (default 30): ") or "30")
                tracker.generate_activity_report(user_id, days)
                
            elif choice == '4':
                print("Goodbye!")
                break
                
            else:
                print("Invalid choice. Please try again.")
                
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if 'tracker' in locals():
            tracker.conn.close()

if __name__ == "__main__":
    main() 