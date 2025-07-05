import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from BMI_calc import load_current_user, get_user_data, calculate_age
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas as pdf_canvas
import sqlite3
from datetime import datetime, timedelta

# --- Language Support ---
LANGUAGES = {
    'en': {
        'title': 'Steps & Diet Plan Calculator',
        'user_info': 'User Information & Goals',
        'medical_conditions': 'Medical Conditions',
        'results': 'Results & Plan',
        'progress': 'Progress Tracking',
        'meal_planning': 'Meal Planning',
        'reminders': 'Reminders & Alerts',
        'settings': 'Settings',
        'calculate': 'Calculate Personalized Plan',
        'current_weight': 'Current Weight (kg)',
        'target_weight': 'Target Weight (kg)',
        'days_to_target': 'Days to reach target',
        'activity_level': 'Activity Level',
        'custom_calories': 'Custom daily calories (optional)',
        'validation_error': 'Validation Error',
        'success': 'Success',
        'warning': 'Warning',
        'error': 'Error',
        'info': 'Info',
    },
    'es': {
        'title': 'Calculadora de Pasos y Dieta',
        'user_info': 'Información del Usuario y Objetivos',
        'medical_conditions': 'Condiciones Médicas',
        'results': 'Resultados y Plan',
        'progress': 'Seguimiento del Progreso',
        'meal_planning': 'Planificación de Comidas',
        'reminders': 'Recordatorios y Alertas',
        'settings': 'Configuración',
        'calculate': 'Calcular Plan Personalizado',
        'current_weight': 'Peso Actual (kg)',
        'target_weight': 'Peso Objetivo (kg)',
        'days_to_target': 'Días para alcanzar objetivo',
        'activity_level': 'Nivel de Actividad',
        'custom_calories': 'Calorías diarias personalizadas (opcional)',
        'validation_error': 'Error de Validación',
        'success': 'Éxito',
        'warning': 'Advertencia',
        'error': 'Error',
        'info': 'Información',
    }
}

# --- Helper Classes ---
class ValidationError(Exception):
    pass

class InputValidator:
    @staticmethod
    def validate_weight(weight_str, min_weight=20, max_weight=500):
        try:
            weight = float(weight_str)
            if weight < min_weight or weight > max_weight:
                raise ValidationError(f"Weight must be between {min_weight} and {max_weight} kg")
            return weight
        except ValueError:
            raise ValidationError("Weight must be a valid number")
    @staticmethod
    def validate_days(days_str, min_days=1, max_days=365):
        try:
            days = int(days_str)
            if days < min_days or days > max_days:
                raise ValidationError(f"Days must be between {min_days} and {max_days}")
            return days
        except ValueError:
            raise ValidationError("Days must be a valid integer")
    @staticmethod
    def validate_calories(calories_str, min_cal=800, max_cal=5000):
        if not calories_str.strip():
            return None
        try:
            calories = int(calories_str)
            if calories < min_cal or calories > max_cal:
                raise ValidationError(f"Calories must be between {min_cal} and {max_cal}")
            return calories
        except ValueError:
            raise ValidationError("Calories must be a valid integer")
    @staticmethod
    def validate_target_weight(current_weight, target_weight, max_loss_percent=0.3):
        max_loss = current_weight * max_loss_percent
        min_target = current_weight - max_loss
        max_target = current_weight + (current_weight * 0.2)
        if target_weight < min_target:
            raise ValidationError(f"Target weight too low. Minimum safe target: {min_target:.1f} kg")
        if target_weight > max_target:
            raise ValidationError(f"Target weight too high. Maximum safe target: {max_target:.1f} kg")
        return target_weight

class MedicalCondition:
    def __init__(self, name, severity="none", medications=None, notes=""):
        self.name = name
        self.severity = severity
        self.medications = medications or []
        self.notes = notes
        self.contraindications = []
        self.dietary_restrictions = []
        self.exercise_restrictions = []

class PregnancyTracker:
    @staticmethod
    def get_trimester_recommendations(weeks_pregnant):
        if weeks_pregnant <= 13:
            return {
                "trimester": "First",
                "calories_extra": 0,
                "exercise": "Light to moderate exercise, avoid contact sports",
                "restrictions": ["No alcohol", "Limit caffeine", "Avoid raw fish/meat"],
                "nutrients": ["Folic acid 400mcg", "Iron", "Calcium"],
                "warning": "Morning sickness may affect nutrition"
            }
        elif weeks_pregnant <= 27:
            return {
                "trimester": "Second",
                "calories_extra": 340,
                "exercise": "Moderate exercise, avoid lying on back",
                "restrictions": ["No alcohol", "Limit caffeine", "Avoid high-mercury fish"],
                "nutrients": ["Iron", "Calcium", "Protein increase"],
                "warning": "Energy levels typically higher"
            }
        else:
            return {
                "trimester": "Third",
                "calories_extra": 450,
                "exercise": "Light exercise, walking, swimming",
                "restrictions": ["No alcohol", "Limit caffeine", "Avoid lying flat"],
                "nutrients": ["Iron", "Calcium", "Omega-3"],
                "warning": "Avoid overheating, stay hydrated"
            }

class ProgressTracker:
    def __init__(self, db_path="progress.db"):
        self.db_path = db_path
        self.init_database()
    def init_database(self):
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS progress (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT,
                    date TEXT,
                    weight REAL,
                    steps INTEGER,
                    calories_consumed INTEGER,
                    calories_burned INTEGER,
                    notes TEXT
                )
            ''')
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS reminders (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT,
                    type TEXT,
                    message TEXT,
                    time TEXT,
                    frequency TEXT,
                    active BOOLEAN
                )
            ''')
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Database initialization error: {e}")
    def add_progress_entry(self, user_id, weight, steps, calories_consumed, calories_burned, notes=""):
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO progress (user_id, date, weight, steps, calories_consumed, calories_burned, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (user_id, datetime.now().isoformat(), weight, steps, calories_consumed, calories_burned, notes))
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"Error adding progress entry: {e}")
            return False
    def get_progress_data(self, user_id, days=30):
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            start_date = (datetime.now() - timedelta(days=days)).isoformat()
            cursor.execute('''
                SELECT date, weight, steps, calories_consumed, calories_burned, notes
                FROM progress 
                WHERE user_id = ? AND date >= ?
                ORDER BY date
            ''', (user_id, start_date))
            data = cursor.fetchall()
            conn.close()
            return data
        except Exception as e:
            print(f"Error retrieving progress data: {e}")
            return []

class MealPlanner:
    def __init__(self):
        self.meal_database = self.load_meal_database()
    def load_meal_database(self):
        return {
            "breakfast": [
                {"name": "Masala Dosa with Coconut Chutney", "calories": 350, "carbs": 50, "protein": 8, "fat": 12, "fiber": 6, "tags": ["south-indian", "vegetarian"]},
                {"name": "Poha with Peanuts", "calories": 300, "carbs": 55, "protein": 10, "fat": 8, "fiber": 5, "tags": ["north-indian", "vegetarian", "light"]},
                {"name": "Aloo Paratha with Curd", "calories": 400, "carbs": 60, "protein": 12, "fat": 15, "fiber": 8, "tags": ["north-indian", "vegetarian"]},
                {"name": "Idli with Sambar", "calories": 280, "carbs": 45, "protein": 10, "fat": 5, "fiber": 8, "tags": ["south-indian", "vegetarian", "low-fat"]},
                {"name": "Upma with Vegetables", "calories": 320, "carbs": 50, "protein": 8, "fat": 10, "fiber": 7, "tags": ["south-indian", "vegetarian"]},
                {"name": "Besan Chilla with Mint Chutney", "calories": 280, "carbs": 30, "protein": 15, "fat": 12, "fiber": 6, "tags": ["north-indian", "vegetarian", "high-protein"]}
            ],
            "lunch": [
                {"name": "Rajma Chawal with Salad", "calories": 450, "carbs": 70, "protein": 20, "fat": 10, "fiber": 15, "tags": ["north-indian", "vegetarian", "high-fiber"]},
                {"name": "Sambar Rice with Papad", "calories": 400, "carbs": 65, "protein": 12, "fat": 8, "fiber": 12, "tags": ["south-indian", "vegetarian"]},
                {"name": "Chicken Curry with Roti", "calories": 500, "carbs": 45, "protein": 35, "fat": 20, "fiber": 8, "tags": ["north-indian", "non-vegetarian"]},
                {"name": "Fish Curry with Rice", "calories": 450, "carbs": 50, "protein": 30, "fat": 15, "fiber": 6, "tags": ["south-indian", "non-vegetarian", "heart-healthy"]},
                {"name": "Dal Tadka with Jeera Rice", "calories": 420, "carbs": 65, "protein": 18, "fat": 12, "fiber": 10, "tags": ["north-indian", "vegetarian"]},
                {"name": "Vegetable Biryani with Raita", "calories": 480, "carbs": 75, "protein": 15, "fat": 15, "fiber": 10, "tags": ["hyderabadi", "vegetarian"]}
            ],
            "dinner": [
                {"name": "Roti with Palak Paneer", "calories": 400, "carbs": 40, "protein": 25, "fat": 18, "fiber": 8, "tags": ["north-indian", "vegetarian", "high-protein"]},
                {"name": "Dahi Chana Chaat", "calories": 350, "carbs": 45, "protein": 20, "fat": 10, "fiber": 12, "tags": ["north-indian", "vegetarian", "light"]},
                {"name": "Methi Thepla with Kadhi", "calories": 380, "carbs": 50, "protein": 18, "fat": 12, "fiber": 10, "tags": ["gujarati", "vegetarian"]},
                {"name": "Egg Curry with Rice", "calories": 420, "carbs": 45, "protein": 30, "fat": 15, "fiber": 6, "tags": ["non-vegetarian", "high-protein"]},
                {"name": "Moong Dal Khichdi", "calories": 350, "carbs": 55, "protein": 15, "fat": 8, "fiber": 12, "tags": ["north-indian", "vegetarian", "light"]},
                {"name": "Mushroom Masala with Roti", "calories": 380, "carbs": 40, "protein": 25, "fat": 15, "fiber": 8, "tags": ["vegetarian", "high-protein"]}
            ],
            "snacks": [
                {"name": "Sprouts Chaat", "calories": 150, "carbs": 20, "protein": 12, "fat": 5, "fiber": 8, "tags": ["healthy", "high-protein"]},
                {"name": "Fruit Salad with Chia Seeds", "calories": 120, "carbs": 25, "protein": 3, "fat": 2, "fiber": 5, "tags": ["healthy", "light"]},
                {"name": "Roasted Makhana", "calories": 100, "carbs": 15, "protein": 5, "fat": 3, "fiber": 3, "tags": ["healthy", "low-calorie"]},
                {"name": "Masala Chai with Biscuits", "calories": 180, "carbs": 25, "protein": 4, "fat": 7, "fiber": 1, "tags": ["comfort"]},
                {"name": "Dhokla with Green Chutney", "calories": 200, "carbs": 30, "protein": 8, "fat": 5, "fiber": 4, "tags": ["gujarati", "vegetarian"]},
                {"name": "Bhel Puri", "calories": 220, "carbs": 35, "protein": 6, "fat": 8, "fiber": 5, "tags": ["street-food", "vegetarian"]}
            ]
        }
    def generate_meal_plan(self, daily_calories, dietary_restrictions, days=7):
        try:
            meal_plan = {}
            target_calories_per_meal = {
                "breakfast": daily_calories * 0.25,
                "lunch": daily_calories * 0.35,
                "dinner": daily_calories * 0.35,
                "snacks": daily_calories * 0.05
            }
            for day in range(1, days + 1):
                daily_meals = {}
                daily_total = 0
                for meal_type, target_cal in target_calories_per_meal.items():
                    suitable_meals = []
                    for meal in self.meal_database[meal_type]:
                        if self.meal_fits_restrictions(meal, dietary_restrictions):
                            suitable_meals.append(meal)
                    if suitable_meals:
                        best_meal = min(suitable_meals, key=lambda x: abs(x["calories"] - target_cal))
                        daily_meals[meal_type] = best_meal
                        daily_total += best_meal["calories"]
                meal_plan[f"Day {day}"] = {
                    "meals": daily_meals,
                    "total_calories": daily_total,
                    "macros": self.calculate_daily_macros(daily_meals)
                }
            return meal_plan
        except Exception as e:
            print(f"Error generating meal plan: {e}")
            return {}
    def meal_fits_restrictions(self, meal, restrictions):
        for restriction in restrictions:
            if restriction in meal.get("tags", []):
                return True
        return len(restrictions) == 0
    def calculate_daily_macros(self, daily_meals):
        total_carbs = sum(meal.get("carbs", 0) for meal in daily_meals.values())
        total_protein = sum(meal.get("protein", 0) for meal in daily_meals.values())
        total_fat = sum(meal.get("fat", 0) for meal in daily_meals.values())
        total_fiber = sum(meal.get("fiber", 0) for meal in daily_meals.values())
        return {
            "carbs": total_carbs,
            "protein": total_protein,
            "fat": total_fat,
            "fiber": total_fiber
        }

class ReminderSystem:
    def __init__(self, progress_tracker):
        self.progress_tracker = progress_tracker
        self.active_reminders = []
        self.reminder_window = None
    def add_reminder(self, user_id, reminder_type, message, time_str, frequency):
        try:
            conn = sqlite3.connect(self.progress_tracker.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO reminders (user_id, type, message, time, frequency, active)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (user_id, reminder_type, message, time_str, frequency, True))
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"Error adding reminder: {e}")
            return False
    def get_active_reminders(self, user_id):
        try:
            conn = sqlite3.connect(self.progress_tracker.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                SELECT id, type, message, time, frequency
                FROM reminders 
                WHERE user_id = ? AND active = 1
            ''', (user_id,))
            reminders = cursor.fetchall()
            conn.close()
            return reminders
        except Exception as e:
            print(f"Error retrieving reminders: {e}")
            return []
    def show_reminder_popup(self, message, reminder_type):
        if self.reminder_window and self.reminder_window.winfo_exists():
            self.reminder_window.destroy()
        self.reminder_window = tk.Toplevel()
        self.reminder_window.title(f"Reminder - {reminder_type}")
        self.reminder_window.geometry("400x200")
        self.reminder_window.attributes('-topmost', True)
        icon = {"medication": "💊", "exercise": "🏃", "hydration": "💧", "meal": "🍽️"}.get(reminder_type, "⏰")
        ttk.Label(self.reminder_window, text=icon, font=("Arial", 24)).pack(pady=10)
        ttk.Label(self.reminder_window, text=message, font=("Arial", 12), wraplength=350).pack(pady=10)
        button_frame = ttk.Frame(self.reminder_window)
        button_frame.pack(pady=20)
        ttk.Button(button_frame, text="OK", command=self.reminder_window.destroy).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Snooze (10 min)", command=lambda: self.snooze_reminder(10)).pack(side=tk.LEFT, padx=5)
    def snooze_reminder(self, minutes):
        self.reminder_window.destroy()
        print(f"Reminder snoozed for {minutes} minutes")

# --- Main Application Class ---
class EnhancedStepsCalculatorApp:
    def __init__(self, root):
        self.root = root
        self.current_language = 'en'
        self.texts = LANGUAGES[self.current_language]
        self.root.title(self.texts['title'])
        self.root.geometry("1200x800")
        self.root.configure(bg='#f0f0f0')
        self.progress_tracker = ProgressTracker()
        self.meal_planner = MealPlanner()
        self.reminder_system = ReminderSystem(self.progress_tracker)
        self.high_contrast = False
        self.user_id = self._load_or_prompt_user_id()
        self.user_data = self._load_or_prompt_user_data(self.user_id)
        self.medical_conditions = {}
        self.create_menu()
        self.create_widgets()
        self.start_reminder_checker()

    def _load_or_prompt_user_id(self):
        user_id = None
        try:
            user_id = load_current_user()
        except Exception as e:
            print(f"Error loading current user ID: {e}")
        if not user_id:
            user_id = self._prompt_for_value("Enter your User ID:")
        return user_id

    def _load_or_prompt_user_data(self, user_id):
        # Try to load from DB
        try:
            user_row = get_user_data(user_id)
            # user_row is a sqlite3.Row, convert to dict
            user_data = dict(user_row)
            # Map DB fields to expected keys
            data = {
                "username": user_data.get("username", ""),
                "gender": user_data.get("gender", ""),
                "date_of_birth": user_data.get("date_of_birth", ""),
                "height": str(user_data.get("height", "")),
                "weight": str(user_data.get("weight", ""))
            }
        except Exception as e:
            print(f"User data not found in DB: {e}")
            data = {
                "username": "",
                "gender": "",
                "date_of_birth": "",
                "height": "",
                "weight": ""
            }
        # Prompt for any missing fields
        for key, prompt in [
            ("username", "Enter your name:"),
            ("gender", "Enter your gender (Male/Female):"),
            ("date_of_birth", "Enter your date of birth (YYYY-MM-DD):"),
            ("height", "Enter your height in cm:"),
            ("weight", "Enter your weight in kg:")
        ]:
            if not data[key]:
                data[key] = self._prompt_for_value(prompt)
        return data

    def _prompt_for_value(self, prompt):
        import tkinter.simpledialog
        value = None
        while not value:
            value = tkinter.simpledialog.askstring("Input Required", prompt, parent=self.root)
            if value is None:
                messagebox.showwarning("Input Required", "This field is required.")
        return value

    def create_menu(self):
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        # Language menu
        settings_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label=self.texts['settings'], menu=settings_menu)
        language_menu = tk.Menu(settings_menu, tearoff=0)
        settings_menu.add_cascade(label="Language", menu=language_menu)
        language_menu.add_command(label="English", command=lambda: self.change_language('en'))
        language_menu.add_command(label="Español", command=lambda: self.change_language('es'))

    def change_language(self, lang_code):
        if lang_code in LANGUAGES:
            self.current_language = lang_code
            self.texts = LANGUAGES[lang_code]
            self.update_ui_language()

    def update_ui_language(self):
        self.root.title(self.texts['title'])
        self.notebook.tab(0, text=self.texts['user_info'])
        self.notebook.tab(1, text=self.texts['medical_conditions'])
        self.notebook.tab(2, text=self.texts['results'])
        self.notebook.tab(3, text=self.texts['progress'])
        self.notebook.tab(4, text=self.texts['meal_planning'])
        self.notebook.tab(5, text=self.texts['reminders'])

    def create_widgets(self):
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.create_info_tab()
        self.create_medical_tab()
        self.create_results_tab()
        self.create_progress_tab()
        self.create_meal_planning_tab()
        self.create_reminders_tab()

    def create_info_tab(self):
        self.info_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.info_frame, text=self.texts['user_info'])
        user = self.user_data
        info_label_frame = ttk.LabelFrame(self.info_frame, text="Current User Information")
        info_label_frame.pack(fill=tk.X, padx=10, pady=5)
        left_frame = ttk.Frame(info_label_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=5)
        right_frame = ttk.Frame(info_label_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=5)
        ttk.Label(left_frame, text=f"User: {user['username']}", font=("Arial", 12, "bold")).pack(anchor=tk.W)
        ttk.Label(left_frame, text=f"Gender: {user['gender']}").pack(anchor=tk.W)
        ttk.Label(left_frame, text=f"Date of Birth: {user['date_of_birth']}").pack(anchor=tk.W)
        age = calculate_age(user['date_of_birth'])
        height_cm = float(user['height'])
        weight = float(user['weight'])
        bmi = round(weight / ((height_cm/100) ** 2), 2)
        ttk.Label(right_frame, text=f"Age: {age} years", font=("Arial", 10, "bold")).pack(anchor=tk.W)
        ttk.Label(right_frame, text=f"Height: {user['height']} cm").pack(anchor=tk.W)
        ttk.Label(right_frame, text=f"Current Weight: {user['weight']} kg").pack(anchor=tk.W)
        ttk.Label(right_frame, text=f"Current BMI: {bmi}", font=("Arial", 10, "bold")).pack(anchor=tk.W)
        goals_frame = ttk.LabelFrame(self.info_frame, text="Your Goals")
        goals_frame.pack(fill=tk.X, padx=10, pady=5)
        input_grid = ttk.Frame(goals_frame)
        input_grid.pack(fill=tk.X, padx=10, pady=10)
        ttk.Label(input_grid, text=self.texts['target_weight']).grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.target_weight_var = tk.StringVar(value=str(weight - 2))
        target_weight_entry = ttk.Entry(input_grid, textvariable=self.target_weight_var, width=12)
        target_weight_entry.grid(row=0, column=1, padx=5, pady=5)
        target_weight_entry.bind('<FocusOut>', self.validate_target_weight)
        ttk.Label(input_grid, text=self.texts['days_to_target']).grid(row=0, column=2, sticky=tk.W, padx=5, pady=5)
        self.days_var = tk.StringVar(value="30")
        days_entry = ttk.Entry(input_grid, textvariable=self.days_var, width=12)
        days_entry.grid(row=0, column=3, padx=5, pady=5)
        days_entry.bind('<FocusOut>', self.validate_days)
        ttk.Label(input_grid, text=self.texts['activity_level']).grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.activity_var = tk.StringVar(value="moderate")
        activity_combo = ttk.Combobox(input_grid, textvariable=self.activity_var, 
                                    values=["sedentary", "light", "moderate", "active", "very_active"], 
                                    width=12, state="readonly")
        activity_combo.grid(row=1, column=1, padx=5, pady=5)
        ttk.Label(input_grid, text=self.texts['custom_calories']).grid(row=1, column=2, sticky=tk.W, padx=5, pady=5)
        self.custom_cal_var = tk.StringVar()
        cal_entry = ttk.Entry(input_grid, textvariable=self.custom_cal_var, width=12)
        cal_entry.grid(row=1, column=3, padx=5, pady=5)
        cal_entry.bind('<FocusOut>', self.validate_calories)
        self.validation_label = ttk.Label(goals_frame, text="", foreground="green")
        self.validation_label.pack(pady=5)

    def create_medical_tab(self):
        self.medical_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.medical_frame, text=self.texts['medical_conditions'])
        instructions = ttk.Label(self.medical_frame, 
                               text="Please select your medical conditions and their severity levels:",
                               font=("Arial", 12, "bold"))
        instructions.pack(pady=10)
        main_container = ttk.Frame(self.medical_frame)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        canvas = tk.Canvas(main_container)
        scrollbar = ttk.Scrollbar(main_container, orient="vertical", command=canvas.yview)
        self.scrollable_frame = ttk.Frame(canvas)
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        self.condition_widgets = {}
        self.create_condition_group("Cardiovascular", [
            ("Heart Disease", "heart_disease"),
            ("High Blood Pressure", "high_blood_pressure"),
            ("Heart Arrhythmia", "arrhythmia"),
            ("High Cholesterol", "high_cholesterol")
        ])
        self.create_condition_group("Metabolic", [
            ("Type 1 Diabetes", "diabetes_type1"),
            ("Type 2 Diabetes", "diabetes_type2"),
            ("Prediabetes", "prediabetes"),
            ("Hyperthyroidism", "hyperthyroid"),
            ("Hypothyroidism", "hypothyroid"),
            ("Metabolic Syndrome", "metabolic_syndrome")
        ])
        self.create_condition_group("Hormonal", [
            ("PCOS", "pcos"),
            ("Menopause", "menopause"),
            ("Insulin Resistance", "insulin_resistance"),
            ("Adrenal Disorders", "adrenal_disorders")
        ])
        pregnancy_frame = self.create_condition_group("Pregnancy/Reproductive", [
            ("Currently Pregnant", "pregnant"),
            ("Currently Breastfeeding", "breastfeeding")
        ])
        self.pregnancy_weeks_var = tk.StringVar()
        pregnancy_weeks_frame = ttk.Frame(pregnancy_frame)
        pregnancy_weeks_frame.pack(fill=tk.X, padx=20, pady=5)
        ttk.Label(pregnancy_weeks_frame, text="If pregnant, weeks:").pack(side=tk.LEFT)
        ttk.Entry(pregnancy_weeks_frame, textvariable=self.pregnancy_weeks_var, width=5).pack(side=tk.LEFT)
        self.create_condition_group("Gastrointestinal", [
            ("IBS", "ibs"),
            ("GERD/Acid Reflux", "gerd"),
            ("Celiac Disease", "celiac"),
            ("Crohn's Disease", "crohns"),
            ("Ulcerative Colitis", "colitis")
        ])
        self.create_condition_group("Musculoskeletal", [
            ("Osteoporosis", "osteoporosis"),
            ("Osteoarthritis", "osteoarthritis"),
            ("Rheumatoid Arthritis", "rheumatoid"),
            ("Chronic Back Pain", "back_pain")
        ])
        self.create_condition_group("Other Conditions", [
            ("Depression", "depression"),
            ("Anxiety", "anxiety"),
            ("Migraines", "migraines"),
            ("Asthma", "asthma"),
            ("Sleep Apnea", "sleep_apnea")
        ])
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        notes_frame = ttk.LabelFrame(self.medical_frame, text="Additional Medical Notes")
        notes_frame.pack(fill=tk.X, padx=10, pady=10)
        self.medical_notes_var = tk.StringVar()
        ttk.Entry(notes_frame, textvariable=self.medical_notes_var, width=60).pack(padx=5, pady=5)

    def create_condition_group(self, group_name, conditions):
        group_frame = ttk.LabelFrame(self.scrollable_frame, text=group_name)
        group_frame.pack(fill=tk.X, padx=10, pady=5)
        for condition_name, condition_key in conditions:
            condition_frame = ttk.Frame(group_frame)
            condition_frame.pack(fill=tk.X, padx=20, pady=2)
            var = tk.BooleanVar(value=False)
            chk = ttk.Checkbutton(condition_frame, text=condition_name, variable=var,
                                 command=lambda c=condition_key, v=var: self.toggle_condition(c, v))
            chk.pack(side=tk.LEFT, anchor=tk.W)
            severity_var = tk.StringVar(value="none")
            severity_combo = ttk.Combobox(condition_frame, textvariable=severity_var,
                                        values=["none", "mild", "moderate", "severe"],
                                        width=10, state="readonly")
            severity_combo.pack(side=tk.RIGHT, padx=5)
            self.condition_widgets[condition_key] = {
                "var": var,
                "severity": severity_var,
                "name": condition_name
            }
        return group_frame

    def toggle_condition(self, condition_key, var):
        """Toggle medical condition on/off and fill restrictions"""
        restrictions_map = {
            'diabetes_type1': {'dietary': ['no_extreme_calorie_restriction', 'monitor_blood_sugar', 'no_fasting'], 'exercise': ['avoid_high_intensity']},
            'diabetes_type2': {'dietary': ['limit_simple_sugars', 'monitor_blood_sugar'], 'exercise': ['gradual_exercise_increase']},
            'hypertension': {'dietary': ['limit_sodium'], 'exercise': ['avoid_intense_isometric_exercises']},
            'heart_disease': {'dietary': ['limit_sodium'], 'exercise': ['no_high_intensity_without_clearance']},
            'osteoporosis': {'dietary': [], 'exercise': ['avoid_high_impact_exercises', 'avoid_forward_spinal_flexion']},
            'arthritis': {'dietary': ['anti_inflammatory_diet'], 'exercise': ['low_impact_exercises']},
            'kidney_disease': {'dietary': ['limit_protein', 'limit_potassium', 'limit_phosphorus'], 'exercise': ['gentle_exercise']},
            'pregnant': {'dietary': ['avoid_alcohol', 'limit_caffeine', 'avoid_raw_foods'], 'exercise': ['avoid_contact_sports']},
            'asthma': {'dietary': [], 'exercise': ['avoid_trigger_environments']},
            'thyroid_disorder': {'dietary': [], 'exercise': []},
            # Add more as needed
        }
        if var.get():
            cond = MedicalCondition(
                name=self.condition_widgets[condition_key]["name"],
                severity=self.condition_widgets[condition_key]["severity"].get()
            )
            # Fill restrictions
            if condition_key in restrictions_map:
                cond.dietary_restrictions = restrictions_map[condition_key].get('dietary', [])
                cond.exercise_restrictions = restrictions_map[condition_key].get('exercise', [])
            self.medical_conditions[condition_key] = cond
        else:
            self.medical_conditions.pop(condition_key, None)

    def create_results_tab(self):
        self.results_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.results_frame, text=self.texts['results'])
        self.results_text = tk.Text(self.results_frame, wrap=tk.WORD, height=20, padx=10, pady=10)
        self.results_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        btn_frame = ttk.Frame(self.results_frame)
        btn_frame.pack(fill=tk.X, padx=10, pady=10)
        ttk.Button(btn_frame, text=self.texts['calculate'], command=self.calculate_plan).pack(pady=5)

    def create_progress_tab(self):
        self.progress_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.progress_frame, text=self.texts['progress'])
        ttk.Label(self.progress_frame, text="(Progress tracking coming soon)").pack(pady=20)

    def create_meal_planning_tab(self):
        self.meal_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.meal_frame, text=self.texts['meal_planning'])
        ttk.Label(self.meal_frame, text="(Meal planning coming soon)").pack(pady=20)

    def create_reminders_tab(self):
        self.reminders_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.reminders_frame, text=self.texts['reminders'])
        ttk.Label(self.reminders_frame, text="(Reminders coming soon)").pack(pady=20)

    def validate_target_weight(self, event=None):
        try:
            weight = float(self.target_weight_var.get().strip())
            if weight <= 0:
                raise ValidationError("Target weight must be greater than 0")
            self.target_weight_var.set(str(weight))
            self.validation_label.config(text="✓ Valid target weight", foreground="green")
            return True
        except Exception as e:
            self.validation_label.config(text=str(e), foreground="red")
            return False

    def validate_days(self, event=None):
        try:
            days = int(self.days_var.get().strip())
            if days <= 0:
                raise ValidationError("Days must be greater than 0")
            self.days_var.set(str(days))
            self.validation_label.config(text="✓ Valid days", foreground="green")
            return True
        except Exception as e:
            self.validation_label.config(text=str(e), foreground="red")
            return False

    def validate_calories(self, event=None):
        try:
            calories = self.custom_cal_var.get().strip()
            if not calories:
                self.custom_cal_var.set("2000")
                self.validation_label.config(text="Using default calories", foreground="blue")
                return True
            calories = int(calories)
            if calories < 0:
                raise ValidationError("Calories must be non-negative")
            self.custom_cal_var.set(str(calories))
            self.validation_label.config(text="✓ Valid calories", foreground="green")
            return True
        except Exception as e:
            self.validation_label.config(text=str(e), foreground="red")
            return False

    def calculate_plan(self):
        try:
            if not all([
                self.validate_target_weight(),
                self.validate_days(),
                self.validate_calories()
            ]):
                messagebox.showerror(self.texts['validation_error'], "Please fix validation errors before calculating")
                return
            weight = float(self.user_data['weight'])
            target_weight = float(self.target_weight_var.get())
            days = int(self.days_var.get())
            height_cm = float(self.user_data['height'])
            height = height_cm / 100
            name = self.user_data['username']
            gender = self.user_data['gender']
            age = calculate_age(self.user_data['date_of_birth'])
            if self.custom_cal_var.get():
                total_calories = int(self.custom_cal_var.get())
            else:
                total_calories = 2000
        except Exception as e:
            messagebox.showerror(self.texts['error'], f"Invalid input: {e}")
            return
        bmi = round(weight / (height ** 2), 2)
        bmi_cat = bmi_category(bmi)
        bmr = calculate_bmr(weight, height_cm, age, gender)
        diet = recommend_diet(bmi_cat, total_calories)
        weight_diff = weight - target_weight
        total_calories_to_burn = weight_diff * 7700 if weight_diff > 0 else 0
        daily_calorie_burn_goal = total_calories_to_burn / days if weight_diff > 0 else 0
        net_calories_to_burn = max(0, (total_calories + daily_calorie_burn_goal) - bmr)
        steps_per_day = int(net_calories_to_burn / 0.04)
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, f"\n📊 Health Plan for {name}\n")
        self.results_text.insert(tk.END, f"🎯 Target: From {weight} kg → {target_weight} kg in {days} days\n")
        self.results_text.insert(tk.END, f"⚖️ Current BMI: {bmi} ({bmi_cat})\n")
        self.results_text.insert(tk.END, f"🧬 Estimated BMR ({gender.title()}): {bmr} kcal/day\n")
        if weight_diff > 0:
            self.results_text.insert(tk.END, f"🔥 You need to burn: {total_calories_to_burn:.0f} kcal total\n")
            self.results_text.insert(tk.END, f"📅 Daily burn target (for weight loss): {daily_calorie_burn_goal:.0f} kcal\n")
        else:
            self.results_text.insert(tk.END, "✅ No weight loss targeted.\n")
        self.results_text.insert(tk.END, f"🚶 Steps required per day: {steps_per_day} steps (after subtracting BMR)\n")
        self.results_text.insert(tk.END, f"\n🥗 Suggested Diet Plan (Calories: {total_calories} kcal):\n")
        self.results_text.insert(tk.END, f"🍚 Carbs: {diet['carbs_g']}g\n")
        self.results_text.insert(tk.END, f"🥑 Fat: {diet['fat_g']}g\n")
        self.results_text.insert(tk.END, f"🌾 Fiber: {diet['fiber_g']}g\n")

    def generate_meal_plan(self):
        try:
            results_text = self.results_text.get(1.0, tk.END)
            for line in results_text.split("\n"):
                if "Target Daily Calories:" in line:
                    target_calories = float(line.split(":")[1].strip().split(" ")[0])
                    break
            else:
                target_calories = 2000
            dietary_restrictions = []
            for condition in self.medical_conditions.values():
                if condition.severity != "none":
                    dietary_restrictions.extend(condition.dietary_restrictions)
            # Remove duplicates
            dietary_restrictions = list(set(dietary_restrictions))
            meal_plan = self.meal_planner.generate_meal_plan(target_calories, dietary_restrictions)
            self.display_meal_plan(meal_plan)
            messagebox.showinfo(self.texts['success'], "Indian meal plan generated successfully!")
        except Exception as e:
            messagebox.showerror(self.texts['error'], f"Error generating meal plan: {str(e)}")

    def export_pdf(self):
        try:
            file_path = filedialog.asksaveasfilename(
                defaultextension=".pdf",
                filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")],
                title="Export Plan as PDF"
            )
            if not file_path:
                return
            c = pdf_canvas.Canvas(file_path, pagesize=letter)
            width, height = letter
            y = height - 40
            c.setFont("Helvetica-Bold", 16)
            c.drawString(40, y, self.texts['title'])
            y -= 30
            c.setFont("Helvetica", 10)
            for section, text_widget in [("Results", self.results_text), ("Meal Plan", self.meal_plan_text)]:
                c.setFont("Helvetica-Bold", 12)
                c.drawString(40, y, section)
                y -= 20
                c.setFont("Helvetica", 10)
                for line in text_widget.get(1.0, tk.END).split("\n"):
                    if y < 40:
                        c.showPage()
                        y = height - 40
                    c.drawString(40, y, line[:110])
                    y -= 14
                y -= 10
            c.save()
            messagebox.showinfo(self.texts['success'], "PDF exported successfully!")
        except Exception as e:
            messagebox.showerror(self.texts['error'], f"Error exporting PDF: {str(e)}")

    def delete_reminder(self):
        selection = self.reminders_listbox.curselection()
        if not selection:
            messagebox.showwarning(self.texts['warning'], "Please select a reminder to delete")
            return
        try:
            idx = selection[0]
            reminders = self.reminder_system.get_active_reminders(self.user_id)
            reminder_id = reminders[idx][0]
            conn = sqlite3.connect(self.progress_tracker.db_path)
            cursor = conn.cursor()
            cursor.execute("UPDATE reminders SET active = 0 WHERE id = ?", (reminder_id,))
            conn.commit()
            conn.close()
            self.update_reminders_list()
            messagebox.showinfo(self.texts['success'], "Reminder deleted successfully!")
        except Exception as e:
            messagebox.showerror(self.texts['error'], f"Error deleting reminder: {str(e)}")

    def toggle_high_contrast(self):
        self.high_contrast = not self.high_contrast
        style = ttk.Style()
        if self.high_contrast:
            self.root.configure(bg='black')
            style.theme_use('clam')
            style.configure('.', background='black', foreground='white')
            style.configure('TLabel', background='black', foreground='white')
            style.configure('TFrame', background='black')
            style.configure('TButton', background='black', foreground='white')
            style.configure('TEntry', fieldbackground='black', foreground='white')
        else:
            self.root.configure(bg='#f0f0f0')
            style.theme_use('default')
            style.configure('.', background='#f0f0f0', foreground='black')
            style.configure('TLabel', background='#f0f0f0', foreground='black')
            style.configure('TFrame', background='#f0f0f0')
            style.configure('TButton', background='#f0f0f0', foreground='black')
            style.configure('TEntry', fieldbackground='white', foreground='black')

    def add_tooltip(self, widget, text):
        def on_enter(event):
            self.tooltip = tk.Toplevel(widget)
            self.tooltip.wm_overrideredirect(True)
            x = widget.winfo_rootx() + 20
            y = widget.winfo_rooty() + 20
            self.tooltip.wm_geometry(f"+{x}+{y}")
            label = tk.Label(self.tooltip, text=text, background="yellow", relief='solid', borderwidth=1, font=("Arial", 10))
            label.pack()
        def on_leave(event):
            if hasattr(self, 'tooltip'):
                self.tooltip.destroy()
        widget.bind("<Enter>", on_enter)
        widget.bind("<Leave>", on_leave)

    def start_reminder_checker(self):
        # Placeholder for reminder checking logic
        pass

if __name__ == "__main__":
    root = tk.Tk()
    app = EnhancedStepsCalculatorApp(root)
    root.mainloop()
