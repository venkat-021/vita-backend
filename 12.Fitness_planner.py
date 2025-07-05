from typing import List, Dict, Optional
from enum import Enum
from dataclasses import dataclass, field
import sqlite3
import sys
from datetime import datetime

# ================== Enums ==================
class Goal(Enum):
    WEIGHT_LOSS = "Weight Loss"
    MUSCLE_GAIN = "Muscle Gain"
    ENDURANCE = "Endurance"
    FLEXIBILITY = "Flexibility"
    GENERAL_FITNESS = "General Fitness"

class Equipment(Enum):
    NONE = "None"
    DUMBBELLS = "Dumbbells"
    RESISTANCE_BANDS = "Resistance Bands"
    YOGA_MAT = "Yoga Mat"
    TREADMILL = "Treadmill"
    CYCLE = "Cycle"
    BARBELL = "Barbell"
    KETTLEBELL = "Kettlebell"
    BODYWEIGHT = "Bodyweight"

class DietType(Enum):
    VEG = "Vegetarian"
    NON_VEG = "Non-Vegetarian"
    VEGAN = "Vegan"
    KETO = "Keto"
    PALEO = "Paleo"
    MEDITERRANEAN = "Mediterranean"
    OTHER = "Other"

# ================== Data Models ==================
@dataclass
class UserProfile:
    user_id: int
    name: str
    age: int
    height_cm: float
    weight_kg: float
    gender: str
    goal: Goal
    stamina_level: int
    health_conditions: List[str] = field(default_factory=list)
    restrictions: List[str] = field(default_factory=list)
    equipment: List[str] = field(default_factory=list)
    diet_type: DietType = DietType.OTHER
    preferences: List[str] = field(default_factory=list)
    target_days_per_week: int = 3
    allergies: List[str] = field(default_factory=list)
    cuisine_preferences: List[str] = field(default_factory=list)
    budget: str = "medium"
    progress: List[Dict] = field(default_factory=list)

# ================== Database Access ==================
def get_db_connection():
    try:
        conn = sqlite3.connect("health_analysis.db")
        conn.row_factory = sqlite3.Row
        return conn
    except sqlite3.Error as e:
        print(f"Database connection error: {e}")
        raise

# ================== User Input Functions ==================
def get_user_input(prompt: str, input_type=str, options: List = None, default=None):
    while True:
        try:
            user_input = input(prompt).strip()
            if not user_input and default is not None:
                return default
            if options and user_input.lower() not in [str(o).lower() for o in options]:
                print(f"Please choose from: {', '.join(options)}")
                continue
            return input_type(user_input)
        except ValueError:
            print(f"Please enter a valid {input_type.__name__}")

def select_from_enum(enum_class, prompt: str) -> Enum:
    print(prompt)
    for i, option in enumerate(enum_class, 1):
        print(f"{i}. {option.value}")
    while True:
        try:
            choice = int(input("Enter your choice (number): ")) - 1
            if 0 <= choice < len(list(enum_class)):
                return list(enum_class)[choice]
            print(f"Please enter 1-{len(list(enum_class))}")
        except ValueError:
            print("Please enter a number")

def get_multiselect(prompt: str, options: List[str]) -> List[str]:
    print(prompt)
    for i, option in enumerate(options, 1):
        print(f"{i}. {option}")
    print("Enter selections separated by commas (e.g., 1,3)")
    while True:
        try:
            selections = input("Your choices: ").strip().split(',')
            selected = [options[int(s.strip())-1] for s in selections if s.strip().isdigit() and 1 <= int(s.strip()) <= len(options)]
            if selected:
                return selected
            print("Please select at least one option")
        except (ValueError, IndexError):
            print("Invalid selection")

# ================== Interactive Profile Creation ==================
def create_user_profile() -> UserProfile:
    print("\n" + "="*40)
    print(" FITNESS PLANNER PROFILE SETUP ".center(40, "="))
    print("="*40 + "\n")

    # Try to get user_id from database (simulate login or ask for username)
    conn = get_db_connection()
    cursor = conn.cursor()
    user_id = None
    name = None
    # Try to load user by username
    username = input("Enter your username (leave blank to create new): ").strip()
    if username:
        cursor.execute("SELECT id, username FROM users WHERE username = ?", (username,))
        row = cursor.fetchone()
        if row:
            user_id = row[0]
            name = row[1]
            print(f"Loaded user: {name}")
        else:
            print("User not found. Creating new profile.")
    if not user_id:
        # Register new user
        name = get_user_input("Your name: ")
        age = get_user_input("Your age: ", int)
        gender = get_user_input("Gender (male/female/other): ", options=["male", "female", "other"])
        # Insert into users table
        cursor.execute("INSERT INTO users (username, password, profile_completed) VALUES (?, ?, 1)", (name, "fitness", 1))
        user_id = cursor.lastrowid
        conn.commit()
    # Try to load user profile
    cursor.execute("SELECT * FROM user_profiles WHERE user_id = ?", (user_id,))
    profile_row = cursor.fetchone()
    if profile_row:
        age = profile_row["date_of_birth"]
        gender = profile_row["gender"]
        height_cm = profile_row["height"]
        weight_kg = profile_row["weight"]
    else:
        age = get_user_input("Your age: ", int)
        height_cm = get_user_input("Height (cm): ", float)
        weight_kg = get_user_input("Weight (kg): ", float)
        gender = get_user_input("Gender (male/female/other): ", options=["male", "female", "other"])
        # Insert into user_profiles
        cursor.execute(
            "INSERT INTO user_profiles (user_id, date_of_birth, gender, height, weight, arm_length, arm_circumference, hip, waist, leg_length) VALUES (?, ?, ?, ?, ?, 0, 0, 0, 0, 0)",
            (user_id, str(datetime.now().date()), gender, height_cm, weight_kg)
        )
        conn.commit()
    # Fitness Goals
    goal = select_from_enum(Goal, "\nSelect your primary goal:")
    stamina_level = get_user_input("\nFitness level (1-5 where 1 is beginner, 5 is advanced): ", int, options=["1","2","3","4","5"])
    target_days = get_user_input("How many days per week can you workout? (1-7): ", int, options=[str(i) for i in range(1,8)])
    # Health Considerations
    print("\nHealth Considerations:")
    conditions = get_multiselect(
        "Select any health conditions (leave blank if none):",
        ["diabetes", "hypertension", "knee_surgery", "back_pain", "heart_condition"]
    )
    restrictions = get_multiselect(
        "Select any movement restrictions (leave blank if none):",
        ["knee_pain", "shoulder_pain", "back_pain", "wrist_issues"]
    )
    # Equipment
    equipment = get_multiselect(
        "\nSelect equipment you have available:",
        [e.value for e in Equipment]
    )
    # Diet Preferences
    diet_type = select_from_enum(DietType, "\nSelect your diet type:")
    allergies = get_user_input(
        "List any food allergies (comma separated, leave blank if none): "
    ).split(',') if input("Any allergies? (y/n): ").lower() == 'y' else []
    # Preferences
    cuisine_prefs = get_multiselect(
        "\nSelect preferred cuisines:",
        ["indian", "western", "mediterranean", "asian", "no_preference"]
    )
    budget = get_user_input(
        "Food budget (low/medium/high): ",
        options=["low", "medium", "high"],
        default="medium"
    )
    profile = UserProfile(
        user_id=user_id,
        name=name,
        age=age,
        height_cm=height_cm,
        weight_kg=weight_kg,
        gender=gender,
        goal=goal,
        stamina_level=stamina_level,
        health_conditions=conditions,
        restrictions=restrictions,
        equipment=equipment,
        diet_type=diet_type,
        preferences=[],
        target_days_per_week=target_days,
        allergies=[a.strip() for a in allergies if a.strip()],
        cuisine_preferences=cuisine_prefs,
        budget=budget
    )
    print("\n" + "="*40)
    print(" PROFILE CREATED SUCCESSFULLY! ".center(40, "="))
    print("="*40 + "\n")
    conn.close()
    return profile

# ================== Fitness Planner Logic ==================
class WorkoutExercise:
    def __init__(self, name, sets, reps, muscles, description=""):
        self.name = name
        self.sets = sets
        self.reps = reps
        self.muscles = muscles
        self.description = description

class FitnessPlanner:
    def __init__(self, profile: UserProfile):
        self.profile = profile

    def generate_workout(self) -> List[WorkoutExercise]:
        # Simple logic based on goal and equipment
        goal = self.profile.goal
        equipment = set(self.profile.equipment)
        restrictions = set(self.profile.restrictions)
        workout = []
        # Example exercise pool
        pool = [
            ("Push-ups", 3, 12, ["chest", "triceps"], "bodyweight", ["shoulder_pain"]),
            ("Squats", 3, 15, ["legs", "glutes"], "bodyweight", ["knee_pain"]),
            ("Dumbbell Rows", 3, 10, ["back", "biceps"], "Dumbbells", []),
            ("Plank", 3, 30, ["core"], "bodyweight", []),
            ("Resistance Band Pulls", 3, 15, ["back", "shoulders"], "Resistance Bands", []),
            ("Yoga Stretch", 2, 60, ["flexibility"], "Yoga Mat", []),
            ("Treadmill Walk", 2, 20, ["cardio"], "Treadmill", ["knee_pain"]),
        ]
        for name, sets, reps, muscles, req_equipment, avoid in pool:
            if req_equipment != "bodyweight" and req_equipment not in equipment:
                continue
            if any(r in restrictions for r in avoid):
                continue
            if goal == Goal.FLEXIBILITY and "flexibility" not in muscles:
                continue
            if goal == Goal.ENDURANCE and "cardio" not in muscles:
                continue
            workout.append(WorkoutExercise(name, sets, reps, muscles, ""))
        if not workout:
            workout.append(WorkoutExercise("Walking", 3, 20, ["cardio"], "Fallback: Just walk!"))
        return workout

# ================== Meal Planner Logic ==================
class Meal:
    def __init__(self, name, calories, protein, carbs, fats, ingredients, prep_time, cost):
        self.name = name
        self.calories = calories
        self.macros = {'protein_g': protein, 'carbs_g': carbs, 'fats_g': fats}
        self.ingredients = ingredients
        self.prep_time = prep_time
        self.cost = cost

class MealPlanner:
    def __init__(self, profile: UserProfile):
        self.profile = profile

    def generate_meal_plan(self, days: int) -> Dict[str, List[Meal]]:
        # Simple meal plan logic
        meals = {}
        base_meals = [
            Meal("Oats & Fruit", 350, 10, 60, 5, ["oats", "banana", "milk"], 10, "low"),
            Meal("Grilled Chicken & Veggies", 500, 40, 30, 15, ["chicken", "broccoli", "carrot"], 20, "medium"),
            Meal("Paneer Salad", 400, 20, 25, 18, ["paneer", "lettuce", "tomato"], 15, "medium"),
            Meal("Lentil Soup", 300, 18, 40, 4, ["lentils", "spinach", "onion"], 25, "low"),
            Meal("Egg Bhurji", 350, 22, 5, 25, ["egg", "onion", "tomato"], 10, "low"),
        ]
        # Filter by diet type and allergies
        allowed = []
        for meal in base_meals:
            if self.profile.diet_type == DietType.VEG and any(i in meal.ingredients for i in ["chicken", "egg"]):
                continue
            if self.profile.diet_type == DietType.VEGAN and ("paneer" in meal.ingredients or "egg" in meal.ingredients or "milk" in meal.ingredients):
                continue
            if any(a.lower() in (i.lower() for i in meal.ingredients) for a in self.profile.allergies):
                continue
            allowed.append(meal)
        for d in range(days):
            day = f"Day {d+1}"
            meals[day] = allowed[:3] if len(allowed) >= 3 else allowed
        return meals

# ================== Progress Logger ==================
class ProgressLogger:
    def __init__(self, profile: UserProfile):
        self.profile = profile
        self.conn = get_db_connection()

    def log_progress(self, weight, body_fat=None):
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT INTO user_measurement_history (user_id, date_recorded, weight_kg, body_fat) VALUES (?, ?, ?, ?)",
            (self.profile.user_id, datetime.now().date(), weight, body_fat)
        )
        self.conn.commit()
        print("Progress logged!")

    def get_progress(self):
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT date_recorded, weight_kg, body_fat FROM user_measurement_history WHERE user_id = ? ORDER BY date_recorded ASC",
            (self.profile.user_id,)
        )
        return cursor.fetchall()

# ================== Gamification ==================
class FitnessGame:
    def __init__(self, profile: UserProfile):
        self.profile = profile
        self.points = 0
        self.level = 1
        self.badges = []

    def award_points(self, action):
        if action == "workout_completed":
            self.points += 10
        elif action == "meal_logged":
            self.points += 5
        elif action == "progress_logged":
            self.points += 3
        if self.points >= self.level * 20:
            self.level += 1
            self.badges.append(f"Level {self.level} Achieved!")
            return f"🎉 Congratulations! You've reached Level {self.level}!"
        return None

# ================== Main Menu ==================
def main_menu(profile: UserProfile):
    planner = FitnessPlanner(profile)
    meal_planner = MealPlanner(profile)
    progress_logger = ProgressLogger(profile)
    game = FitnessGame(profile)
    while True:
        print("\n" + "="*40)
        print(" MAIN MENU ".center(40, "="))
        print("="*40)
        print(f"\nHello {profile.name}! (Level {game.level} | Points: {game.points})")
        options = [
            "Generate Workout",
            "Generate Meal Plan",
            "Log Progress",
            "View Achievements",
            "View Progress History",
            "Exit"
        ]
        for i, option in enumerate(options, 1):
            print(f"{i}. {option}")
        choice = get_user_input("\nSelect an option: ", int, options=[str(i) for i in range(1, len(options)+1)])
        if choice == 1:  # Generate Workout
            workout = planner.generate_workout()
            print("\n" + "💪 YOUR WORKOUT PLAN ".center(40, "-"))
            for ex in workout:
                print(f"\n{ex.name.upper()}")
                print(f"Sets: {ex.sets} | Reps: {ex.reps}")
                print(f"Muscles: {', '.join(ex.muscles)}")
                if ex.description:
                    print(f"Notes: {ex.description}")
            if input("\nDid you complete this workout? (y/n): ").lower() == 'y':
                game.award_points("workout_completed")
                print("✅ Workout logged successfully!")
                if achievement := game.award_points("workout_completed"):
                    print(achievement)
        elif choice == 2:  # Generate Meal Plan
            days = get_user_input("How many days to plan? (1-7): ", int, options=[str(i) for i in range(1,8)])
            meals = meal_planner.generate_meal_plan(days)
            print("\n" + "🍽️ YOUR MEAL PLAN ".center(40, "-"))
            for day, day_meals in meals.items():
                print(f"\n{day.upper()}")
                for meal in day_meals:
                    print(f"\n{meal.name}")
                    print(f"Calories: {meal.calories}")
                    print(f"Protein: {meal.macros['protein_g']}g | Carbs: {meal.macros['carbs_g']}g | Fats: {meal.macros['fats_g']}g")
                    print(f"Ingredients: {', '.join(meal.ingredients)}")
                    print(f"Prep Time: {meal.prep_time} mins | Cost: {meal.cost}")
            game.award_points("meal_logged")
        elif choice == 3:  # Log Progress
            weight = get_user_input("Current weight (kg): ", float)
            body_fat = get_user_input("Body fat % (optional, press enter to skip): ", float, default=None)
            progress_logger.log_progress(weight, body_fat)
            game.award_points("progress_logged")
        elif choice == 4:  # View Achievements
            print("\n" + "🏆 YOUR ACHIEVEMENTS ".center(40, "-"))
            if game.badges:
                for badge in game.badges:
                    print(f"- {badge}")
            else:
                print("No achievements yet. Keep working hard!")
            print(f"\nTotal Points: {game.points} | Level: {game.level}")
        elif choice == 5:  # View Progress History
            print("\n" + "📈 PROGRESS HISTORY ".center(40, "-"))
            progress = progress_logger.get_progress()
            if not progress:
                print("No progress data yet.")
                continue
            print(f"\n{'Date':<15} {'Weight':<10} {'Body Fat':<10}")
            print("-"*35)
            for entry in progress:
                print(f"{entry['date_recorded']:<15} {entry['weight_kg']:<10} {entry['body_fat'] or '-':<10}")
            # Show simple progress chart
            weights = [entry['weight_kg'] for entry in progress]
            if len(weights) > 1:
                print("\nWeight Trend:")
                min_w, max_w = min(weights), max(weights)
                scale = 20 / (max_w - min_w) if max_w != min_w else 1
                for entry in progress:
                    pos = int((entry['weight_kg'] - min_w) * scale)
                    print(f"{entry['date_recorded'][5:]}: {' '*(pos-1)}* ({entry['weight_kg']}kg)")
        elif choice == 6:  # Exit
            print("\nThank you for using Fitness Planner!")
            sys.exit()

if __name__ == "__main__":
    print("\n" + "="*50)
    print(" FITNESS PLANNER ".center(50, "="))
    print("="*50 + "\n")
    user_profile = create_user_profile()
    main_menu(user_profile) 