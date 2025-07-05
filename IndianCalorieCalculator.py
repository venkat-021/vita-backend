import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from collections import defaultdict
from PIL import Image, ImageTk
import os
from BMI_calc import load_current_user, get_user_data  # <-- Add this import

# 📦 Enhanced Indian food database with regional specialties
categorized_food_data = {
    # ... existing food data ...
}

# Flatten food data for easy calorie lookup
flat_food_data = {
    food: data for category in categorized_food_data.values() for food, data in category.items()
}

class IndianCalorieCalculator:
    def __init__(self, root):
        self.root = root
        self.user_id = self._load_or_prompt_user_id()
        self.user_data = self._load_or_prompt_user_data(self.user_id)
        self.root.title(f"Indian Calorie Calculator - {self.user_data['username']}")
        self.root.geometry("900x700")
        self.root.configure(bg="#FFF8E1")  # Light warm background
        # ... existing style setup ...
        self.style = ttk.Style()
        self.style.configure('TFrame', background='#FFF8E1')
        self.style.configure('TLabel', background='#FFF8E1', font=('Arial', 10))
        self.style.configure('TButton', font=('Arial', 10), background='#FF9933')  # Saffron
        self.style.map('TButton', background=[('active', '#FF5722')])  # Darker orange on click
        try:
            self.root.iconbitmap('indian_flag.ico')
        except:
            pass
        self.cart = defaultdict(list)
        self.create_header()
        self.main_frame = ttk.Frame(root, padding="10")
        self.main_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.main_frame.configure(style='TFrame')
        root.grid_rowconfigure(1, weight=1)
        root.grid_columnconfigure(0, weight=1)
        self.create_widgets()
        self.update_food_options()
        self.root.after(100, lambda: messagebox.showinfo("Namaste!", f"Welcome {self.user_data['username']}!\n\nTrack calories for authentic Indian meals."))

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
        try:
            user_row = get_user_data(user_id)
            user_data = dict(user_row)
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

    def create_header(self):
        header_frame = ttk.Frame(self.root, height=80)
        header_frame.grid(row=0, column=0, sticky="ew")
        header_frame.grid_propagate(False)
        # Add title with Indian theme and user name
        title_label = tk.Label(header_frame, text=f"🇮🇳 Indian Calorie Calculator - {self.user_data['username']}", 
                             font=('Arial', 16, 'bold'), bg='#FF9933', fg='white')
        title_label.pack(fill='both', expand=True)

    # ... rest of your class code remains unchanged ... 