import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from collections import defaultdict

print("Starting application...")

# 📦 Categorized food data: {Category: {Food: (calories per 100g, standard weight)}}
categorized_food_data = {
    "Breakfast": {
        "Idli": (110, 50), "Dosa": (166, 100), "Masala Dosa": (200, 150), "Vada": (290, 70),
        "Medu Vada": (330, 80), "Uttapam": (157, 120), "Poha": (143, 150), "Upma": (132, 150),
        "Paratha": (300, 80), "Thepla": (265, 60), "Moong Dal Chilla": (128, 100),
        "Pongal": (143, 150), "Chai": (90, 150), "Filter Coffee": (80, 150)
    },
    "Lunch/Dinner": {
        "Roti": (264, 40), "Chapati": (260, 35), "Rice": (130, 150), "Dal": (104, 150),
        "Vegetable Curry": (90, 100), "Aloo Curry": (140, 100), "Chicken Curry": (190, 150),
        "Fish Curry": (180, 150), "Mutton Curry": (250, 150), "Paneer": (265, 50),
        "Curd": (60, 100), "Rajma": (110, 150), "Chole": (140, 150), "Bhindi Fry": (104, 100),
        "Baingan Bharta": (102, 100), "Biryani": (245, 200)
    },
    "Snacks/Drinks": {
        "Samosa": (308, 100), "Pakora": (280, 50), "Dhokla": (160, 100), "Khandvi": (152, 100),
        "Pani Puri": (45, 1), "Pav Bhaji": (250, 200), "Chole Bhature": (450, 350),
        "Sabudana Khichdi": (170, 150), "Misal Pav": (232, 200),
        "Gulab Jamun": (175, 50), "Jalebi": (150, 50), "Ladoo": (180, 40),
        "Kheer": (140, 150), "Rasgulla": (125, 60), "Banana": (89, 120), "Apple": (52, 150),
        "Milk": (62, 200), "Lassi": (160, 250), "Chaas": (35, 250)
    }
}

print("Food data loaded...")

# Flatten food data for easy calorie lookup
flat_food_data = {
    food: data for category in categorized_food_data.values() for food, data in category.items()
}

class CalorieCalculator:
    def __init__(self, root):
        print("Initializing calculator...")
        self.root = root
        self.root.title("Calorie Calculator")
        self.root.geometry("800x600")
        
        # Make sure the window is on top
        self.root.lift()
        self.root.attributes('-topmost', True)
        self.root.after_idle(self.root.attributes, '-topmost', False)
        
        # Cart to store items
        self.cart = defaultdict(list)
        
        # Create main frame
        self.main_frame = ttk.Frame(root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        root.grid_rowconfigure(0, weight=1)
        root.grid_columnconfigure(0, weight=1)
        
        # Create widgets
        self.create_widgets()
        
        # Update food options initially
        self.update_food_options()
        print("Calculator initialized!")
        
        # Show popup message at startup
        self.root.after(100, lambda: messagebox.showinfo("Welcome!", "Calorie Calculator is ready!"))
        
    def create_widgets(self):
        print("Creating widgets...")
        # Meal Type
        ttk.Label(self.main_frame, text="Meal Type:").grid(row=0, column=0, sticky=tk.W)
        self.category_var = tk.StringVar()
        self.category_combo = ttk.Combobox(self.main_frame, textvariable=self.category_var)
        self.category_combo['values'] = list(categorized_food_data.keys())
        self.category_combo.grid(row=0, column=1, sticky=tk.W)
        self.category_combo.bind('<<ComboboxSelected>>', lambda e: self.update_food_options())
        
        # Food Selection
        ttk.Label(self.main_frame, text="Select Food:").grid(row=1, column=0, sticky=tk.W)
        self.food_var = tk.StringVar()
        self.food_combo = ttk.Combobox(self.main_frame, textvariable=self.food_var)
        self.food_combo.grid(row=1, column=1, sticky=tk.W)
        
        # Quantity
        ttk.Label(self.main_frame, text="Quantity:").grid(row=2, column=0, sticky=tk.W)
        self.quantity_var = tk.StringVar(value="1")
        self.quantity_entry = ttk.Entry(self.main_frame, textvariable=self.quantity_var)
        self.quantity_entry.grid(row=2, column=1, sticky=tk.W)
        
        # Custom Weight
        ttk.Label(self.main_frame, text="Custom Weight (g/item):").grid(row=3, column=0, sticky=tk.W)
        self.weight_var = tk.StringVar(value="0")
        self.weight_entry = ttk.Entry(self.main_frame, textvariable=self.weight_var)
        self.weight_entry.grid(row=3, column=1, sticky=tk.W)
        
        # Buttons
        button_frame = ttk.Frame(self.main_frame)
        button_frame.grid(row=4, column=0, columnspan=2, pady=10)
        
        ttk.Button(button_frame, text="Add Item", command=self.add_item).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Remove Item", command=self.remove_item).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Show Chart", command=self.show_chart).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Clear Meal", command=self.clear_cart).pack(side=tk.LEFT, padx=5)
        
        # Summary Text
        self.summary_text = tk.Text(self.main_frame, height=10, width=50)
        self.summary_text.grid(row=5, column=0, columnspan=2, pady=10)
        print("Widgets created!")
        
    def update_food_options(self):
        selected_category = self.category_var.get()
        if selected_category:
            self.food_combo['values'] = list(categorized_food_data[selected_category].keys())
            self.food_combo.set('')
            
    def add_item(self):
        food = self.food_var.get()
        if not food:
            messagebox.showwarning("Warning", "Please select a food item")
            return
            
        try:
            quantity = int(self.quantity_var.get())
            custom_weight = float(self.weight_var.get())
        except ValueError:
            messagebox.showwarning("Warning", "Please enter valid numbers for quantity and weight")
            return
            
        if quantity <= 0:
            messagebox.showwarning("Warning", "Quantity must be greater than 0")
            return
            
        calories_per_100g, default_weight = flat_food_data[food]
        weight = custom_weight if custom_weight > 0 else default_weight
        total_weight = weight * quantity
        total_calories = (calories_per_100g / 100) * total_weight
        
        self.cart[food].append({
            "quantity": quantity,
            "weight_per_item": weight,
            "total_weight": total_weight,
            "calories": total_calories
        })
        self.show_summary()
        
    def remove_item(self):
        food = self.food_var.get()
        if food in self.cart:
            self.cart[food].pop()
            if not self.cart[food]:
                del self.cart[food]
        self.show_summary()
        
    def show_chart(self):
        if not self.cart:
            messagebox.showinfo("Info", "Your meal is empty")
            return
            
        # Create a new window for the chart
        chart_window = tk.Toplevel(self.root)
        chart_window.title("Meal Calorie Breakdown")
        
        # Create figure and plot
        fig, ax = plt.subplots(figsize=(6, 6))
        labels = []
        values = []
        
        for item, entries in self.cart.items():
            total_cals = sum(e['calories'] for e in entries)
            labels.append(item)
            values.append(total_cals)
            
        ax.pie(values, labels=labels, autopct='%1.1f%%', startangle=140)
        ax.set_title("Meal Calorie Breakdown")
        ax.axis('equal')
        
        # Embed the chart in the window
        canvas = FigureCanvasTkAgg(fig, master=chart_window)
        canvas.draw()
        canvas.get_tk_widget().pack()
        
    def clear_cart(self):
        self.cart.clear()
        self.show_summary()
        
    def show_summary(self):
        self.summary_text.delete(1.0, tk.END)
        if not self.cart:
            self.summary_text.insert(tk.END, "🧺 Your meal is empty.")
            return
            
        self.summary_text.insert(tk.END, "🛒 Your Meal Summary:\n\n")
        total_cals = 0
        total_weight = 0
        
        for item, entries in self.cart.items():
            item_weight = sum(e['total_weight'] for e in entries)
            item_calories = sum(e['calories'] for e in entries)
            total_cals += item_calories
            total_weight += item_weight
            self.summary_text.insert(tk.END, f"{item}: {item_weight:.0f}g → {item_calories:.2f} kcal\n")
            
        self.summary_text.insert(tk.END, f"\n✅ Total Calories: {total_cals:.2f} kcal\n")
        self.summary_text.insert(tk.END, f"📏 Total Weight: {total_weight:.0f}g\n")
        self.summary_text.insert(tk.END, f"🍽️ Items in Meal: {sum(len(v) for v in self.cart.values())}")

if __name__ == "__main__":
    print("Creating main window...")
    root = tk.Tk()
    app = CalorieCalculator(root)
    print("Starting main loop...")
    root.mainloop()
    print("Application closed.") 