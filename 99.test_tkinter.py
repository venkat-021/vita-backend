import tkinter as tk
from tkinter import messagebox

def main():
    # Create the main window
    root = tk.Tk()
    root.title("Tkinter Test")
    root.geometry("300x200")
    
    # Add a label
    label = tk.Label(root, text="If you can see this window,\ntkinter is working correctly!")
    label.pack(pady=20)
    
    # Add a button
    def show_message():
        messagebox.showinfo("Test", "Button clicked! Tkinter is working!")
    
    button = tk.Button(root, text="Click Me!", command=show_message)
    button.pack(pady=20)
    
    # Make sure window is on top
    root.lift()
    root.attributes('-topmost', True)
    root.after_idle(root.attributes, '-topmost', False)
    
    # Start the main loop
    root.mainloop()

if __name__ == "__main__":
    main() 