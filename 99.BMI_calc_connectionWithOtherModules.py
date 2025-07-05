# Import helper for 2.1BMI_calc.py - directly import its contents
import os
import sys
import importlib.util

# Get the full path to the 2.1BMI_calc.py file
file_path = os.path.join(os.path.dirname(__file__), '2.1BMI_calc.py')

# Use importlib to load the module from file path
spec = importlib.util.spec_from_file_location('bmi_calc_module', file_path)
module = importlib.util.module_from_spec(spec)
sys.modules['bmi_calc_module'] = module
spec.loader.exec_module(module)

# Copy all attributes to make them available when importing this module
for attr_name in dir(module):
    if not attr_name.startswith('__'):
        globals()[attr_name] = getattr(module, attr_name)

# Explicitly expose common functions for clarity
load_current_user = module.load_current_user if hasattr(module, 'load_current_user') else None
get_user_data = module.get_user_data if hasattr(module, 'get_user_data') else None
get_body_composition_summary = module.get_body_composition_summary if hasattr(module, 'get_body_composition_summary') else None
