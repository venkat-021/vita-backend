# Import helper for 3.suggestion.py - directly import its contents
import os
import sys
import importlib.util

# Get the full path to the 3.suggestion.py file
file_path = os.path.join(os.path.dirname(__file__), '3.suggestion.py')

# Use importlib to load the module from file path
spec = importlib.util.spec_from_file_location('suggestion_module', file_path)
module = importlib.util.module_from_spec(spec)
sys.modules['suggestion_module'] = module
spec.loader.exec_module(module)

# Copy all attributes to make them available when importing this module
for attr_name in dir(module):
    if not attr_name.startswith('__'):
        globals()[attr_name] = getattr(module, attr_name)

# Explicitly expose common functions for clarity
generate_health_advice = module.generate_health_advice if hasattr(module, 'generate_health_advice') else None
print_health_advice = module.print_health_advice if hasattr(module, 'print_health_advice') else None
