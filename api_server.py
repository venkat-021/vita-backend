from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '12.Fitness.py'))
import Lab_reportAnalyzer
from flask_cors import CORS
CORS(app)

app = Flask(__name__)
CORS(app)
analyzer = Lab_reportAnalyzer.EnhancedLabReportAnalyzer()

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/api/analyze_lab_report', methods=['POST'])
def analyze_lab_report():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    gender = request.form.get('gender', 'male')
    age = int(request.form.get('age', 30))
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)
    user_data = {'gender': gender, 'age': age}
    result = analyzer.analyze_file(filepath, user_data)
    os.remove(filepath)  # Clean up after analysis
    return jsonify(result)

# Example: Add more endpoints for other modules
# from 11.Diabetes_prediction import predict_diabetes
# @app.route('/api/diabetes_predict', methods=['POST'])
# def diabetes_predict():
#     ...

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True) 