from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the saved model
model = joblib.load('random_forest_model.pkl')

# Define acceptable ranges for inputs
RANGES = {
    'airTemp': (290.0, 310.0),  # Kelvin
    'processTemp': (300.0, 320.0),  # Kelvin
    'rotSpeed': (1100.0, 3000.0),  # rpm
    'torque': (2.0, 90.0),  # Nm
    'toolWear': (0.0, 300.0),  # minutes
}

# Route for the home page
@app.route('/')
def index():
    return render_template('index.html')

# Route for predictions
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    errors = {}

    # Validate inputs
    for key, (min_val, max_val) in RANGES.items():
        if key not in data:
            errors[key] = 'This field is required.'
        elif not (min_val <= data[key] <= max_val):
            errors[key] = f'Value must be between {min_val} and {max_val}.'

    if errors:
        return jsonify({'error': errors}), 400

    # Prepare input and predict
    input_data = np.array([[data['airTemp'], data['processTemp'], data['rotSpeed'], data['torque'], data['toolWear']]])
    prediction = model.predict(input_data)[0]
    return jsonify({'prediction': int(prediction)})

if __name__ == '__main__':
    app.run(debug=True)
