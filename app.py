from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pickle
import numpy as np
import pandas as pd
import os
from datetime import datetime

# Initialize the Flask app and allow cross-origin requests
app = Flask(__name__)
CORS(app)

# The path to our trained model file
MODEL_PATH = 'model_package.pkl'

# Helper function to load the pickle file safely
def load_model():
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, 'rb') as f:
            return pickle.load(f)
    return None

# Load the model once when the server starts
try:
    pkg = load_model()
    if pkg:
        model = pkg['model']
        scaler = pkg['scaler']
        # We need these to make sure we feed features in the exact same order as training
        feature_names = pkg.get('feature_names', ['Month', 'Day', 'DayOfWeek', 'Is_Weekend', 'Hour'])
        model_name = pkg.get('model_name', 'Unknown')
        feature_importance = pkg.get('feature_importance', {})
        print(f"Server started. Loaded model: {model_name}")
    else:
        model = None
except Exception as e:
    print(f"Error loading the model file: {e}")
    model = None

# Serve the frontend HTML page
@app.route('/')
def home():
    return render_template('index.html')

# The main API endpoint for making predictions
@app.route('/predict', methods=['POST'])
def predict():
    if model is None: 
        return jsonify({'error': 'Model is not loaded properly on the server.'}), 500
        
    try:
        # Get the JSON data sent from the frontend
        data = request.json
        date_str = data.get('datetime', '')
        
        if not date_str:
            return jsonify({'error': 'No datetime provided.'}), 400
            
        # Parse the ISO datetime string (e.g., "2025-05-16T14:00")
        dt = datetime.fromisoformat(date_str)
        
        # Extract features manually just like we did in train.py
        month = dt.month
        day = dt.day
        dayofweek = dt.weekday()
        # 5 is Saturday, 6 is Sunday
        is_weekend = 1 if dayofweek in [5, 6] else 0
        hour = dt.hour
        
        # Create a single-row dataframe for the model input
        input_df = pd.DataFrame([{
            'Month': month,
            'Day': day,
            'DayOfWeek': dayofweek,
            'Is_Weekend': is_weekend,
            'Hour': hour
        }])
        
        # Reorder columns just to be safe
        input_df = input_df[feature_names]
        
        # Scale the inputs using the scaler we fit during training
        input_scaled = scaler.transform(input_df)
        
        # Get the prediction (it returns an array, so we take the first item)
        prediction = model.predict(input_scaled)[0]
        
        # Sort the feature importances so the frontend can display them nicely
        sorted_importance = {}
        if feature_importance:
            sorted_importance = {k: round(v, 4) for k, v in sorted(feature_importance.items(), key=lambda item: item[1], reverse=True)}

        # Make sure we don't return negative energy production (doesn't make sense physically)
        prediction_val = max(0.0, float(prediction))
        
        # --- Requirement 7 from the project brief: Practical Implications ---
        # Generate some strategic advice based on the predicted load
        if prediction_val > 40000:
            implication = "High Grid Load Expected! Keep Natural Gas and Coal power plant reserves ready to cover the energy deficit during peak hours. Renewable sources (Solar/Wind) alone might be insufficient."
            status_level = "Critical Load"
            color = "#ef4444" # red
        elif prediction_val > 32000:
            implication = "Normal Grid Load. The grid appears balanced. You can optimize costs by maximizing Hydroelectric (Dam) and Wind energy production."
            status_level = "Balanced Load"
            color = "#eab308" # yellow
        else:
            implication = "Low Grid Load. Due to low demand, maintenance work can be scheduled for base-load power plants (e.g., Nuclear or Thermal). If there is excess production, storage systems (Pumped-storage hydroelectricity) can be activated."
            status_level = "Low Load"
            color = "#22c55e" # green
        
        # Send everything back as a JSON response
        return jsonify({
            'prediction': round(prediction_val, 2),
            'prediction_unit': 'MWh',
            'active_model': model_name,
            'feature_importance': sorted_importance,
            'implication': implication,
            'status_level': status_level,
            'status_color': color,
            'status': 'success'
        })
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 400

if __name__ == '__main__':
    # Run the app locally on port 8000
    app.run(debug=True, port=8000)
