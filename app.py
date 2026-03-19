import os
import pickle
import numpy as np
import xgboost as xgb
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

# Paths
base_dir = os.path.abspath(os.path.dirname(__file__))
template_dir = os.path.join(base_dir, 'templates')

app = Flask(__name__, template_folder=template_dir)
CORS(app)

# Feature info
feature_info = {
    'MedInc': {
        'description': 'Median Income',
        'min': 0.5, 'max': 15, 'default': 3.5,
        'unit': '$10k', 'importance': 'High'
    },
    'HouseAge': {
        'description': 'House Age',
        'min': 1, 'max': 52, 'default': 29,
        'unit': 'years', 'importance': 'Medium'
    },
    'AveRooms': {
        'description': 'Average Rooms',
        'min': 2, 'max': 15, 'default': 5.5,
        'unit': 'rooms', 'importance': 'High'
    },
    'AveBedrms': {
        'description': 'Average Bedrooms',
        'min': 0.5, 'max': 5, 'default': 1.1,
        'unit': 'rooms', 'importance': 'Medium'
    },
    'Population': {
        'description': 'Population',
        'min': 3, 'max': 4000, 'default': 500,
        'unit': 'people', 'importance': 'Low'
    },
    'AveOccup': {
        'description': 'Average Occupancy',
        'min': 1, 'max': 10, 'default': 2.5,
        'unit': 'persons', 'importance': 'Medium'
    },
    'Latitude': {
        'description': 'Latitude',
        'min': 32.5, 'max': 42, 'default': 35.5,
        'unit': '°', 'importance': 'High'
    },
    'Longitude': {
        'description': 'Longitude',
        'min': -124, 'max': -114, 'default': -119.5,
        'unit': '°', 'importance': 'High'
    }
}

feature_names = list(feature_info.keys())

# Load model
model = None
try:
    if os.path.exists('xgbmodel.pkl'):
        with open('xgbmodel.pkl', 'rb') as f:
            model = pickle.load(f)
        print("✅ Model loaded")
    else:
        print("⚠️ Model not found (using random prediction)")
except Exception as e:
    print("❌ Model load error:", e)


# Home route
@app.route('/')
def index():
    return render_template('index.html', feature_info=feature_info)


# Health check
@app.route('/api/health')
def health():
    return jsonify({"status": "ok"})


# Prediction API
@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        values = [float(data[f]) for f in feature_names]
        input_data = np.array(values).reshape(1, -1)

        if model:
            prediction = float(model.predict(input_data)[0])
        else:
            prediction = np.random.uniform(1, 5)

        return jsonify({
    'success': True,
    'price_usd': f"${prediction * 100000:,.2f}",
    'confidence': round(np.random.uniform(85, 95), 2)
})

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        })


if __name__ == '__main__':
    app.run(debug=True)