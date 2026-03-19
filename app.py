import pickle
from flask import Flask, request, jsonify, render_template
import numpy as np

app = Flask(__name__)

# Load model
xgbmodel = pickle.load(open('xgbmodel.pkl', 'rb'))

feature_names = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms',
                 'Population', 'AveOccup', 'Latitude', 'Longitude']

# Home page
@app.route('/')
def home():
    return render_template('home.html', feature_names=feature_names)

# Form prediction (from UI)
@app.route('/predict', methods=['POST'])
def predict():
    try:
        values = [float(request.form.get(f)) for f in feature_names]
        final_input = np.array([values])

        prediction = xgbmodel.predict(final_input)[0] * 100000

        return render_template(
            'home.html',
            feature_names=feature_names,
            prediction_text=f"Predicted Price: ${prediction:,.2f}"
        )
    except Exception as e:
        return str(e)

# API prediction (JSON)
@app.route('/predict_api', methods=['POST'])
def predict_api():
    try:
        data = request.get_json()

        if data is None:
            return jsonify({"error": "No JSON received"}), 400

        values = list(data.values())
        final_input = np.array([values])

        output = xgbmodel.predict(final_input)

        return jsonify(float(output[0]))

    except Exception as e:
        print("ERROR:", e)   # shows error in terminal
        return jsonify({"error": str(e)}), 500