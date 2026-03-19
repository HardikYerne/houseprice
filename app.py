import pickle
from flask import Flask, request, jsonify, render_template
import numpy as np

app = Flask(__name__)

# Load model
xgbmodel = pickle.load(open('xgbmodel.pkl', 'rb'))

# OPTIONAL: Load scaler if used during training
# scaler = pickle.load(open('scaler.pkl', 'rb'))

feature_names = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms',
                 'Population', 'AveOccup', 'Latitude', 'Longitude']


# Home page
@app.route('/')
def home():
    return render_template('home.html', feature_names=feature_names)


# UI Prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        values = [float(request.form.get(f)) for f in feature_names]
        final_input = np.array([values])

        # If scaler used:
        # final_input = scaler.transform(final_input)

        prediction = xgbmodel.predict(final_input)[0] * 100000

        return render_template(
            'home.html',
            feature_names=feature_names,
            prediction_text=f"Predicted Price: ${prediction:,.2f}"
        )
    except Exception as e:
        return str(e)


# API Prediction (JSON)
@app.route('/predict_api', methods=['POST'])
def predict_api():
    try:
        data = request.get_json()

        values = [float(data[f]) for f in feature_names]
        final_input = np.array([values])

        # If scaler used:
        # final_input = scaler.transform(final_input)

        prediction = xgbmodel.predict(final_input)[0] * 100000

        return jsonify({
            "prediction": prediction,
            "status": "success"
        })

    except Exception as e:
        return jsonify({
            "error": str(e),
            "status": "failed"
        })


if __name__ == "__main__":
    app.run(debug=True)