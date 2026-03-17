
# 🏠 California Housing Price Prediction using XGBoost

## 📌 Overview

This project focuses on predicting house prices using the  **California Housing Dataset** .
The final model uses  **XGBoost with Early Stopping** , achieving strong performance on tabular data.

---

## 🚀 Key Highlights

* Implemented multiple models: Linear Regression, ANN, Random Forest, Gradient Boosting
* Applied **feature scaling** and preprocessing
* Used **cross-validation** for model evaluation
* Implemented **hyperparameter tuning**
* Applied **Early Stopping in XGBoost** for optimal performance
* Built **ensemble models** and compared results

---

## 🏆 Final Model Performance

| Model                    | RMSE               |
| ------------------------ | ------------------ |
| XGBoost (Early Stopping) | **0.4403**✅ |

👉 Final Model: **XGBoost with Early Stopping**
👉 Error ≈ **$44,030**

---

## 🧠 Why XGBoost?

* Handles tabular data efficiently
* Captures nonlinear relationships
* Built-in regularization
* Works well without heavy preprocessing

---

## 🛠️ Tech Stack

* Python
* Scikit-learn
* XGBoost
* TensorFlow / Keras
* NumPy, Pandas
* Matplotlib

---

## 📂 Project Structure

```
├── data/
├── notebooks/
├── models/
├── README.md
└── main.py
```

---

## ⚙️ Installation

```bash
pip install numpy pandas scikit-learn xgboost tensorflow matplotlib
```

---

## ▶️ Usage

```python
from xgboost import XGBRegressor

model = XGBRegressor()
model.fit(X_train, y_train)

prediction = model.predict(X_test)
```

---

## 🔮 Predict on New Data

```python
import numpy as np

new_data = np.array([[8.0, 20, 6.0, 1.0, 1000, 3.0, 34.0, -118.0]])

prediction = model.predict(new_data)
print("Predicted Price:", prediction[0] * 100000)
```

---

## 📊 Features Used

* MedInc (Median Income)
* HouseAge
* AveRooms
* AveBedrms
* Population
* AveOccup
* Latitude
* Longitude

---

## 📈 Learning Outcomes

* Regression modeling techniques
* Model comparison and evaluation
* Hyperparameter tuning
* Ensemble learning
* Real-world ML pipeline

---

## 🔥 Future Improvements

* Hyperparameter tuning using Optuna
* Model deployment (FastAPI / Flask)
* SHAP explainability
* Stacking ensemble models

---

## 🤝 Contributing

Feel free to fork this repo and improve the model or add new features.

---

## ⭐ Acknowledgment

Dataset provided by **Scikit-learn (California Housing Dataset)**

---

## 📌 Author

**Hardik Yerne**

---
