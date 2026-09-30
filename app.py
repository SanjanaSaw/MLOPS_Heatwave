
import pandas as pd
import joblib
from flask import Flask, request, jsonify

app = Flask(__name__)
model = joblib.load("model/heatwave_model.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
        input_data = pd.DataFrame([[
            data["max_temp"], data["min_temp"], data["humidity"],
            data["wind_speed"], data["previous_temp"]
        ]], columns=["max_temp", "min_temp", "humidity", "wind_speed", "previous_temp"])
        prediction = model.predict(input_data)[0]
        return jsonify({"prediction": "Heatwave" if prediction == 1 else "No Heatwave"})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
