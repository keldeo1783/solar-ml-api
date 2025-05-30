# Temp comment to force Git update

from flask import Flask, request, jsonify, render_template
import joblib
import requests
from datetime import datetime

class DummyModel:
    def predict(self, X):
        import numpy as np
        return np.random.uniform(10, 20, size=(len(X),))

class DummyEncoder:
    def transform(self, labels):
        return [0] * len(labels)

app = Flask(__name__)

# Load dummy model and encoders (for demo purposes)
model = joblib.load("solar_efficiency_model.pkl")
weather_encoder = joblib.load("weather_encoder.pkl")
demand_encoder = joblib.load("demand_encoder.pkl")

API_KEY = '312fb7ad1f564f0db7b90923252705'  
LAT, LON = 12.9716, 77.5946  # Bangalore, India

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    print("✅ /predict POST received")
    data = request.json
    try:
        user_date = data['date']
        forecast = []

        url = f"http://api.weatherapi.com/v1/forecast.json?key={API_KEY}&q={LAT},{LON}&days=3&aqi=no&alerts=no"
        print(f"Request body: {data}")
        resp = requests.get(url, timeout=5)
        weather_data = resp.json()

        if 'forecast' not in weather_data or 'forecastday' not in weather_data['forecast']:
            return jsonify({"error": "Weather data unavailable."}), 500

        demand = "medium"
        demand_encoded = demand_encoder.transform([demand])[0]
        selected_date = datetime.strptime(user_date, "%Y-%m-%d").date()

        for day in weather_data['forecast']['forecastday']:
            day_date = datetime.strptime(day['date'], "%Y-%m-%d").date()
            if day_date == selected_date:
                for hour_data in day['hour']:
                    hour_dt = datetime.strptime(hour_data['time'], "%Y-%m-%d %H:%M")
                    hour = hour_dt.hour
                    if 6 <= hour <= 19:
                        temp = hour_data['temp_c']
                        cloud_pct = hour_data['cloud']
                        condition_text = hour_data['condition']['text'].lower()
                        irradiance = 1000 * (1 - cloud_pct / 100)

                        if 'clear' in condition_text:
                            weather_label = 'clear'
                        elif 'cloud' in condition_text:
                            weather_label = 'cloudy'
                        elif 'rain' in condition_text or 'drizzle' in condition_text:
                            weather_label = 'rainy'
                        else:
                            weather_label = 'cloudy'

                        try:
                            weather_encoded = weather_encoder.transform([weather_label])[0]
                        except ValueError:
                            weather_encoded = weather_encoder.transform(['cloudy'])[0]

                        import pandas as pd
                        features = pd.DataFrame([[
                        irradiance, temp, weather_encoded, demand_encoded
                        ]], columns=['solar_irradiance_wm2', 'ambient_temperature_c', 'weather_condition', 'user_demand'])

                        efficiency = model.predict(features)[0]

                        forecast.append({
                            "hour": hour,
                            "efficiency": round(efficiency, 2),
                            "irradiance": round(irradiance, 2),
                            "weather": weather_label
                        })

        if not forecast:
            return jsonify({"error": "No forecast available for selected date."}), 404

        sorted_hours = sorted(forecast, key=lambda f: f["efficiency"], reverse=True)
        best_hours = [f["hour"] for f in sorted_hours[:3]]

        return jsonify({
            "forecast": forecast,
            "best_hours": best_hours
        })

    except Exception as e:
        print("❌ Exception during prediction:", e)
        return jsonify({"error": str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True)
