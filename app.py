from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
from data_manager import DataManager
from model_manager import ModelManager
import os

app = Flask(__name__)
CORS(app)

# Initialize managers
data_dir = os.path.dirname(os.path.abspath(__file__))
dm = DataManager(data_dir=data_dir)
mm = ModelManager(model_dir=os.path.join(data_dir, 'models'))

# Global data storage to avoid re-loading on every request
cached_data = None

def get_data():
    global cached_data
    if cached_data is None:
        cached_data = dm.load_data()
    return cached_data

@app.route('/')
def index():
    return render_template('dashboard.html')

@app.route('/predict_page')
def predict_page():
    return render_template('predict.html')

@app.route('/forecast_page')
def forecast_page():
    return render_template('forecast.html')

@app.route('/api/stats')
def get_stats():
    get_data()
    return jsonify(dm.get_kpis())

@app.route('/api/trends')
def get_trends():
    get_data()
    return jsonify(dm.get_sales_trend())

@app.route('/api/predict', methods=['POST'])
def predict():
    feature_dict = request.json
    prediction = mm.predict_sales(feature_dict)
    return jsonify({"prediction": prediction})

@app.route('/api/forecast')
def get_forecast():
    df = get_data()
    forecast_data = mm.generate_forecast(df)
    return jsonify(forecast_data)

@app.route('/api/train')
def train_model():
    df = get_data()
    mm.train_rf_model(df)
    return jsonify({"status": "Model trained and saved."})

@app.route('/api/model_status')
def model_status():
    exists = os.path.exists(mm.rf_model_path)
    return jsonify({
        "trained": exists,
        "last_trained": os.path.getmtime(mm.rf_model_path) if exists else None
    })

if __name__ == '__main__':
    # Ensure data is loaded and model is ready before starting
    print("Initializing application...")
    get_data()
    if not mm.load_rf_model():
        print("Model not found. Please run training.")
    
    app.run(debug=True, port=5000)
