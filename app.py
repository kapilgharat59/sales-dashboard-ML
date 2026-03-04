import os
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
import plotly
from flask import Flask, render_template, request, redirect, url_for, flash

# Import custom services
from services.preprocess import load_data, clean_data, scale_data, transform_single_row
from services.inference import load_autoencoder_model, detect_anomalies, perform_clustering, calculate_mse

app = Flask(__name__)
app.secret_key = "marketing_dashboard_pro"

# Paths
DATA_PATH = os.path.join('data', 'marketing_data.csv')
MODEL_PATH = os.path.join('model', 'autoencoder.h5')

# Persistence (Global state for the demo)
STATE = {
    'df': None,
    'model': None,
    'scaler': None,
    'threshold': None,
    'feature_columns': None
}

def bootstrap_app():
    """Load data and model on startup as per strict rules."""
    print("Bootstrapping Marketing Dashboard...")
    
    # 1. Load and Clean
    raw_df = load_data(DATA_PATH)
    if raw_df is None:
        print(f"CRITICAL ERROR: Could not find data at {DATA_PATH}")
        return
        
    clean_df = clean_data(raw_df)
    
    # 2. Scale
    scaled_data, scaler = scale_data(clean_df)
    
    # 3. Load Model
    model = load_autoencoder_model(MODEL_PATH)
    
    if model is not None:
        # 4. Run Inference
        anomalies, mse, threshold = detect_anomalies(scaled_data, model)
        
        # 5. Clustering (Segments)
        clusters = perform_clustering(scaled_data)
        
        # 6. Store in State
        clean_df['anomaly'] = anomalies
        clean_df['mse'] = mse
        clean_df['cluster'] = clusters
        
        STATE['df'] = clean_df
        STATE['model'] = model
        STATE['scaler'] = scaler
        STATE['threshold'] = threshold
        STATE['feature_columns'] = clean_df.columns.drop(['anomaly', 'mse', 'cluster']).tolist()
        print("Bootstrap complete.")
    else:
        print(f"CRITICAL ERROR: Could not load model at {MODEL_PATH}")

# Call bootstrap
bootstrap_app()

@app.route('/')
def index():
    """Stats Dashboard with filters."""
    if STATE['df'] is None:
        return render_template('error.html', message="System initializing or error loading data. Check console.")
    
    df = STATE['df']
    
    # Filters (Segment and Tenure)
    segment_filter = request.args.get('segment', 'All')
    tenure_filter = request.args.get('tenure', 'All')
    
    filtered_df = df.copy()
    if segment_filter != 'All':
        filtered_df = filtered_df[filtered_df['cluster'] == int(segment_filter)]
    if tenure_filter != 'All':
        filtered_df = filtered_df[filtered_df['TENURE'] == int(tenure_filter)]
    
    # KPI Calculations
    total_records = len(filtered_df)
    anomaly_count = int(filtered_df['anomaly'].sum())
    normal_count = total_records - anomaly_count
    anomaly_percent = round((anomaly_count / total_records * 100), 2) if total_records > 0 else 0
    
    # Visualizations
    # 1. Anomaly Distribution (Pie Chart)
    fig_pie = px.pie(names=['Normal', 'Anomaly'], 
                    values=[normal_count, anomaly_count],
                    title="Traffic Distribution",
                    color_discrete_sequence=['#4e73df', '#e74a3b'],
                    hole=0.4)
    graph_pie = json.dumps(fig_pie, cls=plotly.utils.PlotlyJSONEncoder)
    
    # 2. MSE Distribution (Histogram)
    fig_hist = px.histogram(filtered_df, x="mse", color="anomaly",
                           title="Reconstruction Error (MSE) Distribution",
                           color_discrete_map={0: '#4e73df', 1: '#e74a3b'},
                           nbins=50)
    graph_hist = json.dumps(fig_hist, cls=plotly.utils.PlotlyJSONEncoder)
    
    # 3. Top Anomalies Table
    top_anomalies = filtered_df[filtered_df['anomaly'] == 1].sort_values(by='mse', ascending=False).head(10).to_dict('records')
    
    # Select Options
    segments = sorted(df['cluster'].unique().tolist())
    tenures = sorted(df['TENURE'].unique().tolist())
    
    return render_template('dashboard.html', 
                           total_records=total_records,
                           normal_count=normal_count,
                           anomaly_count=anomaly_count,
                           anomaly_percent=anomaly_percent,
                           graph_pie=graph_pie,
                           graph_hist=graph_hist,
                           top_anomalies=top_anomalies,
                           segments=segments,
                           tenures=tenures,
                           current_segment=segment_filter,
                           current_tenure=tenure_filter)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """Future Prediction (What-If Analysis)."""
    if STATE['df'] is None:
        return redirect(url_for('index'))
    
    result = None
    if request.method == 'POST':
        # Collect form data
        form_data = {}
        for col in STATE['feature_columns']:
            form_data[col] = float(request.form.get(col, 0))
            
        # Transform and Predict
        scaled_row = transform_single_row(form_data, STATE['scaler'], STATE['feature_columns'])
        mse = calculate_mse(scaled_row, STATE['model'])[0]
        
        is_anomaly = mse > STATE['threshold']
        
        result = {
            'mse': round(float(mse), 6),
            'threshold': round(float(STATE['threshold']), 6),
            'is_anomaly': is_anomaly,
            'percent_of_threshold': round((mse / STATE['threshold'] * 100), 2)
        }
    
    return render_template('predict.html', 
                           features=STATE['feature_columns'], 
                           result=result)

if __name__ == '__main__':
    app.run(debug=True)
