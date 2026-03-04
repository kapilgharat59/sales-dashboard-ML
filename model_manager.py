import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestRegressor
from prophet import Prophet

class ModelManager:
    def __init__(self, model_dir='models'):
        self.model_dir = model_dir
        self.rf_model_path = os.path.join(self.model_dir, 'rf_model.pkl')
        self.rf_model = None
        self.mappings = {}
        self.features = []
        
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

    def train_rf_model(self, df):
        """Trains a Random Forest Regressor for instant predictions."""
        print("Training Random Forest model...")
        
        # Select features for prediction
        features = ['Store', 'DayOfWeek', 'Promo', 'StateHoliday', 'SchoolHoliday', 
                    'StoreType', 'Assortment', 'CompetitionDistance', 'Year', 'Month', 'Day']
        
        # Preprocess categorical features for sklearn
        X = df[features].copy()
        
        # Create a mapping for categorical variables
        categorical_cols = ['StateHoliday', 'StoreType', 'Assortment']
        mappings = {}
        
        for col in categorical_cols:
            # Get unique values and create mapping
            unique_vals = X[col].unique()
            mappings[col] = {val: i for i, val in enumerate(unique_vals)}
            X[col] = X[col].map(mappings[col])
            
        y = df['Sales']

        # Use a subset if data is too large for fast training in this environment
        if len(X) > 100000:
            X = X.sample(100000, random_state=42)
            y = y.loc[X.index]

        self.rf_model = RandomForestRegressor(n_estimators=50, max_depth=15, random_state=42, n_jobs=-1)
        self.rf_model.fit(X, y)
        
        # Save model and mappings as a bundle
        model_bundle = {
            'model': self.rf_model,
            'mappings': mappings,
            'features': features
        }
        joblib.dump(model_bundle, self.rf_model_path)
        print(f"Model bundle saved to {self.rf_model_path}")
        return self.rf_model

    def load_rf_model(self):
        """Loads the pre-trained Random Forest model bundle."""
        if os.path.exists(self.rf_model_path):
            bundle = joblib.load(self.rf_model_path)
            if isinstance(bundle, dict) and 'model' in bundle:
                self.rf_model = bundle['model']
                self.mappings = bundle.get('mappings', {})
                self.features = bundle.get('features', [])
            else:
                # Legacy support
                self.rf_model = bundle
                self.mappings = {}
                self.features = ['Store', 'DayOfWeek', 'Promo', 'StateHoliday', 'SchoolHoliday', 
                                'StoreType', 'Assortment', 'CompetitionDistance', 'Year', 'Month', 'Day']
            return self.rf_model
        return None

    def predict_sales(self, feature_dict):
        """Predicts sales for a given set of features."""
        if self.rf_model is None:
            self.load_rf_model()
        
        if self.rf_model:
            # Prepare input data
            input_data = {}
            for feat in self.features:
                val = feature_dict.get(feat, 0)
                # Apply mapping if it's a categorical column
                if hasattr(self, 'mappings') and feat in self.mappings:
                    mapping = self.mappings[feat]
                    # If value not in mapping, default to first or 0
                    val = mapping.get(str(val), mapping.get(val, 0))
                input_data[feat] = val
                
            input_df = pd.DataFrame([input_data])
            prediction = self.rf_model.predict(input_df)
            return float(prediction[0])
        return 0.0

    def generate_forecast(self, df):
        """Generates future sales forecast using Facebook Prophet."""
        print("Generating Prophet forecast...")
        # Prepare data for Prophet
        prophet_df = df[['Date', 'Sales']].groupby('Date').sum().reset_index()
        prophet_df.columns = ['ds', 'y']
        
        model = Prophet(interval_width=0.95)
        model.fit(prophet_df)
        
        future = model.make_future_dataframe(periods=42) # 6 weeks
        forecast = model.predict(future)
        
        # Return only a clean version of forecast
        return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(42).to_dict(orient='records')

if __name__ == "__main__":
    from data_manager import DataManager
    dm = DataManager(data_dir='d:/project/sales')
    df = dm.load_data()
    
    mm = ModelManager()
    mm.train_rf_model(df)
    print("Training complete.")
