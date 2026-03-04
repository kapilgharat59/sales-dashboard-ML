import pandas as pd
import numpy as np
import os

class DataManager:
    def __init__(self, data_dir='.'):
        self.data_dir = data_dir
        self.train_df = None
        self.store_df = None
        self.merged_df = None

    def load_data(self):
        """Loads and cleans the initial CSV files."""
        train_path = os.path.join(self.data_dir, 'train.csv')
        store_path = os.path.join(self.data_dir, 'store.csv')

        print(f"Loading data from {self.data_dir}...")
        
        # Load datasets
        self.train_df = pd.read_csv(train_path, low_memory=False)
        self.store_df = pd.read_csv(store_path)

        # Basic cleaning as per notebook
        self.train_df['Date'] = pd.to_datetime(self.train_df['Date'])
        
        # Merge datasets
        self.merged_df = pd.merge(self.train_df, self.store_df, on='Store', how='inner')
        
        # Feature Engineering (extracted from notebook)
        self.merged_df['Year'] = self.merged_df['Date'].dt.year
        self.merged_df['Month'] = self.merged_df['Date'].dt.month
        self.merged_df['Day'] = self.merged_df['Date'].dt.day
        self.merged_df['WeekOfYear'] = self.merged_df['Date'].dt.isocalendar().week.astype(int)
        
        # Fill missing values in store data (simple imputation for now)
        self.merged_df['CompetitionDistance'] = self.merged_df['CompetitionDistance'].fillna(self.merged_df['CompetitionDistance'].mean())
        self.merged_df.fillna(0, inplace=True)
        
        return self.merged_df

    def get_kpis(self):
        """Calculates key performance indicators."""
        if self.merged_df is None:
            self.load_data()
            
        kpis = {
            "total_sales": float(self.merged_df['Sales'].sum()),
            "avg_sales": float(self.merged_df['Sales'].mean()),
            "total_stores": int(self.merged_df['Store'].nunique()),
            "total_customers": int(self.merged_df['Customers'].sum()),
            "avg_customers": float(self.merged_df['Customers'].mean())
        }
        return kpis

    def get_sales_trend(self):
        """Gets sales trend for charting."""
        if self.merged_df is None:
            self.load_data()
        
        # Monthly average sales
        trend = self.merged_df.groupby('Month')['Sales'].mean().reset_index()
        return trend.to_dict(orient='records')

if __name__ == "__main__":
    # Test the data manager
    dm = DataManager(data_dir='d:/project/sales')
    df = dm.load_data()
    print("Data loaded. KPI stats:")
    print(dm.get_kpis())
