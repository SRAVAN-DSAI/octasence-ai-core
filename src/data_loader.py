# src/data_loader.py
import pandas as pd
import kagglehub
import glob
import os
from sklearn.preprocessing import LabelEncoder

class DataLoader:
    def __init__(self, dataset_path=None):
        self.dataset_path = dataset_path
        self.encoders = {}

    def load_data(self):
        """Downloads/Loads dataset from Kaggle or local path."""
        if not self.dataset_path:
            print("⬇️ Downloading latest dataset from Kaggle...")
            self.dataset_path = kagglehub.dataset_download("ziya07/bim-ai-integrated-dataset")
        
        csv_files = glob.glob(f"{self.dataset_path}/*.csv")
        if not csv_files:
            raise FileNotFoundError("No CSV files found in the dataset path.")
        
        print(f"✅ Data loaded from {csv_files[0]}")
        return pd.read_csv(csv_files[0])

    def preprocess(self, df):
        """Cleans and encodes the data for ML."""
        df = df.copy()
        
        # Drop ID columns (irrelevant for patterns)
        if 'Project_ID' in df.columns:
            df = df.drop(columns=['Project_ID'])

        # Feature Engineering: Duration
        if 'Start_Date' in df.columns and 'End_Date' in df.columns:
            df['Start_Date'] = pd.to_datetime(df['Start_Date'])
            df['End_Date'] = pd.to_datetime(df['End_Date'])
            df['Planned_Duration'] = (df['End_Date'] - df['Start_Date']).dt.days
            df = df.drop(columns=['Start_Date', 'End_Date'])

        # Encode Categorical Variables
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            self.encoders[col] = le  # Save encoder for inference later
            
        return df