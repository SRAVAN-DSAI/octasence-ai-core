# src/model_trainer.py
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.preprocessing import LabelEncoder
import joblib
import json
import numpy as np

class OctaSenceRiskAgent:
    def __init__(self):
        # XGBoost is faster and often more accurate than Random Forest
        self.model = XGBClassifier(
            n_estimators=100, 
            learning_rate=0.1, 
            max_depth=5, 
            eval_metric='mlogloss',
            use_label_encoder=False
        )
        self.metrics = {}
        self.feature_names = []

    def train(self, df, target_col='Risk Level'):
        """Trains the XGBoost agent."""
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Save feature names for the App to use later
        self.feature_names = X.columns.tolist()

        # Ensure target is integers (0, 1, 2)
        le = LabelEncoder()
        y = le.fit_transform(y)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        print(f"🧠 Training OctaSence XGBoost Agent on {len(X)} records...")
        self.model.fit(X_train, y_train)
        
        # --- Calculate Metrics ---
        y_pred = self.model.predict(X_test)
        
        self.metrics = {
            "accuracy": round(accuracy_score(y_test, y_pred), 2),
            "precision": round(precision_score(y_test, y_pred, average='weighted'), 2),
            "recall": round(recall_score(y_test, y_pred, average='weighted'), 2),
            "f1": round(f1_score(y_test, y_pred, average='weighted'), 2)
        }
        
        print(f"✅ Training Complete. Accuracy: {self.metrics['accuracy']}")
        return X_test, y_test

    def save_model(self, model_path='models/octasence_agent.pkl', metrics_path='models/metrics.json'):
        # Save the Model
        joblib.dump(self.model, model_path)
        
        # Save Metrics AND Feature Names (Crucial for App robustness)
        data_to_save = {
            "metrics": self.metrics,
            "features": self.feature_names
        }
        
        with open(metrics_path, 'w') as f:
            json.dump(data_to_save, f)
            
        print(f"💾 Model saved to {model_path}")
        print(f"📊 Metadata saved to {metrics_path}")

    def evaluate(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        print("\n--- XGBoost Agent Report ---")
        print(classification_report(y_test, y_pred))