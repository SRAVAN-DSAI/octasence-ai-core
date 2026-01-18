# src/model_trainer.py
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import joblib
import json

class OctaSenceRiskAgent:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.metrics = {}

    def train(self, df, target_col='Risk Level'):
        """Trains the model and calculates performance metrics."""
        X = df.drop(columns=[target_col])
        y = df[target_col]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        print("🧠 Training OctaSence Agent...")
        self.model.fit(X_train, y_train)
        
        # --- Calculate Metrics ---
        y_pred = self.model.predict(X_test)
        
        # 'weighted' average handles multi-class (Low/Med/High) better
        self.metrics = {
            "accuracy": round(accuracy_score(y_test, y_pred), 2),
            "precision": round(precision_score(y_test, y_pred, average='weighted'), 2),
            "recall": round(recall_score(y_test, y_pred, average='weighted'), 2),
            "f1": round(f1_score(y_test, y_pred, average='weighted'), 2)
        }
        
        print(f"✅ Training Complete. Metrics: {self.metrics}")
        return X_test, y_test

    def save_model(self, model_path='models/octasence_agent.pkl', metrics_path='models/metrics.json'):
        # Save the Model
        joblib.dump(self.model, model_path)
        
        # Save the Metrics
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics, f)
            
        print(f"💾 Model saved to {model_path}")
        print(f"📊 Metrics saved to {metrics_path}")

    def evaluate(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        print("\n--- Agent Report ---")
        print(classification_report(y_test, y_pred))