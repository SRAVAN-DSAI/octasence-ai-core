# train.py (Updated with Debugging)
from src.data_loader import DataLoader
from src.model_trainer import OctaSenceRiskAgent
import os

def main():
    # 1. Setup
    if not os.path.exists('models'):
        os.makedirs('models')

    # 2. ETL (Extract, Transform, Load)
    loader = DataLoader()
    raw_data = loader.load_data()
    clean_data = loader.preprocess(raw_data)
    
    # --- DEBUGGING STEP ---
    # This prints all column names so you can see the exact spelling
    print("\n🔍 ACTUAL DATASET COLUMNS:", clean_data.columns.tolist())
    
    # 3. Dynamic Target Selection
    # We check for common variations of the target column
    possible_targets = ['Risk Level', 'Risk_Level', 'RiskLevel', 'risk_level']
    target_col = 'Risk Level' # Default
    
    for col in possible_targets:
        if col in clean_data.columns:
            target_col = col
            print(f"🎯 Target Column Found: '{target_col}'")
            break
    
    # 4. Training
    agent = OctaSenceRiskAgent()
    
    # We pass the detected target_col to the train function
    try:
        X_test, y_test = agent.train(clean_data, target_col=target_col)
        
        # 5. Evaluation
        agent.evaluate(X_test, y_test)
        
        # 6. Serialization (Saving)
        agent.save_model('models/octasence_agent.pkl')
        
    except KeyError as e:
        print(f"\n❌ ERROR: The pipeline could not find the target column '{target_col}'.")
        print("Check the 'ACTUAL DATASET COLUMNS' list printed above and update the target_col variable.")

if __name__ == "__main__":
    main()