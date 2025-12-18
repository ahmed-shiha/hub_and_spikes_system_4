
from flask import Flask, request, jsonify
import lightgbm as lgb
import pandas as pd
from medins_ml_utils.utils import load_config

app = Flask(__name__)
model = None

@app.before_request
def load_model():
    global model
    if model is None:
        config = load_config()
        model_path = Path(config['paths']['output_dir']) / "models/financial_model.txt"
        model = lgb.Booster(model_file=str(model_path))

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    # Logic to compute features on the fly
    # Inputs: claim_amount, prior_spend, plan_limit (optional, default 10000)
    claim_amount = float(data.get('claim_amount', 0))
    prior_spend = float(data.get('prior_spend', 0))
    plan_limit = float(data.get('plan_limit', 10000.0))
    
    plan_utilization_percent = (prior_spend + claim_amount) / plan_limit
    
    features = {
        'plan_utilization_percent': plan_utilization_percent,
        'claim_amount': claim_amount
    }
    
    df = pd.DataFrame([features])
    pred = model.predict(df)[0]
    return jsonify({'financial_rejection_prob': float(pred)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
