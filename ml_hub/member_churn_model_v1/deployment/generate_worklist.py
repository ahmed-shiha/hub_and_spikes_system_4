
import mlflow
import pandas as pd
import shap
import logging
from pathlib import Path
import numpy as np
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent))
from medins_ml_utils.utils import load_config
from medins_ml_utils.features import safe_literal_eval


logging.basicConfig(level=logging.INFO)

def main():
    config = load_config()
    logging.info("--- Member Churn Model: Generating Retention Worklist ---")
    
    try:
        abt_path = Path(config['paths']['pricing_model_output_dir']) / 'data/abt_renewal.csv'
        renewal_candidates = pd.read_csv(abt_path)
        list_cols = [col for col in renewal_candidates.columns if col.endswith('_list')]
        for col in list_cols:
            renewal_candidates[col] = renewal_candidates[col].apply(safe_literal_eval)
    except FileNotFoundError:
        logging.error(f"Renewal ABT not found at {abt_path}. Run the pricing model pipeline first.")
        return

    # Engineer the same 'friction' features as in the training script
    renewal_candidates['out_of_pocket_ratio'] = (renewal_candidates['annualized_cost'] / renewal_candidates['premium']).fillna(0).clip(0, 2)
    renewal_candidates['premium_increase_percentage'] = (renewal_candidates['premium'] / renewal_candidates['premium'].shift(1, fill_value=renewal_candidates['premium'].mean())) - 1
    renewal_candidates['pre_auth_denial_rate'] = renewal_candidates['pre_auth_outcomes_list'].apply(lambda x: x.count('rejected') / len(x) if isinstance(x, list) and len(x) > 0 else 0)
    renewal_candidates['manual_adjudication_rate'] = np.random.uniform(0, 1, len(renewal_candidates))

    try:
        model_uri = "models:/member-churn-model/Production"
        model_wrapper = mlflow.pyfunc.load_model(model_uri=model_uri)
        model = model_wrapper._model_impl.lgb
        
        features = ['out_of_pocket_ratio', 'premium_increase_percentage', 'pre_auth_denial_rate', 'policy_duration_days', 'manual_adjudication_rate']
        X = renewal_candidates[features].fillna(0)
        
        renewal_candidates['churn_probability'] = model.predict_proba(X)[:, 1]

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        
        def get_top_reasons(shap_row, feature_names):
            abs_shap = np.abs(shap_row)
            top_indices = np.argsort(abs_shap)[-3:]
            return [feature_names[i] for i in reversed(top_indices)]

        churn_shap_values = shap_values[1] if isinstance(shap_values, list) else shap_values
        renewal_candidates['top_churn_drivers'] = [get_top_reasons(row, X.columns) for row in churn_shap_values]
        
        worklist = renewal_candidates[['Patient_KEY', 'churn_probability', 'top_churn_drivers']].sort_values('churn_probability', ascending=False)
        
        output_path = Path("output/retention_worklist.csv")
        output_path.parent.mkdir(exist_ok=True)
        worklist.to_csv(output_path, index=False)
        logging.info(f"Actionable retention worklist saved to {output_path}")

    except Exception as e:
        logging.error(f"Failed to generate worklist: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
