
import json
import pandas as pd
import logging
from typing import Dict, List, Any

def parse_user_json(json_str: str) -> Dict[str, Any]:
    """Parses the User JSON blob into a flat dictionary of features."""
    try:
        if not json_str:
            return {}
        data = json.loads(json_str)
        
        # Extract valuable features
        features = {
            "user_id": data.get("name"),
            "gender": data.get("gender"),
            "birth_date": data.get("birth_date"),
            "salary_range": data.get("salary"),
            "had_chronic_diseases": int(data.get("had_chronic_diseases", 0)),
            "marital_status": data.get("marital_status"),
            "nationality": data.get("nationality")
        }
        
        # Calculate Age (Approximate)
        if features['birth_date']:
            try:
                birth_year = int(str(features['birth_date'])[:4])
                features['age'] = 2025 - birth_year # Using current year fixed for simplicity
            except:
                features['age'] = 30 # Default
        
        return features
    except json.JSONDecodeError:
        logging.error("Failed to parse User JSON")
        return {}

def flatten_claim_dump(csv_path: str) -> pd.DataFrame:
    """
    Reads the 'Tuba' style dump and flattens it into a standard ABT format.
    One row per Service/Medication item.
    """
    try:
        # Load all as string to handle varying formats
        df = pd.read_csv(csv_path, dtype=str)
        
        # Identify Item Type (Medication or Service)
        # We can create a unified 'Items' dataframe
        
        items = []
        
        for _, row in df.iterrows():
            # Common Claim Info
            claim_id = row.get('Name', row.get('ID', '')) # Assuming first col is ID? Header is messed up in example
            # The example header is very wide. Let's assume standard keys mapping.
            
            # Extract Service Items
            if pd.notna(row.get('Service Code (Service Items)')):
                item = {
                    'claim_id': claim_id,
                    'type': 'Service',
                    'code': row.get('Service Code (Service Items)'),
                    'diagnosis': row.get('Diagnosis Code (Service Items)'),
                    'amount': float(row.get('Billed Net Amount (Service Items)', 0) or 0),
                    'user_json': None # Logic to fetch user json would be here if linked
                }
                items.append(item)
                
            # Extract Medication Items
            if pd.notna(row.get('Product Code (Medications)')):
                item = {
                    'claim_id': claim_id,
                    'type': 'Medication',
                    'code': row.get('Product Code (Medications)'),
                    'diagnosis': row.get('Diagnosis Code (Medications)'),
                    'amount': float(row.get('Net Amount (Medications)', 0) or 0),
                    'user_json': None
                }
                items.append(item)
                
        return pd.DataFrame(items)
        
    except Exception as e:
        logging.error(f"Failed to flatten dump: {e}")
        return pd.DataFrame()
