
import logging
from pyspark.sql import DataFrame
from pyspark.sql.functions import col, lit, when, size, array_contains, explode, sum as spark_sum, count, avg, stddev, sort_array, udf
from pyspark.sql.types import FloatType, IntegerType, ArrayType, StringType
import pyspark.sql.functions as F
import ast
import numpy as np
import itertools

# --- Helper UDFs (only used when native functions are too complex) ---
# Note: In a true Big Data environment, we'd avoid UDFs or use Pandas UDFs (Arrow) for performance.
# For simplicity and "simulation" correctness, we use standard Python UDFs here.

def safe_literal_eval(val):
    try:
        if val is None: return []
        if isinstance(val, list): return val
        if isinstance(val, str) and val.startswith('[') and val.endswith(']'):
            return ast.literal_eval(val)
        return []
    except (ValueError, SyntaxError, TypeError):
        return []

# Registering UDFs that might be needed if columns are strings instead of arrays
safe_eval_udf = udf(safe_literal_eval, ArrayType(StringType()))

def create_onboarding_features(df: DataFrame, artifacts: dict) -> DataFrame:
    """Creates features for new business applicants with limited data."""
    logging.info("Creating onboarding features using Spark...")

    cost_weights = artifacts.get('chronic_condition_cost_weights', {})
    freq_weights = artifacts.get('chronic_condition_freq_weights', {})

    # For performance, avoid UDFs. Use native Spark functions.
    # The best approach is to convert dicts to Spark maps and use a SQL expression.
    if not cost_weights or not freq_weights:
        df_out = df.withColumn('self_reported_chronic_score', lit(0.0))
    else:
        # Create map literals for joining/lookup within the DataFrame context
        cost_map = F.create_map([F.lit(x) for x in itertools.chain.from_iterable(cost_weights.items())])
        freq_map = F.create_map([F.lit(x) for x in itertools.chain.from_iterable(freq_weights.items())])

        # Use a Spark SQL expression for transforming the array of conditions into an array of scores, then summing.
        # This is highly performant and avoids expensive UDFs or explodes.
        score_expr = f"""
        aggregate(
            transform(self_reported_conditions, c ->
                (coalesce({cost_map._jc.toString()}[c], 1.0) + coalesce({freq_map._jc.toString()}[c], 1.0)) / 2.0
            ),
            0.0,
            (acc, val) -> acc + val
        )
        """

        # Ensure the column exists and handle nulls before applying the expression
        df_with_conditions = df.withColumn(
            'self_reported_conditions',
            F.when(col('self_reported_conditions').isNull(), F.array()).otherwise(col('self_reported_conditions'))
        )

        df_with_maps = df_with_conditions.withColumn("cost_map", cost_map).withColumn("freq_map", freq_map)

        df_out = df_with_maps.withColumn(
            'self_reported_chronic_score',
            F.expr(score_expr)
        )

    df_out = df_out.withColumn('age_x_chronic_score', col('age') * col('self_reported_chronic_score'))
    
    # Add a new feature for demographic risk
    location_risk_map = F.create_map([
        F.lit('urban'), F.lit(1.2),
        F.lit('suburban'), F.lit(1.0),
        F.lit('rural'), F.lit(0.9)
    ])
    
    df_out = df_out.withColumn(
        'demographic_risk_score',
        (col('age') / 100.0) * F.coalesce(location_risk_map[col('location')], 1.0)
    )
    
    return df_out

def create_renewal_vice_champion_features(df: DataFrame, artifacts: dict) -> DataFrame:
    """Creates all advanced features EXCEPT provider-specific ones."""
    logging.info("Creating renewal vice-champion features using Spark...")
    
    cost_weights = artifacts.get('chronic_condition_cost_weights', {})
    freq_weights = artifacts.get('chronic_condition_freq_weights', {})
    chronic_defs = artifacts.get('chronic_conditions_definitions', {})

    # Create maps for efficient lookups
    cost_map = F.create_map([F.lit(x) for x in itertools.chain.from_iterable(cost_weights.items())])
    freq_map = F.create_map([F.lit(x) for x in itertools.chain.from_iterable(freq_weights.items())])
    
    # This is a complex transformation. We'll build a mapping from diagnosis prefixes to condition names.
    # To do this efficiently in Spark, we can create a series of `when` clauses.

    # Flatten the chronic_defs for easier processing
    diag_to_cond_list = []
    for cond, codes in chronic_defs.items():
        for code in codes:
            diag_to_cond_list.append((code, cond))

    # Build the CASE WHEN expression for mapping diagnosis to condition
    map_expr = F.create_map([F.lit(x) for x in itertools.chain.from_iterable(diag_to_cond_list)])

    # UDF to resolve diagnosis to a chronic condition. While we aim to avoid UDFs,
    # the string matching logic `startswith` is complex to vectorize perfectly without one.
    # A Pandas UDF would be the next step up in a real-world scenario.
    @udf(ArrayType(StringType()))
    def map_diags_to_conditions(diag_codes):
        if not diag_codes:
            return []
        found_conditions = set()
        for diag in diag_codes:
            if not diag:
                continue
            for cond, prefixes in chronic_defs.items():
                if any(diag.startswith(prefix) for prefix in prefixes):
                    found_conditions.add(cond)
        return list(found_conditions)

    df_with_conditions = df.withColumn(
        'chronic_conditions_list',
        map_diags_to_conditions(col('diagnosis_codes_list'))
    )

    # Now calculate score based on the mapped conditions, similar to the onboarding logic
    score_expr = f"""
    aggregate(
        transform(chronic_conditions_list, c ->
            (coalesce({cost_map._jc.toString()}[c], 1.0) + coalesce({freq_map._jc.toString()}[c], 1.0)) / 2.0
        ),
        0.0,
        (acc, val) -> acc + val
    )
    """

    df_with_maps = df_with_conditions.withColumn("cost_map", cost_map).withColumn("freq_map", freq_map)

    df_out = df_with_maps.withColumn('data_driven_risk_score', F.expr(score_expr))

    # Cost Volatility (IQR) - Hard to do per-row in Spark without UDF or complex Array functions (Spark 3.1+ has percentile_approx)
    # We'll use a UDF for row-wise array statistics.
    @udf(FloatType())
    def cost_volatility_iqr(costs):
        if not costs or len(costs) <= 1: return 0.0
        try:
            return float(np.subtract(*np.percentile(costs, [75, 25])))
        except: return 0.0

    df_out = df_out.withColumn('cost_volatility_iqr', cost_volatility_iqr(col('claim_costs_list')))
    
    # Pre-auth denial rate
    # Spark 2.4+ array functions
    # transform array to boolean (is_rejected), then aggregate?
    # Easier with UDF for list logic.
    @udf(FloatType())
    def pre_auth_denial_rate(outcomes):
        if not outcomes: return 0.0
        rejected = sum(1 for x in outcomes if x == 'rejected')
        return float(rejected) / len(outcomes)

    df_out = df_out.withColumn('pre_auth_denial_rate', pre_auth_denial_rate(col('pre_auth_outcomes_list')))

    # Days between visits STD
    @udf(FloatType())
    def get_days_between_visits_std(dates):
        if not dates or len(dates) < 2: return 0.0
        try:
            # Need pandas/numpy for convenient date diff logic usually
            import pandas as pd
            import numpy as np
            sorted_dates = sorted(pd.to_datetime(dates, errors='coerce').dropna())
            if len(sorted_dates) < 2: return 0.0
            days_between = np.diff(sorted_dates).astype('timedelta64[D]').astype(int)
            return float(np.std(days_between)) if len(days_between) > 0 else 0.0
        except:
            return 0.0

    df_out = df_out.withColumn('days_between_visits_std', get_days_between_visits_std(col('service_dates_list')))

    # Cost Trend Slope
    @udf(FloatType())
    def get_cost_trend_slope(dates, costs):
        if not dates or not costs or len(dates) < 2 or len(dates) != len(costs): return 0.0
        try:
            import pandas as pd
            from scipy import stats
            df = pd.DataFrame({'date': pd.to_datetime(dates, errors='coerce'), 'cost': costs}).dropna().sort_values('date')
            if len(df) < 2: return 0.0
            time_deltas = (df['date'] - df['date'].min()).dt.days
            if len(set(time_deltas)) < 2: return 0.0
            slope, _, _, _, _ = stats.linregress(time_deltas, df['cost'])
            return float(slope) if not np.isnan(slope) else 0.0
        except:
            return 0.0

    df_out = df_out.withColumn('cost_trend_slope', get_cost_trend_slope(col('service_dates_list'), col('claim_costs_list')))

    # Care team complexity (unique count)
    # Native Spark: size(array_distinct(col))
    df_out = df_out.withColumn('care_team_complexity', size(F.array_distinct(col('practitioner_specialty_list'))))

    # Activity Ratios
    @udf(FloatType())
    def medication_ratio(activities):
        if not activities: return 0.0
        return float(activities.count('Medication')) / len(activities)

    @udf(FloatType())
    def imaging_ratio(activities):
        if not activities: return 0.0
        return float(activities.count('Imaging')) / len(activities)
        
    df_out = df_out.withColumn('medication_ratio', medication_ratio(col('activity_type_list')))
    df_out = df_out.withColumn('imaging_ratio', imaging_ratio(col('activity_type_list')))

    # Medical Necessity
    medical_necessity_codes = artifacts.get('denial_codes', {}).get('medical_necessity', [])
    # Broadcast or closure capture
    
    @udf(IntegerType())
    def medical_necessity_denial_count(reasons):
        if not reasons: return 0
        return sum(1 for code in reasons if code in medical_necessity_codes)
    
    df_out = df_out.withColumn('medical_necessity_denial_count', medical_necessity_denial_count(col('adjudication_reason_list')))

    return df_out

def create_renewal_champion_features(df: DataFrame, original_abt: DataFrame, artifacts: dict) -> DataFrame:
    """Adds provider-specific features."""
    logging.info("Creating renewal champion features using Spark...")
    
    # Spark DataFrames are immutable, so we chain transformations.
    # original_abt in Spark might be the same as df if not dropped.
    
    provider_stats = artifacts.get('provider_stats', {})
    provider_to_specialty = artifacts.get('provider_to_specialty_map', {})
    specialty_avg_costs = artifacts.get('specialty_avg_costs', {})
    provider_to_type = artifacts.get('provider_to_type_map', {})
    global_avg_provider_cost = float(np.mean([v['avg_cost'] for v in provider_stats.values()]) if provider_stats else 0)

    @udf(FloatType())
    def get_credibility_weighted_score(providers_list):
        if not providers_list: return global_avg_provider_cost
        scores = []
        for provider_id in providers_list:
            provider_id = str(provider_id)
            stats = provider_stats.get(provider_id)
            if stats:
                provider_avg_cost = stats['avg_cost']
                claim_count = stats['claim_count']
                specialty = provider_to_specialty.get(provider_id)
                specialty_avg = specialty_avg_costs.get(specialty, global_avg_provider_cost)
                credibility = claim_count / (claim_count + 20.0) 
                weighted_score = (provider_avg_cost * credibility) + (specialty_avg * (1 - credibility))
                scores.append(weighted_score)
            else:
                scores.append(global_avg_provider_cost)
        return float(np.mean(scores)) if scores else global_avg_provider_cost

    @udf(FloatType())
    def get_hospital_visit_ratio(providers_list):
        if not providers_list: return 0.0
        hospital_visits = 0
        for provider_id in providers_list:
            provider_type = provider_to_type.get(str(provider_id), '')
            if isinstance(provider_type, str) and 'hospital' in provider_type.lower():
                hospital_visits += 1
        return float(hospital_visits) / len(providers_list)

    # Note: original_abt in Spark implies we need to join back if we lost columns, 
    # but typically we just keep adding columns to the same DF. 
    # We assume 'df' has 'providers_visited_list'.
    
    df_out = df.withColumn('avg_provider_risk_score', get_credibility_weighted_score(col('providers_visited_list')))
    df_out = df_out.withColumn('hospital_visit_ratio', get_hospital_visit_ratio(col('providers_visited_list')))

    return df_out


def create_financial_approval_features(df: DataFrame, artifacts: dict) -> DataFrame:
    """Creates features for financial claim approval."""
    logging.info("Creating financial approval features using Spark...")
    
    plan_limits = artifacts.get('plan_limits', {})
    provider_master = artifacts.get('provider_master', {}) # Mocked provider master
    
    @udf(FloatType())
    def calculate_plan_utilization(plan_id, current_claim_amount, prior_spend):
        limit = plan_limits.get(plan_id, 10000.0)
        total_spend = (prior_spend if prior_spend else 0.0) + current_claim_amount
        return float(total_spend) / limit

    @udf(IntegerType())
    def get_provider_network_tier(provider_id):
        # 1 = In-Network, 2 = Out-of-Network, 3 = Blacklisted
        if not provider_id: return 2
        return provider_master.get(str(provider_id), {}).get('tier', 2)

    df_out = df.withColumn('plan_utilization_percent', calculate_plan_utilization(col('plan_id'), col('claim_amount'), col('ytd_spend')))
    
    # Provider Network Tier
    df_out = df_out.withColumn('provider_network_tier', get_provider_network_tier(col('provider_license_key')))

    # Duplicate Suspect (Simulated for this row based on history available in ABT)
    # In reality, this would check against a history table. 
    # Here, we assume 'prior_claims_hashes' is a list of hashes of (date, amount, provider) passed in ABT.
    
    @udf(IntegerType())
    def is_duplicate_suspect(curr_hash, history_hashes):
        if not history_hashes: return 0
        return 1 if curr_hash in history_hashes else 0

    # We assume the ABT builder creates 'claim_hash' and 'history_hashes'
    if 'claim_hash' in df.columns and 'history_hashes' in df.columns:
        df_out = df_out.withColumn('is_duplicate_suspect', is_duplicate_suspect(col('claim_hash'), col('history_hashes')))
    
    return df_out

def create_claim_fwa_features(df: DataFrame, artifacts: dict) -> DataFrame:
    """Creates features for Claim-level FWA detection."""
    logging.info("Creating claim FWA features using Spark...")
    
    # 1. Round Amount Check (500.00 vs 493.21)
    @udf(IntegerType())
    def is_round_amount(amount):
        if amount is None: return 0
        try:
            return 1 if amount % 100 == 0 else 0
        except: return 0

    df_out = df.withColumn('is_round_amount', is_round_amount(col('payer_share_amount')))
    
    # 2. Digit Analysis (Benford's Law - Simplified)
    # Check if first digit is '9' (often over-used by fraudsters avoiding thresholds like 1000)
    # Actually Benford says 1 is most common (30%), 9 is least (4%). High 9s might mean "just under limit".
    @udf(IntegerType())
    def first_digit_is_nine(amount):
        if not amount: return 0
        s = str(abs(int(amount)))
        return 1 if s[0] == '9' else 0
        
    df_out = df_out.withColumn('starts_with_nine', first_digit_is_nine(col('payer_share_amount')))
    
    return df_out

def create_patient_fwa_features(df: DataFrame, artifacts: dict) -> DataFrame:
    """Creates features for Patient-level FWA detection (e.g., Doctor Shopping)."""
    logging.info("Creating patient FWA features using Spark...")
    
    # Expects aggregated history in 'provider_history_list' (list of provider IDs visited in last 30 days)
    # This must be prepared by the ABT step.
    
    @udf(IntegerType())
    def count_unique_providers(providers):
        if not providers: return 0
        return len(set(providers))

    df_out = df.withColumn('unique_providers_30d', count_unique_providers(col('provider_history_list')))
    
    # High Velocity: Claims in last 7 days
    # Expects 'claim_dates_list'
    @udf(IntegerType())
    def count_claims_7d(dates, current_date):
        if not dates or not current_date: return 0
        import pandas as pd
        curr = pd.to_datetime(current_date)
        count = 0
        for d in dates:
            dt = pd.to_datetime(d)
            if (curr - dt).days <= 7 and (curr - dt).days >= 0:
                count += 1
        return count
    
    # Optimization: passing current_date as a column if row-specific, or lit() if today. 
    # Usually ABT has 'max_serviced_date' as 'current_date'.
    df_out = df_out.withColumn('claim_velocity_7d', count_claims_7d(col('service_dates_list'), col('max_serviced_date')))
    
    return df_out

def create_medical_approval_features(df: DataFrame, artifacts: dict) -> DataFrame:
    """Creates features for Medical Necessity."""
    logging.info("Creating medical approval features using Spark...")
    
    soc = artifacts.get('standard_of_care', {})
    gender_rules = artifacts.get('gender_restricted_procedures', {})
    
    @udf(IntegerType())
    def is_standard_of_care(diag, proc):
        if not diag or not proc: return 1 # Benefit of doubt
        allowed = soc.get(diag, [])
        return 1 if proc in allowed else 0

    @udf(IntegerType())
    def is_gender_compliant(proc, gender):
        if not proc or not gender: return 1
        allowed_genders = gender_rules.get(proc)
        if not allowed_genders: return 1
        return 1 if gender in allowed_genders else 0

    df_out = df.withColumn('is_soc_compliant', is_standard_of_care(col('diagnosis_code'), col('procedure_code')))
    df_out = df_out.withColumn('is_gender_compliant', is_gender_compliant(col('procedure_code'), col('Gender')))
    
    return df_out
