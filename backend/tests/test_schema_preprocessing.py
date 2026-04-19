import pandas as pd

from app.services.column_matcher import normalize_column_name, fuzzy_match_score, find_best_matches
from app.services.intelligent_imputation import extract_imputation_stats, impute_missing_columns
from app.services.type_coercion import coerce_series_to_expected_type


def test_column_matching_basic():
    assert normalize_column_name("Revenue") == "revenue"
    assert fuzzy_match_score("Revenue", "revenue") == 1.0
    assert fuzzy_match_score("website_visits", "website visits") >= 0.85

    expected = {"revenue", "website_visits"}
    actual = {"Revenue", "website visits", "other"}
    match_result = find_best_matches(expected, actual)

    assert len(match_result["matches"]) == 2
    assert match_result["unmatched_expected"] == []


def test_imputation_and_coercion():
    df = pd.DataFrame({
        "revenue": [100.0, None, 300.0],
        "industry": ["Tech", None, "Finance"],
    })
    column_types = {
        "revenue": "numeric",
        "industry": "categorical",
        "last_seen": "temporal",
    }
    stats = extract_imputation_stats(df, column_types)

    df_imputed, report = impute_missing_columns(df.copy(), column_types.keys(), stats)
    assert "last_seen" in df_imputed.columns
    assert report["imputed_count"] == 1

    coerced_numeric, numeric_report = coerce_series_to_expected_type(df_imputed["revenue"], "numeric")
    assert numeric_report["coerced_non_null"] >= 2
    assert coerced_numeric.notna().sum() >= 2

    coerced_temporal, temporal_report = coerce_series_to_expected_type(df_imputed["last_seen"], "temporal")
    assert temporal_report["expected_type"] == "temporal"
    assert coerced_temporal.isna().sum() == len(df_imputed)
