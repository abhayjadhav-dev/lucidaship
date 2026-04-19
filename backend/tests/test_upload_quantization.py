import pandas as pd

from app.services.dataset_relationships import DatasetAsset, analyze_dataset_collection, execute_merge_plan, prepare_combined_dataset
from app.services.upload_quantization import ingest_uploaded_dataset


def test_upload_quantization_protects_structural_columns_and_round_trips_numeric_block():
    df = pd.DataFrame(
        {
            "lead_id": [f"id-{i}" for i in range(200)],
            "email": [f"user{i}@example.com" for i in range(200)],
            "created_at": pd.date_range("2026-01-01", periods=200, freq="D").astype(str),
            "segment": ["A", "B"] * 100,
            "converted": [0, 1] * 100,
            "spend": [100.0 + i for i in range(200)],
            "visits": [float((i % 13) + 1) for i in range(200)],
        }
    )

    asset = ingest_uploaded_dataset(
        "leads.csv",
        df,
        enabled=True,
        mode="shadow",
        numeric_only=True,
        min_rows=32,
        max_allowed_mse=1.0,
        max_allowed_ip_error=1.0,
        target_column="converted",
    )

    diagnostics = asset.diagnostics
    assert "lead_id" in diagnostics["protected_columns"]
    assert "email" in diagnostics["protected_columns"]
    assert "created_at" in diagnostics["protected_columns"]
    assert "segment" in diagnostics["protected_columns"]
    assert "converted" in diagnostics["protected_columns"]
    assert diagnostics["compressed_numeric_columns"] == ["spend", "visits"]
    assert diagnostics["distortion_metrics"]["mse"] >= 0.0
    assert asset.quantized_block is not None
    assert asset.dequantized_df.shape == df.shape
    assert asset.dequantized_df["spend"].isna().sum() == df["spend"].isna().sum()


def test_upload_quantization_bypasses_small_or_ineligible_datasets():
    df = pd.DataFrame(
        {
            "lead_id": ["a", "b", "c"],
            "segment": ["A", "B", "A"],
            "converted": [0, 1, 0],
        }
    )

    asset = ingest_uploaded_dataset(
        "small.csv",
        df,
        enabled=True,
        mode="shadow",
        numeric_only=True,
        min_rows=32,
        max_allowed_mse=0.05,
        max_allowed_ip_error=0.10,
        target_column="converted",
    )

    assert asset.quantized_block is None
    assert asset.diagnostics["bypass_reason"] == "below_min_rows"


def test_merge_plan_is_stable_between_raw_and_execution_assets():
    base_df = pd.DataFrame(
        {
            "lead_id": [f"id-{i}" for i in range(200)],
            "revenue": [float(i) for i in range(200)],
            "visits": [float((i % 11) + 1) for i in range(200)],
        }
    )
    activity_df = pd.DataFrame(
        {
            "lead_id": [f"id-{i}" for i in range(200)],
            "sessions": [float((i % 7) + 1) for i in range(200)],
            "opens": [float((i % 5) + 2) for i in range(200)],
        }
    )

    ingested_base = ingest_uploaded_dataset(
        "base.csv",
        base_df,
        enabled=True,
        mode="safe_default_on",
        numeric_only=True,
        min_rows=32,
        max_allowed_mse=1.0,
        max_allowed_ip_error=1.0,
    )
    ingested_activity = ingest_uploaded_dataset(
        "activity.csv",
        activity_df,
        enabled=True,
        mode="safe_default_on",
        numeric_only=True,
        min_rows=32,
        max_allowed_mse=1.0,
        max_allowed_ip_error=1.0,
    )

    raw_assets = [
        DatasetAsset(
            name="base.csv",
            df=ingested_base.raw_df,
            raw_df=ingested_base.raw_df,
            protected_df=ingested_base.protected_df,
            dequantized_df=ingested_base.dequantized_df,
            compression=ingested_base.diagnostics,
            execution_mode=ingested_base.mode,
        ),
        DatasetAsset(
            name="activity.csv",
            df=ingested_activity.raw_df,
            raw_df=ingested_activity.raw_df,
            protected_df=ingested_activity.protected_df,
            dequantized_df=ingested_activity.dequantized_df,
            compression=ingested_activity.diagnostics,
            execution_mode=ingested_activity.mode,
        ),
    ]

    analysis = analyze_dataset_collection(raw_assets)
    assert analysis["relationships"][0]["should_consider_merge"] is True

    _, raw_plan = prepare_combined_dataset(raw_assets)
    execution_assets = [asset.for_execution() for asset in raw_assets]
    execution_df, execution_plan = execute_merge_plan(execution_assets, raw_plan)

    assert raw_plan["base_dataset"] == execution_plan["base_dataset"]
    assert execution_plan["executed_steps"][0]["left_column"] == "lead_id"
    assert execution_df.shape[0] == base_df.shape[0]


def test_many_to_many_relationship_is_aggregated_instead_of_raising():
    orders_df = pd.DataFrame(
        {
            "Style": ["A", "A", "B", "B", "C"],
            "qty": [1, 2, 3, 4, 5],
        }
    )
    catalog_df = pd.DataFrame(
        {
            "Style": ["A", "A", "B", "C", "C"],
            "price": [10.0, 11.0, 12.0, 13.0, 14.0],
            "discount": [1.0, 2.0, 0.5, 0.0, 1.5],
        }
    )

    assets = [
        DatasetAsset(name="orders.csv", df=orders_df),
        DatasetAsset(name="catalog.csv", df=catalog_df),
    ]

    merged_df, plan = prepare_combined_dataset(assets)
    assert merged_df.shape[0] == orders_df.shape[0]
    assert "catalog.csv__row_count" in merged_df.columns
    assert any(step.get("strategy") == "aggregate_many_to_many_right" for step in plan.get("executed_steps", []))
