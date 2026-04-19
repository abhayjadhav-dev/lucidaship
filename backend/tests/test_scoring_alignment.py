import pandas as pd

from adaptive_scorer import UniversalAdaptiveScorer, EngagementScorer, ActionRecommender


def test_sorted_results_keep_engagement_and_action_aligned_to_same_row():
    train_df = pd.DataFrame(
        {
            "lead_id": [f"id-{i}" for i in range(24)],
            "segment": ["A", "B", "C", "A", "B", "C"] * 4,
            "revenue": [80, 210, 140, 90, 230, 160] * 4,
            "email_open_count": [0, 10, 2, 1, 12, 3] * 4,
            "converted": [0, 1, 0, 0, 1, 0] * 4,
        }
    )

    score_df = pd.DataFrame(
        {
            "lead_id": ["cold-row", "hot-row"],
            "segment": ["A", "B"],
            "revenue": [70, 250],
            "email_open_count": [0, 15],
        }
    )

    scorer = UniversalAdaptiveScorer()
    scorer.train(train_df, target_col="converted", client_id="alignment-test")
    results = scorer.score(score_df)

    # Results are sorted by score, so the first item may not match score_df.iloc[0].
    assert results[0]["data"]["lead_id"] == "hot-row"
    assert results[1]["data"]["lead_id"] == "cold-row"

    engagement_scorer = EngagementScorer()
    engagement_scorer.analyze(score_df)
    recommender = ActionRecommender()

    enriched = []
    for result in results:
        row = pd.Series(result["data"])
        engagement = engagement_scorer.score_lead(row)
        action = recommender.recommend(result["score"], engagement["engagement_score"])
        enriched.append(
            {
                "lead_id": result["data"]["lead_id"],
                "score": result["score"],
                "engagement_score": engagement["engagement_score"],
                "action": action["action"],
            }
        )

    by_id = {item["lead_id"]: item for item in enriched}
    assert by_id["hot-row"]["engagement_score"] > by_id["cold-row"]["engagement_score"]
    assert by_id["hot-row"]["action"] != by_id["cold-row"]["action"]


def test_low_profile_scores_are_not_all_deprioritized():
    recommender = ActionRecommender()

    a = recommender.recommend(45, None)
    b = recommender.recommend(30, 70)
    c = recommender.recommend(10, 10)

    assert a["action"] != "DEPRIORITIZE"
    assert b["action"] != "DEPRIORITIZE"
    assert c["action"] == "DEPRIORITIZE"
