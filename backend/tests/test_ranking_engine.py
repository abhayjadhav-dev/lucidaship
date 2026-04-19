"""
Unit tests for Patent-Strengthening Ranking Engine - Phase 1

Tests cover:
- SignalExtractor (all signal types)
- TopsisRanker (normalization, scoring, ranking)
- AHPWeighting (pairwise comparison, consistency)
- ConfidenceIntervals (bootstrap estimation)
- RankingEngine (end-to-end pipeline)
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from app.services.ranking_engine import (
    SignalExtractor,
    SignalType,
    TopsisRanker,
    AHPWeighting,
    ConfidenceIntervals,
    RankingEngine,
)


@pytest.fixture
def sample_numeric_df():
    """Create sample DataFrame with numeric columns."""
    np.random.seed(42)
    return pd.DataFrame({
        "revenue": np.random.exponential(100000, 100),
        "employees": np.random.randint(10, 1000, 100),
        "rating": np.random.uniform(1, 5, 100),
        "years_active": np.random.randint(1, 50, 100),
    })


@pytest.fixture
def sample_mixed_df():
    """Create sample DataFrame with mixed types."""
    np.random.seed(42)
    n = 100
    
    return pd.DataFrame({
        "name": [f"Company_{i}" for i in range(n)],
        "revenue": np.random.exponential(100000, n),
        "status": np.random.choice(["active", "inactive", "pending"], n),
        "created_date": [
            datetime.now() - timedelta(days=int(x))
            for x in np.random.exponential(365, n)
        ],
        "employees": np.random.randint(10, 1000, n),
        "completeness": np.random.uniform(0.5, 1.0, n),
    })


class TestSignalExtractor:
    """Test signal extraction from various data types."""
    
    def test_extract_numeric_signals(self, sample_numeric_df):
        """Test extraction of numeric signals."""
        extractor = SignalExtractor(sample_numeric_df)
        signal_df, signal_info = extractor.extract_all()
        
        # Check output shapes
        assert signal_df.shape[0] == sample_numeric_df.shape[0]
        assert signal_df.shape[1] > sample_numeric_df.shape[1]  # More signals than inputs
        
        # Check signal info
        assert len(signal_info) == signal_df.shape[1]
        
        # Check that signals are in [0, 1]
        assert (signal_df >= 0).all().all()
        assert (signal_df <= 1).all().all()
    
    def test_numeric_signal_types(self, sample_numeric_df):
        """Test that expected numeric signal types are extracted."""
        extractor = SignalExtractor(sample_numeric_df)
        _, signal_info = extractor.extract_all()
        
        signal_types = {info.signal_type for info in signal_info.values()}
        
        # Should have numeric signals
        assert SignalType.NUMERIC_ABSOLUTE in signal_types
        assert SignalType.NUMERIC_Z_SCORE in signal_types
        assert SignalType.NUMERIC_PERCENTILE in signal_types
        assert SignalType.NUMERIC_DISTANCE in signal_types
    
    def test_categorical_signals(self, sample_mixed_df):
        """Test extraction of categorical signals."""
        extractor = SignalExtractor(sample_mixed_df)
        signal_df, signal_info = extractor.extract_all()
        
        # Check categorical signals
        categorical_info = [
            info for info in signal_info.values()
            if "categorical" in info.signal_type.value
        ]
        assert len(categorical_info) > 0
        
        # Categorical signals should be in [0, 1]
        assert (signal_df >= 0).all().all()
        assert (signal_df <= 1).all().all()
    
    def test_temporal_signals(self, sample_mixed_df):
        """Test extraction of temporal signals."""
        extractor = SignalExtractor(sample_mixed_df)
        signal_df, signal_info = extractor.extract_all()
        
        # Check temporal signals
        temporal_info = [
            info for info in signal_info.values()
            if "temporal" in info.signal_type.value
        ]
        assert len(temporal_info) > 0
    
    def test_composite_signals(self, sample_mixed_df):
        """Test extraction of composite signals."""
        extractor = SignalExtractor(sample_mixed_df)
        signal_df, signal_info = extractor.extract_all()
        
        # Check composite signals
        composite_info = [
            info for info in signal_info.values()
            if "composite" in info.signal_type.value
        ]
        assert len(composite_info) > 0
        
        # Profile completeness should exist
        assert any("completeness" in name for name in signal_info.keys())
    
    def test_signal_metadata(self, sample_numeric_df):
        """Test signal metadata accuracy."""
        extractor = SignalExtractor(sample_numeric_df)
        _, signal_info = extractor.extract_all()
        
        for signal_name, info in signal_info.items():
            # All signals should have valid count
            assert info.valid_count > 0
            # All signals should have non-negative null count
            assert info.null_count >= 0


class TestTopsisRanker:
    """Test TOPSIS ranking algorithm."""
    
    def test_score_range(self, sample_numeric_df):
        """Test that TOPSIS scores are in [0, 1]."""
        extractor = SignalExtractor(sample_numeric_df)
        signal_df, _ = extractor.extract_all()
        
        ranker = TopsisRanker(signal_df)
        scores = ranker.score()
        
        assert scores.shape[0] == len(signal_df)
        assert (scores >= 0).all()
        assert (scores <= 1).all()
    
    def test_perfect_signal(self):
        """Test TOPSIS with perfect signal."""
        # Create signal where perfect rows are obvious
        signal_matrix = pd.DataFrame({
            "signal1": [1, 0, 0, 0],
            "signal2": [1, 0, 0, 0],
        })
        
        ranker = TopsisRanker(signal_matrix)
        scores = ranker.score()
        
        # First row should have highest score
        assert np.argmax(scores) == 0
        # Scores should be different
        assert len(np.unique(scores)) > 1
    
    def test_normalization(self, sample_numeric_df):
        """Test that normalization works correctly."""
        extractor = SignalExtractor(sample_numeric_df)
        signal_df, _ = extractor.extract_all()
        
        ranker = TopsisRanker(signal_df)
        normalized = ranker._normalize()
        
        # Normalized matrix should have Euclidean norms close to 1
        col_norms = np.sqrt(np.sum(normalized ** 2, axis=0))
        np.testing.assert_array_almost_equal(col_norms, np.ones(normalized.shape[1]))
    
    def test_ranking_order(self, sample_numeric_df):
        """Test that ranking returns ordered results."""
        extractor = SignalExtractor(sample_numeric_df)
        signal_df, _ = extractor.extract_all()
        
        ranker = TopsisRanker(signal_df)
        rankings = ranker.rank(top_n=5)
        
        assert len(rankings) <= 5
        # Scores should be in descending order
        scores = [score for _, score in rankings]
        assert scores == sorted(scores, reverse=True)
    
    def test_custom_weights(self, sample_numeric_df):
        """Test TOPSIS with custom weights."""
        extractor = SignalExtractor(sample_numeric_df)
        signal_df, _ = extractor.extract_all()
        
        # Create weights that favor first signal
        weights = np.zeros(signal_df.shape[1])
        weights[0] = 0.9
        weights[1:] = 0.1 / (signal_df.shape[1] - 1)
        
        ranker = TopsisRanker(signal_df, weights=weights)
        scores = ranker.score()
        
        assert (scores >= 0).all() and (scores <= 1).all()


class TestAHPWeighting:
    """Test AHP (Analytic Hierarchy Process) weighting."""
    
    def test_weights_sum_to_one(self, sample_numeric_df):
        """Test that AHP weights sum to 1."""
        extractor = SignalExtractor(sample_numeric_df)
        signal_df, _ = extractor.extract_all()
        
        ahp = AHPWeighting(signal_df, top_n=5)
        weights = ahp.compute_weights()
        
        np.testing.assert_almost_equal(np.sum(weights), 1.0)
    
    def test_weights_are_positive(self, sample_numeric_df):
        """Test that AHP weights are positive."""
        extractor = SignalExtractor(sample_numeric_df)
        signal_df, _ = extractor.extract_all()
        
        ahp = AHPWeighting(signal_df, top_n=5)
        weights = ahp.compute_weights()
        
        assert (weights >= 0).all()
    
    def test_consistency_ratio(self, sample_numeric_df):
        """Test consistency ratio calculation."""
        extractor = SignalExtractor(sample_numeric_df)
        signal_df, _ = extractor.extract_all()
        
        ahp = AHPWeighting(signal_df, top_n=5)
        ahp.compute_weights()
        
        # CR should be between 0 and 1
        assert 0 <= ahp.consistency_ratio <= 1
    
    def test_top_n_constraint(self, sample_numeric_df):
        """Test that top_n constraint is respected."""
        extractor = SignalExtractor(sample_numeric_df)
        signal_df, _ = extractor.extract_all()
        
        ahp = AHPWeighting(signal_df, top_n=3)
        weights = ahp.compute_weights()
        
        # Should have same length as input
        assert len(weights) == signal_df.shape[1]
    
    def test_is_consistent(self, sample_numeric_df):
        """Test consistency check method."""
        extractor = SignalExtractor(sample_numeric_df)
        signal_df, _ = extractor.extract_all()
        
        ahp = AHPWeighting(signal_df, top_n=5)
        ahp.compute_weights()
        
        # Should return boolean
        is_consistent = ahp.is_consistent()
        assert isinstance(is_consistent, (bool, np.bool_))


class TestConfidenceIntervals:
    """Test bootstrap-based confidence intervals."""
    
    def test_ci_bounds(self, sample_numeric_df):
        """Test that CI are properly bounded."""
        extractor = SignalExtractor(sample_numeric_df)
        signal_df, _ = extractor.extract_all()
        
        base_scores = np.random.uniform(0, 1, len(signal_df))
        
        ci = ConfidenceIntervals(signal_df, n_bootstrap=100)
        lower, upper = ci.estimate_ci(base_scores)
        
        # Lower should be < upper
        assert (lower <= upper).all()
        
        # CI should be in [0, 1]
        assert (lower >= 0).all()
        assert (upper <= 1).all()
    
    def test_ci_width(self, sample_numeric_df):
        """Test CI width calculation."""
        extractor = SignalExtractor(sample_numeric_df)
        signal_df, _ = extractor.extract_all()
        
        base_scores = np.random.uniform(0, 1, len(signal_df))
        
        ci = ConfidenceIntervals(signal_df, n_bootstrap=100)
        ci.estimate_ci(base_scores)
        width = ci.get_ci_width()
        
        # Width should be non-negative
        assert (width >= 0).all()
        
        # Width should be reasonable (not larger than [0,1])
        assert (width <= 1).all()
    
    def test_bootstrap_sample_count(self, sample_numeric_df):
        """Test that correct number of bootstrap samples created."""
        extractor = SignalExtractor(sample_numeric_df)
        signal_df, _ = extractor.extract_all()
        
        n_bootstrap = 50
        base_scores = np.random.uniform(0, 1, len(signal_df))
        
        ci = ConfidenceIntervals(signal_df, n_bootstrap=n_bootstrap)
        ci.estimate_ci(base_scores)
        
        assert ci.bootstrap_scores.shape == (len(signal_df), n_bootstrap)


class TestRankingEngine:
    """Test end-to-end ranking pipeline."""
    
    def test_ranking_pipeline(self, sample_mixed_df):
        """Test complete ranking pipeline."""
        engine = RankingEngine(sample_mixed_df)
        result = engine.rank(top_n=10)
        
        # Check outputs
        assert result.topsis_scores.shape[0] == len(sample_mixed_df)
        assert result.combined_scores.shape[0] == len(sample_mixed_df)
        assert len(result.rankings) <= 10
        
        # Rankings should be sorted by score
        scores = [score for _, score, _, _ in result.rankings]
        assert scores == sorted(scores, reverse=True)
    
    def test_ranking_with_weights(self, sample_mixed_df):
        """Test ranking with different weight configurations."""
        engine = RankingEngine(sample_mixed_df)
        
        # Pure TOPSIS
        result1 = engine.rank(topsis_weight=1.0, ahp_weight=0.0, top_n=5)
        
        # Pure AHP-weighted
        result2 = engine.rank(topsis_weight=0.0, ahp_weight=1.0, top_n=5)
        
        # Mixed
        result3 = engine.rank(topsis_weight=0.6, ahp_weight=0.4, top_n=5)
        
        # All should produce valid results
        assert len(result1.rankings) > 0
        assert len(result2.rankings) > 0
        assert len(result3.rankings) > 0
    
    def test_ranking_to_dict(self, sample_mixed_df):
        """Test serialization to dict."""
        engine = RankingEngine(sample_mixed_df)
        result = engine.rank(top_n=5)
        
        result_dict = result.to_dict()
        
        # Check structure
        assert "rankings" in result_dict
        assert "signal_count" in result_dict
        assert "signal_info" in result_dict
        assert "ahp_consistency_ratio" in result_dict
        assert "statistics" in result_dict
        
        # Check rankings structure
        assert len(result_dict["rankings"]) > 0
        for ranking in result_dict["rankings"]:
            assert "rank" in ranking
            assert "index" in ranking
            assert "score" in ranking
            assert "confidence_lower" in ranking
            assert "confidence_upper" in ranking
    
    def test_confidence_intervals_in_result(self, sample_mixed_df):
        """Test that confidence intervals are included in result."""
        engine = RankingEngine(sample_mixed_df)
        result = engine.rank(top_n=5)
        
        # Each ranking should have CI
        # Note: Bootstrap CI may not tightly bound the original score due to resampling
        for idx, score, lower, upper in result.rankings:
            # Just verify CI bounds are sensible
            assert lower <= upper + 1e-6  # Allow small numerical tolerance
            assert lower >= 0 and upper <= 1
    
    def test_large_dataset(self):
        """Test ranking with large dataset (performance check)."""
        np.random.seed(42)
        n = 5000
        
        large_df = pd.DataFrame({
            "feature_1": np.random.normal(100, 20, n),
            "feature_2": np.random.exponential(50, n),
            "feature_3": np.random.choice(["A", "B", "C"], n),
            "feature_4": [
                datetime.now() - timedelta(days=int(x))
                for x in np.random.exponential(365, n)
            ],
        })
        
        engine = RankingEngine(large_df)
        result = engine.rank(top_n=20)
        
        # Should complete successfully
        assert len(result.rankings) > 0
        # Scores should be valid
        assert (result.combined_scores >= 0).all()
        assert (result.combined_scores <= 1).all()


class TestComplexity:
    """Test algorithmic complexity claims."""
    
    def test_topsis_complexity_linear(self):
        """Test that TOPSIS is O(n×m) - linear in both."""
        import time
        
        # Small dataset
        df_small = pd.DataFrame({
            f"signal_{i}": np.random.uniform(0, 1, 100)
            for i in range(10)
        })
        
        # Large dataset (10x in rows, same signals)
        df_large = pd.DataFrame({
            f"signal_{i}": np.random.uniform(0, 1, 1000)
            for i in range(10)
        })
        
        ranker_small = TopsisRanker(df_small)
        ranker_large = TopsisRanker(df_large)
        
        start = time.time()
        ranker_small.score()
        time_small = time.time() - start
        
        start = time.time()
        ranker_large.score()
        time_large = time.time() - start
        
        # Should be roughly 10x slower for 10x more rows
        assert time_large < time_small * 20  # Allow some overhead


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
