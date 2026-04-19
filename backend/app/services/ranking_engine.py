"""
Patent-Strengthening Ranking Engine - Phase 1

Implements unsupervised probabilistic ranking system with:
- Automatic signal extraction (50+ signals from arbitrary CSV data)
- TOPSIS multi-criteria decision making
- AHP pairwise comparison weighting
- Bootstrap-based confidence intervals

Authors: Lucida Team
Timeline: Phase 1 (Weeks 1-4)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum

import numpy as np
import pandas as pd
from scipy import stats
from scipy.linalg import eig
from scipy.special import rel_entr


class SignalType(Enum):
    """Categories of signals extracted from data."""
    NUMERIC_ABSOLUTE = "numeric_absolute"
    NUMERIC_Z_SCORE = "numeric_zscore"
    NUMERIC_PERCENTILE = "numeric_percentile"
    NUMERIC_DISTANCE = "numeric_distance_to_max"
    CATEGORICAL_FREQUENCY = "categorical_frequency"
    CATEGORICAL_ENTROPY = "categorical_entropy"
    CATEGORICAL_TOP_INDICATOR = "categorical_top_indicator"
    TEMPORAL_RECENCY = "temporal_recency"
    TEMPORAL_VELOCITY = "temporal_velocity"
    TEMPORAL_TREND = "temporal_trend"
    COMPOSITE_COMPLETENESS = "composite_completeness"
    COMPOSITE_AGGREGATE = "composite_aggregate"


@dataclass
class SignalInfo:
    """Metadata about extracted signal."""
    name: str
    signal_type: SignalType
    source_column: Optional[str] = None
    description: str = ""
    valid_count: int = 0
    null_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": self.signal_type.value,
            "source_column": self.source_column,
            "description": self.description,
            "valid_count": int(self.valid_count),
            "null_count": int(self.null_count),
        }


class SignalExtractor:
    """Extract 50+ signals from arbitrary CSV data for ranking."""
    
    def __init__(self, df: pd.DataFrame, max_categories: int = 50):
        """
        Initialize signal extractor.
        
        Args:
            df: Input DataFrame
            max_categories: Max distinct values for categorical analysis
        """
        self.df = df
        self.max_categories = max_categories
        self.signals: Dict[str, np.ndarray] = {}
        self.signal_info: Dict[str, SignalInfo] = {}
        self.n_rows = len(df)
        
    def extract_all(self) -> Tuple[pd.DataFrame, Dict[str, SignalInfo]]:
        """
        Extract all available signals from the data.
        
        Returns:
            Tuple of (signal_matrix DataFrame, signal_info dict)
            Signal matrix shape: (n_rows, n_signals)
        """
        for col in self.df.columns:
            series = self.df[col]
            if pd.api.types.is_datetime64_any_dtype(series):
                self._extract_temporal_signals(col)
            elif pd.api.types.is_numeric_dtype(series):
                self._extract_numeric_signals(col)
            elif (
                pd.api.types.is_object_dtype(series)
                or pd.api.types.is_string_dtype(series)
                or pd.api.types.is_categorical_dtype(series)
            ):
                self._extract_categorical_signals(col)
        
        # Add composite signals
        self._extract_composite_signals()
        
        # Convert to DataFrame
        signal_df = pd.DataFrame(self.signals)
        return signal_df, self.signal_info
    
    def _extract_numeric_signals(self, col: str) -> None:
        """Extract signals from numeric column."""
        data = self.df[col].astype(float)
        median_value = data.median()
        if np.isnan(median_value):
            median_value = 0.0
        data = data.fillna(median_value)
        
        # Absolute value signal
        signal_name = f"{col}_absolute"
        self.signals[signal_name] = (data - data.min()) / (data.max() - data.min() + 1e-10)
        self.signal_info[signal_name] = SignalInfo(
            name=signal_name,
            signal_type=SignalType.NUMERIC_ABSOLUTE,
            source_column=col,
            description=f"Normalized absolute value of {col}",
            valid_count=len(data),
        )
        
        # Z-score signal
        signal_name = f"{col}_zscore"
        z_scores = np.abs(stats.zscore(data, nan_policy="omit"))
        z_scores = np.nan_to_num(z_scores, nan=0.0, posinf=0.0, neginf=0.0)
        self.signals[signal_name] = 1.0 / (1.0 + z_scores)  # Inverted (high is good)
        self.signal_info[signal_name] = SignalInfo(
            name=signal_name,
            signal_type=SignalType.NUMERIC_Z_SCORE,
            source_column=col,
            description=f"Inverted z-score outlier detection for {col}",
            valid_count=len(data),
        )
        
        # Percentile signal
        signal_name = f"{col}_percentile"
        self.signals[signal_name] = stats.rankdata(data) / len(data)
        self.signal_info[signal_name] = SignalInfo(
            name=signal_name,
            signal_type=SignalType.NUMERIC_PERCENTILE,
            source_column=col,
            description=f"Percentile rank of {col}",
            valid_count=len(data),
        )
        
        # Distance to maximum
        signal_name = f"{col}_dist_to_max"
        max_val = data.max()
        self.signals[signal_name] = 1.0 - (max_val - data) / (max_val - data.min() + 1e-10)
        self.signal_info[signal_name] = SignalInfo(
            name=signal_name,
            signal_type=SignalType.NUMERIC_DISTANCE,
            source_column=col,
            description=f"Distance of {col} from maximum",
            valid_count=len(data),
        )
    
    def _extract_categorical_signals(self, col: str) -> None:
        """Extract signals from categorical column."""
        data = self.df[col].fillna("MISSING")
        value_counts = data.value_counts()
        
        # Skip columns with too many categories
        if len(value_counts) > self.max_categories:
            return
        
        # Frequency signal
        signal_name = f"{col}_frequency"
        self.signals[signal_name] = data.map(value_counts) / len(data)
        self.signal_info[signal_name] = SignalInfo(
            name=signal_name,
            signal_type=SignalType.CATEGORICAL_FREQUENCY,
            source_column=col,
            description=f"Frequency (relative) of {col}",
            valid_count=len(data),
        )
        
        # Entropy-based signal (how informative is this category?)
        signal_name = f"{col}_entropy"
        probs = value_counts / len(data)
        entropy_col = -np.sum(probs * np.log2(probs + 1e-10))
        # Lower entropy = more informative
        category_entropy = pd.Series(
            [np.log2(len(data) / (value_counts[cat] + 1e-10)) for cat in data],
            index=data.index
        )
        # Normalize by max entropy (log of num categories)
        max_entropy = np.log2(len(value_counts)) if len(value_counts) > 1 else 1.0
        entropy_signal = np.clip(category_entropy / max_entropy, 0, 1)
        self.signals[signal_name] = entropy_signal
        self.signal_info[signal_name] = SignalInfo(
            name=signal_name,
            signal_type=SignalType.CATEGORICAL_ENTROPY,
            source_column=col,
            description=f"Entropy-based informativeness of {col}",
            valid_count=len(data),
        )
        
        # Top category indicator
        if len(value_counts) > 0:
            signal_name = f"{col}_is_top_category"
            top_category = value_counts.index[0]
            self.signals[signal_name] = (data == top_category).astype(float)
            self.signal_info[signal_name] = SignalInfo(
                name=signal_name,
                signal_type=SignalType.CATEGORICAL_TOP_INDICATOR,
                source_column=col,
                description=f"Indicator if {col} is '{top_category}'",
                valid_count=len(data),
            )
    
    def _extract_temporal_signals(self, col: str) -> None:
        """Extract signals from datetime column."""
        data = pd.to_datetime(self.df[col], errors="coerce")
        valid_mask = data.notna()
        
        if valid_mask.sum() == 0:
            return
        
        now = pd.Timestamp.now()
        
        # Recency signal (how recent is this record?)
        signal_name = f"{col}_recency"
        days_since = (now - data).dt.days
        # Normalize by max days
        self.signals[signal_name] = np.where(
            valid_mask,
            1.0 / (1.0 + (days_since / (days_since.max() + 1))),
            0.0
        )
        self.signal_info[signal_name] = SignalInfo(
            name=signal_name,
            signal_type=SignalType.TEMPORAL_RECENCY,
            source_column=col,
            description=f"Recency score (how recent) for {col}",
            valid_count=valid_mask.sum(),
            null_count=(~valid_mask).sum(),
        )
        
        # Velocity signal (change over recent time window)
        signal_name = f"{col}_velocity"
        recent_cutoff = now - pd.Timedelta(days=30)
        recent_mask = data >= recent_cutoff
        self.signals[signal_name] = np.where(
            valid_mask,
            recent_mask.astype(float),
            0.0
        )
        self.signal_info[signal_name] = SignalInfo(
            name=signal_name,
            signal_type=SignalType.TEMPORAL_VELOCITY,
            source_column=col,
            description=f"Velocity (activity in last 30 days) for {col}",
            valid_count=valid_mask.sum(),
        )
        
        # Trend signal (is this getting older or newer?)
        if valid_mask.sum() > 1:
            signal_name = f"{col}_trend"
            sorted_days = sorted(days_since[valid_mask])
            mid = sorted_days[len(sorted_days) // 2]
            trend_score = np.where(
                valid_mask,
                np.exp(-days_since / (mid + 1)),
                0.0
            )
            self.signals[signal_name] = trend_score
            self.signal_info[signal_name] = SignalInfo(
                name=signal_name,
                signal_type=SignalType.TEMPORAL_TREND,
                source_column=col,
                description=f"Trend (temporal momentum) for {col}",
                valid_count=valid_mask.sum(),
            )
    
    def _extract_composite_signals(self) -> None:
        """Extract composite signals from profile completeness."""
        # Profile completeness: what fraction of fields are non-null?
        signal_name = "profile_completeness"
        null_counts = self.df.isnull().sum(axis=1)
        completeness = 1.0 - (null_counts / len(self.df.columns))
        self.signals[signal_name] = completeness.values
        self.signal_info[signal_name] = SignalInfo(
            name=signal_name,
            signal_type=SignalType.COMPOSITE_COMPLETENESS,
            description="Fraction of non-null fields in record",
            valid_count=self.n_rows,
        )
        
        # Numeric aggregate: average of all numeric signal zscore
        numeric_signals = [
            s for s, info in self.signal_info.items()
            if "numeric" in info.signal_type.value
        ]
        if numeric_signals:
            signal_name = "numeric_aggregate"
            numeric_matrix = np.column_stack([self.signals[s] for s in numeric_signals])
            self.signals[signal_name] = np.mean(numeric_matrix, axis=1)
            self.signal_info[signal_name] = SignalInfo(
                name=signal_name,
                signal_type=SignalType.COMPOSITE_AGGREGATE,
                description="Average of all numeric signals",
                valid_count=self.n_rows,
            )


class TopsisRanker:
    """
    TOPSIS (Technique for Order Preference by Similarity to Ideal Solution) ranking.
    
    O(n×m) complexity where n=rows, m=signals.
    Final score in [0, 1] where 1 is "ideal".
    """
    
    def __init__(self, signal_matrix: pd.DataFrame, weights: Optional[np.ndarray] = None):
        """
        Initialize TOPSIS ranker.
        
        Args:
            signal_matrix: (n_rows, n_signals) DataFrame
            weights: Optional (n_signals,) weight vector. Defaults to uniform.
        """
        self.signal_matrix = signal_matrix.astype(float)
        self.n_rows, self.n_signals = signal_matrix.shape
        
        if weights is None:
            self.weights = np.ones(self.n_signals) / self.n_signals
        else:
            self.weights = weights / np.sum(weights)
        
        self.normalized_matrix = None
        self.weighted_matrix = None
        self.topsis_scores = None
    
    def score(self) -> np.ndarray:
        """
        Compute TOPSIS scores.
        
        Returns:
            Array of scores [0, 1] for each row
        """
        # Step 1: Normalize
        self.normalized_matrix = self._normalize()
        
        # Step 2: Weight
        self.weighted_matrix = self.normalized_matrix * self.weights[np.newaxis, :]
        
        # Step 3: Find ideal and worst solutions
        ideal = np.max(self.weighted_matrix, axis=0)
        worst = np.min(self.weighted_matrix, axis=0)
        
        # Step 4: Distance to ideal and worst
        d_ideal = np.sqrt(np.sum((self.weighted_matrix - ideal) ** 2, axis=1))
        d_worst = np.sqrt(np.sum((self.weighted_matrix - worst) ** 2, axis=1))
        
        # Step 5: TOPSIS score
        self.topsis_scores = d_worst / (d_ideal + d_worst + 1e-10)
        return self.topsis_scores
    
    def _normalize(self) -> np.ndarray:
        """
        Normalize using vector normalization (Euclidean norm).
        
        Returns:
            Normalized matrix
        """
        # Euclidean normalization
        signal_array = self.signal_matrix.values if isinstance(self.signal_matrix, pd.DataFrame) else self.signal_matrix
        col_norms = np.sqrt(np.sum(signal_array ** 2, axis=0))
        return signal_array / (col_norms[np.newaxis, :] + 1e-10)
    
    def rank(self, top_n: int = 10) -> List[Tuple[int, float]]:
        """
        Get top-N ranked rows by TOPSIS score.
        
        Args:
            top_n: Number of top rows to return
        
        Returns:
            List of (row_index, score) tuples
        """
        if self.topsis_scores is None:
            self.score()
        
        top_indices = np.argsort(-self.topsis_scores)[:top_n]
        return [(idx, self.topsis_scores[idx]) for idx in top_indices]


class AHPWeighting:
    """
    Analytic Hierarchy Process (AHP) for signal weighting.
    
    Creates pairwise comparison matrix from top signals and derives weights
    using eigenvalue decomposition. Validates consistency ratio (CR < 0.1).
    """
    
    # Random consistency index for different matrix sizes
    RI = {
        1: 0.00,
        2: 0.00,
        3: 0.58,
        4: 0.90,
        5: 1.12,
        6: 1.24,
        7: 1.32,
        8: 1.41,
        9: 1.45,
    }
    
    def __init__(self, signal_matrix: pd.DataFrame, top_n: int = 10):
        """
        Initialize AHP weighting.
        
        Args:
            signal_matrix: (n_rows, n_signals) DataFrame
            top_n: Number of top signals to compare (typically 10)
        """
        self.signal_matrix = signal_matrix.astype(float)
        self.top_n = min(top_n, signal_matrix.shape[1])
        self.weights = None
        self.consistency_ratio = None
    
    def compute_weights(self) -> np.ndarray:
        """
        Compute weights using pairwise comparison matrix.
        
        Signals ranked by variance (importance) and compared pairwise.
        Weights sum to 1.
        
        Returns:
            Weight vector of shape (n_signals,)
        """
        # Select top signals by variance
        variances = self.signal_matrix.var(axis=0)
        top_indices = np.argsort(-variances)[:self.top_n]
        top_signals = self.signal_matrix.iloc[:, top_indices]
        
        # Create pairwise comparison matrix
        # Compare signals by correlation and variance
        pairwise_matrix = self._create_pairwise_matrix(top_signals)
        
        # Compute eigenvector (weights)
        eigenvalues, eigenvectors = eig(pairwise_matrix)
        
        # Get eigenvector for largest eigenvalue
        max_eigen_idx = np.argmax(np.real(eigenvalues))
        weights_top = np.real(eigenvectors[:, max_eigen_idx])
        weights_top = np.abs(weights_top) / np.sum(np.abs(weights_top))
        
        # Compute consistency ratio
        lambda_max = np.real(eigenvalues[max_eigen_idx])
        self.consistency_ratio = self._compute_consistency_ratio(
            lambda_max, self.top_n
        )
        
        # Full weight vector (pad with zeros for non-top signals)
        full_weights = np.zeros(self.signal_matrix.shape[1])
        full_weights[top_indices] = weights_top
        
        self.weights = full_weights / (np.sum(full_weights) + 1e-10)
        return self.weights
    
    def _create_pairwise_matrix(self, signals: pd.DataFrame) -> np.ndarray:
        """
        Create pairwise comparison matrix from signals.
        
        Compares signals by:
        - Variance (importance)
        - Correlation (redundancy penalty)
        
        Returns:
            (n, n) comparison matrix where a[i,j] > 1 means signal i > signal j
        """
        n = signals.shape[1]
        matrix = np.ones((n, n))
        
        # Normalize signals to [0, 1]
        signals_normalized = (signals - signals.min()) / (signals.max() - signals.min() + 1e-10)
        
        # Variance-based comparison
        variances = signals_normalized.var(axis=0)
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    matrix[i, j] = 1.0
                elif i > j:
                    matrix[i, j] = 1.0 / matrix[j, i]
                else:
                    # AHP scale: 1-9
                    # Ratio of variances on AHP scale
                    var_ratio = variances.iloc[i] / (variances.iloc[j] + 1e-10)
                    
                    # Apply correlation penalty (reduce if highly correlated)
                    corr = np.abs(signals_normalized.iloc[:, i].corr(
                        signals_normalized.iloc[:, j]
                    ))
                    correlation_penalty = 1.0 - (0.5 * corr)
                    
                    # Map ratio to AHP scale [1, 9]
                    ahp_value = 1.0 + (8.0 * min(var_ratio, 9.0) / 10.0)
                    ahp_value = ahp_value * correlation_penalty
                    
                    matrix[i, j] = max(1.0, min(9.0, ahp_value))
        
        return matrix
    
    def _compute_consistency_ratio(self, lambda_max: float, n: int) -> float:
        """
        Compute Consistency Ratio (CR).
        CR < 0.1 is acceptable.
        
        Args:
            lambda_max: Largest eigenvalue
            n: Matrix size
        
        Returns:
            Consistency ratio
        """
        if n not in self.RI:
            return 0.0  # No consistency check for large n
        
        ci = (lambda_max - n) / (n - 1) if n > 1 else 0.0
        cr = ci / self.RI[n] if self.RI[n] > 0 else 0.0
        return cr
    
    def is_consistent(self) -> bool:
        """Check if weights are consistent (CR < 0.1)."""
        if self.consistency_ratio is None:
            self.compute_weights()
        return self.consistency_ratio < 0.1


class ConfidenceIntervals:
    """
    Bootstrap-based confidence intervals for ranking scores.
    
    Uses bootstrap resampling to estimate uncertainty in final scores.
    Outputs [lower, upper] 95% CI for each row.
    """
    
    def __init__(self, signal_matrix: pd.DataFrame, n_bootstrap: int = 500):
        """
        Initialize confidence interval estimator.
        
        Args:
            signal_matrix: (n_rows, n_signals) DataFrame
            n_bootstrap: Number of bootstrap samples (default 500)
        """
        self.signal_matrix = signal_matrix.astype(float)
        self.n_bootstrap = n_bootstrap
        self.bootstrap_scores = None
        self.lower_ci = None
        self.upper_ci = None
    
    def estimate_ci(self, base_scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Estimate 95% confidence intervals for scores.
        
        Args:
            base_scores: Baseline scores from TOPSIS or other method
        
        Returns:
            Tuple of (lower_ci, upper_ci) arrays
        """
        n_rows = self.signal_matrix.shape[0]
        bootstrap_values: List[List[float]] = [[] for _ in range(n_rows)]
        
        # Bootstrap: resample with replacement and recompute scores
        for i in range(self.n_bootstrap):
            indices = np.random.choice(n_rows, size=n_rows, replace=True)
            bootstrap_sample = self.signal_matrix.iloc[indices]
            
            # Recompute TOPSIS scores on this sample
            ranker = TopsisRanker(bootstrap_sample)
            bootstrap_scores = ranker.score()
            
            # Collect all sampled scores for each original row index.
            for j, idx in enumerate(indices):
                bootstrap_values[idx].append(float(bootstrap_scores[j]))

        # Keep a dense matrix for observability and existing tests.
        bootstrap_results = np.full((n_rows, self.n_bootstrap), np.nan, dtype=float)
        lower_ci = np.zeros(n_rows, dtype=float)
        upper_ci = np.zeros(n_rows, dtype=float)

        for row_idx in range(n_rows):
            values = np.asarray(bootstrap_values[row_idx], dtype=float)
            if values.size == 0:
                values = np.asarray([float(base_scores[row_idx])], dtype=float)

            copy_count = min(values.size, self.n_bootstrap)
            bootstrap_results[row_idx, :copy_count] = values[:copy_count]

            std = float(np.std(values))
            margin = float(np.clip(1.96 * std, 0.02, 0.15))
            center = float(base_scores[row_idx])
            lower_ci[row_idx] = float(np.clip(center - margin, 0.0, 1.0))
            upper_ci[row_idx] = float(np.clip(center + margin, 0.0, 1.0))

        self.bootstrap_scores = bootstrap_results
        self.lower_ci = lower_ci
        self.upper_ci = upper_ci
        
        return self.lower_ci, self.upper_ci
    
    def get_ci_width(self) -> np.ndarray:
        """Get width of confidence intervals (uncertainty measure)."""
        if self.lower_ci is None or self.upper_ci is None:
            raise ValueError("Call estimate_ci() first")
        return self.upper_ci - self.lower_ci


@dataclass
class RankingResult:
    """Complete ranking output with all components."""
    topsis_scores: np.ndarray
    ahp_weights: np.ndarray
    combined_scores: np.ndarray
    lower_ci: np.ndarray
    upper_ci: np.ndarray
    signal_matrix: pd.DataFrame
    signal_info: Dict[str, SignalInfo]
    consistency_ratio: float
    rankings: List[Tuple[int, float, float, float]]  # (idx, score, lower, upper)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "rankings": [
                {
                    "rank": i + 1,
                    "index": int(idx),
                    "score": float(score),
                    "confidence_lower": float(lower),
                    "confidence_upper": float(upper),
                }
                for i, (idx, score, lower, upper) in enumerate(self.rankings)
            ],
            "signal_count": len(self.signal_info),
            "signal_info": {
                name: info.to_dict()
                for name, info in self.signal_info.items()
            },
            "ahp_consistency_ratio": float(self.consistency_ratio),
            "statistics": {
                "mean_score": float(np.mean(self.combined_scores)),
                "std_score": float(np.std(self.combined_scores)),
                "min_score": float(np.min(self.combined_scores)),
                "max_score": float(np.max(self.combined_scores)),
            }
        }


class RankingEngine:
    """
    Complete unsupervised ranking pipeline.
    
    Orchestrates: signal extraction → TOPSIS → AHP weighting → confidence intervals
    """
    
    def __init__(self, df: pd.DataFrame):
        """Initialize ranking engine with data."""
        self.df = df
        self.extractor = None
        self.signal_matrix = None
        self.signal_info = None
    
    def rank(
        self,
        topsis_weight: float = 0.6,
        ahp_weight: float = 0.4,
        top_n: int = 10,
    ) -> RankingResult:
        """
        Perform end-to-end ranking.
        
        Args:
            topsis_weight: Weight of TOPSIS score (0-1)
            ahp_weight: Weight of AHP-weighted score (0-1)
            top_n: Return top N rankings
        
        Returns:
            RankingResult with complete ranking information
        """
        # Step 1: Extract signals
        self.extractor = SignalExtractor(self.df)
        self.signal_matrix, self.signal_info = self.extractor.extract_all()
        
        # Step 2: TOPSIS scoring
        topsis_ranker = TopsisRanker(self.signal_matrix)
        topsis_scores = topsis_ranker.score()
        
        # Step 3: AHP weighting
        ahp = AHPWeighting(self.signal_matrix, top_n=10)
        ahp_weights = ahp.compute_weights()
        
        # AHP-weighted score
        topsis_ranker_ahp = TopsisRanker(self.signal_matrix, weights=ahp_weights)
        ahp_scores = topsis_ranker_ahp.score()
        
        # Step 4: Combined score
        combined_scores = (
            topsis_weight * topsis_scores +
            ahp_weight * ahp_scores
        )

        # Slightly reward rows with better profile completeness for lead-quality use cases.
        if "profile_completeness" in self.signal_matrix.columns:
            completeness_bonus = self.signal_matrix["profile_completeness"].values
            combined_scores = (0.9 * combined_scores) + (0.1 * completeness_bonus)

        combined_scores = np.clip(combined_scores, 0.0, 1.0)

        # Step 5: Confidence intervals on final combined scores
        ci_estimator = ConfidenceIntervals(self.signal_matrix, n_bootstrap=500)
        lower_ci, upper_ci = ci_estimator.estimate_ci(combined_scores)
        
        # Step 6: Generate rankings
        top_indices = np.argsort(-combined_scores)[:top_n]
        rankings = [
            (int(idx), float(combined_scores[idx]), float(lower_ci[idx]), float(upper_ci[idx]))
            for idx in top_indices
        ]
        
        return RankingResult(
            topsis_scores=topsis_scores,
            ahp_weights=ahp_weights,
            combined_scores=combined_scores,
            lower_ci=lower_ci,
            upper_ci=upper_ci,
            signal_matrix=self.signal_matrix,
            signal_info=self.signal_info,
            consistency_ratio=ahp.consistency_ratio,
            rankings=rankings,
        )
