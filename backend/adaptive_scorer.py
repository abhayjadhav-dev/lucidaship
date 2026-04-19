# adaptive_scorer.py - Complete Working Implementation

"""
Universal Adaptive Lead Scorer
- Zero hard-coded keywords
- Works with ANY CSV structure
- Learns feature importance from data
- No configuration needed
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
os.environ.setdefault("JOBLIB_MULTIPROCESSING", "0")

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, brier_score_loss
from sklearn.calibration import CalibratedClassifierCV
from imblearn.over_sampling import ADASYN, SMOTE
from scipy.stats import spearmanr
import joblib

try:
    from sklearn.frozen import FrozenEstimator
except Exception:
    FrozenEstimator = None

try:
    from xgboost import XGBClassifier
except:
    XGBClassifier = None

try:
    import shap
except:
    shap = None

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.decomposition import PCA
except:
    SentenceTransformer = None


RF_N_JOBS = max(1, int(os.getenv("LUCIDA_RF_N_JOBS", "1")))
NULL_LIKE_TOKENS = {"", "nan", "none", "null", "n/a", "na"}
POSITIVE_BINARY_TOKENS = {"1", "yes", "true", "won", "converted", "y", "success"}


def _safe_to_datetime(series: pd.Series) -> pd.Series:
    """Use pandas mixed parsing to avoid noisy format inference warnings."""
    return pd.to_datetime(series, errors='coerce', format='mixed')


def _normalize_binary_token(value) -> Optional[str]:
    """Normalize raw values so binary detection is robust to nulls/whitespace/mixed types."""
    if pd.isna(value):
        return None

    if isinstance(value, (bool, np.bool_)):
        return "1" if bool(value) else "0"

    if isinstance(value, (int, np.integer)):
        return str(int(value))

    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            return None
        rounded = round(float(value))
        if abs(float(value) - rounded) < 1e-9:
            return str(int(rounded))
        return f"{float(value):g}"

    token = str(value).strip().lower()
    if token in NULL_LIKE_TOKENS:
        return None

    try:
        numeric_value = float(token)
        if np.isfinite(numeric_value):
            rounded = round(numeric_value)
            if abs(numeric_value - rounded) < 1e-9:
                return str(int(rounded))
            return f"{numeric_value:g}"
    except Exception:
        pass

    return token


def _normalized_binary_series(series: pd.Series) -> pd.Series:
    normalized = series.map(_normalize_binary_token)
    return normalized.dropna()


def _is_binary_series(series: pd.Series) -> bool:
    non_null = _normalized_binary_series(series)
    if non_null.empty:
        return False
    return bool(non_null.nunique() == 2)


# ═══════════════════════════════════════════════════════════════════════════════
#  LAYER 1: INTELLIGENT DATA ANALYSIS (NO KEYWORDS!)
# ═══════════════════════════════════════════════════════════════════════════════

class DataAnalyzer:
    """
    Analyze ANY dataset without keywords.
    Learn: column types, target variable, feature importance.
    """

    def __init__(self, df: pd.DataFrame, target_col: Optional[str] = None):
        self.df = df.copy()
        self.target_col = target_col
        self.column_types = {}
        self.feature_importance = {}
        self.is_binary_target = False
        self.binary_encoders = {}  # maps col -> {val: 0/1}
        self.target_scores = {}
        self.target_diagnostics = {}
        self.imputation_stats = {}

    def _encode_binary(self, col: str) -> pd.Series:
        """
        Safely convert any binary column (Yes/No, True/False, 0/1, Won/Lost)
        to numeric 0/1. Stores the mapping for reuse.
        """
        normalized = self.df[col].map(_normalize_binary_token)
        non_null = normalized.dropna()
        unique_vals = sorted(non_null.unique())
        if len(unique_vals) != 2:
            raise ValueError(
                f"Column '{col}' is not binary after normalization "
                f"(found {len(unique_vals)} unique non-null values)."
            )

        # Heuristic: common positive values map to 1.
        if unique_vals[0] in POSITIVE_BINARY_TOKENS and unique_vals[1] not in POSITIVE_BINARY_TOKENS:
            mapping = {unique_vals[0]: 1, unique_vals[1]: 0}
        elif unique_vals[1] in POSITIVE_BINARY_TOKENS and unique_vals[0] not in POSITIVE_BINARY_TOKENS:
            mapping = {unique_vals[0]: 0, unique_vals[1]: 1}
        else:
            mapping = {unique_vals[0]: 0, unique_vals[1]: 1}  # deterministic fallback

        self.binary_encoders[col] = mapping
        encoded = normalized.map(mapping)
        if encoded.isna().all():
            return pd.Series(0, index=self.df.index, dtype=int)
        fill_value = int(encoded.dropna().mode().iloc[0]) if not encoded.dropna().empty else 0
        return encoded.fillna(fill_value).astype(int)

    def encode_binary_series(self, col: str, series: pd.Series) -> pd.Series:
        """Encode a binary series using stored mapping from training where possible."""
        normalized = series.map(_normalize_binary_token)
        mapping = self.binary_encoders.get(col)
        if mapping is None:
            unique_vals = sorted(normalized.dropna().unique())
            if len(unique_vals) != 2:
                raise ValueError(f"Column '{col}' is not binary in provided series.")
            if unique_vals[0] in POSITIVE_BINARY_TOKENS and unique_vals[1] not in POSITIVE_BINARY_TOKENS:
                mapping = {unique_vals[0]: 1, unique_vals[1]: 0}
            elif unique_vals[1] in POSITIVE_BINARY_TOKENS and unique_vals[0] not in POSITIVE_BINARY_TOKENS:
                mapping = {unique_vals[0]: 0, unique_vals[1]: 1}
            else:
                mapping = {unique_vals[0]: 0, unique_vals[1]: 1}

        encoded = normalized.map(mapping)
        if encoded.isna().all():
            return pd.Series(0, index=series.index, dtype=int)
        fill_value = int(encoded.dropna().mode().iloc[0]) if not encoded.dropna().empty else 0
        return encoded.fillna(fill_value).astype(int)

    def infer_column_types(self) -> Dict[str, str]:
        """
        Infer column type by analyzing content, not name.
        
        Returns: {col_name: type}
        Types: numeric, categorical, binary, text, temporal, id, ignore
        """
        types = {}

        for col in self.df.columns:
            col_data = self.df[col]
            null_ratio = col_data.isnull().sum() / len(col_data)

            # Skip mostly-null columns
            if null_ratio > 0.9:
                types[col] = 'ignore'
                continue

            # Skip constant columns (all same value or empty)
            if col_data.nunique() <= 1:
                types[col] = 'ignore'
                continue

            if _is_binary_series(col_data):
                types[col] = 'binary'
                continue

            # Try datetime parsing
            if col_data.dtype == 'object':
                try:
                    parsed_dates = _safe_to_datetime(col_data.dropna())
                    valid_ratio = parsed_dates.notna().sum() / len(col_data.dropna())
                    if valid_ratio > 0.8:
                        types[col] = 'temporal'
                        continue
                except:
                    pass

            # Numeric columns
            if col_data.dtype in ['int64', 'float64']:
                unique_count = col_data.nunique()

                # Binary (exactly 2 values)
                if unique_count == 2:
                    types[col] = 'binary'
                # ID (nearly all unique)
                elif unique_count / len(col_data) > 0.95:
                    types[col] = 'id'
                else:
                    types[col] = 'numeric'
                continue

            # String columns
            if col_data.dtype == 'object':
                unique_count = col_data.nunique()
                avg_length = col_data.astype(str).str.len().mean()

                # ID (nearly all unique)
                if unique_count / len(col_data) > 0.95:
                    types[col] = 'id'
                # Binary categorical
                elif unique_count == 2:
                    types[col] = 'binary'
                # Text (high cardinality + long strings)
                elif avg_length > 20 and unique_count > 50:
                    types[col] = 'text'
                # Categorical (low cardinality)
                elif unique_count < 50:
                    types[col] = 'categorical'
                else:
                    types[col] = 'text'

        self.column_types = types
        return types

    def auto_detect_target(self) -> str:
        """
        Find target column (binary column with strongest predictive power).
        """
        if self.target_col and self.target_col in self.df.columns:
            self.target_scores = {self.target_col: 1.0}
            self.target_diagnostics = self.get_target_diagnostics()
            return self.target_col

        binary_cols = [
            col for col, type_ in self.column_types.items()
            if type_ == 'binary'
        ]

        if not binary_cols:
            synthetic_target = self._create_synthetic_target()
            self.target_scores = {synthetic_target: 0.0}
            self.target_col = synthetic_target
            self.is_binary_target = True
            self.target_diagnostics = self.get_target_diagnostics()
            return synthetic_target

        # Score each by correlation with numerics
        scores = {}
        for target in binary_cols:
            try:
                y = self._encode_binary(target)
            except:
                scores[target] = 0
                continue
            numeric_cols = [c for c, t in self.column_types.items() if t == 'numeric']

            if numeric_cols:
                correlations = [
                    abs(self.df[c].fillna(0).corr(y))
                    for c in numeric_cols
                ]
                scores[target] = np.mean(correlations) if correlations else 0
            else:
                scores[target] = 0.5  # No numeric cols, just pick

        target = max(scores, key=scores.get)
        self.target_scores = dict(sorted(scores.items(), key=lambda item: item[1], reverse=True))
        self.target_col = target
        self.is_binary_target = True
        self.target_diagnostics = self.get_target_diagnostics()
        return target

    def _create_synthetic_target(self) -> str:
        """Create a deterministic pseudo-target so supervised training can proceed."""
        target_name = "__synthetic_target__"

        numeric_candidates = [
            col for col, type_ in self.column_types.items()
            if type_ == "numeric" and self.df[col].dropna().nunique() > 1
        ]

        if numeric_candidates:
            best_col = max(
                numeric_candidates,
                key=lambda c: float(self.df[c].fillna(self.df[c].median()).std() or 0.0),
            )
            ranked = self.df[best_col].fillna(self.df[best_col].median()).rank(method="first", pct=True)
            self.df[target_name] = (ranked >= 0.5).astype(int)
        else:
            categorical_candidates = [
                col for col, type_ in self.column_types.items()
                if type_ in {"categorical", "text"} and self.df[col].dropna().nunique() > 1
            ]
            if categorical_candidates:
                best_col = max(categorical_candidates, key=lambda c: int(self.df[c].nunique(dropna=True)))
                normalized = self.df[best_col].fillna("unknown").astype(str).str.strip().str.lower()
                hashed = pd.util.hash_pandas_object(normalized, index=False).astype("int64")
                ranked = pd.Series(hashed, index=self.df.index).rank(method="first", pct=True)
                self.df[target_name] = (ranked >= 0.5).astype(int)
            else:
                # Last resort: stable balanced split.
                self.df[target_name] = (self.df.index.to_series().rank(method="first", pct=True) >= 0.5).astype(int)

        self.column_types[target_name] = "binary"
        return target_name

    def get_target_diagnostics(self) -> Dict:
        """Summarize how confident the engine is about target detection."""
        if not self.target_col:
            return {}

        ranked = [
            {"column": str(col), "score": float(score)}
            for col, score in sorted(self.target_scores.items(), key=lambda item: item[1], reverse=True)
        ]
        top_score = ranked[0]["score"] if ranked else 0.0
        runner_up_score = ranked[1]["score"] if len(ranked) > 1 else 0.0
        score_gap = max(top_score - runner_up_score, 0.0)

        encoded_target = self._encode_binary(self.target_col)
        class_counts = {
            str(label): int(count)
            for label, count in zip(*np.unique(encoded_target, return_counts=True))
        }
        minority_count = min(class_counts.values()) if class_counts else 0
        total_count = sum(class_counts.values()) if class_counts else 0
        minority_ratio = float(minority_count / total_count) if total_count else 0.0

        review_flags = []
        if top_score < 0.15:
            review_flags.append("weak_signal")
        if len(ranked) > 1 and score_gap < 0.05:
            review_flags.append("ambiguous_target")
        if minority_ratio < 0.1:
            review_flags.append("class_imbalance")

        recommendation = "strong_auto_target"
        if review_flags:
            recommendation = "manual_review_recommended"
        if self.target_col == "__synthetic_target__":
            recommendation = "synthetic_target_created"

        return {
            "selected_target": str(self.target_col),
            "candidate_count": int(len(ranked)),
            "ranked_candidates": ranked,
            "score_gap": float(score_gap),
            "minority_ratio": float(round(minority_ratio, 4)),
            "class_balance": class_counts,
            "review_flags": review_flags,
            "recommendation": recommendation,
            "target_source": "synthetic" if self.target_col == "__synthetic_target__" else "detected_or_provided",
        }

    def compute_feature_importance(self) -> Dict[str, float]:
        """
        Compute which columns predict target.
        - Numeric: Spearman correlation
        - Categorical: Mutual Information
        - Temporal: Recency correlation
        """
        if not self.target_col:
            self.auto_detect_target()

        y = self._encode_binary(self.target_col)
        importances = {}

        # Numeric: correlation
        for col, type_ in self.column_types.items():
            if type_ == 'numeric':
                try:
                    col_data = self.df[col]
                    # Skip constant columns (Spearman undefined)
                    if col_data.nunique() <= 1:
                        importances[col] = 0
                        continue
                    corr, _ = spearmanr(col_data.fillna(col_data.median()), y)
                    importances[col] = abs(corr) if not np.isnan(corr) else 0
                except:
                    importances[col] = 0

        # Categorical: mutual information
        for col, type_ in self.column_types.items():
            if type_ == 'categorical':
                try:
                    X = self.df[[col]].fillna('Unknown')
                    le = LabelEncoder()
                    X_encoded = le.fit_transform(X[col])
                    mi = mutual_info_classif(X_encoded.reshape(-1, 1), y, random_state=42)
                    importances[col] = mi[0]
                except:
                    importances[col] = 0

        # Temporal: recency
        for col, type_ in self.column_types.items():
            if type_ == 'temporal':
                try:
                    df_temp = self.df.copy()
                    df_temp[col] = _safe_to_datetime(df_temp[col])
                    today = pd.Timestamp.today()
                    days_ago = (today - df_temp[col]).dt.days
                    median_days = days_ago.dropna().median()
                    if pd.isna(median_days):
                        median_days = 30
                    recency = 1 / (1 + days_ago.fillna(median_days))
                    corr = recency.corr(y)
                    importances[col] = abs(corr) if not np.isnan(corr) else 0
                except:
                    importances[col] = 0

        # Normalize to 0-1
        if importances:
            max_imp = max(importances.values())
            if max_imp > 0:
                importances = {k: v / max_imp for k, v in importances.items()}

        self.feature_importance = importances
        return importances

    def filter_relevant_columns(self, threshold: float = 0.1) -> List[str]:
        """Keep only important columns."""
        if not self.feature_importance:
            self.compute_feature_importance()

        exclude_types = ['id', 'ignore']
        exclude_cols = [c for c, t in self.column_types.items() if t in exclude_types]

        relevant = [
            col for col, imp in self.feature_importance.items()
            if imp >= threshold and col not in exclude_cols and col != self.target_col
        ]

        return relevant

    def summary(self) -> Dict:
        """Return analysis summary."""
        return {
            'column_types': self.column_types,
            'target_column': self.target_col,
            'target_diagnostics': self.get_target_diagnostics(),
            'feature_importance': self.feature_importance,
            'relevant_columns': self.filter_relevant_columns(),
            'n_features': int(len(self.filter_relevant_columns())),
            'target_balance': {str(k): int(v) for k, v in self.df[self.target_col].value_counts().items()} if self.target_col else None
        }

    def compute_imputation_stats(self) -> Dict[str, Dict[str, object]]:
        """Compute default imputation values for each column type."""
        if not self.column_types:
            self.infer_column_types()

        stats: Dict[str, Dict[str, object]] = {}
        for col, col_type in self.column_types.items():
            if col == self.target_col:
                continue
            if col_type in {'ignore', 'id'}:
                continue

            col_data = self.df[col] if col in self.df.columns else pd.Series(dtype=object)

            if col_type == 'numeric':
                median_value = float(col_data.median()) if col_data.notna().any() else 0.0
                mean_value = float(col_data.mean()) if col_data.notna().any() else 0.0
                stats[col] = {
                    'type': 'numeric',
                    'median': median_value,
                    'mean': mean_value,
                    'min': float(col_data.min()) if col_data.notna().any() else 0.0,
                    'max': float(col_data.max()) if col_data.notna().any() else 0.0,
                    'default': median_value,
                }
            elif col_type in {'categorical', 'text'}:
                mode_value = col_data.dropna().mode().iloc[0] if not col_data.dropna().empty else 'Unknown'
                stats[col] = {
                    'type': col_type,
                    'mode': str(mode_value),
                    'default': str(mode_value),
                }
            elif col_type == 'binary':
                mode_value = col_data.dropna().mode().iloc[0] if not col_data.dropna().empty else 0
                mode_value = int(mode_value) if str(mode_value).isdigit() else 0
                stats[col] = {
                    'type': 'binary',
                    'mode': mode_value,
                    'default': mode_value,
                }
            elif col_type == 'temporal':
                try:
                    parsed = _safe_to_datetime(col_data)
                    days_ago = (pd.Timestamp.today() - parsed).dt.days
                    median_days = float(days_ago.dropna().median()) if days_ago.notna().any() else 30.0
                except Exception:
                    median_days = 30.0
                stats[col] = {
                    'type': 'temporal',
                    'median_days': median_days,
                    'default': None,
                }

        self.imputation_stats = stats
        return stats


# ═══════════════════════════════════════════════════════════════════════════════
#  LAYER 2: ADAPTIVE FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════════

class AdaptiveFeatureEngineering:
    """
    Build features WITHOUT knowing column names.
    Stores pipeline for consistent application to new data.
    """

    def __init__(self, df: pd.DataFrame, analyzer: DataAnalyzer):
        self.df = df.copy()
        self.analyzer = analyzer
        self.scalers = {}
        self.encoders = {}
        self.pca_models = {}
        self.one_hot_categories = {}
        self.numeric_imputers = {}
        self.temporal_baselines = {}
        self.feature_names = []
        self.feature_lineage = {}
        self.feature_baselines = {}

    def _register_lineage(self, feature_name: str, source_column: str, strategy: str, category: Optional[str] = None):
        self.feature_lineage[feature_name] = {
            "source_column": source_column,
            "strategy": strategy,
            "category": category,
        }

    def build_features(self) -> pd.DataFrame:
        """
        Build features from training data.
        Stores scalers/encoders for later use on test data.
        """
        relevant_cols = self.analyzer.filter_relevant_columns()
        column_types = self.analyzer.column_types

        X = pd.DataFrame()

        # 1. NUMERIC columns
        numeric_cols = [c for c in relevant_cols if column_types[c] == 'numeric']
        if numeric_cols:
            X_numeric = self.df[numeric_cols].copy()
            for col in numeric_cols:
                missing_flag = X_numeric[col].isna().astype(int)
                X[f"{col}_missing"] = missing_flag
                self._register_lineage(f"{col}_missing", col, "missing_indicator")
                median_value = X_numeric[col].median()
                if pd.isna(median_value):
                    median_value = 0.0
                self.numeric_imputers[col] = float(median_value)
                X_numeric[col] = X_numeric[col].fillna(median_value)

            scaler = MinMaxScaler()
            X_numeric_scaled = pd.DataFrame(
                scaler.fit_transform(X_numeric),
                columns=[f"{col}_scaled" for col in numeric_cols],
                index=X_numeric.index
            )
            X = pd.concat([X, X_numeric_scaled], axis=1)
            self.scalers['numeric'] = (numeric_cols, scaler)
            for col in numeric_cols:
                self._register_lineage(f"{col}_scaled", col, "scaled")

        # 2. CATEGORICAL columns
        categorical_cols = [c for c in relevant_cols if column_types[c] == 'categorical']
        if categorical_cols:
            X_cat = self.df[categorical_cols].copy()

            for col in categorical_cols:
                X[f"{col}_missing"] = X_cat[col].isna().astype(int)
                self._register_lineage(f"{col}_missing", col, "missing_indicator")
                X_cat[col] = X_cat[col].fillna('Unknown')
                n_unique = X_cat[col].nunique()

                if n_unique <= 10:
                    # One-hot
                    one_hot = pd.get_dummies(X_cat[col], prefix=col)
                    X = pd.concat([X, one_hot], axis=1)
                    self.one_hot_categories[col] = X_cat[col].astype(str).unique().tolist()
                    for feature_name in one_hot.columns:
                        category = feature_name.replace(f"{col}_", "", 1)
                        self._register_lineage(feature_name, col, "one_hot", category=category)
                else:
                    # Frequency encode
                    freq = X_cat[col].value_counts(normalize=True)
                    freq_col = X_cat[col].map(freq).fillna(0)
                    X[f"{col}_freq"] = freq_col
                    self.encoders[col] = freq
                    self._register_lineage(f"{col}_freq", col, "frequency")

        # 3. TEXT columns
        text_cols = [c for c in relevant_cols if column_types[c] == 'text']
        if text_cols and SentenceTransformer:
            try:
                model = SentenceTransformer('all-MiniLM-L6-v2')

                for col in text_cols:
                    texts = self.df[col].fillna('').astype(str).tolist()
                    embeddings = model.encode(texts, show_progress_bar=False)

                    pca = PCA(n_components=3)
                    compressed = pca.fit_transform(embeddings)

                    for i in range(3):
                        X[f"{col}_emb_{i}"] = compressed[:, i]
                        self._register_lineage(f"{col}_emb_{i}", col, "text_embedding_component", category=str(i))

                    self.pca_models[col] = pca
            except Exception as e:
                print(f"Warning: Text embedding skipped ({e})")

        # 4. TEMPORAL columns
        temporal_cols = [c for c in relevant_cols if column_types[c] == 'temporal']
        if temporal_cols:
            for col in temporal_cols:
                try:
                    df_temp = self.df.copy()
                    df_temp[col] = _safe_to_datetime(df_temp[col])
                    today = pd.Timestamp.today()
                    days_ago = (today - df_temp[col]).dt.days
                    median_days = days_ago.dropna().median()
                    if pd.isna(median_days):
                        median_days = 30
                    self.temporal_baselines[col] = float(median_days)
                    X[f"{col}_missing"] = days_ago.isna().astype(int)
                    self._register_lineage(f"{col}_missing", col, "missing_indicator")
                    recency = 1 / (1 + days_ago.fillna(median_days))
                    X[f"{col}_recency"] = recency
                    self._register_lineage(f"{col}_recency", col, "recency")
                except:
                    pass

        self.feature_names = X.columns.tolist()
        self.feature_baselines = X.mean(numeric_only=True).fillna(0).to_dict()
        return X

    def build_features_from_new_data(self, df_new: pd.DataFrame) -> pd.DataFrame:
        """
        Apply SAME transformations to new data.
        Uses stored scalers/encoders/PCA models.
        """
        relevant_cols = self.analyzer.filter_relevant_columns()
        column_types = self.analyzer.column_types

        X = pd.DataFrame()

        # 1. NUMERIC (apply stored scaler)
        numeric_cols, scaler = self.scalers.get('numeric', ([], None))
        if scaler and numeric_cols:
            X_numeric = df_new[numeric_cols].copy()
            for col in numeric_cols:
                X[f"{col}_missing"] = X_numeric[col].isna().astype(int)
                fill_value = self.numeric_imputers.get(col, 0.0)
                X_numeric[col] = X_numeric[col].fillna(fill_value)
            X_numeric_scaled = pd.DataFrame(
                scaler.transform(X_numeric),
                columns=[f"{col}_scaled" for col in numeric_cols],
                index=X_numeric.index
            )
            X = pd.concat([X, X_numeric_scaled], axis=1)

        # 2. CATEGORICAL (apply stored encoders)
        for col, categories in self.one_hot_categories.items():
            if col in df_new.columns:
                raw_values = df_new[col]
                X[f"{col}_missing"] = raw_values.isna().astype(int)
                values = raw_values.fillna('Unknown').astype(str)
                for category in categories:
                    X[f"{col}_{category}"] = (values == category).astype(int)

        for col, freq_map in self.encoders.items():
            if col in df_new.columns:
                freq_col = df_new[col].fillna('Unknown').map(freq_map).fillna(0)
                X[f"{col}_freq"] = freq_col

        # 3. TEXT (apply stored PCA)
        for col, pca in self.pca_models.items():
            if col in df_new.columns and SentenceTransformer:
                try:
                    model = SentenceTransformer('all-MiniLM-L6-v2')
                    texts = df_new[col].fillna('').astype(str).tolist()
                    embeddings = model.encode(texts, show_progress_bar=False)
                    compressed = pca.transform(embeddings)

                    for i in range(3):
                        X[f"{col}_emb_{i}"] = compressed[:, i]
                except:
                    pass

        # 4. TEMPORAL
        temporal_cols = [c for c in relevant_cols if column_types[c] == 'temporal']
        if temporal_cols:
            for col in temporal_cols:
                if col in df_new.columns:
                    try:
                        df_temp = df_new.copy()
                        df_temp[col] = _safe_to_datetime(df_temp[col])
                        today = pd.Timestamp.today()
                        days_ago = (today - df_temp[col]).dt.days
                        X[f"{col}_missing"] = days_ago.isna().astype(int)
                        baseline_days = self.temporal_baselines.get(col, 30.0)
                        recency = 1 / (1 + days_ago.fillna(baseline_days))
                        X[f"{col}_recency"] = recency
                    except:
                        pass

        # Align to training features (fill missing with 0)
        for fname in self.feature_names:
            if fname not in X.columns:
                X[fname] = 0

        return X[self.feature_names]

    def summarize_feature_blueprint(self) -> Dict:
        """Summarize the engineered feature system for explainability and reporting."""
        strategy_counts = {}
        for meta in self.feature_lineage.values():
            strategy = meta["strategy"]
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1

        return {
            "n_engineered_features": int(len(self.feature_names)),
            "strategy_counts": strategy_counts,
            "sample_features": list(self.feature_lineage.items())[:8],
        }


# ═══════════════════════════════════════════════════════════════════════════════
#  LAYER 3: UNIVERSAL MODEL TRAINING
# ═══════════════════════════════════════════════════════════════════════════════

class AdaptiveLeadScorer:
    """
    Train model on client's unique data distribution.
    Pure data-driven, no transfer learning.
    """

    def __init__(self):
        self.model = None
        self.calibrated_model = None
        self.explainer = None
        self.feature_names = []
        self.metadata = {}
        self.feature_importances = []
        self.feature_baselines = {}
        self.ranking_version = "lucida_rank_v2"
        self.rationale_version = "lucida_rationale_v2"
        self.model_family = None
        self.decision_threshold = 0.5
        self.training_diagnostics = {}

    def _precision_at_percent(self, y_true: np.ndarray, y_scores: np.ndarray, percent: float) -> float:
        top_n = max(1, int(np.ceil(len(y_scores) * percent)))
        ranked = np.argsort(y_scores)[::-1][:top_n]
        return float(np.mean(y_true[ranked])) if len(ranked) else 0.0

    def _lift_at_percent(self, y_true: np.ndarray, y_scores: np.ndarray, percent: float) -> float:
        baseline = float(np.mean(y_true)) if len(y_true) else 0.0
        if baseline == 0:
            return 0.0
        return float(self._precision_at_percent(y_true, y_scores, percent) / baseline)

    def _expected_calibration_error(self, y_true: np.ndarray, y_scores: np.ndarray, bins: int = 10) -> float:
        if len(y_true) == 0:
            return 0.0

        y_true = np.asarray(y_true, dtype=float)
        y_scores = np.asarray(y_scores, dtype=float)
        edges = np.linspace(0.0, 1.0, bins + 1)
        ece = 0.0

        for idx in range(bins):
            left = edges[idx]
            right = edges[idx + 1]
            if idx == bins - 1:
                mask = (y_scores >= left) & (y_scores <= right)
            else:
                mask = (y_scores >= left) & (y_scores < right)
            if not np.any(mask):
                continue
            bucket_confidence = float(np.mean(y_scores[mask]))
            bucket_accuracy = float(np.mean(y_true[mask]))
            ece += (np.sum(mask) / len(y_scores)) * abs(bucket_accuracy - bucket_confidence)

        return float(ece)

    def _target_sampling_ratio(self, y_train: np.ndarray) -> float:
        positives = int(np.sum(y_train))
        negatives = int(len(y_train) - positives)
        if positives == 0 or negatives == 0:
            return 1.0
        minority = min(positives, negatives)
        majority = max(positives, negatives)
        current_ratio = minority / majority
        # Avoid the common 1:1 oversampling trap on CRM-style low-conversion data.
        return float(max(current_ratio, 1 / 6.6))

    def _resample_training_data(self, X_train: pd.DataFrame, y_train: np.ndarray) -> Tuple[pd.DataFrame, np.ndarray, Dict]:
        positives = int(np.sum(y_train))
        negatives = int(len(y_train) - positives)
        minority = min(positives, negatives)
        majority = max(positives, negatives)

        diagnostics = {
            "strategy": "none",
            "sampling_ratio": float(minority / majority) if majority else 0.0,
            "rows_before": int(len(X_train)),
            "rows_after": int(len(X_train)),
        }

        if minority < 6 or majority == 0:
            diagnostics["reason"] = "too_few_minority_examples"
            return X_train, y_train, diagnostics

        target_ratio = self._target_sampling_ratio(y_train)
        current_ratio = minority / majority
        if current_ratio >= target_ratio:
            diagnostics["reason"] = "class_ratio_already_acceptable"
            return X_train, y_train, diagnostics

        neighbors = min(5, minority - 1)
        samplers = [
            ("adasyn", ADASYN(random_state=42, n_neighbors=neighbors, sampling_strategy=target_ratio)),
            ("smote", SMOTE(random_state=42, k_neighbors=neighbors, sampling_strategy=target_ratio)),
        ]

        for name, sampler in samplers:
            try:
                X_res, y_res = sampler.fit_resample(X_train, y_train)
                resampled_positives = int(np.sum(y_res))
                resampled_negatives = int(len(y_res) - resampled_positives)
                resampled_minority = min(resampled_positives, resampled_negatives)
                resampled_majority = max(resampled_positives, resampled_negatives)
                diagnostics.update({
                    "strategy": name,
                    "sampling_ratio": float(resampled_minority / resampled_majority) if resampled_majority else 0.0,
                    "rows_after": int(len(X_res)),
                    "target_ratio": float(target_ratio),
                })
                return X_res, y_res, diagnostics
            except Exception:
                continue

        diagnostics["reason"] = "resampling_failed"
        diagnostics["target_ratio"] = float(target_ratio)
        return X_train, y_train, diagnostics

    def _calibrate_model(self, estimator, X_calibration: pd.DataFrame, y_calibration: np.ndarray):
        try:
            if FrozenEstimator is not None:
                calibrated = CalibratedClassifierCV(FrozenEstimator(estimator), method="sigmoid")
            else:
                calibrated = CalibratedClassifierCV(estimator, method="sigmoid", cv="prefit")
            calibrated.fit(X_calibration, y_calibration)
            return calibrated
        except Exception:
            return None

    def _candidate_models(self, y_train: np.ndarray) -> List[Tuple[str, object]]:
        positives = max(1, int(np.sum(y_train)))
        negatives = max(1, int(len(y_train) - positives))
        scale_pos_weight = negatives / positives

        candidates = [
            (
                "gradient_boosting",
                GradientBoostingClassifier(
                    n_estimators=160,
                    learning_rate=0.05,
                    max_depth=3,
                    subsample=0.9,
                    random_state=42,
                ),
            ),
            (
                "random_forest",
                RandomForestClassifier(
                    n_estimators=200,
                    max_depth=8,
                    random_state=42,
                    n_jobs=RF_N_JOBS,
                    class_weight="balanced_subsample",
                ),
            ),
        ]

        if XGBClassifier is not None:
            candidates.insert(
                0,
                (
                    "xgboost",
                    XGBClassifier(
                        n_estimators=220,
                        max_depth=4,
                        learning_rate=0.05,
                        subsample=0.9,
                        colsample_bytree=0.8,
                        reg_lambda=1.0,
                        min_child_weight=2,
                        eval_metric="logloss",
                        random_state=42,
                        n_jobs=RF_N_JOBS,
                        scale_pos_weight=scale_pos_weight,
                    ),
                ),
            )

        return candidates

    def _optimize_threshold(self, y_true: np.ndarray, y_scores: np.ndarray) -> Dict[str, float]:
        if len(y_true) == 0:
            return {
                "threshold": 0.5,
                "accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
            }

        candidate_thresholds = np.unique(np.clip(np.round(y_scores, 3), 0.05, 0.95))
        candidate_thresholds = np.unique(np.concatenate([candidate_thresholds, np.array([0.2, 0.3, 0.4, 0.5])]))

        baseline_pred = (y_scores >= 0.5).astype(int)
        baseline_recall = recall_score(y_true, baseline_pred, zero_division=0)

        best = None
        for threshold in candidate_thresholds:
            y_pred = (y_scores >= threshold).astype(int)
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            # Prefer better recall on costly false negatives, but never at the cost of a large accuracy drop.
            score = (
                (accuracy * 0.55)
                + (recall * 0.35)
                + (precision * 0.10)
                - max(0.0, baseline_recall - recall) * 0.25
            )
            candidate = {
                "threshold": float(threshold),
                "accuracy": float(accuracy),
                "precision": float(precision),
                "recall": float(recall),
                "objective": float(score),
            }
            if best is None or candidate["objective"] > best["objective"]:
                best = candidate

        return best or {
            "threshold": 0.5,
            "accuracy": float(accuracy_score(y_true, baseline_pred)),
            "precision": float(precision_score(y_true, baseline_pred, zero_division=0)),
            "recall": float(baseline_recall),
            "objective": 0.0,
        }

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_eval: pd.DataFrame,
        y_eval: np.ndarray,
        client_id: str = "default",
        feature_baselines: Optional[Dict[str, float]] = None,
        target_diagnostics: Optional[Dict] = None,
        validation_context: Optional[Dict] = None,
    ) -> Dict:
        """
        Train on provided training split and evaluate on provided holdout split.
        """
        self.feature_names = X_train.columns.tolist()

        try:
            model_train_X, calibration_X, model_train_y, calibration_y = train_test_split(
                X_train,
                y_train,
                test_size=0.25,
                random_state=42,
                stratify=y_train,
            )
        except Exception:
            model_train_X, calibration_X, model_train_y, calibration_y = X_train, X_train, y_train, y_train
        X_res, y_res, resampling_diagnostics = self._resample_training_data(model_train_X, model_train_y)

        candidate_results = []
        for candidate_name, estimator in self._candidate_models(model_train_y):
            try:
                estimator.fit(X_res, y_res)
                calibrated = self._calibrate_model(estimator, calibration_X, calibration_y)
                predictor = calibrated or estimator
                calibration_scores = predictor.predict_proba(calibration_X)[:, 1]
                candidate_results.append(
                    {
                        "name": candidate_name,
                        "estimator": estimator,
                        "calibrated": calibrated,
                        "lift_at_20_percent": float(self._lift_at_percent(calibration_y, calibration_scores, 0.20)),
                        "roc_auc": float(roc_auc_score(calibration_y, calibration_scores)),
                        "brier_score": float(brier_score_loss(calibration_y, calibration_scores)),
                        "ece": float(self._expected_calibration_error(calibration_y, calibration_scores)),
                    }
                )
            except Exception:
                continue

        if not candidate_results:
            raise ValueError("Model training failed for every candidate estimator.")

        candidate_results.sort(
            key=lambda item: (
                item["lift_at_20_percent"],
                item["roc_auc"],
                -item["brier_score"],
                -item["ece"],
            ),
            reverse=True,
        )

        best_candidate = candidate_results[0]
        self.model = best_candidate["estimator"]
        self.calibrated_model = best_candidate["calibrated"]
        self.model_family = best_candidate["name"]

        self.explainer = None
        explanation_method = 'tree_shap' if self._get_explainer() is not None else 'feature_importance_fallback'

        # Evaluate
        predictor = self.calibrated_model or self.model
        y_proba = predictor.predict_proba(X_eval)[:, 1]
        threshold_metrics = self._optimize_threshold(y_eval, y_proba)
        self.decision_threshold = float(threshold_metrics["threshold"])
        y_pred = (y_proba >= self.decision_threshold).astype(int)
        self.feature_importances = self.model.feature_importances_.tolist()
        self.feature_baselines = feature_baselines or X_train.mean(numeric_only=True).fillna(0).to_dict()
        self.training_diagnostics = {
            "resampling": resampling_diagnostics,
            "candidate_models": [
                {
                    "name": item["name"],
                    "lift_at_20_percent": item["lift_at_20_percent"],
                    "roc_auc": item["roc_auc"],
                    "brier_score": item["brier_score"],
                    "ece": item["ece"],
                }
                for item in candidate_results
            ],
            "selected_model": best_candidate["name"],
            "recommended_threshold": self.decision_threshold,
        }

        self.metadata = {
            'client_id': client_id,
            'n_features': len(self.feature_names),
            'feature_names': self.feature_names,
            'accuracy': float(accuracy_score(y_eval, y_pred)),
            'roc_auc': float(roc_auc_score(y_eval, y_proba)),
            'precision': float(threshold_metrics['precision']),
            'recall': float(threshold_metrics['recall']),
            'precision_at_10_percent': float(self._precision_at_percent(y_eval, y_proba, 0.10)),
            'precision_at_20_percent': float(self._precision_at_percent(y_eval, y_proba, 0.20)),
            'lift_at_10_percent': float(self._lift_at_percent(y_eval, y_proba, 0.10)),
            'lift_at_20_percent': float(self._lift_at_percent(y_eval, y_proba, 0.20)),
            'brier_score': float(brier_score_loss(y_eval, y_proba)),
            'expected_calibration_error': float(self._expected_calibration_error(y_eval, y_proba)),
            'n_train': len(X_train),
            'n_test': len(X_eval),
            'class_distribution': {str(k): int(v) for k, v in zip(*np.unique(np.concatenate([y_train, y_eval]), return_counts=True))},
            'ranking_version': self.ranking_version,
            'rationale_version': self.rationale_version,
            'target_diagnostics': target_diagnostics or {},
            'validation_context': validation_context or {},
            'probability_calibration': 'sigmoid' if self.calibrated_model is not None else 'none',
            'recommended_threshold': self.decision_threshold,
            'threshold_tuning': threshold_metrics,
            'imbalance_strategy': resampling_diagnostics,
            'model_family': self.model_family,
            'candidate_models': self.training_diagnostics['candidate_models'],
            'explanation_method': explanation_method,
        }

        return self.metadata

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return probabilities."""
        X_aligned = X[self.feature_names].fillna(0)
        predictor = self.calibrated_model or self.model
        return predictor.predict_proba(X_aligned)[:, 1]

    def _format_feature_label(self, feature_name: str, feature_lineage: Optional[Dict[str, Dict]]) -> str:
        meta = (feature_lineage or {}).get(feature_name, {})
        source = meta.get("source_column", feature_name)
        strategy = meta.get("strategy")
        category = meta.get("category")

        if strategy == "one_hot" and category is not None:
            return f"{source} = {category}"
        if strategy == "frequency":
            return f"{source} category frequency"
        if strategy == "scaled":
            return f"{source} scaled value"
        if strategy == "recency":
            return f"{source} recency"
        if strategy == "text_embedding_component" and category is not None:
            return f"{source} semantic component {category}"
        return str(source)

    def _confidence_band(self, score: float) -> str:
        if score >= 80:
            return "high"
        if score >= 55:
            return "medium"
        return "low"

    def _get_explainer(self):
        """Build a SHAP explainer lazily when available."""
        if self.explainer is not None:
            return self.explainer
        if shap is None or self.model is None:
            return None
        try:
            self.explainer = shap.TreeExplainer(self.model)
            return self.explainer
        except Exception:
            self.explainer = None
            return None

    def _compute_shap_matrix(self, X: pd.DataFrame) -> Optional[np.ndarray]:
        """
        Return row-level feature attributions for the positive class when available.
        Falls back to None when SHAP cannot safely explain the current model/input.
        """
        explainer = self._get_explainer()
        if explainer is None:
            return None

        try:
            shap_output = explainer.shap_values(X)
        except Exception:
            try:
                shap_output = explainer(X)
            except Exception:
                return None

        values = getattr(shap_output, "values", shap_output)

        if isinstance(values, list):
            if not values:
                return None
            values = values[-1]

        values = np.asarray(values, dtype=float)

        if values.ndim == 1:
            values = values.reshape(1, -1)
        elif values.ndim == 3:
            # SHAP may return (rows, features, classes) or (rows, classes, features).
            if values.shape[-1] == 2:
                values = values[:, :, -1]
            elif values.shape[1] == 2:
                values = values[:, -1, :]
            else:
                return None

        if values.ndim != 2:
            return None

        if values.shape[1] != len(self.feature_names) and values.shape[0] == len(self.feature_names):
            values = values.T

        if values.shape[1] != len(self.feature_names):
            return None

        return values

    def _build_row_rationale(
        self,
        row: pd.Series,
        shap_row: Optional[np.ndarray] = None,
        feature_lineage: Optional[Dict[str, Dict]] = None,
        feature_baselines: Optional[Dict[str, float]] = None,
    ) -> Dict:
        contributions = []
        baselines = feature_baselines or self.feature_baselines or {}
        explanation_method = "tree_shap" if shap_row is not None else "feature_importance_fallback"

        for idx, feature_name in enumerate(self.feature_names):
            importance = self.feature_importances[idx] if idx < len(self.feature_importances) else 0.0
            value = float(row.get(feature_name, 0.0))
            baseline = float(baselines.get(feature_name, 0.0))
            contribution = float(shap_row[idx]) if shap_row is not None and idx < len(shap_row) else (value - baseline) * importance
            label = self._format_feature_label(feature_name, feature_lineage)
            meta = (feature_lineage or {}).get(feature_name, {})

            contributions.append({
                "label": label,
                "engineered_feature": feature_name,
                "source_column": meta.get("source_column", feature_name),
                "strategy": meta.get("strategy", "derived"),
                "contribution": float(round(contribution, 6)),
                "value": float(round(value, 6)),
                "baseline": float(round(baseline, 6)),
                "contribution_type": "shap_value" if shap_row is not None else "heuristic_delta",
            })

        positive = [item for item in contributions if item["contribution"] > 0]
        negative = [item for item in contributions if item["contribution"] < 0]
        positive.sort(key=lambda item: item["contribution"], reverse=True)
        negative.sort(key=lambda item: item["contribution"])

        top_positive = positive[:3]
        top_negative = negative[:3]

        summary_parts = []
        if top_positive:
            summary_parts.append(
                "Boosted by " + ", ".join(item["label"] for item in top_positive[:2])
            )
        if top_negative:
            summary_parts.append(
                "held back by " + ", ".join(item["label"] for item in top_negative[:2])
            )

        return {
            "top_positive": top_positive,
            "top_negative": top_negative,
            "summary": "; ".join(summary_parts) if summary_parts else "No strong driver signals detected.",
            "method": explanation_method,
        }

    def predict_with_explanation(
        self,
        X: pd.DataFrame,
        feature_lineage: Optional[Dict[str, Dict]] = None,
        feature_baselines: Optional[Dict[str, float]] = None,
    ) -> List[Dict]:
        """
        Score and explain.
        """
        scores = self.predict(X)
        shap_matrix = self._compute_shap_matrix(X)

        importances = np.array(self.feature_importances or self.model.feature_importances_)
        top_3_indices = np.argsort(importances)[-3:][::-1] if len(importances) else []

        results = []
        for idx, score in enumerate(scores):
            row = X.iloc[idx]
            rationale = self._build_row_rationale(
                row,
                shap_row=shap_matrix[idx] if shap_matrix is not None and idx < len(shap_matrix) else None,
                feature_lineage=feature_lineage,
                feature_baselines=feature_baselines,
            )
            top_features = [self._format_feature_label(self.feature_names[i], feature_lineage) for i in top_3_indices]
            score_pct = round(float(score) * 100, 2)
            results.append({
                'index': idx,
                'score': score_pct,
                'top_drivers': [item["label"] for item in rationale["top_positive"]] or top_features,
                'rationale': rationale,
                'rationale_summary': rationale["summary"],
                'score_band': self._confidence_band(score_pct),
                'ranking_version': self.ranking_version,
                'explanation_method': rationale["method"],
            })

        return results

    def save(self, path: str):
        """Save model."""
        joblib.dump({
            'model': self.model,
            'calibrated_model': self.calibrated_model,
            'feature_names': self.feature_names,
            'metadata': self.metadata,
            'feature_importances': self.feature_importances,
            'feature_baselines': self.feature_baselines,
            'ranking_version': self.ranking_version,
            'rationale_version': self.rationale_version,
            'model_family': self.model_family,
            'decision_threshold': self.decision_threshold,
            'training_diagnostics': self.training_diagnostics,
        }, path)

    def load(self, path: str):
        """Load model."""
        bundle = joblib.load(path)
        self.model = bundle['model']
        self.calibrated_model = bundle.get('calibrated_model')
        self.explainer = None
        self.feature_names = bundle['feature_names']
        self.metadata = bundle['metadata']
        self.feature_importances = bundle.get('feature_importances', [])
        self.feature_baselines = bundle.get('feature_baselines', {})
        self.ranking_version = bundle.get('ranking_version', self.ranking_version)
        self.rationale_version = bundle.get('rationale_version', self.rationale_version)
        self.model_family = bundle.get('model_family')
        self.decision_threshold = bundle.get('decision_threshold', 0.5)
        self.training_diagnostics = bundle.get('training_diagnostics', {})


# ═══════════════════════════════════════════════════════════════════════════════
#  COMPLETE PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

class UniversalAdaptiveScorer:
    """
    End-to-end pipeline: CSV → Auto-analysis → Train → Score
    """

    def __init__(self):
        self.analyzer = None
        self.engineer = None
        self.scorer = None

    def _split_raw_df(
        self,
        df: pd.DataFrame,
        target_col: str,
        column_types: Dict[str, str],
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
        """Create a holdout split with time-awareness when possible."""
        temporal_candidates = [col for col, type_ in column_types.items() if type_ == 'temporal' and col != target_col]
        for temporal_col in temporal_candidates:
            try:
                parsed = _safe_to_datetime(df[temporal_col])
                valid = parsed.notna().sum()
                if valid < max(10, int(len(df) * 0.5)):
                    continue
                ordered = df.assign(_sort_time=parsed).sort_values("_sort_time")
                holdout_size = max(1, int(np.ceil(len(ordered) * 0.2)))
                train_df = ordered.iloc[:-holdout_size].drop(columns=["_sort_time"])
                eval_df = ordered.iloc[-holdout_size:].drop(columns=["_sort_time"])
                if (
                    len(train_df) >= 10
                    and len(eval_df) >= 2
                    and _is_binary_series(train_df[target_col])
                    and _is_binary_series(eval_df[target_col])
                ):
                    return train_df, eval_df, {
                        "strategy": "time_split",
                        "reference_column": temporal_col,
                    }
            except Exception:
                pass

        stratify_labels = df[target_col].map(_normalize_binary_token)
        non_null_labels = stratify_labels.dropna()
        if non_null_labels.empty or non_null_labels.nunique() < 2:
            raise ValueError(
                f"Target column '{target_col}' does not have enough normalized binary signal for stratified split."
            )
        fill_token = str(non_null_labels.mode().iloc[0])

        train_df, eval_df = train_test_split(
            df,
            test_size=0.2,
            random_state=42,
            stratify=stratify_labels.fillna(fill_token),
        )
        return train_df.copy(), eval_df.copy(), {
            "strategy": "random_split",
            "reference_column": None,
        }

    def train_from_csv(self, csv_path: str, target_col: Optional[str] = None) -> Dict:
        """Train on CSV file."""
        df = pd.read_csv(csv_path)
        return self.train(df, target_col=target_col)

    def train(self, df: pd.DataFrame, target_col: Optional[str] = None, client_id: str = "default") -> Dict:
        """
        Full training pipeline.
        
        Input: DataFrame (any structure)
        Output: Trained model ready for scoring
        """
        # Step 1: Analyze raw schema and determine target
        bootstrap_analyzer = DataAnalyzer(df, target_col=target_col)
        bootstrap_analyzer.infer_column_types()
        target = bootstrap_analyzer.auto_detect_target()
        bootstrap_df = bootstrap_analyzer.df.copy()
        train_df, eval_df, validation_context = self._split_raw_df(
            bootstrap_df,
            target,
            bootstrap_analyzer.column_types,
        )

        # Step 2: Fit analyzer only on training data to avoid leakage
        self.analyzer = DataAnalyzer(train_df, target_col=target)
        column_types = self.analyzer.infer_column_types()
        target = self.analyzer.auto_detect_target()
        importance = self.analyzer.compute_feature_importance()
        self.analyzer.compute_imputation_stats()
        analysis = self.analyzer.summary()

        # Step 3: Engineer features from training only, then transform holdout
        self.engineer = AdaptiveFeatureEngineering(train_df, self.analyzer)
        X_train = self.engineer.build_features()
        X_eval = self.engineer.build_features_from_new_data(eval_df)
        y_train = self.analyzer._encode_binary(target).values
        y_eval = self.analyzer.encode_binary_series(target, eval_df[target]).values

        # Step 4: Train/evaluate
        self.scorer = AdaptiveLeadScorer()
        metrics = self.scorer.train(
            X_train,
            y_train,
            X_eval,
            y_eval,
            client_id=client_id,
            feature_baselines=self.engineer.feature_baselines,
            target_diagnostics=self.analyzer.get_target_diagnostics(),
            validation_context=validation_context,
        )

        return {
            'status': 'success',
            'client_id': client_id,
            'analysis': {
                'column_types': column_types,
                'target_column': target,
                'target_diagnostics': self.analyzer.get_target_diagnostics(),
                'n_features': int(analysis['n_features']),
                'validation_context': validation_context,
                'feature_blueprint': self.engineer.summarize_feature_blueprint(),
                'top_features': [
                    (str(feat), float(imp)) 
                    for feat, imp in sorted(
                        importance.items(),
                        key=lambda x: x[1],
                        reverse=True
                    )[:5]
                ]
            },
            'metrics': metrics
        }

    def score(self, df: pd.DataFrame) -> List[Dict]:
        """
        Score new leads.
        
        Input: DataFrame (any structure, same as training data)
        Output: List of scores + explanations
        """
        if not self.scorer:
            raise ValueError("Must train first!")

        X = self.engineer.build_features_from_new_data(df)
        results = self.scorer.predict_with_explanation(
            X,
            feature_lineage=self.engineer.feature_lineage,
            feature_baselines=self.engineer.feature_baselines,
        )

        # Add original data
        output = []
        for i, result in enumerate(results):
            # Convert row to native Python types for JSON compatibility
            raw_data = df.iloc[i].to_dict()
            clean_data = {}
            for k, v in raw_data.items():
                if hasattr(v, 'item'): # numpy type
                    clean_data[k] = v.item()
                elif pd.isna(v):
                    clean_data[k] = None
                else:
                    clean_data[k] = v

            output.append({
                'index': i,
                'data': clean_data,
                'score': result['score'],
                'top_drivers': result['top_drivers'],
                'rationale': result['rationale'],
                'rationale_summary': result['rationale_summary'],
                'score_band': result['score_band'],
                'ranking_version': result['ranking_version'],
                'explanation_method': result['explanation_method'],
            })

        # Sort by score
        output.sort(key=lambda x: x['score'], reverse=True)
        return output

    def save(self, path: str):
        """Save trained pipeline."""
        joblib.dump({
            'analyzer': self.analyzer,
            'engineer': self.engineer,
            'scorer': self.scorer
        }, path)

    def load(self, path: str):
        """Load trained pipeline."""
        bundle = joblib.load(path)
        self.analyzer = bundle['analyzer']
        self.engineer = bundle['engineer']
        self.scorer = bundle['scorer']


# ═══════════════════════════════════════════════════════════════════════════════
#  ENGAGEMENT MOMENTUM SCORER (Rule-Based, No Training Required)
# ═══════════════════════════════════════════════════════════════════════════════

class EngagementScorer:
    """
    Calculates Engagement Momentum Score (0-100) from behavioral signals.
    
    This is rule-based (not ML) for:
    - Speed: No training required
    - Interpretability: Easy to explain
    - Real-time: Can score on the fly
    
    Auto-detects engagement columns by name patterns.
    """
    
    # Column name patterns for auto-detection
    ENGAGEMENT_PATTERNS = {
        'reply': ['reply', 'replied', 'response', 'responded'],
        'call': ['call', 'answered', 'connected', 'phone'],
        'meeting': ['meeting', 'demo', 'appointment', 'scheduled'],
        'recency': ['days_since', 'last_contact', 'last_activity', 'recency', 'last_interaction'],
        'email_open': ['open', 'opened', 'email_open'],
        'click': ['click', 'clicked', 'ctr'],
        'visit': ['visit', 'page_view', 'website', 'session'],
    }
    
    # Weights for each signal type (total = 100)
    SIGNAL_WEIGHTS = {
        'reply': 25,      # Highest signal - they responded!
        'meeting': 25,    # Very strong - they committed time
        'call': 20,       # Strong - verbal engagement
        'recency': 15,    # Time decay matters
        'email_open': 8,  # Weak but positive signal
        'click': 5,       # Clicked a link
        'visit': 2,       # Visited website
    }
    
    def __init__(self):
        self.detected_columns: Dict[str, str] = {}  # signal_type -> column_name
        self.column_stats: Dict[str, Dict] = {}     # For normalization
    
    def _normalize_column_name(self, col: str) -> str:
        """Normalize column name for pattern matching."""
        return col.lower().replace('_', ' ').replace('-', ' ')
    
    def detect_engagement_columns(self, df: pd.DataFrame) -> Dict[str, str]:
        """
        Auto-detect which columns contain engagement signals.
        Returns mapping: signal_type -> column_name
        """
        self.detected_columns = {}
        
        for col in df.columns:
            col_normalized = self._normalize_column_name(col)
            
            for signal_type, patterns in self.ENGAGEMENT_PATTERNS.items():
                if signal_type in self.detected_columns:
                    continue  # Already found this signal type
                    
                for pattern in patterns:
                    if pattern in col_normalized:
                        # Validate it's a usable column (numeric or boolean-ish)
                        if self._is_engagement_column(df[col]):
                            self.detected_columns[signal_type] = col
                            break
        
        return self.detected_columns
    
    def _is_engagement_column(self, series: pd.Series) -> bool:
        """Check if column can be used as engagement signal."""
        # Numeric columns
        if pd.api.types.is_numeric_dtype(series):
            return True
        
        # Boolean-like strings
        sample = series.dropna().head(100).astype(str).str.lower()
        bool_values = {'yes', 'no', 'true', 'false', '1', '0', 'y', 'n'}
        if sample.isin(bool_values).mean() > 0.8:
            return True
        
        return False
    
    def _parse_value(self, value, signal_type: str) -> float:
        """Parse a value to a numeric score component."""
        if pd.isna(value):
            return 0.0
        
        # Handle boolean-like strings
        if isinstance(value, str):
            val_lower = value.lower().strip()
            if val_lower in ('yes', 'true', 'y', '1'):
                return 1.0
            if val_lower in ('no', 'false', 'n', '0'):
                return 0.0
            try:
                return float(value)
            except:
                return 0.0
        
        # Handle numeric
        try:
            return float(value)
        except:
            return 0.0
    
    def _compute_signal_score(self, value: float, signal_type: str, stats: Dict) -> float:
        """
        Convert raw value to 0-1 score for a signal type.
        """
        if signal_type == 'recency':
            # Recency: lower days = better (invert)
            # Score = max(0, 1 - days/30)
            days = max(0, value)
            return max(0, 1 - (days / 30))
        
        # For count-based signals (reply, call, meeting, etc.)
        if value <= 0:
            return 0.0
        
        # Binary signal (0 or 1 max)
        if stats.get('max', 1) <= 1:
            return min(1.0, value)
        
        # Count signal: normalize by max with diminishing returns
        max_val = max(stats.get('max', 1), 1)
        # Use log scaling for counts to prevent outliers from dominating
        normalized = np.log1p(value) / np.log1p(max_val)
        return min(1.0, normalized)
    
    def compute_stats(self, df: pd.DataFrame) -> Dict[str, Dict]:
        """Compute stats for normalization."""
        self.column_stats = {}
        
        for signal_type, col in self.detected_columns.items():
            values = df[col].apply(lambda x: self._parse_value(x, signal_type))
            self.column_stats[signal_type] = {
                'min': float(values.min()),
                'max': float(values.max()),
                'mean': float(values.mean()),
            }
        
        return self.column_stats
    
    def score_lead(self, row: pd.Series) -> Dict:
        """
        Score a single lead's engagement momentum.
        
        Returns:
            {
                'engagement_score': 0-100,
                'signals': {signal_type: {value, contribution, ...}},
                'top_signals': ['signal1', 'signal2'],
                'engagement_band': 'high'|'medium'|'low'
            }
        """
        if not self.detected_columns:
            return {
                'engagement_score': None,
                'signals': {},
                'top_signals': [],
                'engagement_band': None,
                'has_engagement_data': False,
            }
        
        signals = {}
        total_score = 0.0
        total_possible = 0.0
        
        for signal_type, col in self.detected_columns.items():
            weight = self.SIGNAL_WEIGHTS.get(signal_type, 0)
            total_possible += weight
            
            raw_value = row.get(col)
            parsed_value = self._parse_value(raw_value, signal_type)
            stats = self.column_stats.get(signal_type, {})
            signal_score = self._compute_signal_score(parsed_value, signal_type, stats)
            contribution = signal_score * weight
            
            total_score += contribution
            
            signals[signal_type] = {
                'column': col,
                'raw_value': raw_value if not pd.isna(raw_value) else None,
                'parsed_value': round(parsed_value, 2),
                'signal_score': round(signal_score, 3),
                'weight': weight,
                'contribution': round(contribution, 2),
            }
        
        # Normalize to 0-100
        if total_possible > 0:
            engagement_score = round((total_score / total_possible) * 100, 1)
        else:
            engagement_score = 0.0
        
        # Sort signals by contribution
        sorted_signals = sorted(
            signals.items(), 
            key=lambda x: x[1]['contribution'], 
            reverse=True
        )
        top_signals = [s[0] for s in sorted_signals if s[1]['contribution'] > 0][:3]
        
        # Confidence band
        if engagement_score >= 70:
            band = 'high'
        elif engagement_score >= 40:
            band = 'medium'
        else:
            band = 'low'
        
        return {
            'engagement_score': engagement_score,
            'signals': signals,
            'top_signals': top_signals,
            'engagement_band': band,
            'has_engagement_data': True,
        }
    
    def score_dataframe(self, df: pd.DataFrame) -> List[Dict]:
        """Score all leads in a DataFrame."""
        results = []
        for idx, row in df.iterrows():
            result = self.score_lead(row)
            result['index'] = idx
            results.append(result)
        return results
    
    def analyze(self, df: pd.DataFrame) -> Dict:
        """
        Analyze a DataFrame for engagement columns.
        Call this before scoring to set up the scorer.
        """
        detected = self.detect_engagement_columns(df)
        stats = self.compute_stats(df) if detected else {}
        
        return {
            'detected_columns': detected,
            'column_stats': stats,
            'signals_found': list(detected.keys()),
            'signals_missing': [s for s in self.SIGNAL_WEIGHTS.keys() if s not in detected],
            'coverage': len(detected) / len(self.SIGNAL_WEIGHTS) * 100,
        }


# ═══════════════════════════════════════════════════════════════════════════════
#  ACTION RECOMMENDER (Profile + Engagement → Action)
# ═══════════════════════════════════════════════════════════════════════════════

class ActionRecommender:
    """
    Maps (profile_score, engagement_score) to recommended action.
    
    Action matrix with profile-score stratification.
    Low-profile leads are still split into actionable buckets.
    """
    
    ACTIONS = {
        'close_now': {
            'action': 'CLOSE NOW',
            'emoji': '🟢',
            'color': 'green',
            'priority': 1,
            'description': 'Hot lead - prioritize immediately!',
            'next_steps': ['Schedule call today', 'Send proposal', 'Involve decision maker'],
        },
        'nurture': {
            'action': 'NURTURE',
            'emoji': '🟡',
            'color': 'yellow',
            'priority': 2,
            'description': 'Perfect fit, not engaged yet. Reach out now.',
            'next_steps': ['Send personalized outreach', 'Share relevant case study', 'Add to high-touch sequence'],
        },
        'auto_sequence': {
            'action': 'AUTO-SEQUENCE',
            'emoji': '🟠',
            'color': 'orange',
            'priority': 3,
            'description': 'Interested but not ideal fit. Automate nurturing.',
            'next_steps': ['Add to automated email sequence', 'Monitor for qualification changes', 'Deprioritize manual effort'],
        },
        'deprioritize': {
            'action': 'DEPRIORITIZE',
            'emoji': '🔴',
            'color': 'red',
            'priority': 6,
            'description': 'Low priority - focus elsewhere.',
            'next_steps': ['Add to long-term nurture', 'Revisit in 6 months', 'Remove from active pipeline'],
        },
        'monitor': {
            'action': 'MONITOR',
            'emoji': '🟣',
            'color': 'purple',
            'priority': 5,
            'description': 'Not ready now, but keep under watch for signal changes.',
            'next_steps': ['Track weekly activity', 'Trigger alert on engagement spike', 'Re-score after new interactions'],
        },
        'light_nurture': {
            'action': 'LIGHT NURTURE',
            'emoji': '🔵',
            'color': 'blue',
            'priority': 4,
            'description': 'Moderate potential - keep warm with lightweight automation.',
            'next_steps': ['Add to low-frequency sequence', 'Share one relevant use case', 'Re-evaluate after 14 days'],
        },
        'priority_review': {
            'action': 'PRIORITY REVIEW',
            'emoji': '🟠',
            'color': 'orange',
            'priority': 3,
            'description': 'Near-threshold profile - review quickly for contextual fit.',
            'next_steps': ['Check account context', 'Validate fit criteria', 'Escalate if intent strengthens'],
        },
    }
    
    def __init__(self, profile_threshold: float = 50, engagement_threshold: float = 50):
        self.profile_threshold = profile_threshold
        self.engagement_threshold = engagement_threshold
    
    def recommend(self, profile_score: float, engagement_score: Optional[float]) -> Dict:
        """
        Get action recommendation based on both scores.
        
        Args:
            profile_score: 0-100 (from existing ML model)
            engagement_score: 0-100 or None (from EngagementScorer)
        
        Returns:
            Action recommendation with metadata
        """
        # If no engagement data, still stratify by profile score bands.
        if engagement_score is None:
            if profile_score >= 75:
                action_key = 'nurture'
            elif profile_score >= self.profile_threshold:
                action_key = 'priority_review'
            elif profile_score >= 40:
                action_key = 'light_nurture'
            elif profile_score >= 20:
                action_key = 'monitor'
            else:
                action_key = 'deprioritize'
            
            result = self.ACTIONS[action_key].copy()
            result['profile_score'] = profile_score
            result['engagement_score'] = None
            result['quadrant'] = 'unknown_engagement'
            result['confidence'] = 'low'  # Can't be confident without engagement
            return result
        
        # Engagement-aware routing with profile-score banding.
        high_profile = profile_score >= self.profile_threshold
        high_engagement = engagement_score >= self.engagement_threshold
        
        if high_profile and high_engagement:
            action_key = 'close_now'
            quadrant = 'high_profile_high_engagement'
        elif high_profile and not high_engagement:
            action_key = 'nurture' if profile_score >= 75 else 'priority_review'
            quadrant = 'high_profile_low_engagement'
        elif not high_profile and high_engagement:
            if profile_score >= 40:
                action_key = 'auto_sequence'
            elif profile_score >= 20:
                action_key = 'light_nurture'
            else:
                action_key = 'monitor'
            quadrant = 'low_profile_high_engagement'
        else:
            if profile_score >= 40:
                action_key = 'light_nurture'
            elif profile_score >= 20:
                action_key = 'monitor'
            else:
                action_key = 'deprioritize'
            quadrant = 'low_profile_low_engagement'
        
        # Calculate confidence based on how far from thresholds
        profile_distance = abs(profile_score - self.profile_threshold)
        engagement_distance = abs(engagement_score - self.engagement_threshold)
        min_distance = min(profile_distance, engagement_distance)
        
        if min_distance >= 20:
            confidence = 'high'
        elif min_distance >= 10:
            confidence = 'medium'
        else:
            confidence = 'low'  # Close to threshold boundary
        
        result = self.ACTIONS[action_key].copy()
        result['profile_score'] = profile_score
        result['engagement_score'] = engagement_score
        result['quadrant'] = quadrant
        result['confidence'] = confidence
        
        return result
    
    def recommend_batch(self, scores: List[Dict]) -> List[Dict]:
        """
        Recommend actions for a batch of scored leads.
        
        Args:
            scores: List of dicts with 'profile_score' and 'engagement_score'
        
        Returns:
            List of recommendations
        """
        return [
            self.recommend(
                s.get('profile_score', s.get('score', 0)),
                s.get('engagement_score')
            )
            for s in scores
        ]


# ═══════════════════════════════════════════════════════════════════════════════
#  USAGE EXAMPLE
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    
    # Create sample data (any structure!)
    df_train = pd.DataFrame({
        'prospect_email': ['a@b.com', 'c@d.com', 'e@f.com', 'g@h.com', 'i@j.com'] * 20,
        'num_interactions': np.random.randint(0, 50, 100),
        'company_size_employees': np.random.randint(10, 10000, 100),
        'industry_sector': np.random.choice(['Tech', 'Finance', 'Retail'], 100),
        'last_contact_date': pd.date_range('2024-01-01', periods=100),
        'interested': np.random.choice([0, 1], 100)
    })
    
    print("=" * 80)
    print("UNIVERSAL ADAPTIVE LEAD SCORER - DEMO")
    print("=" * 80)
    
    # TRAIN
    print("\n[1] TRAINING...")
    scorer = UniversalAdaptiveScorer()
    train_result = scorer.train(df_train, client_id="demo")
    
    print(f"\nStatus: {train_result['status']}")
    print(f"\nAnalysis:")
    print(f"  Target: {train_result['analysis']['target_column']}")
    print(f"  Features used: {train_result['analysis']['n_features']}")
    print(f"  Top predictors:")
    for feat, imp in train_result['analysis']['top_features']:
        print(f"    - {feat}: {imp:.3f}")
    
    print(f"\nModel Performance:")
    print(f"  Accuracy: {train_result['metrics']['accuracy']:.3f}")
    print(f"  ROC-AUC: {train_result['metrics']['roc_auc']:.3f}")
    
    # SCORE
    print("\n[2] SCORING NEW LEADS...")
    df_test = pd.DataFrame({
        'prospect_email': ['new1@b.com', 'new2@d.com', 'new3@f.com'],
        'num_interactions': [45, 5, 20],
        'company_size_employees': [5000, 500, 8000],
        'industry_sector': ['Tech', 'Retail', 'Finance'],
        'last_contact_date': ['2024-12-01', '2024-11-01', '2024-12-15']
    })
    
    results = scorer.score(df_test)
    
    print(f"\nRanked Leads:")
    for i, result in enumerate(results, 1):
        print(f"\n  #{i} (Score: {result['score']})")
        print(f"     Data: {result['data']}")
        print(f"     Top drivers: {', '.join(result['top_drivers'][:2])}")
    
    print("\n" + "=" * 80)
    print("✓ Universal Adaptive Scorer Ready!")
    print("=" * 80)
