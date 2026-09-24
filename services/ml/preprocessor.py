from typing import Dict, Any, List, Tuple, Optional
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler
from services.profiling.schema_detector import SchemaDetector


class DatetimeFeatureExtractor(BaseEstimator, TransformerMixin):
    """Custom transformer to extract temporal features from datetime columns."""

    def __init__(self, datetime_cols: List[str]):
        self.datetime_cols = datetime_cols
        self.feature_names_: List[str] = []

    def fit(self, X, y=None):
        feature_names = []
        for col in self.datetime_cols:
            feature_names.extend([
                f"{col}_year",
                f"{col}_month",
                f"{col}_day",
                f"{col}_dayofweek"
            ])
        self.feature_names_ = feature_names
        return self

    def transform(self, X):
        df = pd.DataFrame(X, columns=self.datetime_cols)
        extracted = []
        for col in self.datetime_cols:
            dt_series = pd.to_datetime(df[col], errors="coerce")
            year = dt_series.dt.year.fillna(2020).astype(int).values.reshape(-1, 1)
            month = dt_series.dt.month.fillna(1).astype(int).values.reshape(-1, 1)
            day = dt_series.dt.day.fillna(1).astype(int).values.reshape(-1, 1)
            dayofweek = dt_series.dt.dayofweek.fillna(0).astype(int).values.reshape(-1, 1)
            extracted.extend([year, month, day, dayofweek])

        if extracted:
            return np.hstack(extracted)
        return np.empty((len(X), 0))

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_)


class MLPreprocessor:
    """Automated, leak-free scikit-learn preprocessing pipeline factory."""

    @classmethod
    def identify_feature_types(
        cls, 
        df: pd.DataFrame, 
        target_column: str,
        excluded_columns: Optional[List[str]] = None
    ) -> Dict[str, List[str]]:
        """Identify feature types and select active feature subsets."""
        excluded = set(excluded_columns or [])
        excluded.add(target_column)

        schema = SchemaDetector.detect_schema(df)
        numerical = []
        categorical = []
        boolean_cols = []
        datetime_cols = []
        identifiers = []

        for col, col_info in schema["columns"].items():
            if col in excluded:
                continue

            itype = col_info["inferred_type"]
            if itype == "IDENTIFIER":
                identifiers.append(col)
            elif itype == "NUMERICAL":
                numerical.append(col)
            elif itype == "DATETIME":
                datetime_cols.append(col)
            else:  # CATEGORICAL, BOOLEAN or TEXT
                categorical.append(col)

        return {
            "numerical": numerical,
            "categorical": categorical,
            "boolean": [],
            "datetime": datetime_cols,
            "identifiers": identifiers,
            "active_features": numerical + categorical + datetime_cols
        }

    @classmethod
    def build_preprocessor(
        cls,
        feature_types: Dict[str, List[str]],
        scale_numerical: bool = True
    ) -> ColumnTransformer:
        """
        Build a scikit-learn ColumnTransformer strictly adhering to data hygiene.
        """
        transformers = []

        # 1. Numerical Pipeline: Imputation (median) + Optional Robust Scaling
        if feature_types["numerical"]:
            num_steps = [("imputer", SimpleImputer(strategy="median"))]
            if scale_numerical:
                num_steps.append(("scaler", StandardScaler()))
            num_pipe = Pipeline(num_steps)
            transformers.append(("num", num_pipe, feature_types["numerical"]))

        # 2. Boolean Pipeline: Imputer (most_frequent)
        if feature_types["boolean"]:
            bool_pipe = Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent"))
            ])
            transformers.append(("bool", bool_pipe, feature_types["boolean"]))

        # 3. Categorical Pipeline: Imputation (missing) + OneHotEncoder (unknown='ignore')
        if feature_types["categorical"]:
            cat_pipe = Pipeline([
                ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
            ])
            transformers.append(("cat", cat_pipe, feature_types["categorical"]))

        # 4. Datetime Pipeline
        if feature_types["datetime"]:
            dt_pipe = Pipeline([
                ("dt_extract", DatetimeFeatureExtractor(feature_types["datetime"])),
                ("scaler", StandardScaler())
            ])
            transformers.append(("dt", dt_pipe, feature_types["datetime"]))

        preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder="drop",
            sparse_threshold=0
        )
        return preprocessor
