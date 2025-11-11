from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np
import warnings
from dependency import check_null_values, remove_the_duplicates

warnings.filterwarnings('ignore', category=UserWarning, module='pkg_resources')
warnings.filterwarnings('ignore',
                        category=UserWarning,
                        module='_distutils_hack')


class Preprocessing:
    """
    Pure preprocessing module (no MLflow logging).
    Handles:
      - Continuous and categorical preprocessing
      - Likert-scale detection for IRT analysis
    """

    def __init__(self, use_ordinal_for_categorical=True):
        self.numeric_features = []
        self.categorical_features = []
        self.likert_features = []
        self.continuous_features = []
        self.preprocessor = None
        self.likert_preprocessor = None
        self.has_nulls = False
        self.use_ordinal_for_categorical = use_ordinal_for_categorical
        self.likert_data = None

    def _is_likert_scale(self, series: pd.Series) -> bool:
        """Detect numeric Likert-like features (1–5, 1–7, 0–10)."""
        if not pd.api.types.is_numeric_dtype(series):
            return False
        unique_values = series.dropna().unique()
        if len(unique_values) <= 10:
            min_val, max_val = unique_values.min(), unique_values.max()
            if (min_val >= 0 and max_val <= 10) or (min_val >= 1
                                                    and max_val <= 7):
                return True
        return False

    def _create_likert_preprocessor(self, df: pd.DataFrame):
        """Build separate preprocessor for Likert data (for IRT)."""
        if not self.likert_features:
            return None
        steps = []
        if self.has_nulls:
            steps.append(('imputer', SimpleImputer(strategy='median')))
        self.likert_preprocessor = Pipeline(steps=steps) if steps else None
        return self.likert_preprocessor

    @check_null_values
    @remove_the_duplicates
    def _create_preprocessor(self, df: pd.DataFrame):
        """Dynamically builds main preprocessor (excludes Likert)."""
        self.numeric_features = df.select_dtypes(
            include=['int64', 'float64']).columns.tolist()
        self.categorical_features = df.select_dtypes(
            include=['object', 'category']).columns.tolist()

        # Identify Likert and continuous numeric features
        self.likert_features = [
            col for col in self.numeric_features
            if self._is_likert_scale(df[col])
        ]
        self.continuous_features = [
            col for col in self.numeric_features
            if col not in self.likert_features
        ]

        # Create Likert preprocessor
        self._create_likert_preprocessor(df)

        transformers = []

        # Continuous numeric
        if self.continuous_features:
            cont_steps = []
            if self.has_nulls:
                cont_steps.append(('imputer', SimpleImputer(strategy='mean')))
            cont_steps.append(('scaler', StandardScaler()))
            transformers.append(
                ('cont_num', Pipeline(cont_steps), self.continuous_features))

        # Categorical
        if self.categorical_features:
            cat_steps = []
            if self.has_nulls:
                cat_steps.append(
                    ('imputer', SimpleImputer(strategy='most_frequent')))
            if self.use_ordinal_for_categorical:
                cat_steps.append(
                    ('ordinal',
                     OrdinalEncoder(handle_unknown='use_encoded_value',
                                    unknown_value=-1)))
            else:
                cat_steps.append(('onehot',
                                  OneHotEncoder(handle_unknown='ignore',
                                                sparse_output=False)))
            transformers.append(
                ('cat', Pipeline(cat_steps), self.categorical_features))

        self.preprocessor = ColumnTransformer(
            transformers=transformers) if transformers else None

    def fit_transform(self, df: pd.DataFrame):
        """Return processed_df, likert_df (no logging)."""
        if self.preprocessor is None and self.likert_preprocessor is None:
            self._create_preprocessor(df)

        likert_df = None
        if self.likert_features:
            likert_data = df[self.likert_features].copy()
            if self.likert_preprocessor:
                likert_array = self.likert_preprocessor.fit_transform(
                    likert_data)
                likert_df = pd.DataFrame(likert_array,
                                         columns=self.likert_features)
            else:
                likert_df = likert_data
            self.likert_data = likert_df

        processed_df = None
        if self.preprocessor:
            non_likert_cols = self.continuous_features + self.categorical_features
            if non_likert_cols:
                processed_array = self.preprocessor.fit_transform(
                    df[non_likert_cols])
                if hasattr(processed_array, "toarray"):
                    processed_array = processed_array.toarray()
                processed_df = pd.DataFrame(processed_array)

        return processed_df, likert_df

    def get_likert_data(self):
        """Return the preprocessed Likert subset."""
        return self.likert_data
