from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
 
from src.FeatureEngineering import (
    DateExtractoer,
    FlagFeatureAdder,
    DropColumns,
)
 
 
def build_preprocessing_pipeline() -> Pipeline:

    encode_seasons = ColumnTransformer(
        transformers=[
            ("season_ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["Seasons"]),
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")
 
    return Pipeline(steps=[
        ("date_features", DateExtractoer()),
        ("flags", FlagFeatureAdder()),
        ("column_select", DropColumns()),
        ("encode_seasons", encode_seasons),
    ])
 
 
def build_full_pipeline(model) -> Pipeline:
    return Pipeline(steps=[
        ("preprocessing", build_preprocessing_pipeline()),
        ("model", model),
    ])