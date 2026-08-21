import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# Dataset path
FEATURED_DATA_PATH = "cars_dataset_features.csv"


# Target variable
TARGET_COLUMN = "price_usd"


# Numerical features
NUMERIC_COLUMNS = [
    "year",
    "mileage_kilometers",
    "volume_cm3",
    "car_age",
    "mileage_per_year",
    "engine_volume_liters"
]


# Categorical features
CATEGORICAL_COLUMNS = [
    "make",
    "model",
    "condition",
    "fuel_type",
    "color",
    "transmission",
    "drive_unit",
    "segment"
]


def create_preprocessing_pipeline() -> ColumnTransformer:
    """
    Create preprocessing pipeline for numerical
    and categorical features.
    """

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
                )
            )
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ( "numeric", numeric_pipeline, NUMERIC_COLUMNS),
            ("categorical", categorical_pipeline, CATEGORICAL_COLUMNS)
        ]
    )

    return preprocessor


def main() -> None:
    """Load dataset and create preprocessing pipeline."""

    print("Loading dataset...")

    df = pd.read_csv(FEATURED_DATA_PATH)

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    print("Target variable:", TARGET_COLUMN)
    print("Number of samples:", len(df))

    print("\nNumerical columns:")
    print(NUMERIC_COLUMNS)

    print("\nCategorical columns:")
    print(CATEGORICAL_COLUMNS)

    preprocessor = create_preprocessing_pipeline()

    print("\nPreprocessing pipeline created successfully.")
    print(preprocessor)


if __name__ == "__main__":
    main()