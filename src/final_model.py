import pandas as pd
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from data_preprocessing import (
    TARGET_COLUMN,
    create_preprocessing_pipeline
)


# File paths
FEATURED_DATA_PATH = "cars_dataset_features.csv"
FINAL_MODEL_PATH = "final_random_forest_model.pkl"


def main() -> None:
    """Train and save the final Random Forest model."""

    print("Loading feature-engineered dataset...")

    df = pd.read_csv(FEATURED_DATA_PATH)

    # Separate features and target
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    print(f"Dataset shape: {df.shape}")
    print(f"Target variable: {TARGET_COLUMN}")

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")

    # Create preprocessing pipeline
    preprocessor = create_preprocessing_pipeline()

    # Define final Random Forest model
    random_forest = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )

    # Combine preprocessing and model
    final_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", random_forest)
        ]
    )

    print("\nTraining final Random Forest model...")

    # Train the complete pipeline
    final_pipeline.fit(X_train, y_train)

    print("Training completed.")

    # Save preprocessing + model together
    joblib.dump(
        final_pipeline,
        FINAL_MODEL_PATH
    )

    print(
        f"\nFinal model saved to: {FINAL_MODEL_PATH}"
    )


if __name__ == "__main__":
    main()