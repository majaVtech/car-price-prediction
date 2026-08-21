import joblib
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from data_preprocessing import (
    TARGET_COLUMN,
    create_preprocessing_pipeline
)


FEATURED_DATA_PATH = "cars_dataset_features.csv"
MODEL_PATH = "linear_regression_model.pkl"


def main() -> None:
    """Load data, train the baseline regression model and save it."""

    print("Loading dataset...")

    df = pd.read_csv(FEATURED_DATA_PATH)

    # Separate features and target variable
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    print(f"Dataset shape: {df.shape}")
    print(f"Target variable: {TARGET_COLUMN}")

    # Split data into training and testing sets
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

    # Create complete model pipeline
    model_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", LinearRegression())
        ]
    )

    print("\nTraining Linear Regression model...")

    # Train the model
    model_pipeline.fit(X_train, y_train)

    print("Model training completed.")

    # Save the complete pipeline
    joblib.dump(
        model_pipeline,
        MODEL_PATH
    )

    print(f"Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()