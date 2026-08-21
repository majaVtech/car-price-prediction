import joblib
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
    r2_score
)
from sklearn.model_selection import train_test_split

from data_preprocessing import TARGET_COLUMN


# File paths
FEATURED_DATA_PATH = "cars_dataset_features.csv"
MODEL_PATH = "linear_regression_model.pkl"


def main() -> None:
    """Load the dataset and trained model, generate predictions,
    and evaluate the model using regression metrics.
    """

    print("Loading dataset...")

    df = pd.read_csv(FEATURED_DATA_PATH)

    # Separate features and target variable
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    print(f"Dataset shape: {df.shape}")
    print(f"Target variable: {TARGET_COLUMN}")

    # Split data into training and testing sets
    # The same random_state and test_size used during training
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")

    # Load trained model pipeline
    print("\nLoading trained model...")

    model = joblib.load(MODEL_PATH)

    print("Model loaded successfully.")

    # Generate predictions for the test set
    print("\nGenerating predictions...")

    y_pred = model.predict(X_test)

    # Calculate regression metrics
    mae = mean_absolute_error(
        y_test,
        y_pred
    )

    rmse = mean_squared_error(
        y_test,
        y_pred,
    )

    r2 = r2_score(
        y_test,
        y_pred
    )

    mape = mean_absolute_percentage_error(
        y_test,
        y_pred
    )

    # Display evaluation results
    print("\nModel Evaluation")
    print("-" * 40)

    print(f"MAE:  {mae:.2f} USD")
    print(f"RMSE: {rmse:.2f} USD")
    print(f"R²:   {r2:.4f}")
    print(f"MAPE: {mape * 100:.2f}%")

    # Interpretation
    print("\nMetric Interpretation")
    print("-" * 40)

    print(
        f"MAE indicates that the model's predictions "
        f"are off by approximately {mae:.2f} USD on average."
    )

    print(
        f"RMSE is {rmse:.2f} USD and gives more weight "
        f"to larger prediction errors."
    )

    print(
        f"R² indicates that the model explains approximately "
        f"{r2 * 100:.2f}% of the variability in car prices."
    )

    print(
        f"MAPE indicates an average relative prediction error "
        f"of approximately {mape * 100:.2f}%."
    )


if __name__ == "__main__":
    main()