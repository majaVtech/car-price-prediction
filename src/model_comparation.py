import pandas as pd

from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
    r2_score
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

from data_preprocessing import (
    TARGET_COLUMN,
    create_preprocessing_pipeline
)


# File paths
FEATURED_DATA_PATH = "cars_dataset_features.csv"
RESULTS_PATH = "model_comparison_results.csv"


def evaluate_model(
    model_name,
    model,
    X_train,
    X_test,
    y_train,
    y_test
):
    """
    Train and evaluate a regression model.
    """

    print(f"\nTraining {model_name}...")

    # Create a new preprocessing pipeline for each model
    preprocessor = create_preprocessing_pipeline()

    # Combine preprocessing and model
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    # Train the model
    pipeline.fit(X_train, y_train)

    # Generate predictions
    y_pred = pipeline.predict(X_test)

    # Calculate evaluation metrics
    mae = mean_absolute_error(
        y_test,
        y_pred
    )

    rmse = mean_squared_error(
        y_test,
        y_pred
    ) ** 0.5

    r2 = r2_score(
        y_test,
        y_pred
    )

    mape = mean_absolute_percentage_error(
        y_test,
        y_pred
    ) * 100

    print(f"{model_name} training completed.")

    print(f"MAE:  {mae:.2f} USD")
    print(f"RMSE: {rmse:.2f} USD")
    print(f"R²:   {r2:.4f}")
    print(f"MAPE: {mape:.2f}%")

    return {
        "Model": model_name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "MAPE": mape
    }


def main() -> None:
    """Compare multiple regression models."""

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

    # Define regression models
    models = {
        "Linear Regression": LinearRegression(),

        "Decision Tree": DecisionTreeRegressor(
            random_state=42
        ),

        "Random Forest": RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        ),

        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42
        )
    }

    # Store model evaluation results
    results = []

    # Train and evaluate all models
    for model_name, model in models.items():

        result = evaluate_model(
            model_name=model_name,
            model=model,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test
        )

        results.append(result)

    # Create results DataFrame
    results_df = pd.DataFrame(results)

    # Sort models by R² score
    results_df = results_df.sort_values(
        by="R2",
        ascending=False
    ).reset_index(drop=True)

    # Display comparison
    print("\nModel Comparison")
    print("=" * 75)

    print(
        results_df.to_string(
            index=False,
            formatters={
                "MAE": "{:.2f}".format,
                "RMSE": "{:.2f}".format,
                "R2": "{:.4f}".format,
                "MAPE": "{:.2f}".format
            }
        )
    )

    # Identify the best model
    best_model = results_df.iloc[0]

    print("\nBest Model")
    print("=" * 75)

    print(f"Model: {best_model['Model']}")
    print(f"MAE:   {best_model['MAE']:.2f} USD")
    print(f"RMSE:  {best_model['RMSE']:.2f} USD")
    print(f"R²:    {best_model['R2']:.4f}")
    print(f"MAPE:  {best_model['MAPE']:.2f}%")

    # Save comparison results
    results_df.to_csv(
        RESULTS_PATH,
        index=False
    )

    print(
        f"\nComparison results saved to: {RESULTS_PATH}"
    )


if __name__ == "__main__":
    main()