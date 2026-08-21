import pandas as pd


CLEANED_DATA_PATH = "cars_dataset_cleaned.csv"
FEATURED_DATA_PATH = "cars_dataset_features.csv"


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    
    df = df.copy()
    
    current_year = 2026
    
    # Calculate car age
    df["car_age"] = current_year - df["year"]
    
    # Convert engine volume from cm3 to liters
    df["engine_volume_liters"] = (
        df["volume_cm3"] / 1000
    ).round(1)
    
    # Calculate average mileage per year
    df["mileage_per_year"] = (
        df["mileage_kilometers"] /
        df["car_age"].replace(0, pd.NA)
    ).round(2)
    
    return df


def main() -> None:
    """Load cleaned data, create features, and save the result."""
    
    print("Loading cleaned dataset...")
    
    df = pd.read_csv(CLEANED_DATA_PATH)
    
    print("Creating new features...")
    
    df_features = create_features(df)
    
    print("Saving dataset with new features...")
    
    df_features.to_csv(
        FEATURED_DATA_PATH,
        index=False
    )
    
    print(
        f"Dataset with features saved to: {FEATURED_DATA_PATH}"
    )


if __name__ == "__main__":
    main()