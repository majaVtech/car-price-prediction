import pandas as pd
import re
RAW_DATA_PATH = "cars.csv"
CLEANED_DATA_PATH = "cars_dataset_cleaned.csv"
# Standardization of columns name
def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    new_columns = []
    
    for col in df.columns:
        clean_col = col.strip().lower()
    
        clean_col = clean_col.replace("(", "_")
        clean_col = clean_col.replace(")", "")
        clean_col = clean_col.replace("-", "_")
        clean_col = clean_col.replace("/", "_")

        clean_col = re.sub(r"\s+", "_", clean_col)
        clean_col = re.sub(r"[^a-z0-9_]", "", clean_col)
        clean_col = re.sub(r"_+", "_", clean_col)
        clean_col = clean_col.strip("_")

        if clean_col == "priceusd": 
            clean_col = "price_usd"

        new_columns.append(clean_col)
        
    df.columns = new_columns
        
    return df
# Strip string values
def strip_string_values(df: pd.DataFrame) -> pd.DataFrame:
    
    df = df.copy()
    
    text_columns = df.select_dtypes(include=["object"]).columns

    for column in text_columns:
        df[column] = df[column].str.strip()

    return df
# Defying missing like values
MISSING_LIKE_VALUES = {
    "",
    " ",
    "nan",
    "NaN",
    "NAN",
    "null",
    "Null",
    "NULL",
    "none",
    "None",
    "NONE",
}
# Replacing missing values
def replace_missing_like_values(df: pd.DataFrame) -> pd.DataFrame:
 
    df = df.copy()
 
    df = df.replace(list(MISSING_LIKE_VALUES), pd.NA)
 
    return df
# Converting numeric columns
def convert_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
 
    df = df.copy()
 
    numeric_columns = [
    "mileage_kilometers",
    "volume_cm3",
    "year",
    "price_usd"
]
 
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
 
    return df

# Standardization category values 
def clean_categorical_values(df: pd.DataFrame) -> pd.DataFrame:
    
    df = df.copy()

    categorical_columns = [
    "make",
    "model",
    "condition",
    "fuel_type",
    "color",
    "transmission",
    "drive_unit",
    "segment"
]
    
    for col in categorical_columns:
        if col in df.columns:
            df[col] = df[col].astype("string").str.strip().str.lower()
    
    if 'drive_unit' in df.columns:
        df['drive_unit'] = df['drive_unit'].replace({
            "front-wheel drive": "FWD",
            "rear drive": "RWD",
            "all-wheel drive": "AWD",
            "part-time four-wheel drive": "4WD"
        })
        
    return df

# Removing missing valuse from higly imortant column
def remove_rows_with_missing_target(df: pd.DataFrame) -> pd.DataFrame:
 
    df = df.copy()
 
    df = df.dropna(subset=["price_usd"])
 
    return df

# gethering all into pipline
def clean(df: pd.DataFrame) -> pd.DataFrame:
 
    df_clean = (
        df
        .pipe(standardize_column_names)
        .pipe(strip_string_values)
        .pipe(replace_missing_like_values)
        .pipe(convert_numeric_columns)
        .pipe(clean_categorical_values)
        .pipe(remove_rows_with_missing_target)
        .reset_index(drop=True)
    )
 
    return df_clean
# loading raw datay cleaning it and saving it in new CSV file
def main() -> None:
    """Load raw data, clean it, and save the cleaned dataset."""
    print("Loading raw dataset...")
 
    df_raw = pd.read_csv(RAW_DATA_PATH)
 
    print("Cleaning dataset...")
 
    df_cleaned = clean(df_raw)
 
    print("Saving cleaned dataset...")
 
    df_cleaned.to_csv(CLEANED_DATA_PATH, index=False)
 
    print(f"Cleaned dataset saved to: {CLEANED_DATA_PATH}")
    
    	
if __name__ == "__main__":
    main()