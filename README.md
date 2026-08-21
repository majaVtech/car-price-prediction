# Car Price Prediction

## Project Overview

This project is a machine learning regression project for predicting used car prices based on vehicle characteristics.

The project follows a complete machine learning workflow, starting with exploratory data analysis and continuing through data cleaning, feature engineering, preprocessing, model training, evaluation, model comparison, and final model selection.

The main goal is to develop a regression model that can predict the price of a used car based on information such as:

- Make
- Model
- Year
- Condition
- Mileage
- Fuel type
- Engine volume
- Color
- Transmission
- Drive unit
- Segment

The final selected model is a **Random Forest Regressor**, which achieved the best overall performance among the tested models.

---

## Project Objectives

The main objectives of this project are:

- Explore and understand the car dataset.
- Analyze numerical and categorical variables.
- Identify missing values.
- Detect completely duplicated records.
- Investigate potential outliers.
- Clean and standardize the dataset.
- Create additional features useful for prediction.
- Build a preprocessing pipeline.
- Train several regression models.
- Evaluate model performance.
- Compare different regression algorithms.
- Select the best-performing model.
- Save the final model for future predictions.

---

# Dataset

The project uses a dataset containing information about used cars.

The original dataset is stored in:

Data/cars.csv

The dataset contains information about:

Make

Model

Price

Year

Condition

Mileage

Fuel type

Engine volume

Color

Transmission

Drive unit

Segment

The target variable is:

price_usd
which represents the price of the vehicle in US dollars.

Project Structure
car-price-prediction/
│
├── Data/
│   ├── cars.csv
│   ├── cars_dataset_cleaned.csv
│   └── cars_dataset_features.csv
│
├── Notebooks/
│   └── 01_eda.ipynb
│
├── scripts/
│   ├── data_cleaning.py
│   ├── feature_engineering.py
│   ├── data_preprocessing.py
│   ├── model_training.py
│   ├── model_evaluation.py
│   └── model_comparison.py
│
├── models/
│   ├── final_random_forest_model.pkl
│   └── linear_regression_model.pkl
│
├── README.md
└── requirements.txt
The final model files are stored using Git Large File Storage (Git LFS) because the Random Forest model exceeds GitHub's standard 100 MB file-size limit.

1. Exploratory Data Analysis
Exploratory Data Analysis is performed in:

Notebooks/01_eda.ipynb
The purpose of EDA is to understand the structure and quality of the original dataset before applying data cleaning and machine learning techniques.

The analysis includes:

Dataset dimensions

Data types

Descriptive statistics

Missing value analysis

Unique values

Categorical variable distributions

Numerical variable distributions

Duplicate detection

Outlier detection

Boxplots

Statistical analysis

Numerical Variables
The main numerical variables include:

price_usd
year
mileage_kilometers
volume_cm3
Descriptive statistics were used to examine:

Mean

Standard deviation

Minimum

Maximum

Quartiles

Potentially extreme values were also investigated using the IQR method.

Outlier Analysis
Potential outliers were identified using the Interquartile Range (IQR) method.

The IQR is calculated as:

IQR = Q3 - Q1
The lower and upper boundaries are:

Lower Bound = Q1 - 1.5 × IQR

Upper Bound = Q3 + 1.5 × IQR
Boxplots were used to visually investigate potentially extreme observations.

It is important to note that an extreme value is not automatically an invalid value.

For example, a vehicle with very high mileage or a high price may still represent a legitimate observation.

Therefore, outlier detection was used as part of the exploratory analysis rather than automatically removing every detected outlier.

2. Duplicate Analysis
The dataset was checked for completely duplicated rows using:

df.duplicated().sum()
The analysis identified:

47
completely duplicated rows.

Additional duplicate checks were performed using combinations of selected columns, including:

year + mileage
year + mileage + volume
year + mileage + volume + price
These combinations produced many more repeated records.

However, many of these records differed in other vehicle characteristics such as:

Make

Model

Condition

Fuel type

Color

Transmission

Drive unit

Segment

Price

Therefore, partial duplicates were not automatically removed.

The analysis focused on completely duplicated rows because they represent exact duplicate records.

3. Data Cleaning
Data cleaning is implemented in:

scripts/data_cleaning.py
The cleaning pipeline performs several transformations.

Column Name Standardization
Column names are standardized using lowercase letters and underscores.

Examples:

priceUSD
becomes:

price_usd
and:

mileage(kilometers)
becomes:

mileage_kilometers
while:

volume(cm3)
becomes:

volume_cm3
This provides a consistent naming convention throughout the project.

String Cleaning
Leading and trailing whitespace is removed from text columns.

This prevents values such as:

"BMW"
and:

" BMW "
from being treated as different values.

Missing Values
Different representations of missing values are standardized.

Examples include:

""
" "
"nan"
"NaN"
"null"
"None"
These values are converted to proper missing values.

Numerical Conversion
The following columns are converted to numerical data types:

mileage_kilometers
volume_cm3
year
price_usd
Invalid numerical values are converted to missing values using:

pd.to_numeric(..., errors="coerce")
Categorical Standardization
Categorical variables are standardized using lowercase values and whitespace removal.

The main categorical columns are:

make
model
condition
fuel_type
color
transmission
drive_unit
segment
Drive unit values are also standardized.

The following mapping is used:

front-wheel drive              → FWD
rear drive                     → RWD
all-wheel drive                → AWD
part-time four-wheel drive     → 4WD
Target Variable
The target variable is:

price_usd
Rows with missing target values are removed because a supervised regression model requires a known target value during training.

4. Feature Engineering
Feature engineering is implemented in:

scripts/feature_engineering.py
Additional features were created to provide the models with more useful information.

Car Age
The car_age feature represents the approximate age of the vehicle.

Conceptually:

car_age = current_year - year
This allows the model to work directly with vehicle age.

Mileage Per Year
The following feature was created:

mileage_per_year
It represents the approximate mileage accumulated per year.

The calculation is:

mileage_per_year =
mileage_kilometers / car_age
Special handling is applied to avoid division by zero.

Engine Volume in Liters
The original engine volume is provided in cubic centimeters.

A new feature called:

engine_volume_liters
is created using:

engine_volume_liters = volume_cm3 / 1000
This provides a more intuitive representation of engine size.

5. Data Preprocessing
Data preprocessing is implemented in:

scripts/data_preprocessing.py
The features are separated into numerical and categorical variables.

Numerical Features
Examples include:

year
mileage_kilometers
volume_cm3
car_age
mileage_per_year
engine_volume_liters
Numerical preprocessing uses:

SimpleImputer

StandardScaler

SimpleImputer is used to handle missing numerical values.

StandardScaler standardizes numerical features.

Categorical Features
Categorical features include:

make
model
condition
fuel_type
color
transmission
drive_unit
segment
Categorical preprocessing uses:

SimpleImputer

OneHotEncoder

Missing categorical values are handled using SimpleImputer.

OneHotEncoder converts categorical values into numerical features suitable for machine learning algorithms.

6. Regression Models
Four regression algorithms were trained and compared:

Linear Regression

Decision Tree Regressor

Random Forest Regressor

Gradient Boosting Regressor

Linear Regression
Linear Regression was used as a baseline model.

It assumes a linear relationship between the input features and the target variable.

It provides a simple reference point for comparing more complex models.

Decision Tree Regressor
Decision Tree Regressor is a non-linear regression algorithm that can capture relationships between different vehicle characteristics and price.

It can model non-linear relationships but may overfit if not properly controlled.

Random Forest Regressor
Random Forest Regressor is an ensemble algorithm that combines predictions from multiple decision trees.

It is capable of learning complex and non-linear relationships between vehicle characteristics and price.

Random Forest achieved the best overall performance in this project and was therefore selected as the final model.

Gradient Boosting Regressor
Gradient Boosting Regressor builds an ensemble of trees sequentially, with each new tree attempting to correct errors made by previous trees.

It was included as another advanced regression algorithm for comparison.

7. Model Evaluation
The models were evaluated using standard regression metrics.

The following metrics were used:

MAE — Mean Absolute Error
MAE measures the average absolute difference between predicted and actual prices.

MAE = average(|actual - predicted|)
Lower values indicate better performance.

RMSE — Root Mean Squared Error
RMSE measures the square root of the average squared prediction error.

RMSE = sqrt(mean((actual - predicted)^2))
RMSE gives greater importance to larger prediction errors.

Lower values indicate better performance.

R² — R-squared
R² measures how much of the variation in the target variable is explained by the model.

Higher values indicate better performance.

A value closer to 1 indicates that the model explains a larger proportion of the variation in car prices.

8. Model Comparison
The models were compared using the same dataset and evaluation procedure.

The tested models were:

Linear Regression
Decision Tree Regressor
Random Forest Regressor
Gradient Boosting Regressor
The comparison was based primarily on:

MAE

RMSE

R²

The model comparison is implemented in:

scripts/model_comparison.py
The Random Forest Regressor achieved the best overall results and was selected as the final model.

9. Final Model Selection
The final model selected for this project is:

Random Forest Regressor
Random Forest was selected because it provided the strongest overall performance among the tested models.

This model is particularly suitable for this dataset because car prices can depend on complex and non-linear relationships between:

Vehicle age

Mileage

Engine volume

Make

Model

Fuel type

Transmission

Drive unit

Vehicle segment

Random Forest can capture these non-linear relationships and interactions between features more effectively than a simple linear model.

10. Saved Final Model
The final Random Forest model is stored in:

models/final_random_forest_model.pkl
A Linear Regression model was also saved for comparison:

models/linear_regression_model.pkl
The .pkl files are managed using Git LFS because the final Random Forest model is larger than GitHub's standard 100 MB file limit.

11. Loading the Final Model
The final model can be loaded using joblib:

import joblib

model = joblib.load(
    "models/final_random_forest_model.pkl"
)

print("Final Random Forest model loaded successfully.")
Before loading the model after cloning the repository, make sure Git LFS files have been downloaded:

git lfs install
git lfs pull
12. Technologies Used
The project was developed using:

Programming Language
Python

Data Analysis
Pandas

NumPy

Machine Learning
Scikit-learn

Visualization
Matplotlib

Seaborn

Model Persistence
Joblib

Development Environment
Jupyter Notebook

Version Control
Git

GitHub

Git LFS

13. Requirements
The required Python packages are listed in:

requirements.txt
The main libraries used in the project are:

pandas
numpy
scikit-learn
matplotlib
seaborn
joblib
jupyter
14. Installation
Clone the repository:

git clone https://github.com/majaVtech/car-price-prediction.git
Navigate to the project directory:

cd car-price-prediction
Create a virtual environment:

python -m venv venv
Activate the virtual environment on Windows:

venv\Scripts\activate
Install the required packages:

pip install -r requirements.txt
15. Git LFS
Git LFS is required because the saved machine learning models are large files.

Install and initialize Git LFS:

git lfs install
Download the model files:

git lfs pull
The final model should then be available at:

models/final_random_forest_model.pkl
16. Running the Project
The recommended workflow is:

Step 1 — Exploratory Data Analysis
Open:

Notebooks/01_eda.ipynb
and run the notebook.

Step 2 — Data Cleaning
Run:

python scripts/data_cleaning.py
This creates the cleaned dataset.

Step 3 — Feature Engineering
Run:

python scripts/feature_engineering.py
This creates additional features such as:

car_age
mileage_per_year
engine_volume_liters
Step 4 — Data Preprocessing
Run:

python scripts/data_preprocessing.py
This prepares the numerical and categorical features for machine learning.

Step 5 — Model Training
Run:

python scripts/model_training.py
This trains the regression model and saves the trained model.

Step 6 — Model Evaluation
Run:

python scripts/model_evaluation.py
This calculates:

MAE
RMSE
R²
Step 7 — Model Comparison
Run:

python scripts/model_comparison.py
This compares the four tested regression algorithms.

17. Machine Learning Workflow
The complete workflow can be summarized as:

Raw Dataset
     │
     ▼
Exploratory Data Analysis
     │
     ▼
Data Cleaning
     │
     ▼
Feature Engineering
     │
     ▼
Data Preprocessing
     │
     ▼
Model Training
     │
     ▼
Model Evaluation
     │
     ▼
Model Comparison
     │
     ▼
Final Model Selection
     │
     ▼
Random Forest Regressor
     │
     ▼
Saved Final Model
18. Future Improvements
The project can be further improved through:

Hyperparameter tuning

Cross-validation

Grid Search

Randomized Search

Additional feature engineering

Feature importance analysis

More detailed error analysis

Testing additional regression algorithms

Gradient boosting optimization

XGBoost or similar algorithms

Model interpretability

Deployment as a web application

19. Conclusion
This project demonstrates a complete machine learning workflow for used car price prediction.

The process started with exploratory data analysis to understand the dataset and identify missing values, duplicates, unusual observations and variable distributions.

The dataset was then cleaned and standardized.

Feature engineering was performed to create additional variables such as:

car_age
mileage_per_year
engine_volume_liters
Numerical and categorical features were prepared using Scikit-learn preprocessing techniques including:

SimpleImputer
StandardScaler
OneHotEncoder
Four regression algorithms were trained and compared:

Linear Regression
Decision Tree Regressor
Random Forest Regressor
Gradient Boosting Regressor
Based on the evaluation results, the Random Forest Regressor achieved the best overall performance and was selected as the final model.

The final trained model is stored as:

models/final_random_forest_model.pkl
Because of its large file size, the model is stored using Git Large File Storage (Git LFS).

The project provides a reproducible machine learning pipeline that can be further extended with hyperparameter optimization, cross-validation, additional feature engineering and model deployment.

Author
Maja Vtech

GitHub:

https://github.com/majaVtech
