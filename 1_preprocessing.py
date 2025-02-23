import pandas as pd
import numpy as np
import sys
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
# Setting UTF-8 output
sys.stdout.reconfigure(encoding='utf-8')

# Expanding output options to view all rows
pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)

# ==========================================
# 1. Loading data
# ==========================================
# Loading CSV file into DataFrame
housing = pd.read_csv('datasets/AmesHousing.csv', delimiter=',', na_values=[], keep_default_na=False)

# Checking the first rows of the dataset
print("=== First 5 rows ===")
print(housing.head())

# Information about the initial state of the DataFrame
print("\n=== Data Information ===")
print(housing.info())

# ==========================================
# 2. Processing numerical variables with missing values
# ==========================================
# List of numerical columns that should be Int64
columns_to_int = [
    'Lot Frontage', 'Mas Vnr Area', 'BsmtFin SF 1', 'BsmtFin SF 2', 
    'Bsmt Unf SF', 'Total Bsmt SF', 'Garage Yr Blt', 'Garage Cars', 
    'Garage Area', 'Bsmt Full Bath', 'Bsmt Half Bath'
]

# Replacing empty strings with NaN
housing[columns_to_int] = housing[columns_to_int].replace('', np.nan)

# Converting numerical variables to Int64 type
housing[columns_to_int] = housing[columns_to_int].astype('Int64')

# Checking updated information about the DataFrame
print("\n=== Information after processing numerical columns ===")
print(housing.info())

# ==========================================
# 3. Processing categorical variables
# ==========================================
# Replacing empty values with 'None'
housing['Mas Vnr Type'] = housing['Mas Vnr Type'].replace('', 'None')
housing['Garage Type'] = housing['Garage Type'].replace('NA', 'None')

# Replacing empty values in the 'Mas Vnr Area' column with 0
housing['Mas Vnr Area'] = housing['Mas Vnr Area'].fillna(0)

# List of columns to fill
bsmt_columns = [
    'Bsmt Qual', 'Bsmt Cond', 'Bsmt Exposure',
    'BsmtFin Type 1', 'BsmtFin SF 1', 'BsmtFin Type 2',
    'BsmtFin SF 2', 'Bsmt Unf SF', 'Total Bsmt SF'
]

# Default values for columns
default_values = {
    'Bsmt Qual': 'NA',       # No basement
    'Bsmt Cond': 'NA',       # No basement
    'Bsmt Exposure': 'NA',   # No windows
    'BsmtFin Type 1': 'NA',  # No finished area
    'BsmtFin SF 1': 0,       # Zero area
    'BsmtFin Type 2': 'NA',  # No second finished area
    'BsmtFin SF 2': 0,       # Zero area
    'Bsmt Unf SF': 0,        # Zero unfinished area
    'Total Bsmt SF': 0       # Total area = 0
}

# Replacing empty values in columns with default values
housing[bsmt_columns] = housing[bsmt_columns].replace("", default_values)
# Filling missing values with zeros
housing[bsmt_columns] = housing[bsmt_columns].fillna(0)

# Checking if all missing values are filled
print("\n=== Checking missing values after filling Bsmt columns ===")
print(housing[bsmt_columns].isnull().sum())  # Should be 0

# ==========================================
# 4. Converting text columns to categorical
# ==========================================
# Converting text (object) columns to categorical
object_columns = housing.select_dtypes(include=['object']).columns
housing[object_columns] = housing[object_columns].astype('category')

# Converting numerical variables that are actually categorical
categorical_columns = ['MS SubClass', 'Overall Qual', 'Overall Cond', 'Mo Sold']
housing[categorical_columns] = housing[categorical_columns].astype('category')

# Checking updated information about the DataFrame
print("\n=== Information after processing categorical columns ===")
print(housing.info())

# ==========================================
# 5. Analyzing missing data
# ==========================================
# Checking the number of missing values in each column
print("\n=== Number of missing values in columns ===")
print(housing.isnull().sum())

# ==========================================
# 6. Processing and filling missing values in Garage columns
# ==========================================

# 6.1. Filling missing values for 'Garage Yr Blt'
# Creating a mask for rows where 'Garage Type' == 'None'
mask = housing['Garage Type'] == 'None'

# Filling 'Garage Yr Blt' with 0 for rows that match the condition
housing.loc[mask, 'Garage Yr Blt'] = 0

# Filtering rows where 'Garage Yr Blt' is missing and 'Garage Type' is not 'None'
mask = housing['Garage Yr Blt'].isnull() & (housing['Garage Type'] != 'None')

# Filling 'Garage Yr Blt' with values from the 'Year Built' column
housing.loc[mask, 'Garage Yr Blt'] = housing.loc[mask, 'Year Built']

# Checking the number of missing values in the 'Garage Yr Blt' column
print("\n=== Number of missing values in 'Garage Yr Blt' ===")
print(housing['Garage Yr Blt'].isnull().sum())  # Should be 0

# 6.2. Filling missing values in 'Garage Finish', 'Garage Cond', and 'Garage Qual'
# Filling missing values in 'Garage Finish' with 'Unf' (unfinished garage)
housing['Garage Finish'] = housing['Garage Finish'].fillna('Unf')

# Filling missing values in 'Garage Cond' and 'Garage Qual' with 'TA' (Typical/Average)
housing['Garage Cond'] = housing['Garage Cond'].fillna('TA')
housing['Garage Qual'] = housing['Garage Qual'].fillna('TA')

# Checking the result
print("\n=== Checking missing values after filling Garage columns ===")
print(housing[['Garage Finish', 'Garage Cond', 'Garage Qual']].isnull().sum())  # Should be 0

# 6.3. Processing missing values in 'Garage Cars' for type 'Detchd'
# Filtering rows where 'Garage Type' == 'Detchd'
detchd_garages = housing[housing['Garage Type'] == 'Detchd']

# Calculating the average number of cars for each 'Bedroom AbvGr' value
average_cars_per_bedroom = detchd_garages.groupby('Bedroom AbvGr')['Garage Cars'].mean().round()

# Displaying the result
print("\nAverage number of cars for detached garages (Detchd):")
print(average_cars_per_bedroom)

# Function to fill missing values in 'Garage Cars'
def fill_garage_cars(row):
    if pd.isnull(row['Garage Cars']) and row['Garage Type'] == 'Detchd':
        return average_cars_per_bedroom.get(row['Bedroom AbvGr'], 1)  # Default to 1 if no match
    return row['Garage Cars']

# Applying the function to the DataFrame
housing['Garage Cars'] = housing.apply(fill_garage_cars, axis=1)

# Converting 'Garage Cars' to integer type
housing['Garage Cars'] = housing['Garage Cars'].astype(int)

# Checking the result
print(housing['Garage Cars'].isnull().sum())  # Should be 0 if all missing values are filled

# 6.4. Processing missing values in 'Garage Area' for type 'Detchd'
# Filtering rows where 'Garage Type' == 'Detchd' and 'Garage Area' is not missing
detchd_garages = housing[(housing['Garage Type'] == 'Detchd') & housing['Garage Area'].notnull()]

# Calculating the average garage area for each number of cars
average_area_per_cars = detchd_garages.groupby('Garage Cars')['Garage Area'].mean().round()

# Function to fill missing values in 'Garage Area'
def fill_garage_area(row):
    if pd.isnull(row['Garage Area']) and row['Garage Type'] == 'Detchd':
        return average_area_per_cars.get(row['Garage Cars'], 200)  # Default to 200 if no match
    return row['Garage Area']

# Applying the function to the DataFrame
housing['Garage Area'] = housing.apply(fill_garage_area, axis=1)

# Checking the result
print(housing['Garage Area'].isnull().sum())  # Should be 0 if all missing values are filled

# ==========================================
# 7. Filling missing values in Basement columns
# ==========================================
# Filling missing values with zeros for 'Bsmt Full Bath' and 'Bsmt Half Bath'
housing['Bsmt Full Bath'] = housing['Bsmt Full Bath'].fillna(0).astype(int)
housing['Bsmt Half Bath'] = housing['Bsmt Half Bath'].fillna(0).astype(int)

# Checking if there are any remaining missing values
print("\n=== Checking missing values after filling Basement columns ===")
print(housing[['Bsmt Full Bath', 'Bsmt Half Bath']].isnull().sum())  # Should be 0

# ==========================================
# 8. Filling missing values in Lot Frontage
# ==========================================
# Calculating median Lot Frontage values for each MS Zoning type
median_values = housing.groupby('MS Zoning')['Lot Frontage'].median()

# Function to fill missing Lot Frontage values
def fill_lot_frontage(row):
    if pd.isnull(row['Lot Frontage']):
        if row['MS Zoning'] in ['C (all)', 'FV', 'RH', 'RM']:
            # Fill with the median value for the corresponding MS Zoning
            return median_values[row['MS Zoning']]
        elif row['MS Zoning'] == 'I (all)':
            # Set to 0 for industrial zone
            return 0
    return row['Lot Frontage']

# Applying the function to each row
housing['Lot Frontage'] = housing.apply(fill_lot_frontage, axis=1)

# Converting to numerical type Int64, filling NaNs with zeros (or other value if needed)
housing['Lot Frontage'] = housing['Lot Frontage'].astype('Int64')

# Checking if there are no missing values
print("\n=== Number of missing values in Lot Frontage after filling ===")
print(housing['Lot Frontage'].isnull().sum())  # Should be 0

# ==========================================
# 9. Creating a model to predict Lot Frontage
# ==========================================

# Filtering rows with MS Zoning == 'RL'
rl_data = housing[housing['MS Zoning'] == 'RL']

# Selecting variables for training (all numerical except 'SalePrice')
features = ['Lot Area', 'Garage Area', '1st Flr SF', 'Gr Liv Area']

# Preparing training data
train_data = rl_data[rl_data['Lot Frontage'].notnull()]
X_train = train_data[features]
y_train = train_data['Lot Frontage']

# Creating a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Preparing data for prediction (rows with missing Lot Frontage values)
predict_data = rl_data[rl_data['Lot Frontage'].isnull()]
X_predict = predict_data[features]

# Predicting missing Lot Frontage values
predicted_values = model.predict(X_predict)

# Filling missing values in the Lot Frontage column
housing.loc[predict_data.index, 'Lot Frontage'] = np.round(predicted_values)

# Displaying the first 10 filled Lot Frontage values
print("\n=== First 10 filled Lot Frontage values ===")
print(housing.loc[predict_data.index, ['Lot Frontage']].head(10))

# ==========================================
# 10. Checking and saving results
# ==========================================
# Checking the number of missing values in each column
print("\n=== Number of missing values in columns after processing ===")
print(housing.isnull().sum())

# ==========================================
# 11. Creating new features based on existing data
# ==========================================

# 11.1. Age of the house
housing['Age of House'] = housing['Yr Sold'] - housing['Year Built']

# 11.2. Age of the last renovation
housing['Age of Renovation'] = housing['Yr Sold'] - housing['Year Remod/Add']
housing['Renovation Flag'] = (housing['Age of Renovation'] > 0).astype(int)  # 1 - if renovation was done, 0 - if not

# 11.4. Ratio of living area to lot area
housing['Living Area to Lot Area Ratio'] = housing['Gr Liv Area'] / housing['Lot Area']

# 11.5. Total number of bathrooms
housing['Total Bathrooms'] = housing['Full Bath'] + housing['Half Bath'] + housing['Bsmt Full Bath'] + housing['Bsmt Half Bath']

# 11.6. Total number of fireplaces
housing['Total Fireplaces'] = housing['Fireplaces'] + housing['Bsmt Full Bath']  # Assuming each fireplace may have a separate bathroom

# 11.7. Garage area per car
housing['Garage Area per Car'] = housing['Garage Area'] / housing['Garage Cars']

# 11.8. Ratio of veneer area to lot area
housing['Mas Vnr Area to Lot Area'] = housing['Mas Vnr Area'] / housing['Lot Area']
housing['Mas Vnr Area to Lot Area'] = housing['Mas Vnr Area to Lot Area'].astype('float64')

# 11.10. Total porch area
housing['Total Porch Area'] = housing['Wood Deck SF'] + housing['Open Porch SF'] + housing['Enclosed Porch'] + housing['3Ssn Porch'] + housing['Screen Porch']

# 11.11. Ratio of sale price to lot area
housing['SalePrice to Lot Area Ratio'] = housing['SalePrice'] / housing['Lot Area']

# 11.12. Price per square foot
housing['Price per Square Foot'] = housing['SalePrice'] / housing['Gr Liv Area']

# ==========================================
# 12. Standardizing (scaling) numerical variables
# ==========================================

# Dropping the 'Order' column
housing = housing.drop(columns=['Order', 'PID'])

# Selecting numerical columns for standardization
numeric_columns = housing.select_dtypes(include=[np.number]).columns

# Excluding the 'SalePrice' column as it may be the dependent variable
numeric_columns = numeric_columns.drop('SalePrice')

# Standardizing all numerical columns
scaler = StandardScaler()
housing[numeric_columns] = scaler.fit_transform(housing[numeric_columns])

# Checking if values changed after standardization
print("\n=== Checking results of standardization ===")
print(housing[numeric_columns].head())

# Saving results to CSV
housing.to_csv('datasets/preproc.csv')
