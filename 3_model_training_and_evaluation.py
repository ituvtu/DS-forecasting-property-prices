# ==========================================
# 1. Import necessary libraries
# ==========================================
import pandas as pd
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import sys

# Setting for UTF-8 output
sys.stdout.reconfigure(encoding='utf-8')

# ==========================================
# 2. Loading data
# ==========================================
# Load CSV file into DataFrame
housing = pd.read_csv('datasets/preproc.csv')

# Check the first rows of the dataset
print("=== First 5 rows ===")
print(housing.head())

# ==========================================
# 3. Selecting features for the model
# ==========================================
# Selecting variables to train the model
selected_features = [
    'Overall Qual', 'Year Built', 'Year Remod/Add', 'Mas Vnr Area', 
    'Total Bsmt SF', 'Gr Liv Area', 'Garage Cars', 
    'Total Bathrooms', 'Total Fireplaces', 'Price per Square Foot'
]

# Creating X (input variables) and y (target variable)
X = housing[selected_features]
y = housing['SalePrice']

# ==========================================
# 4. Splitting into training and test sets
# ==========================================
# Splitting data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==========================================
# 5. Creating and training models
# ==========================================
# Linear Regression (without regularization)
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# Lasso (L1 regularization)
lasso_model = Lasso(alpha=0.1)  # Change alpha for different levels of regularization
lasso_model.fit(X_train, y_train)

# Ridge (L2 regularization)
ridge_model = Ridge(alpha=0.1)  # Change alpha for different levels of regularization
ridge_model.fit(X_train, y_train)

# ==========================================
# 6. Evaluating models
# ==========================================
# Linear Regression
y_pred_linear = linear_model.predict(X_test)
mse_linear = mean_squared_error(y_test, y_pred_linear)
r2_linear = r2_score(y_test, y_pred_linear)

# Lasso
y_pred_lasso = lasso_model.predict(X_test)
mse_lasso = mean_squared_error(y_test, y_pred_lasso)
r2_lasso = r2_score(y_test, y_pred_lasso)

# Ridge
y_pred_ridge = ridge_model.predict(X_test)
mse_ridge = mean_squared_error(y_test, y_pred_ridge)
r2_ridge = r2_score(y_test, y_pred_ridge)

# Output evaluation results for each model
print("\n=== Model Evaluation ===")
print(f"Linear Regression: MSE = {mse_linear:.2f}, R² = {r2_linear:.2f}")
print(f"Lasso: MSE = {mse_lasso:.2f}, R² = {r2_lasso:.2f}")
print(f"Ridge: MSE = {mse_ridge:.2f}, R² = {r2_ridge:.2f}")

# ==========================================
# 7. Visualizing results
# ==========================================
# Visualize actual and predicted values for each model

# Linear Regression
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_linear, color='blue', edgecolor='black', alpha=0.6)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.title('Actual vs Predicted Values (Linear Regression)')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')

# Lasso
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_lasso, color='green', edgecolor='black', alpha=0.6)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.title('Actual vs Predicted Values (Lasso)')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')

# Ridge
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_ridge, color='orange', edgecolor='black', alpha=0.6)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.title('Actual vs Predicted Values (Ridge)')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')


# ==========================================
# 8. Outputting model coefficients
# ==========================================
print("\n=== Model Coefficients ===")
print("\nLinear Regression:")
for feature, coef in zip(selected_features, linear_model.coef_):
    print(f"{feature}: {coef:.2f}")

print("\nLasso:")
for feature, coef in zip(selected_features, lasso_model.coef_):
    print(f"{feature}: {coef:.2f}")

print("\nRidge:")
for feature, coef in zip(selected_features, ridge_model.coef_):
    print(f"{feature}: {coef:.2f}")

plt.show()