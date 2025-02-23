import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys

# Setting for UTF-8 output
sys.stdout.reconfigure(encoding='utf-8')

housing = pd.read_csv('datasets/preproc.csv')
# Creating a correlation matrix for numerical data
numerical_cols = housing.select_dtypes(include=['float64', 'int64']).columns
correlation_matrix = housing[numerical_cols].corr()

# # Visualization of the correlation matrix
# plt.figure(figsize=(12, 10))
# sns.heatmap(correlation_matrix, annot=True, fmt=".5f", cmap="coolwarm", cbar=True)
# plt.title("Correlation Matrix of Numerical Features")
# plt.show()

# Selecting variables with high correlation
high_corr_features = correlation_matrix['SalePrice'][correlation_matrix['SalePrice'].abs() > 0.5]
print("Variables with high correlation with SalePrice (threshold 0.5):")
print(high_corr_features)

# If needed, you can also filter the correlation matrix and show only highly correlated variables
high_corr_matrix = correlation_matrix.loc[high_corr_features.index, high_corr_features.index]
plt.figure(figsize=(8, 6))
sns.heatmap(high_corr_matrix, annot=True, fmt=".5f", cmap="coolwarm", cbar=True)
plt.title("Correlation Matrix for Highly Correlated Variables")
plt.show()
print(high_corr_features.axes)