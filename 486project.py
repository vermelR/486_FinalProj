import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load and preprocess the data
file_path = 'Stat_486/project-data/calendar.csv'
data = pd.read_csv(file_path)

# Drop the 'adjusted_price' column if it exists
if 'adjusted_price' in data.columns:
    data.drop(columns=['adjusted_price'], inplace=True)

# Strip any leading/trailing spaces from column names
data.columns = data.columns.str.strip()

# Convert 'price' to float by removing dollar signs and commas
data['price'] = data['price'].replace({'\$': '', ',': ''}, regex=True).astype(float)

# Visualization Part
sns.set(style="whitegrid")

# 1. Histogram of the price feature
plt.figure(figsize=(10, 6))
sns.histplot(data['price'], bins=30, kde=True)
plt.title('Distribution of Price')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.show()

# 2. Box plot to identify outliers in price
plt.figure(figsize=(10, 6))
sns.boxplot(x=data['price'])
plt.title('Box Plot of Price')
plt.xlabel('Price')
plt.show()

# 3. Scatter plot to visualize relationships (example with minimum_nights)
plt.figure(figsize=(10, 6))
sns.scatterplot(x=data['minimum_nights'], y=data['price'])
plt.title('Scatter Plot between Minimum Nights and Price')
plt.xlabel('Minimum Nights')
plt.ylabel('Price')
plt.show()

# Data Preprocessing

# 1. Encoding Categorical Variables
# Identify categorical columns
categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
print("Categorical columns before encoding:", categorical_cols)

# Apply one-hot encoding
data_encoded = pd.get_dummies(data, columns=categorical_cols, drop_first=True)

# 2. Feature Scaling
# Identify numerical columns for scaling
numerical_cols = data_encoded.select_dtypes(include=['float64', 'int64']).columns.tolist()
print("Numerical columns before scaling:", numerical_cols)

# Initialize the scaler
scaler = StandardScaler()

# Fit and transform the numerical columns
data_encoded[numerical_cols] = scaler.fit_transform(data_encoded[numerical_cols])

# Define the target variable and features
# 'price' is the target variable
X = data_encoded.drop(columns=['price'])
y = data_encoded['price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 1. Linear Regression Model
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# Predictions and evaluation
linear_predictions = linear_model.predict(X_test)
linear_mse = mean_squared_error(y_test, linear_predictions)
linear_r2 = r2_score(y_test, linear_predictions)
print(f'Linear Regression MSE: {linear_mse:.4f}, R^2 Score: {linear_r2:.4f}')

# 2. Random Forest Model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predictions and evaluation
rf_predictions = rf_model.predict(X_test)
rf_mse = mean_squared_error(y_test, rf_predictions)
rf_r2 = r2_score(y_test, rf_predictions)
print(f'Random Forest MSE: {rf_mse:.4f}, R^2 Score: {rf_r2:.4f}')

# 3. Gradient Boosting Model (GBM)
gbm_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
gbm_model.fit(X_train, y_train)

# Predictions and evaluation
gbm_predictions = gbm_model.predict(X_test)
gbm_mse = mean_squared_error(y_test, gbm_predictions)
gbm_r2 = r2_score(y_test, gbm_predictions)
print(f'GBM MSE: {gbm_mse:.4f}, R^2 Score: {gbm_r2:.4f}')
