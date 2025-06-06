import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, f1_score, mean_squared_error
import joblib
import numpy as np

# Step 1: Data Preprocessing
data_files = ['2003.csv', '2004.csv', '2005.csv', '2006.csv', '2007.csv', '2008.csv', '2009.csv', '2010.csv', '2011.csv', '2012.csv', '2013.csv', '2014.csv', '2015.csv', '2016.csv', '2017.csv', '2018.csv', '2019.csv', '2020.csv']
data_list = [pd.read_csv(file) for file in data_files]
data = pd.concat(data_list, ignore_index=True)

# Handle missing values if any
data.dropna(inplace=True)

# Convert date strings to datetime objects with the correct format
data['date'] = pd.to_datetime(data['date'], format='%Y-%m-%d')

# Add a synthetic column indicating forest fire occurrences
data['forest_fire_occurred'] = 1  # Set all values to 1 indicating forest fires occurred

# Step 2: Feature Engineering
X = data[['lat', 'lon', 'vpd_avg', 'vpd_max', 'vpd_min', 'vpd_avg_1', 'vpd_max_1', 'vpd_min_1']]
y = data['forest_fire_occurred']

# Data augmentation
num_synthetic_samples = 500000  # Number of synthetic samples to generate
synthetic_X = []
synthetic_y = []
for _ in range(num_synthetic_samples):
    synthetic_sample = []
    for feature in X.columns:
        feature_values = X[feature].values
        synthetic_value = np.random.choice(feature_values)
        synthetic_sample.append(synthetic_value)
    synthetic_X.append(synthetic_sample)
    synthetic_y.append(0)  # Label as "no forest fire occurrence"

# Convert lists to arrays
synthetic_X = np.array(synthetic_X)
synthetic_y = np.array(synthetic_y)

# Combine original and synthetic data
X_combined = np.concatenate((X.values, synthetic_X))
y_combined = np.concatenate((y.values, synthetic_y))

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_combined, y_combined, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 3: Model Training
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train_scaled, y_train)

# Model evaluation
y_pred = dt_model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("Accuracy:", accuracy)
print("Precision:", precision)
print("F1 Score:", f1)
print("RMSE:", rmse)

# Save the trained Decision Tree model
joblib.dump(dt_model, 'dt_model.pkl')

# Save the scaler used for scaling input data during training
joblib.dump(scaler, 'scaler.pkl')
