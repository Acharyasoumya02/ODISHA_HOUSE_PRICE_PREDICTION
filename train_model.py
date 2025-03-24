import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
import joblib

df = pd.read_csv('odisha_house_prices.csv')

label_encoders = {}
categorical_features = ['Furnishing Status', 'Parking', 'Proximity']
for feature in categorical_features:
    le = LabelEncoder()
    df[feature] = le.fit_transform(df[feature])
    label_encoders[feature] = le

features = ['Area (sq. ft.)', 'BHK', 'Bathrooms', 'Furnishing Status', 'Parking', 'Age of Property', 'Proximity']
X = df[features]
y = df['Price (Lakhs)']

y = np.log1p(y)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rf = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

predictions = best_model.predict(X_test)
predictions = np.expm1(predictions)  
mse = mean_squared_error(np.expm1(y_test), predictions)  
print(f'Mean Squared Error: {mse}')

joblib.dump(best_model, 'house_price_model.pkl')
joblib.dump(label_encoders, 'label_encoders.pkl')
