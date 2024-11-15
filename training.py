# training.py

import pandas as pd
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
import os

# Load the data
data = pd.read_csv('house_price_regression_dataset.csv')

# Data Cleaning (remove rows with missing target or irrelevant columns)
data = data.dropna(subset=['House_Price'])  # Assume 'Price' is the target variable

# Separate features and target
X = data.drop(columns=['House_Price'])
y = data['House_Price']

# Identify categorical and numerical columns
categorical_cols = [col for col in X.columns if X[col].dtype == 'object']
numerical_cols = [col for col in X.columns if X[col].dtype in ['int64', 'float64']]

# Preprocessing pipelines for numerical and categorical data
numerical_pipeline = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_pipeline = Pipeline(steps=[
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

# Combine pipelines using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_pipeline, numerical_cols),
        ('cat', categorical_pipeline, categorical_cols)
    ])

# Define models to train
models = {
    'RandomForest': RandomForestRegressor(),
    'LinearRegression': LinearRegression(),
    'DecisionTree': DecisionTreeRegressor(),
    'SVR': SVR()
}

# Create a pipeline that combines preprocessing with model training
best_model = None
best_score = float('inf')
for model_name, model in models.items():
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
    param_grid = {'model__n_estimators': [50, 100]} if model_name == 'RandomForest' else {}
    grid_search = GridSearchCV(pipeline, param_grid, scoring='neg_mean_squared_error', cv=5)
    grid_search.fit(X, y)
    
    score = -grid_search.best_score_
    if score < best_score:
        best_score = score
        best_model = grid_search.best_estimator_

# Save the best model to a pickle file
with open('best_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

print("Training completed. Best model saved as 'best_model.pkl'")
