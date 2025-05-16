import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score

def load_data(file_path):
    
    df = pd.read_csv(file_path)
    
    print(f"data shape: {df.shape}")
    return df

def prepare_data(df):
    # Define features and targets
    X = df[['Inventory Level', 'Price', 'Seasonality_Summer', 'Inventory_Demand', 'UnitsSold_Price']]
    y = df[['Units Sold', 'Demand Forecast']]
    
    return X, y

numeric_features = ['Inventory Level', 'Price', 'Inventory_Demand', 'UnitsSold_Price']
binary_features = ['Seasonality_Summer']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('binary', 'passthrough', binary_features)
    ])

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(random_state=42, n_jobs=-1))
])

param_grid = {
    'regressor__n_estimators': [100, 200],
    'regressor__max_depth': [None, 10],
    'regressor__min_samples_split': [2, 5]
}

if __name__ == "__main__":

    raw_data = load_data(r'datasets\cleaned_data.csv')
    
    X, y = prepare_data(raw_data)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    grid_search = GridSearchCV(pipeline, param_grid, 
                             cv=3, scoring='neg_mean_squared_error',
                             verbose=2, n_jobs=-1)
    
    print("Starting training...")
    grid_search.fit(X_train, y_train)
    
    print(f"\nBest parameters: {grid_search.best_params_}")
    y_pred = grid_search.predict(X_test)
    
    print(f"\nTest MSE: {mean_squared_error(y_test, y_pred):.2f}")
    print(f"Test R²: {r2_score(y_test, y_pred):.2f}")