# Run this cell to import the modules you require
import pandas as pd
import numpy as np

import mlflow
import mlflow.sklearn

from preprocess import preprocess_df
from results import predict_and_evaluate


from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# Import the data and perform exploratory data analysis
weather = pd.read_csv('london_weather.csv')
weather.info()

weather['date'] = pd.to_datetime(weather['date'], format='%Y%m%d')
weather['year'] = weather['date'].dt.year
weather['month'] = weather['date'].dt.month


# Choose features, define the target, and drop null values
feature_selection = ['month', 'cloud_cover', 'sunshine', 'precipitation', 'pressure', 'global_radiation']
target_var = 'mean_temp'
weather = weather.dropna(subset=['mean_temp'])

# Load data and perform exploratory analysis
X_train, X_test, y_train, y_test = preprocess_df(weather, feature_selection, target_var)
  
# Create an experiment
EXPERIMENT_NAME = "weather_mult_model"
EXPERIMENT_ID = mlflow.create_experiment(EXPERIMENT_NAME)

# Predict, evaluate, and log the parameters and metrics of your models
for idx, depth in enumerate([1, 2, 5, 10, 20]):
    parameters = {
        'max_depth': depth
    }    
    run_name = f"run_{idx}"
    with mlflow.start_run(experiment_id=EXPERIMENT_ID, run_name=run_name):
        # Create models
        lin_reg = LinearRegression().fit(X_train, y_train)
        tree_reg = DecisionTreeRegressor(random_state=42, max_depth=depth).fit(X_train, y_train)
        forest_reg = RandomForestRegressor(random_state=42, max_depth=depth).fit(X_train, y_train)
        # Log models
        mlflow.sklearn.log_model(lin_reg, "lin_reg")
        mlflow.sklearn.log_model(tree_reg, "tree_reg")
        mlflow.sklearn.log_model(forest_reg, "forest_reg")

        models = [lin_reg, tree_reg, forest_reg]
        for model in models:
            # Evaluate performance
            rmse = predict_and_evaluate(model, X_test, y_test)
            # Log performance
            mlflow.log_param("max_depth", depth)
            mlflow.log_metric("rmse", rmse)
        
# Search the runs for the experiment's results
experiment_results = mlflow.search_runs(experiment_names=[EXPERIMENT_NAME])
experiment_results