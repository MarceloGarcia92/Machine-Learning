#System
import os
import warnings
import sys

#Support libraries
import pandas as pd
import numpy as np

#Sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression

#Mlflow preferences
from urllib.parse import urlparse
import mlflow
from mlflow.models.signature import infer_signature
import mlflow.sklearn

#Own functions
from evaluation import eval_metrics
from hyperparameters import *

#User control
import logging

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    
    df = pd.read_csv('crx.data', header=None)
    
    #Drop useful columns: DriverLicense, ZipCode
    df = df.drop(columns=[11, 13])

    #Split the Data
    x_train, x_test = train_test_split(df, test_size=0.33, random_state=42)

    #Preprocessing stage: Handle missing values
    x_train = x_train.replace('?', np.NaN)
    x_test = x_test.replace('?', np.NaN)

    for data in [x_train, x_test]:
        for idx, nan in enumerate(data.isna().sum()):
            if nan > 0:
                print(f'The col {data.columns[idx]} have {nan} NaN values')
            else:
                print(f'The col {data.columns[idx]} don\'t have NaN values')

    x_train.fillna(x_train.mean(), inplace=True)
    x_test.fillna(x_test.mean(), inplace=True)

    for col in x_train.columns:
        if x_train[col].dtypes == 'object':
            x_train = x_train.fillna(x_train[col].value_counts().index[0])
            x_test = x_test.fillna(x_train[col].value_counts().index[0])

    for data in [x_train, x_test]:
        for idx, nan in enumerate(data.isna().sum()):
            if nan > 0:
                print(f'The col {data.columns[idx]} have {nan} NaN values')
            else:
                print(f'The col {data.columns[idx]} don\'t have NaN values')

    #Preprocessing stage: Categorical -> Numerical transfomation
    x_train = pd.get_dummies(x_train)
    x_test = pd.get_dummies(x_test)

    x_test = x_test.reindex(columns=x_train.columns, fill_value=0)

    x_train, y_train = x_train.iloc[:,:-1].values, x_train.iloc[:,-1].values
    x_test, y_test = x_test.iloc[:,:-1].values, x_test.iloc[:,-1].values

    # Instantiate MinMaxScaler and use it to rescale X_train and X_test
    scaler = MinMaxScaler()
    rescaledX_train = scaler.fit_transform(x_train)
    rescaledX_test = scaler.transform(x_test)
    
    with mlflow.start_run():
        
        #Model:
        lr = LogisticRegression()

        #Improving models: Hyperparameters
        grid_model = GridSearchCV(estimator=lr, param_grid=param_grid, cv=5)

        # Fit grid_model to the data
        grid_model_result = grid_model.fit(rescaledX_train, y_train)

        # Summarize results
        best_score, best_params = grid_model_result.best_score_, grid_model_result.best_params_
        print("Best: %f using %s" % (best_score, best_params))

        # Extract the best model and evaluate it on the test set
        best_model = grid_model_result.best_estimator_
        print("Accuracy of logistic regression classifier: ", best_model.score(rescaledX_test, y_test))

        predicted_qualities = best_model.predict(rescaledX_test)

        (rmse, mae, r2) = eval_metrics(y_test, predicted_qualities)

        print("Elasticnet model (alpha={:f}, l1_ratio={:f}):".format(alpha, l1_ratio))
        print("  RMSE: %s" % rmse)
        print("  MAE: %s" % mae)
        print("  R2: %s" % r2)

        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)

        predictions = best_model.predict(x_train)
        signature = infer_signature(x_train, predictions)

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        # Model registry does not work with file store
        if tracking_url_type_store != "file":
            # Register the model
            # There are other ways to use the Model Registry, which depends on the use case,
            # please refer to the doc for more information:
            # https://mlflow.org/docs/latest/model-registry.html#api-workflow
            mlflow.sklearn.log_model(
                best_model, "model", registered_model_name="LogisticRegression", signature=signature
            )
        else:
            mlflow.sklearn.log_model(lr, "model", signature=signature)