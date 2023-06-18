#System
import os
import warnings
import sys

#Support libraries
import pandas as pd
import numpy as np

#Sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression

#Mlflow preferences
from urllib.parse import urlparse
import mlflow
from mlflow.models.signature import infer_signature
import mlflow.sklearn

#Own functions
from evaluation import reg_metrics
from hyperparameters import *
from preprocess import clean_unique_cat

#User control
import logging

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


if __name__ == "__main__":

    # Create an experiment
    EXPERIMENT_NAME = "Cup points Kaggle dataset"
    EXPERIMENT_ID = mlflow.create_experiment(EXPERIMENT_NAME)

    warnings.filterwarnings("ignore")
    np.random.seed(1)

    df = pd.read_csv('df_arabica_clean.csv')

    feature_obj = ['Color', 'Processing Method']

    df = clean_unique_cat(feature_obj, df)

    feature_selection = ['Processing Method', 'Aroma', 'Flavor',
       'Aftertaste', 'Acidity', 'Body', 'Balance', 'Uniformity', 'Clean Cup',
       'Sweetness', 'Overall', 'Defects',  'Category One Defects', 'Quakers', 'Color',
       'Category Two Defects']
    
    df_selected = df[feature_selection]
    label = df['Total Cup Points']

    x_train, x_test, y_train, y_test = train_test_split(df_selected, label, random_state=42, test_size=0.33)

    le_method = LabelEncoder()
    x_train['Processing Method'] = le_method.fit_transform(x_train['Processing Method'])
    x_test['Processing Method'] = le_method.transform(x_test['Processing Method'])


    le_color = LabelEncoder()
    x_train['Color'] = le_color.fit_transform(x_train['Color'])
    x_test['Color'] = le_color.transform(x_test['Color'])

    mx = MinMaxScaler()

    x_train = mx.fit_transform(x_train)
    x_test = mx.transform(x_test)

    with mlflow.start_run(experiment_id=EXPERIMENT_ID):
        lr = LinearRegression()
        lr.fit(x_train, y_train)
        y_pred = lr.predict(x_test)

        (rmse, mae, r2) = reg_metrics(y_test, y_pred)


        print("  RMSE: %s" % rmse)
        print("  MAE: %s" % mae)
        print("  R2: %s" % r2)

        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)

        predictions = lr.predict(x_train)
        signature = infer_signature(x_train, predictions)

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        # Model registry does not work with file store
        if tracking_url_type_store != "file":
            # Register the model
            # There are other ways to use the Model Registry, which depends on the use case,
            # please refer to the doc for more information:
            # https://mlflow.org/docs/latest/model-registry.html#api-workflow
            mlflow.sklearn.log_model(
                lr, "model", registered_model_name="cup points Linear Regression", signature=signature
            )
        else:
            mlflow.sklearn.log_model(lr, "model", signature=signature)