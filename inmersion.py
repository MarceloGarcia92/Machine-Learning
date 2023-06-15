#System
import os
import warnings
import sys

#Support libraries
import pandas as pd
import numpy as np

#Sklearn
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

#Mlflow preferences
from urllib.parse import urlparse
import mlflow
from mlflow.models.signature import infer_signature
import mlflow.sklearn

#Own functions
from evaluation import multi_class_metrics
from hyperparameters import *
from preprocess import preprocess_df

#User control
import logging

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    df = pd.read_csv('data.csv')

    feature_selection = list(df.columns)

    feature_selection.remove('ImmersionLevel')
    feature_selection.remove('UserID')

    target_var = 'ImmersionLevel'

    X_train, X_test, y_train, y_test = preprocess_df(df, feature_selection, target_var)

    with mlflow.start_run():
        
        #Model:
        rf = RandomForestClassifier()

        #Improving models: Hyperparameters
        grid_model = GridSearchCV(estimator=rf, param_grid=param_grid['RandomForest'], cv=5)

        # Fit grid_model to the data
        grid_model_result = grid_model.fit(X_train, y_train)

        # Summarize results
        best_score, best_params = grid_model_result.best_score_, grid_model_result.best_params_
        print("Best: %f using %s" % (best_score, best_params))

        # Extract the best model and evaluate it on the test set
        best_model = grid_model_result.best_estimator_
        print("Accuracy of logistic regression classifier: ", best_model.score(X_test, y_test))

        predicted_qualities = best_model.predict(X_test)

        (f1, recall, precision) = multi_class_metrics(y_test, predicted_qualities)


        print("  F1: %s" % f1)
        print("  Recall: %s" % recall)
        print("  Precision: %s" % precision)

        mlflow.log_metric("F1", f1)
        mlflow.log_metric("Recall", recall)
        mlflow.log_metric("Precision", precision)

        predictions = best_model.predict(X_train)
        signature = infer_signature(X_train, predictions)

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

