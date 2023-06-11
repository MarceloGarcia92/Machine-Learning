#System
import os
import warnings
import sys

#Support libraries
import pandas as pd
import numpy as np
from tpot import TPOTClassifier

#Sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

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

    
    transfusion = pd.read_csv('transfusion.data')

    transfusion.rename(
    columns={'whether he/she donated blood in March 2007': 'target'},
    inplace=True)

    #Split the Data
    x_train, x_test, y_train, y_test = train_test_split(
        transfusion.drop(columns='target'),
        transfusion.target,
        test_size=0.25,
        random_state=42,
        stratify=transfusion.target
)

    # Instantiate TPOTClassifier
    tpot = TPOTClassifier(
        generations=5,
        population_size=20,
        verbosity=2,
        scoring='roc_auc',
        random_state=42,
        disable_update_check=True,
        config_dict='TPOT light'
    )

    tpot.fit(x_train, y_train)

    # AUC score for tpot model
    tpot_auc_score = roc_auc_score(y_test, tpot.predict_proba(x_test)[:, 1])
    print(f'\nAUC score: {tpot_auc_score:.4f}')

    # Print best pipeline steps
    print('\nBest pipeline steps:', end='\n')
    for idx, (name, transform) in enumerate(tpot.fitted_pipeline_.steps, start=1):
        # Print idx and transform
        print(f'{name}. {transform}')

    # Copy X_train and X_test into X_train_normed and X_test_normed
    X_train_normed, X_test_normed = x_train.copy(), x_test.copy()

    # Specify which column to normalize
    col_to_normalize = 'Monetary (c.c. blood)' 

    # Log normalization
    for df_ in [X_train_normed, X_test_normed]:
        # Add log normalized column
        df_['monetary_log'] = np.log(df_[col_to_normalize])
        # Drop the original column
        df_.drop(columns=col_to_normalize, inplace=True)

    # Check the variance for X_train_normed
    X_train_normed.var().round(3)
    
    with mlflow.start_run():

        lr = LogisticRegression(
            solver='liblinear',
            random_state=42
        )

        # Train the model
        lr.fit(X_train_normed, y_train)

        #Prediction
        y_pred = lr.predict(X_test_normed)

        (rmse, mae, r2) = eval_metrics(y_test, y_pred)

        print("Elasticnet model (alpha={:f}, l1_ratio={:f}):".format(alpha, l1_ratio))
        print("  RMSE: %s" % rmse)
        print("  MAE: %s" % mae)
        print("  R2: %s" % r2)

        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)

        predictions = lr.predict(X_train_normed)
        signature = infer_signature(X_train_normed, predictions)

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        # Model registry does not work with file store
        if tracking_url_type_store != "file":
            # Register the model
            # There are other ways to use the Model Registry, which depends on the use case,
            # please refer to the doc for more information:
            # https://mlflow.org/docs/latest/model-registry.html#api-workflow
            mlflow.sklearn.log_model(
                lr, "model", registered_model_name="LogisticRegression", signature=signature
            )
        else:
            mlflow.sklearn.log_model(lr, "model", signature=signature)