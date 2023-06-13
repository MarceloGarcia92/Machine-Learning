import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

#Mlflow preferences
from urllib.parse import urlparse
import mlflow
from mlflow.models.signature import infer_signature
import mlflow.sklearn

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import GridSearchCV

from preprocess import *
from evaluation import binary_class_metrics
from hyperparameters import param_grid

df = pd.read_csv('titanic/train.csv')
df = df.drop(columns=['Name', 'Ticket', 'PassengerId', 'Cabin'])

nan = missing_data(df)

if len(nan[nan<1].keys()) > 0:
    for col in nan[nan<1].keys():
        df = df[~df[col].isna()]

for col, dtype in df.dtypes.items():
    if dtype == 'object':
        transformation = pd.get_dummies(df[col])
        df = df.join(transformation)
        df.drop(columns=col, inplace=True)

df.corr()

hm = sns.heatmap(df.corr(), annot=True, square=True, fmt='.2f', annot_kws={'size': 5})
hm.figure.savefig('heatmap.png')

feature_selection = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'female', 'male', 'C', 'Q', 'S']
target_var = 'Survived'

X_train, X_test, y_train, y_test = preprocess_df(df, feature_selection, target_var)

models = {'svm':SVC(), 'randomForest':RandomForestClassifier(), 'adaboost':AdaBoostClassifier()}

with mlflow.start_run():
    best_result = 0

    for name, model in models.items():
        if name == 'randomForest':
            grid_model = GridSearchCV(estimator=model, param_grid=param_grid['RandomForest'], cv=5)
        else:
            grid_model = GridSearchCV(estimator=model, param_grid=param_grid['all'], cv=5)

        grid_model.fit(X_train, y_train)
        best_score, best_params = grid_model.best_score_, grid_model.best_params_
        print(f"Best: {best_score}")

        y_pred = grid_model.predict(X_test)
        result = accuracy_score(y_true=y_test, y_pred=y_pred)
        print(f"Accuracy of logistic regression classifier: {result}") 
        #result = class_likelihood_ratios(y_true=y_test, y_pred=y_pred)

        if result > best_result:
            best_model = grid_model.best_estimator_
        
    y_pred = best_model(X_test)
    (likelihood, acc) = binary_class_metrics(y_test, y_pred)

    print(f'Model: {model} with params: {best_params}')
    print(f'Likelihood: {likelihood}')
    print(f'Accuracy: {acc}')

    mlflow.log_param("Likelihood", likelihood)
    mlflow.log_param("Accuracy", acc)

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
        mlflow.sklearn.log_model(model, "model", signature=signature)    

