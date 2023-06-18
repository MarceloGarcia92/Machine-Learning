from pandas import get_dummies

from sklearn.model_selection import train_test_split

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder

def clean_unique_cat(columns:list, df):
    """
    Delete the values of categorical columns when they have only with values.
    Args:
        columns (list): list of columns with categorical data to check
        df (DataFrame): Dataframe o be processed"""
    
    for col in columns:
        #Iteration for each value/category in the column.
        for value, count in df[col].value_counts().items():
            if count == 1:
                #Delete the value/category from the DataFrame.
                df = df[df[col] != value]

    return df

def missing_data(df):
    """
    Check the percentage of nan values on each column.
    Args:
        df (DataFrame): Data of the experimentation.
    """
    #Calculate the percentage
    nan = df.isnull().sum()/len(df)*100

    #Selection of the columns with nan values
    nan = nan[nan>0].sort_values()

    return nan

def preprocess_df(df, feature_selection, target_var):
    """
    Split dataframe into X and y, and train and test consecutively. 
    Then impute and scale both train and test features, using get_dummies, SimpleImputer and StandarScaler/MinMaxScaler statements.
    Args:
        df (DataFrame): Data of the experimentation
        feature_selection (list): list of columns 
        target_var (str):  name of the label column
    """
    # Subset the data
    X = df[feature_selection]    
    y = df[target_var]
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)

    #Preprocessing stage: Categorical -> Numerical transfomation
    X_train = get_dummies(X_train)
    X_test = get_dummies(X_test)
    
    # Impute missing values
    imputer = SimpleImputer(strategy="mean")
    X_train = imputer.fit_transform(X_train)
    X_test  = imputer.transform(X_test)
    
    # Scale the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test