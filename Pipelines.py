import pandas as pd
import numpy as np
import time
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from imblearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, FunctionTransformer, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from colorama import Fore
from sklearn.ensemble import (RandomForestClassifier, VotingClassifier, BaggingClassifier
                              , AdaBoostClassifier, ExtraTreesClassifier)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import cross_val_score, cross_validate


class columnDropperTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X.drop(self.columns, axis=1)


class FunctionTransformer1(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_ = X.copy()
        X_[self.columns] = np.log(X_[self.columns])
        return X_


class FunctionTransformer2(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_ = X.copy()
        X_[self.columns] = np.log(X_[self.columns] + 0.1)
        return X_


cat_imputer = SimpleImputer(strategy="most_frequent")
num_imputer = SimpleImputer(strategy="median")

cat_pipeline_ord = Pipeline([
    ("cat_imputer", cat_imputer),
    ('encoding', OrdinalEncoder())
])

cat_pipeline_ohe = Pipeline([
    ("cat_imputer", cat_imputer),
    ('encoding', OneHotEncoder())
])

num_pipeline_std = Pipeline([
    ("num_imputer", num_imputer),
    ('standard_scaling', StandardScaler())
])

num_pipeline_minmax = Pipeline([
    ("num_imputer", num_imputer),
    ('standard_scaling', MinMaxScaler())
])


def Full_pipeline1(X_train, y_train):
    """
    - Drop ID, Gender, Arrival Delay
    - log transform deprature delay, Flight distance
    - one hot encode categorical data
    - std scale numerical data
    """

    cat_cols = ['Customer Type', 'Type of Travel', 'Class']

    num_cols = ['Age', 'Departure and Arrival Time Convenience',
                'Ease of Online Booking', 'Check-in Service', 'Online Boarding',
                'Gate Location', 'On-board Service', 'Seat Comfort', 'Leg Room Service',
                'Cleanliness', 'Food and Drink', 'In-flight Service', 'In-flight Wifi Service',
                'In-flight Entertainment', 'Baggage Handling']

    transformer_pipeline = Pipeline([
        ('transformer1', FunctionTransformer1(['Flight Distance'])),
        ('transformer2', FunctionTransformer2(['Departure Delay'])),
        ('standard_scaling', StandardScaler())
    ])

    full_pipeline = ColumnTransformer([
        ("cat", cat_pipeline_ohe, cat_cols),
        ('transform', transformer_pipeline, ['Flight Distance', 'Departure Delay']),
        ("num", num_pipeline_std, num_cols),
        ("drop_cols", "drop", ['Gender', 'ID', 'Arrival Delay'])
    ], remainder="passthrough")

    enc = OrdinalEncoder()

    X_train = full_pipeline.fit_transform(X_train)
    y_train = enc.fit_transform(y_train)

    return X_train, y_train, full_pipeline, enc


def Full_pipeline2(X_train, y_train):
    """
    - Drop ID, Gender, Arrival Delay, departure delay
    - log transform Flight distance
    - ordinal encodde categorical data
    - std scale numerical data
    """

    cat_cols = ['Customer Type', 'Type of Travel', 'Class']

    num_cols = ['Age', 'Departure and Arrival Time Convenience',
                'Ease of Online Booking', 'Check-in Service', 'Online Boarding',
                'Gate Location', 'On-board Service', 'Seat Comfort', 'Leg Room Service',
                'Cleanliness', 'Food and Drink', 'In-flight Service', 'In-flight Wifi Service',
                'In-flight Entertainment', 'Baggage Handling']

    transformer_pipeline = Pipeline([
        ('transformer1', FunctionTransformer1(['Flight Distance'])),
        ('standard_scaling', StandardScaler())
    ])

    full_pipeline = ColumnTransformer([
        ("cat", cat_pipeline_ord, cat_cols),
        ('transform', transformer_pipeline, ['Flight Distance']),
        ("num", num_pipeline_std, num_cols),
        ("drop_cols", "drop", ['Gender', 'ID', 'Arrival Delay', 'Departure Delay'])
    ], remainder="passthrough")

    enc = OrdinalEncoder()

    X_train = full_pipeline.fit_transform(X_train)
    y_train = enc.fit_transform(y_train)

    return X_train, y_train, full_pipeline, enc


def Full_pipeline3(X_train, y_train):
    """
    - Drop ID, Gender, Arrival Delay
    - log transform Flight distance
    - leave departure delay as it is
    - ordinal encodde categorical data
    - minmax scale continous data
    """

    cat_cols = ['Customer Type', 'Type of Travel', 'Class']

    num_cols = ['Age', 'Departure Delay', 'Departure and Arrival Time Convenience',
                'Ease of Online Booking', 'Check-in Service', 'Online Boarding',
                'Gate Location', 'On-board Service', 'Seat Comfort', 'Leg Room Service',
                'Cleanliness', 'Food and Drink', 'In-flight Service', 'In-flight Wifi Service',
                'In-flight Entertainment', 'Baggage Handling']

    transformer_pipeline = Pipeline([
        ('transformer1', FunctionTransformer1(['Flight Distance'])),
        ('standard_scaling', StandardScaler())
    ])

    full_pipeline = ColumnTransformer([
        ("cat", cat_pipeline_ord, cat_cols),
        ('transform', transformer_pipeline, ['Flight Distance']),
        ("num", num_pipeline_minmax, num_cols),
        ("drop_cols", "drop", ['Gender', 'ID', 'Arrival Delay'])
    ], remainder="passthrough")

    enc = OrdinalEncoder()

    X_train = full_pipeline.fit_transform(X_train)
    y_train = enc.fit_transform(y_train)

    return X_train, y_train, full_pipeline, enc


def Full_pipeline4(X_train, y_train):
    """
    - Drop ID, Gender, Arrival Delay
    - log transform Flight distance, departure delay
    - one hot encodde categorical data
    - minmax scale discrete data
    - std scale continous data
    """

    cat_cols = ['Customer Type', 'Type of Travel', 'Class']

    num_cols_discrete = ['Departure and Arrival Time Convenience',
                         'Ease of Online Booking', 'Check-in Service', 'Online Boarding',
                         'Gate Location', 'On-board Service', 'Seat Comfort', 'Leg Room Service',
                         'Cleanliness', 'Food and Drink', 'In-flight Service', 'In-flight Wifi Service',
                         'In-flight Entertainment', 'Baggage Handling']

    num_cols_continous = ['Age']

    transformer_pipeline = Pipeline([
        ('transformer1', FunctionTransformer1(['Flight Distance'])),
        ('transformer2', FunctionTransformer2(['Departure Delay'])),
        ('standard_scaling', StandardScaler())
    ])

    full_pipeline = ColumnTransformer([
        ("cat", cat_pipeline_ohe, cat_cols),
        ('transform', transformer_pipeline, ['Flight Distance', 'Departure Delay']),
        ("num_discrete", num_pipeline_minmax, num_cols_discrete),
        ("num_continous", num_pipeline_std, num_cols_continous),
        ("drop_cols", "drop", ['Gender', 'ID', 'Arrival Delay'])
    ], remainder="passthrough")

    enc = OrdinalEncoder()

    X_train = full_pipeline.fit_transform(X_train)
    y_train = enc.fit_transform(y_train)

    return X_train, y_train, full_pipeline, enc


def Production_pipeline(df,full_pipeline,enc):
    """
    - df : dataframe which will come from the user
    """
    X_test = df.drop('Satisfaction',axis=1)
    y_test = df[['Satisfaction']]
    
    X_test = full_pipeline.transform(X_test)
    y_test = enc.transform(y_test)
    
    return X_test, y_test

    

def compute_pipeline(x,y,models,model_name,n,cv=5,scoring = ['f1','accuracy','precision','recall','roc_auc']):
    """
    x : x data
    y : y data
    n : pipeline number
    cv : number of cross validations (defult = 5)
    scoring : list of required metrics
    """
    c = 0
    pipeline_dict = {}

    print('Pipeline {} : '.format(n))
    print('------------\n\n')

    for i in models:

        start = time.time()
        scores = cross_validate(i, x, y, cv=cv, scoring=scoring)
        end = time.time()
        
        total_time = end - start

        print(Fore.LIGHTBLUE_EX,'time of {} on {} cross validaions : '.format(model_name[c],cv),total_time)
        print(Fore.LIGHTBLUE_EX,'mean f1 score of {} on {} cross validaions : '.format(model_name[c],cv)
              ,scores['test_f1'].mean())
        print(Fore.LIGHTBLUE_EX,'mean accuracy score of {} on {} cross validaions : '.format(model_name[c],cv)
              ,scores['test_accuracy'].mean())
        print(Fore.LIGHTBLUE_EX,'mean precision score of {} on {} cross validaions : '.format(model_name[c],cv)
              ,scores['test_precision'].mean())
        print(Fore.LIGHTBLUE_EX,'mean recall score of {} on {} cross validaions : '.format(model_name[c],cv)
              ,scores['test_recall'].mean())
        print(Fore.LIGHTBLUE_EX,'mean roc_auc score of {} on {} cross validaions : '.format(model_name[c],cv)
              ,scores['test_roc_auc'].mean())

        print(Fore.BLACK,'----------------------------------------------------------------------------------------------\n')
        
        stats = [total_time,scores['test_f1'].mean(),scores['test_accuracy'].mean(),
                scores['test_precision'].mean(),scores['test_recall'].mean(),
                scores['test_roc_auc'].mean()]
        
        pipeline_dict[model_name[c]] = stats
        
        c = c + 1
        
    return pd.DataFrame(pipeline_dict,index=['Time','F1_score','acurracy','precision','recall','roc_auc'])