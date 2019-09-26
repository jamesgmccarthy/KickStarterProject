import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_selection import VarianceThreshold, RFECV
from mlxtend.feature_selection import SequentialFeatureSelector
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
import FeatureEngineeringKS


def impute_missing(X_train, X_test):
    X_train['mean_number_of_backers'].fillna(
        0, inplace=True)
    X_test['mean_number_of_backers'].fillna(
        0, inplace=True)
    return X_train, X_test


def scale_values(X_train, X_test):
    cont_var = ['usd_goal_real', 'duration', 'name_len', 'name_num_words',
                'num_proj_q', 'num_proj_m', 'num_proj_w', 'mean_number_of_backers']
    scaler = StandardScaler()
    scaled_train = pd.DataFrame(
        scaler.fit_transform(X_train[cont_var]), columns=cont_var).set_index(X_train.index)
    scaled_test = pd.DataFrame(
        scaler.fit_transform(X_test[cont_var]), columns=cont_var).set_index(X_test.index)
    X_train[cont_var] = scaled_train[cont_var]
    X_test[cont_var] = scaled_test[cont_var]
    return X_train, X_test


def recursive_feature_selection(X_train, X_test, y_train, model):
    if os.path.isfile("./Models/RFECV.pkl"):
        selector = joblib.load("./Models/RFECV.pkl")
        print("Optimal number of features: %d" % selector.n_features_)
    return X_train.iloc[:, selector.support_], X_test.iloc[:, selector.support_]
    selector = RFECV(estimator=model, step=1, cv=10, scoring='roc_auc')
    selector.fit(X_train, y_train)
    joblib.dump(selector, f"./Models/RFECV.pkl")
    print("Optimal number of features: %d" % selector.n_features_)
    return X_train.iloc[:, selector.support_], X_test.iloc[:, selector.support_]


def main(input, split='random', train=None, test=None):
    X_train, X_test, y_train, y_test = FeatureEngineeringKS.main(
        input, split=split, train=train, test=test)
    X_train, X_test = impute_missing(X_train, X_test)
    # Scaling hurts perfomance
    #X_train, X_test = scale_values(X_train, X_test)
    return X_train, X_test, y_train, y_test
