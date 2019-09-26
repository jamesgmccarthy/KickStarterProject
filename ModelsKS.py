
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.externals import joblib
from sklearn.ensemble import BaggingClassifier, VotingClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.metrics import roc_curve, auc
import FeatureEngineeringKS
import HyperParameterSelection
import FeatureSelection
import sklearn
from FeatureSelection import recursive_feature_selection
import time


def _create_label_encoding(X_train, X_test):
    lb_make = LabelEncoder()
    X_train['main_category'] = lb_make.fit_transform(X_train['main_category'])
    X_train['category'] = lb_make.fit_transform(X_train['category'])
    X_train['country'] = lb_make.fit_transform(X_train['country'])
    X_train['currency'] = lb_make.fit_transform(X_train['currency'])
    X_test['main_category'] = lb_make.fit_transform(X_test['main_category'])
    X_test['category'] = lb_make.fit_transform(X_test['category'])
    X_test['country'] = lb_make.fit_transform(X_test['country'])
    X_test['currency'] = lb_make.fit_transform(X_test['currency'])
    return X_train, X_test


def _create_dummy_features(X_train, X_test):
    """Create dummy features for categorical features using one hot encoding:
    main_category
    category
    country
    currency
    (Launch variables??)
    Arguments:
        df {[type]} -- [description]
    """
    # Main categories
    X_train = pd.get_dummies(X_train, columns=['main_category'])
    X_test = pd.get_dummies(X_test, columns=['main_category'])

    # Sub Categories
    X_train = pd.get_dummies(X_train, columns=['category'])
    X_test = pd.get_dummies(X_test, columns=['category'])

    # Country
    X_train = pd.get_dummies(X_train, columns=['country'])
    X_test = pd.get_dummies(X_test, columns=['country'])
    # Currency
    X_train = pd.get_dummies(X_train, columns=['currency'])
    X_test = pd.get_dummies(X_test, columns=['currency'])

    return X_train, X_test


def decisionTree(X_train, X_test, y_train, y_test, force=False):
    # If model is already saved, load it and test it
    if os.path.isfile("./Models/dtModel.pkl"):
        dt = joblib.load("./Models/dtModel.pkl")
        train_acc = dt.score(X_train, y_train)
        acc = dt.score(X_test, y_test)
        predictions = dt.predict_proba(X_test)
        fpr, tpr, thresholds = roc_curve(y_test, predictions[:,1])
        auc_score = auc(fpr, tpr)
        print("Decision Tree Training accuracy:", train_acc)
        print('Decision Tree accuracy:', acc)
        print('Decision Tree auc:', auc_score)
        return dt
    # If model parameters do not exist or force is true, use grid search to find
    # optimal parameters and then train model and save it
    if not os.path.isfile("./ModelHyperparm/dtParams.pkl") or force:
        params = HyperParameterSelection.dtParamSelect(X_train, y_train)
        dt = DecisionTreeClassifier(**params, class_weight='balanced')
        dt.fit(X_train, y_train)
        joblib.dump(dt, "./Models/dtModel.pkl")
    # If model parameters do exist and force is false, load parameters
    # and then train model and save it
    elif os.path.isfile("./ModelHyperparm/dtParams.pkl") and not force:
        gridsearch_model = joblib.load("./ModelHyperparm/dtParams.pkl")
        dt = DecisionTreeClassifier(
            **gridsearch_model.best_params_, class_weight='balanced')
        dt.fit(X_train, y_train)
        joblib.dump(dt, "./Models/dtModel.pkl")

    train_acc = dt.score(X_train, y_train)
    acc = dt.score(X_test, y_test)
    predictions = dt.predict_proba(X_test)
    fpr, tpr, thresholds = roc_curve(y_test, predictions[:,1])
    auc_score = auc(fpr, tpr)
    print("Decision Tree Training accuracy:", train_acc)
    print('Decision Tree accuracy:', acc)
    print('Decision Tree auc:', auc_score)
    return dt


def logisticRegrs(X_train, X_test, y_train, y_test, force=False):
    if os.path.isfile("./Models/lgrModel.pkl"):
        lgr = joblib.load("./Models/lgrModel.pkl")
        train_acc = lgr.score(X_train, y_train)
        acc = lgr.score(X_test, y_test)
        predicitions = lgr.predict_proba(X_test)
        fpr, tpr, thresholds = roc_curve(y_test, predicitions[:,1])
        auc_score = auc(fpr, tpr)
        print('Logistic Regression training accuracy:', train_acc)
        print('Logistic Regression accuracy:', acc)
        print('Logistic Regression AUC:', auc_score)
        return lgr
    if not os.path.isfile("./ModelHyperparm/lgrParams.pkl") or force:
        params = HyperParameterSelection.lgrParamSelect(X_train, y_train)
        lgr = LogisticRegression(**params, random_state=0)
    elif os.path.isfile("./ModelHyperparm/lgrParams.pkl") and not force:
        gridsearch_model = joblib.load("./ModelHyperparm/lgrParams.pkl")
        lgr = LogisticRegression(
            **gridsearch_model.best_params_, random_state=0)
    lgr.fit(X_train, y_train)
    joblib.dump(lgr, "./Models/lgrModel.pkl")
    train_acc = lgr.score(X_train, y_train)
    acc = lgr.score(X_test, y_test)
    predicitions = lgr.predict_proba(X_test)
    fpr, tpr, thresholds = roc_curve(y_test, predicitions[:,1])
    auc_score = auc(fpr, tpr)
    print('Logistic Regression training accuracy:', train_acc)
    print('Logistic Regression accuracy:', acc)
    print('Logistic Regression AUC:', auc_score)
    return lgr


def randomForest(X_train, X_test, y_train, y_test, force=False):
    if os.path.isfile("./Models/rfModel.pkl"):
        rf = joblib.load("./Models/rfModel.pkl")
        train_acc = rf.score(X_train, y_train)
        acc = rf.score(X_test, y_test)
        predicitons = rf.predict_proba(X_test)
        fpr, tpr, thresholds = roc_curve(y_test, predicitons[:,1])
        auc_score = auc(fpr, tpr)
        print('Random Forrest Neighbours train accuracy:', train_acc)
        print('Random Forrest test accuracy:', acc)
        print('Random Forrest AUC:', auc_score)
        return rf
    if not os.path.isfile('./ModelHyperparm/rfParams.pkl') or force:
        params = HyperParameterSelection.rfParamSelect(X_train, y_train)
        rf = RandomForestClassifier(**params,class_weight='balanced_subsample')
    elif os.path.isfile('./ModelHyperparm/rfParams.pkl') and not force:
        gridsearch_model = joblib.load('./ModelHyperparm/rfParams.pkl')
        rf = RandomForestClassifier(
            **gridsearch_model.best_params_,
            n_estimators=100, random_state=0, class_weight='balanced_subsample')
    rf.fit(X_train, y_train)
    joblib.dump(rf, "./Models/rfModel.pkl")
    train_acc = rf.score(X_train, y_train)
    acc = rf.score(X_test, y_test)
    predicitons = rf.predict_proba(X_test)
    fpr, tpr, thresholds = roc_curve(y_test, predicitons[:,1])
    auc_score = auc(fpr, tpr)
    print('Random Forrest Neighbours train accuracy:', train_acc)
    print('Random Forrest test accuracy:', acc)
    print('Random Forrest AUC:', auc_score)
    return rf


def gradientBooster(X_train, X_test, y_train, y_test, force=False):
    if os.path.isfile("./Models/gbModel.pkl"):
        gb = joblib.load("./Models/gbModel.pkl")
        train_acc = gb.score(X_train, y_train)
        acc = gb.score(X_test, y_test)
        predicitons = gb.predict_proba(X_test)
        fpr, tpr, thresholds = roc_curve(y_test, predicitons[:,1])
        auc_score = auc(fpr, tpr)
        print('Gradient Boosting Classifier train accuracy:', train_acc)
        print('Gradient Boosting Classifier test accuracy:', acc)
        print('Gradient Boosting Classifier AUC:', auc_score)
        return gb

    else:
        gb = GradientBoostingClassifier(
            loss='exponential', n_estimators=100, random_state=0)
        gb.fit(X_train, y_train)
        joblib.dump(gb, './Models/gbModel.pkl')
        train_acc = gb.score(X_train, y_train)
        acc = gb.score(X_test, y_test)
        predicitons = gb.predict_proba(X_test)
        fpr, tpr, thresholds = roc_curve(y_test, predicitons[:,1])
        auc_score = auc(fpr, tpr)
        print('Gradient Boosting Classifier train accuracy:', train_acc)
        print('Gradient Boosting Classifier test accuracy:', acc)
        print('Gradient Boosting Classifier AUC:', auc_score)
        return gb


def adaBooster(X_train, X_test, y_train, y_test, force=False):
    if os.path.isfile("./Models/adaModel.pkl"):
        ada = joblib.load("./Models/adaModel.pkl")
        train_acc = ada.score(X_train, y_train)
        acc = ada.score(X_test, y_test)
        predicitons = ada.predict_proba(X_test)
        fpr, tpr, thresholds = roc_curve(y_test, predicitons[:,1])
        auc_score = auc(fpr, tpr)
        print('AdaBoosting Classifier train accuracy:', train_acc)
        print('AdaBoosting Classifier test accuracy:', acc)
        print('AdaBoosting Classifier AUC:', auc_score)
        return ada

    else:
        ada = AdaBoostClassifier(base_estimator = DecisionTreeClassifier(criterion = 'entropy',
                                max_depth = 20, min_samples_split = 1000, random_state = 0, class_weight = 'balanced'),
                                n_estimators = 100, learning_rate = 0.5)
        ada.fit(X_train, y_train)
        joblib.dump(ada, './Models/adaModel.pkl')
        train_acc = ada.score(X_train, y_train)
        acc = ada.score(X_test, y_test)
        predicitons = ada.predict_proba(X_test)
        fpr, tpr, thresholds = roc_curve(y_test, predicitons[:,1])
        auc_score = auc(fpr, tpr)
        print('AdaBoosting Classifier train accuracy:', train_acc)
        print('AdaBoosting Classifier test accuracy:', acc)
        print('AdaBoosting Classifier AUC:', auc_score)
        return ada


def voting(X_train, X_test, y_train, y_test, estimators):
    if os.path.isfile("./Models/votingModel.pkl"):
        ensemble = joblib.load('./Models/votingModel.pkl')
        predictions = ensemble.predict_proba(X_test)
        fpr, tpr, thresholds = roc_curve(y_test, predictions[:,1])
        auc_score = auc(fpr, tpr)
        train_acc = ensemble.score(X_train, y_train)
        acc = ensemble.score(X_test, y_test)
        print("Voting Ensemble Training Acc:", train_acc)
        print("Voting Ensemble Acc:", acc)
        print("Voting Ensemble AUC:", auc_score)
        return ensemble
    ensemble = VotingClassifier(estimators, voting='soft')
    ensemble.fit(X_train, y_train)
    joblib.dump(ensemble, "./Models/votingModel.pkl")
    predictions = ensemble.predict_proba(X_test)
    fpr, tpr, thresholds = roc_curve(y_test, predictions[:,1])
    auc_score = auc(fpr, tpr)
    acc = ensemble.score(X_test, y_test)
    print("Voting Ensemble Acc:", acc)
    print("Voting Ensemble AUC:", auc_score)
    return ensemble

def bagging(X_train, X_test, y_train, y_test):
    dt = DecisionTreeClassifier(criterion='entropy', max_depth=20,
                         min_samples_split=1000, random_state=0,
                         class_weight = 'balanced')
    if os.path.isfile("./Models/baggingModel.pkl"):
        ensemble = joblib.load('./Models/baggingModel.pkl')
        predictions = ensemble.predict(X_test)
        fpr, tpr, thresholds = roc_curve(y_test, predictions)
        auc_score = auc(fpr, tpr)
        acc = ensemble.score(X_test, y_test)
        print("Bagging Ensemble Acc:", acc)
        print("Bagging Ensemble AUC:", auc_score)
        return ensemble
    ensemble =BaggingClassifier(dt, n_estimators = 100,n_jobs = -1,random_state= 0)
    ensemble.fit(X_train, y_train)
    joblib.dump(ensemble, './Models/baggingModel.pkl')
    predictions = ensemble.predict(X_test)
    fpr, tpr, thresholds = roc_curve(y_test, predictions)
    auc_score = auc(fpr, tpr)
    acc = ensemble.score(X_test, y_test)
    print("Bagging Ensemble Acc:", acc)
    print("Bagging Ensemble AUC:", auc_score)
    return ensemble

def main():
    X_train, X_test, y_train, y_test = FeatureSelection.main(
        './kickstarter-projects/ks-projects-201801.csv')
    X_train_d, X_test_d = _create_dummy_features(X_train,X_test)
    X_train, X_test = _create_label_encoding(X_train, X_test)
    dt = decisionTree(X_train_d, X_test_d, y_train, y_test)
    lgr = logisticRegrs(X_train_d, X_test_d, y_train, y_test)
    rf = randomForest(X_train_d, X_test_d, y_train, y_test)
    gb = gradientBooster(X_train_d, X_test_d, y_train, y_test)
    ada = adaBooster(X_train_d, X_test_d, y_train, y_test)
    bagging_ensemble = bagging(X_train_d, X_test_d, y_train, y_test)
    rf = RandomForestClassifier(criterion='entropy', max_depth=20,
                       min_samples_split=25, n_estimators=100,
                       random_state=0, class_weight='balanced_subsample')
    dt = DecisionTreeClassifier(criterion='entropy', max_depth=20,
                                min_samples_split=1000, random_state=0,
                                class_weight='balanced')
    gb = GradientBoostingClassifier(loss='exponential', n_estimators=100,
                                    random_state=0)
    ada = AdaBoostClassifier(base_estimator=dt, n_estimators=100,
                             learning_rate=0.5)
    estimators = [('dt', dt), ('rf', rf), ('ada', ada), ('gb', gb)]
    voting_ensemble = voting(X_train_d, X_test_d, y_train, y_test, estimators)
    
    #X_train, X_test = recursive_feature_selection(
      # X_train, X_test, y_train, model=rf)
    #print(X_train.columns)
    #bagging_ensemble = bagging(X_train_d, X_test_d, y_train, y_test)
    

if __name__ == "__main__":
    main()

