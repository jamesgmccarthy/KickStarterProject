# %%
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.externals import joblib
import FeatureEngineeringKS
import ParameterSelection

# %%


def decisionTree(X_train, X_test, y_train, y_test, force=False):
    if not os.path.isfile("./ModelHyperparm/dtParams.pkl") or force:
        params = ParameterSelection.dtParamSelect(X_train, y_train)
        dt = DecisionTreeClassifier(**params)
    elif os.path.isfile("./ModelHyperparm/dtParams.pkl") and not force:
        gridsearch_model = joblib.load("./ModelHyperparm/dtParams.pkl")
        dt = DecisionTreeClassifier(**gridsearch_model.best_params_)

    dt.fit(X_train, y_train)
    acc = dt.score(X_test, y_test)
    print('Decision Tree accuracy:', acc)
    return acc


def logisticRegrs(X_train, X_test, y_train, y_test, force=False):
    if not os.path.isfile("./ModelHyperparm/lgrParams.pkl") or force:
        params = ParameterSelection.lgrParamSelect(X_train, y_train)
        lgr = LogisticRegression(**params)
    elif os.path.isfile("./ModelHyperparm/lgrParams.pkl") and not force:
        gridsearch_model = joblib.load("./ModelHyperparm/lgrParams.pkl")
        lgr = LogisticRegression(**gridsearch_model.best_params_)
    lgr.fit(X_train, y_train)
    acc = lgr.score(X_test, y_test)
    print('Logistic Regression accuracy:', acc)
    return acc


def kNeighbor(X_train, X_test, y_train, y_test, force=False):
    if not os.path.isfile('./ModelHyperparm/knnParams.pkl') or force:
        params = ParameterSelection.knnParamSelect(X_train, y_train)
        knn = KNeighborsClassifier(**params)
    elif os.path.isfile("./ModelHyperparm/knnParams.pkl") and not force:
        gridsearch_model = joblib.load('./ModelHyperparm/knnParams.pkl')
        knn = KNeighborsClassifier(**gridsearch_model.best_params_)
    knn.fit(X_train, y_train)
    acc = knn.score(X_test, y_test)
    print('K Nearest Neighbours accuracy:', acc)
    return acc


"""
# %%
ftr_imp = zip(list(X_train.columns.values), dt.feature_importances_)
for i in ftr_imp:
    print(i)

# %%
X_train, X_test, y_train, y_test = FeatureEngineeringKS.main(
    './kickstarter-projects/ks-projects-201801.csv', dummy=False)
train_data = lgb.Dataset(X_train, y_train,
                         categorical_feature=['main_category', 'category', 'currency', 'country'])
validation_data = lgb.Dataset(X_test, y_test, categorical_feature=['main_category', 'category', 'currency', 'country'],
                              reference=train_data)
# %%
gbm_param = {
    'boosting_type': "dart",
    'n_estimators': 1300,
    'learning_rate': 0.08,
    'num_leaves': 35,
    'colsample_bytree': .8,
    'subsample': .9,
    'max_depth': 9,
    'reg_alpha': .1,
    'reg_lambda': .1,
    'min_split_gain': .01
}

# %%
gbm_param['metric'] = ['auc']
gbm_model = lgb.train(gbm_param, train_data, 10, valid_sets=[validation_data])

# %%
gbm_model.best_score
# %%

"""


def main():
    X_train_d, X_test_d, y_train_d, y_test_d = FeatureEngineeringKS.main(
        "./kickstarter-projects/ks-projects-201801.csv")
    dt_acc = decisionTree(X_train_d, X_test_d, y_train_d, y_test_d)
    lgr_ac = logisticRegrs(X_train_d, X_test_d, y_train_d, y_test_d)
    X_train, X_test, y_train, y_test = FeatureEngineeringKS.main(
        './kickstarter-projects/ks-projects-201801.csv', dummy=False)
    knn = kNeighbor(X_train, X_test, y_train, y_test)


main()
"""
# %%
params = joblib.load('dtParams.pkl')

# %%
params.best_params_
dt = DecisionTreeClassifier(**params.best_params_)
# %%

# %%
dt.fit(X_train, y_train)
dt.score(X_test, y_test)

# %%
max(dt.feature_importances_)
# %%
X_train = X_train.drop('backers', 1)
X_test = X_test.drop('backers', 1)

# %%
X_train

# %%

X_train_d, X_test_d, y_train_d, y_test_d = FeatureEngineeringKS.main(
    './kickstarter-projects/ks-projects-201801.csv')

# %%
X_train['goal_buckets'] = X_train.groupby(['main_category'])['usd_goal_real'].transform(
    lambda x: pd.qcut(x, [0, 0.25, 0.5, 0.75, 1], labels=[1, 2, 3, 4]))


# %%
month = X_train.groupby(['goal_buckets', 'main_category',
                         'launch_quarter', 'launch_year'])['category'].count()
month1 = X_train.groupby(
    ['goal_buckets', 'main_category', 'launch_quarter', 'launch_year']).count()
# %%
month

# %%
month1[['category']]
"""
# %%

X_train, X_test, y_train, y_test = FeatureEngineeringKS.main(
    './kickstarter-projects/ks-projects-201801.csv', dummy=False)
X_train.set_index('ID')
X_train['goal_bucket'] = X_train.groupby(['main_category'])['usd_goal_real'].transform(
    lambda x: pd.qcut(x, [0, 0.25, 0.5, 0.75, 1], labels=[1, 2, 3, 4]))

# %%
project_per_quarter = X_train.groupby(['main_category', 'goal_bucket',
                                       'launch_year', 'launch_quarter']).count()
# %%
project_per_quarter = project_per_quarter[['name']]
# %%
project_per_quarter.reset_index(inplace=True)


# %%


# %%
