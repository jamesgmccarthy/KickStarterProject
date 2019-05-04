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
from sklearn.ensemble import BaggingClassifier, VotingClassifier

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
    return dt


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
    return lgr


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
    return knn


def voting(X_train, X_test, y_train, y_test, estimators):
    ensemble = VotingClassifier(estimators, voting='hard')
    ensemble.fit(X_train, y_train)
    acc = ensemble.score(X_test, y_test)
    print(acc)


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
"""
# %%


def main():
    X_train_d, X_test_d, y_train_d, y_test_d = FeatureEngineeringKS.main(
        "./kickstarter-projects/ks-projects-201801.csv")
    X_train, X_test, y_train, y_test = FeatureEngineeringKS.main(
        './kickstarter-projects/ks-projects-201801.csv', dummy=False)
    dt = decisionTree(X_train_d, X_test_d, y_train, y_test)
    lgr = logisticRegrs(X_train_d, X_test_d, y_train_d, y_test_d)
    knn = kNeighbor(X_train, X_test, y_train, y_test)
    estimators = [('log_reg', lgr), ('knn', knn)]
    voting(X_train, X_test, y_train, y_test, estimators)


main()
"""
# %%
X_train, X_test, y_train, y_test = FeatureEngineeringKS.main(
    './kickstarter-projects/ks-projects-201801.csv', dummy=False)
X_train['goal_bucket'] = X_train.groupby(['main_category'])['usd_goal_real'].transform(
    lambda x: pd.qcut(x, [0, 0.25, 0.5, 0.75, 1], labels=[1, 2, 3, 4]))
# %%
projects = X_train.merge(y_train,
                         how='left', right_index=True, left_index=True).set_index(X_train.index)
success_projects = projects[projects['state'] == 1]

# %%
success_projects['pledge_pb'] = success_projects['usd_pledged_real'] / \
    success_projects['backers']
mean_pledge_pb = pd.DataFrame(success_projects.groupby(
    ['main_category', 'launch_year', 'launch_quarter', 'goal_bucket'])['pledge_pb'].mean())
mean_pledge_pb.columns = ['mean_pledge_pb']
mean_pledge_pb.reset_index(inplace=True)

# %%
success_projects = success_projects.merge(mean_pledge_pb, how='left', on=[
                                          'main_category', 'launch_year', 'launch_quarter', 'goal_bucket']).set_index(success_projects.index)
#success_projects.drop('pledge_pb', axis=1)
# %%
success_projects['mean_number_of_backers'] = success_projects['usd_goal_real'] / \
    success_projects['mean_pledge_pb']
# %%
success_projects = success_projects[[
    'main_category', 'launch_year', 'launch_quarter',
    'goal_bucket', 'mean_number_of_backers']]
# %%
success_projects = success_projects.groupby(['main_category', 'launch_year',
                                             'launch_quarter', 'goal_bucket']).mean()

# %%
success_projects.reset_index(inplace=True)
# %%
X_train = pd.merge(X_train, success_projects, how='left',
                   on=['main_category', 'launch_year',
                       'launch_quarter', 'goal_bucket'])
# %%
backer = X_train.groupby(
    ['main_category', 'launch_year', 'launch_quarter'])
# %%

# %%
y_train
# %%
X_train.index

# %%
projects['state']

# %%
X_train, X_test, y_train, y_test = FeatureEngineeringKS.main(
    './kickstarter-projects/ks-projects-201801.csv', dummy=False)

# %%
X_train['mean_number_of_backers']

# %%
X_test.index

# %%
y_train.index

# %%
y_test.index


# %%
X_test.info()

# %%
"""
