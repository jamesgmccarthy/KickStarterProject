import ModelsKS
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc, roc_auc_score, confusion_matrix, f1_score, precision_score, recall_score
import FeatureSelection
from ModelsKS import _create_dummy_features, _create_label_encoding
from FeatureEngineeringKS import prepare_data
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.externals import joblib
import matplotlib.pyplot as plt


def create_strat_k_folds(input, k):
    df = prepare_data(input)
    y = df['state']
    X = df.drop('state', axis=1)
    cv = StratifiedKFold(n_splits=k, random_state=0)
    return cv.split(X, y)


def eval_model(folds, models, metric='roc_curve', dummy=False):
    fprs = {}
    tprs = {}
    aucs = {}
    fold = 0
    for train, test in folds:
        X_train, X_test, y_train, y_test = FeatureSelection.main(
            './kickstarter-projects/ks-projects-201801.csv', split='strat_k_fold', train=train, test=test)
        if dummy == True:
            X_train, X_test = _create_dummy_features(
                X_train, X_test)
        elif dummy == False:
            X_train, X_test = _create_label_encoding(X_train, X_test)

        for name, model in models.items():
            model_ = model.fit(X_train, y_train)
            predicitons = model_.predict_proba(X_test)
            fpr, tpr, thresholds = roc_curve(
                y_test, predicitons[:, 1], pos_label=1)
            auc_score = auc(fpr, tpr)
            aucs[name] = {fold: auc_score}
        fold += 1
    return aucs


def get_auc_score(aucs):
    score_dict = {}
    for model, scores in aucs.items():
        score = []
        for fold, value in scores.items():
            score.append(value)
        score_dict[model] = np.mean(score)
    for model, value in score_dict.items():
        print(f"{model} CV AUC:", value)
    return score_dict.values


def get_roc_curve(fprs, tprs, model, score):
    plt.plot(fprs, tprs, linestyle='-', color='blue')
    plt.plot([0, 1], [0, 1], linestyle='--', color='red')
    plt.title(f"ROC of best model: {model}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.savefig('./Figures/Roc_curve_best_model.png', dpi=300)
    plt.show()


def main():
    splitter = create_strat_k_folds(
        './kickstarter-projects/ks-projects-201801.csv', 3)
    gridsearch_model = joblib.load('./ModelHyperparm/rfParams.pkl')
    rf = RandomForestClassifier(**gridsearch_model.best_params_,
                                n_estimators=100, random_state=0,
                                class_weight='balanced_subsample')
    dt = DecisionTreeClassifier(criterion='entropy', max_depth=20,
                                min_samples_split=1000, random_state=0,
                                class_weight='balanced')
    gb = GradientBoostingClassifier(loss='exponential', n_estimators=100,
                                    random_state=0)
    ada = AdaBoostClassifier(base_estimator=dt, n_estimators=100,
                             learning_rate=0.5)

    estimators = [('dt', dt), ('rf', rf), ('ada', ada), ('gb', gb)]
    voting_ensemble = VotingClassifier(estimators, voting='soft')
    models = {'Random Forest': rf, 'Decision Tree': dt, 'Gradient Booster': gb,
              'AdaBooster': ada, "Voting Ensemble": voting_ensemble}
    fprs, tprs, aucs = eval_model(splitter, models, dummy=True)
    auc_scores = get_auc_score(aucs)

    X_train, X_test, y_train, y_test = FeatureSelection.main(
        "./kickstarter-projects/ks-projects-201801.csv")
    X_train, X_test = _create_dummy_features(X_train, X_test)
    voting_ensemble.fit(X_train, y_train)
    predicitons = voting_ensemble.predict_proba(X_test)
    fpr, tpr, thresholds = roc_curve(
        y_test, predicitons[:, 1], pos_label=1)
    auc_score = auc(fpr, tpr)
    get_roc_curve(fpr, tpr, 'Voting Ensemble', auc_score)
    predicitons = voting_ensemble.predict(X_test)
    conf_matrix = confusion_matrix(y_test, predicitons)
    print(conf_matrix)
    print("F1:", f1_score(y_test, predicitons))
    print("Recall", recall_score(y_test, predicitons))
    print("Percision:", precision_score(y_test, predicitons))


if __name__ == '__main__':
    main()
