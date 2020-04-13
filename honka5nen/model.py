# Bagging ブートストラップ手法を用いて多数決(Hard)と重み付き多数決(soft)
# Boosting XGBoost(勾配ブースティング)と間違いに重みをつけるAdaboost
# Stacking 
import numpy as np
import optuna
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


# XGBoost　最適化
def xgb_object(trial, X_tra, Y_tra, X_tes, Y_tes):
    learning_rate = trial.suggest_loguniform('learning_rate', 0.01, 0.5)
    max_depth = trial.suggest_int('max_depth', 4, 12)
    min_child_weight = trial.suggest_int('min_child_weight', 1, 6)
    subsample = trial.suggest_loguniform('subsample', 0.01, 1)
    colsample_bytree =trial.suggest_loguniform('colsample_bytree', 0.01, 1)
    lambda1 = trial.suggest_loguniform('reg_lambda', 1, 5)
    gamma = trial.suggest_loguniform('gamma', 0.001, 5)

    xgb = XGBClassifier(eta=learning_rate, gamma=gamma, max_depth=max_depth,
                        min_child_weight=min_child_weight,
                        subsample=subsample, reg_lambda=lambda1,
                        colsample_bytree=colsample_bytree,
                        )
    xgb.fit(X_tra, Y_tra)
    y_pred = xgb.predict(X_tes)

    print("=====================================================================")
    print("ACC={:.4f}".format(100*accuracy_score(Y_tes, y_pred)))
    print("再現率R={:.4f}".format(100*recall_score(Y_tes, y_pred)))
    print("適合率P={:.4f}".format(100*precision_score(Y_tes, y_pred)))
    print("F1スコア={:.4f}".format(100*f1_score(Y_tes, y_pred)))
    print(confusion_matrix(Y_tes, y_pred))
    print("=====================================================================")
    return (1.0 - f1_score(Y_tes, y_pred))

#XGBoost　デフォルトパラメータ
def default_xgb_object(X_tra,Y_tra,X_tes,Y_tes):
    xgb = XGBClassifier()
    xgb.fit(X_tra, Y_tra)
    y_pred = xgb.predict(X_tes)

    print("=======================XGBoostデフォルトパラメータ============================")
    print("ACC={:.4f}".format(100*accuracy_score(Y_tes, y_pred)))
    print("再現率R={:.4f}".format(100*recall_score(Y_tes, y_pred)))
    print("適合率P={:.4f}".format(100*precision_score(Y_tes, y_pred)))
    print("F1スコア={:.4f}".format(100*f1_score(Y_tes, y_pred)))
    print(confusion_matrix(Y_tes, y_pred))
    print("============================================================================")

#SVM　デフォ
def default_svm(X_tra,Y_tra,X_tes,Y_tes):
    svm = SVC(gamma='scale', probability=True)
    svm.fit(X_tra,Y_tra)
    y_pred = svm.predict(X_tes)

    print("=========================SVMデフォルトパラメータ==============================")
    print("ACC={:.4f}".format(100*accuracy_score(Y_tes, y_pred)))
    print("再現率R={:.4f}".format(100*recall_score(Y_tes, y_pred)))
    print("適合率P={:.4f}".format(100*precision_score(Y_tes, y_pred)))
    print("F1スコア={:.4f}".format(100*f1_score(Y_tes, y_pred)))
    print(confusion_matrix(Y_tes, y_pred))
    print("============================================================================")

# SVM　最適化
def svm_object(trial, X_tra, Y_tra, X_tes, Y_tes):
    #kernel = trial.suggest_categorical('kernel', ['linear', 'poly', 'rbf'])
    C = trial.suggest_loguniform('C', 1e-4, 1e+4)
    #gamma = trial.suggest_loguniform('gamma', 1e-4, 1e+4)

    svm = SVC(C=C, probability=True)
    svm.fit(X_tra, Y_tra)
    y_pred = svm.predict(X_tes)

    print("=========================SVM最適中================================")
    print("ACC={:.4f}".format(100*accuracy_score(Y_tes, y_pred)))
    print("再現率R={:.4f}".format(100*recall_score(Y_tes, y_pred)))
    print("適合率P={:.4f}".format(100*precision_score(Y_tes, y_pred)))
    print("F1スコア={:.4f}".format(100*f1_score(Y_tes, y_pred)))
    print(confusion_matrix(Y_tes, y_pred))
    print("============================================================================")
    return 1.0 - f1_score(Y_tes, y_pred)

#Voting デフォ ロジスティック回帰　ランダムフォレスト　K近傍法　SVM　ナイーブベイズ
def default_voting(X_tra,Y_tra,X_tes,Y_tes):
    clf1 = LogisticRegression(solver='lbfgs', max_iter=10000)
    clf2 = RandomForestClassifier(n_estimators=100)
    clf3 = KNeighborsClassifier()
    clf4 = SVC(gamma='scale', probability=True)
    clf5 = GaussianNB()

    eclf = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2),
                                        ('knn', clf3), ('svm', clf4),
                                        ('nb', clf5)], voting='hard')
    eclf.fit(X_tra, Y_tra)

    print("Votingデフォルトパラメータ")
    for name, estimator in eclf.named_estimators_.items():
        print(name, ":{:.4f}".format(100*accuracy_score(Y_tes, estimator.predict(X_tes))))
        print("再現率R = {:.4f}".format(100*recall_score(Y_tes, estimator.predict(X_tes))))
        print("適合率P = {:.4f}".format(100*precision_score(Y_tes, estimator.predict(X_tes))))
        print("F1スコア= {:.4f}".format(100*f1_score(Y_tes, estimator.predict(X_tes))))
        print(confusion_matrix(Y_tes, estimator.predict(X_tes)))

    # 総合評価 
    print("正答率 = {:.4f}".format(100*accuracy_score(Y_tes, eclf.predict(X_tes))))
    print("再現率R = {:.4f}".format(100*recall_score(Y_tes, eclf.predict(X_tes))))
    print("適合率P = {:.4f}".format(100*precision_score(Y_tes, eclf.predict(X_tes))))
    print("F1スコア = {:.4f}".format(100*f1_score(Y_tes, eclf.predict(X_tes))))
    print(confusion_matrix(Y_tes, eclf.predict(X_tes)))