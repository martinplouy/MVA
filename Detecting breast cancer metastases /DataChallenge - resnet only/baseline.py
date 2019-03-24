"""Train the baseline model i.e. a logistic regression on the average of the resnet features and
and make a prediction.
"""

import argparse
from pathlib import Path

import os
import numpy as np
import pandas as pd
import sklearn
import sklearn.linear_model
import sklearn.metrics
import sklearn.model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.utils import shuffle
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser()


def get_average_features(filenames):
    """Load and aggregate the resnet features by the average.

    Args:
        filenames: list of filenames of length `num_patients` corresponding to resnet features

    Returns:
        features: np.array of mean resnet features, shape `(num_patients, 2048)`
    """
    # Load numpy arrays
    features = []
    for f in filenames:
        patient_features = np.load(f)

        # Remove location features (but we could use them?)
        patient_features = patient_features[:, 3:]

        aggregated_features_1 = np.max(patient_features, axis=0)
        aggregated_features_2 = np.mean(patient_features, axis=0)
        # aggregated_features_3 = np.min(patient_features, axis=0)
        # aggregated_features = np.concatenate((aggregated_features_1,
        #                                       aggregated_features_2, aggregated_features_3))
        aggregated_features = np.concatenate((aggregated_features_1,
                                              aggregated_features_2))
        # print(aggregated_features.shape)
        features.append(aggregated_features)

    features = np.stack(features, axis=0)
    return features


def computeMeanOfPreds(preds_1, preds_2, preds_3, preds_4, preds_5):
    preds = [(v+w+x + y + z)/5.0 for (v, w, x, y, z)
             in zip(preds_1, preds_2, preds_3, preds_4, preds_5)]
    return preds


def computeEnsemblePreds(X, y, X_test):

    bestLogClassifier = sklearn.linear_model.LogisticRegression(
        penalty="l2", C=0.5, solver="liblinear", tol=0.01)
    bestLogClassifier.fit(X, y)
    preds_Log = bestLogClassifier.predict_proba(X_test)[:, 1]

    bestSVCClassifier = sklearn.svm.SVC(
        probability=True, gamma='scale', C=0.1, tol=0.01, kernel="rbf")
    bestSVCClassifier.fit(X, y)
    preds_SVC = bestSVCClassifier.predict_proba(X_test)[:, 1]

    bestRandomForestClassifier = RandomForestClassifier(
        n_estimators=80, max_depth=None)
    bestRandomForestClassifier.fit(X, y)
    preds_RF = bestRandomForestClassifier.predict_proba(X_test)[:, 1]

    bestRandomXGB = GradientBoostingClassifier(
        n_estimators=200, max_depth=6, subsample=0.8)
    bestRandomXGB.fit(X, y)
    preds_XGB = bestRandomXGB.predict_proba(X_test)[:, 1]

    bestMPLClassifier = MLPClassifier(
        solver="lbfgs", alpha=1e-5, hidden_layer_sizes=(100, 50), activation='tanh', random_state=12)
    bestMPLClassifier.fit(X, y)
    preds_MLP = bestMPLClassifier.predict_proba(X_test)[:, 1]

    # bestMPLClassifier2 = MLPClassifier(
    #     solver="lbfgs", alpha=1e-5, hidden_layer_sizes=(100, 50), activation='tanh', random_state=10)
    # bestMPLClassifier2.fit(X, y)
    # preds_MLP2 = bestMPLClassifier.predict_proba(X_test)[:, 1]

    # bestMPLClassifier3 = MLPClassifier(
    #     solver="lbfgs", alpha=1e-5, hidden_layer_sizes=(100, 50), activation='tanh', random_state=1)
    # bestMPLClassifier3.fit(X, y)
    # preds_MLP3 = bestMPLClassifier.predict_proba(X_test)[:, 1]

    # bestMPLClassifier4 = MLPClassifier(
    #     solver="lbfgs", alpha=1e-5, hidden_layer_sizes=(100, 50), activation='tanh', random_state=8)
    # bestMPLClassifier4.fit(X, y)
    # preds_MLP4 = bestMPLClassifier.predict_proba(X_test)[:, 1]

    # bestMPLClassifier5 = MLPClassifier(
    #     solver="lbfgs", alpha=1e-5, hidden_layer_sizes=(100, 50), activation='tanh', random_state=11)
    # bestMPLClassifier5.fit(X, y)
    # preds_MLP5 = bestMPLClassifier.predict_proba(X_test)[:, 1]

    preds_test = computeMeanOfPreds(
        preds_MLP, preds_RF, preds_XGB, preds_MLP, preds_Log)

    # preds_test = computeMeanOfPreds(
    #     preds_MLP, preds_MLP2, preds_MLP3, preds_MLP4, preds_MLP5)
    return preds_test


def computeMLPPreds(X, y, X_test):
    bestMPLClassifier = MLPClassifier(solver="adam", alpha=1e-5, hidden_layer_sizes=(
        100, 50), activation='relu', learning_rate_init=1e-4, learning_rate='adaptive', batch_size=10, early_stopping=True)
# sklearn.model_selection.cross_val_score(bestMPLClassifier, X=features_train_shuf, y=labels_train_shuf,
#                                         cv=5, scoring="roc_auc", verbose=10)
    bestMPLClassifier.fit(X, y)
    preds_test = bestMPLClassifier.predict_proba(X_test)[:, 1]

    return preds_test


def computeGridSearchOfAllModels(X, y):
    GridSearchLogReg(X, y)
    GridSearchSVM(X, y)
    GridSearchRF(X, y)
    GridSearchGradientBoosting(X, y)
    GridSearchMLP(X, y)


def GridSearchLogReg(X, y):
    print(" --------- Doing Grid Search of Logistic regression")
    params = {
        "C": [1.0, 0.5, 0.1],
        "tol": [1e-4, 1e-2]
    }
    log = sklearn.linear_model.LogisticRegression(
        penalty="l2", solver="liblinear")

    gcv = GridSearchCV(log, params, cv=5, verbose=0,
                       scoring="roc_auc", n_jobs=3)

    gcv.fit(features_train_shuf, labels_train_shuf)
    best_estimator = gcv.best_estimator_

    print("Best score reached is", gcv.best_score_)
    print("Best Params found are", gcv.best_params_)
    return best_estimator


def GridSearchSVM(X, y):
    print(" --------- Doing Grid Search of SVM")
    params = {
        "C": [1.0,  0.1, 0.05],
        "tol": [1e-3, 1e-2],
        "kernel": ["rbf", "sigmoid"],
    }
    svc = sklearn.svm.SVC(probability=True, gamma='scale')

    gcv = GridSearchCV(svc, params, cv=5, verbose=0,
                       scoring="roc_auc", n_jobs=3)

    gcv.fit(features_train_shuf, labels_train_shuf)
    best_estimator = gcv.best_estimator_

    print("Best score reached is", gcv.best_score_)
    print("Best Params found are", gcv.best_params_)
    return best_estimator


def GridSearchRF(X, y):
    print(" --------- Doing Grid Search of Random Forests")
    params = {
        "n_estimators": [50, 80, 150],
        "max_depth": [10,  None],
    }
    rf = RandomForestClassifier()

    gcv = GridSearchCV(rf, params, cv=5, verbose=0,
                       scoring="roc_auc", n_jobs=3)

    gcv.fit(features_train_shuf, labels_train_shuf)
    best_estimator = gcv.best_estimator_

    print("Best score reached is", gcv.best_score_)
    print("Best Params found are", gcv.best_params_)
    return best_estimator


def GridSearchGradientBoosting(X, y):
    print(" --------- Doing Grid Search of Gradient Boosting")
    params = {
        "n_estimators": [80, 150, 200],
        "max_depth": [4, 6, 8],
        "subsample": [0.6, 0.8, 1.0],
    }
    xgb = GradientBoostingClassifier()

    gcv = GridSearchCV(xgb, params, cv=5, verbose=2,
                       scoring="roc_auc", n_jobs=-1)

    gcv.fit(features_train_shuf, labels_train_shuf)
    best_estimator = gcv.best_estimator_

    print("Best score reached is", gcv.best_score_)
    print("Best Params found are", gcv.best_params_)
    return best_estimator


def GridSearchMLP(X, y):
    # params = {
    #     "solver": ['sgd'],
    #     "alpha": [1e-5, 1e-3],
    #     "hidden_layer_sizes": [(100, 50, 20), (100, 50)],
    #     "activation": ['relu'],
    #     "learning_rate_init": [1e-4, 1e-2],
    #     "learning_rate": ['adaptive'],
    #     "batch_size": [2, 10]
    # }
    # mlp = MLPClassifier(early_stopping=True)

    # clf = GridSearchCV(mlp, params, cv=5, verbose=5,
    #                    scoring="roc_auc", n_jobs=3)

    # clf.fit(features_train_shuf, labels_train_shuf)
    # best_estimator = clf.best_estimator_

    # print("for sgd", clf.best_score_)
    # print(clf.best_params_)

    # params = {
    #     "solver": ['adam'],
    #     "alpha": [1e-5, 1e-3],
    #     "hidden_layer_sizes": [(100, 50, 20), (100, 50)],
    #     "activation": ['relu'],
    #     "learning_rate_init": [1e-4, 1e-2],
    #     "batch_size": [2, 10]
    # }
    # mlp = MLPClassifier(early_stopping=True)

    # clf = GridSearchCV(mlp, params, cv=5, verbose=5,
    #                    scoring="roc_auc", n_jobs=3)

    # clf.fit(features_train_shuf, labels_train_shuf)
    # best_estimator = clf.best_estimator_

    # print("for adam", clf.best_score_)
    # print(clf.best_params_)

    params = {
        "solver": ['lbfgs'],
        "alpha": [1e-5, 1e-3],
        "hidden_layer_sizes": [(100, 50, 20), (100, 50)],
        "activation": ['relu', 'tanh'],
    }
    mlp = MLPClassifier()

    clf = GridSearchCV(mlp, params, cv=5, verbose=2,
                       scoring="roc_auc", n_jobs=-1)

    clf.fit(features_train_shuf, labels_train_shuf)
    best_estimator = clf.best_estimator_

    print("for lfbgs", clf.best_score_)
    print(clf.best_params_)

    return best_estimator


if __name__ == "__main__":

    path_root = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
    path = Path(path_root)
    assert path.is_dir()

    # -------------------------------------------------------------------------
    # Load the data
    data_dir = path
    assert data_dir.is_dir()

    train_dir = data_dir / "train_input" / "resnet_features"
    test_dir = data_dir / "test_input" / "resnet_features"

    train_output_filename = data_dir / "train_output.csv"

    train_output = pd.read_csv(train_output_filename)

    # Get the filenames for train
    filenames_train = [train_dir /
                       "{}.npy".format(idx) for idx in train_output["ID"]]
    for filename in filenames_train:
        assert filename.is_file(), filename

    # Get the labels
    labels_train = train_output["Target"].values

    assert len(filenames_train) == len(labels_train)

    # Get the numpy filenames for test
    filenames_test = sorted(test_dir.glob("*.npy"))
    for filename in filenames_test:
        assert filename.is_file(), filename
    ids_test = [f.stem for f in filenames_test]

    # Get the resnet features and aggregate them by the average
    features_train = get_average_features(filenames_train)
    features_test = get_average_features(filenames_test)

    features_train_shuf, labels_train_shuf = shuffle(
        features_train, labels_train, random_state=0)

    X_train, X_valid, y_train, y_valid = train_test_split(
        features_train_shuf, labels_train_shuf, test_size=0.20, random_state=42)

    # computeGridSearchOfAllModels(features_train_shuf, labels_train_shuf)

#     # # Train a final model on the full training set
#     preds_test = computeMLPPreds(X_train, y_train, X_valid)
#     print(sklearn.metrics.roc_auc_score(
#         y_valid, preds_test))

    preds_test = computeEnsemblePreds(X_train, y_train, X_valid)
    print(sklearn.metrics.roc_auc_score(
        y_valid, preds_test))

    preds_test = computeEnsemblePreds(
        features_train, labels_train, features_test)


# # Check that predictions are in [0, 1]
    assert np.max(preds_test) <= 1.0
    assert np.min(preds_test) >= 0.0

# # -------------------------------------------------------------------------
# # Write the predictions in a csv file, to export them in the suitable format
# # to the data challenge platform
    ids_number_test = [i.split("ID_")[1] for i in ids_test]
    test_output = pd.DataFrame(
        {"ID": ids_number_test, "Target": preds_test})
    test_output.set_index("ID", inplace=True)
    test_output.to_csv(data_dir / "preds_test_baseline.csv")
