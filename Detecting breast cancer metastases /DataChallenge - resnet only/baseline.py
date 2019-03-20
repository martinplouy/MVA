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
from sklearn.model_selection import GridSearchCV
from sklearn.utils import shuffle
from sklearn import svm
from sklearn.neural_network import MLPClassifier

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
        aggregated_features_3 = np.min(patient_features, axis=0)
        aggregated_features = np.concatenate((aggregated_features_1,
                                              aggregated_features_2, aggregated_features_3))
        # print(aggregated_features.shape)
        features.append(aggregated_features)

    features = np.stack(features, axis=0)
    return features


def computeEnsemblePred(predsMLP, preds_RF, preds_Log):
    preds = [(x + y + z)/3.0 for (x, y, z)
             in zip(predsMLP, preds_RF, preds_Log)]
    return preds


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

    # -------------------------------------------------------------------------
    # Use the average resnet features to predict the labels

    #rfr = RandomForestClassifier()

    #rfr = svm.SVC()
    rfr = MLPClassifier()

    # params = {
    #     "n_estimators": [10, 20, 50],
    #     "max_depth": [6,10,16,None]
    # }

    # params = {
    #     "solver": ['lbfgs', 'adam'],  # 'sgd','adam'],
    #     "alpha": [1e-5],  # ,1e-6], #2e-5,5e-5
    #     "hidden_layer_sizes": [(100, 50, 20), (100, 50)],
    #     "activation": ['relu'],
    #     "learning_rate_init": [1e-4],  # ,1e-2],#5e-1,1e-4
    #     "learning_rate": ['adaptive'],  # ,'constant','invscaling'],
    #     "batch_size": [10]
    # }

    # clf = GridSearchCV(rfr, params, cv=5, verbose=5,
    #                    scoring="roc_auc", n_jobs=2)

    # clf.fit(features_train_shuf, labels_train_shuf)
    # best_estimator = clf.best_estimator_

    # print(clf.best_score_)
    # print(clf.best_params_)

    bestMPLClassifier = MLPClassifier(solver="lbfgs", alpha=1e-5, hidden_layer_sizes=(
        100, 50), activation='relu', learning_rate_init=1e-4, learning_rate='adaptive', batch_size=10)
    sklearn.model_selection.cross_val_score(bestMPLClassifier, X=features_train, y=labels_train,
                                            cv=5, scoring="roc_auc", verbose=10)
    bestMPLClassifier.fit(features_train, labels_train)
    preds_MLP = bestMPLClassifier.predict(features_test)

    bestRandomForestClassifier = RandomForestClassifier(n_estimators=50)
    bestRandomForestClassifier.fit(features_train, labels_train)
    preds_RF = bestRandomForestClassifier.predict(features_test)

    bestLogClassifier = sklearn.linear_model.LogisticRegression(
        penalty="l2", C=1.0, solver="liblinear")
    bestLogClassifier.fit(features_train, labels_train)
    preds_Log = bestLogClassifier.predict(features_test)

    # # -------------------------------------------------------------------------
    # # Prediction on the test set

    # # Train a final model on the full training set
    preds_test = preds_MLP

    # # aggregate the data of 3 classifiers
    # preds_test = computeEnsemblePred(preds_MLP, preds_RF, preds_Log)


# # Check that predictions are in [0, 1]
    assert np.max(preds_test) <= 1.0
    assert np.min(preds_test) >= 0.0

# # -------------------------------------------------------------------------
# # Write the predictions in a csv file, to export them in the suitable format
# # to the data challenge platform
    ids_number_test = [i.split("ID_")[1] for i in ids_test]
    test_output = pd.DataFrame({"ID": ids_number_test, "Target": preds_test})
    test_output.set_index("ID", inplace=True)
    test_output.to_csv(data_dir / "preds_test_baseline.csv")
