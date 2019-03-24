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
import fnmatch
from   sklearn.ensemble        import RandomForestClassifier
from   sklearn.model_selection import GridSearchCV
from   sklearn.utils           import shuffle
from   sklearn                 import svm 
from   sklearn.neural_network  import MLPClassifier

parser = argparse.ArgumentParser()
parser.add_argument("--num_runs", required=True, type=int,
                    help="Number of runs for the cross validation")
parser.add_argument("--num_splits", default=5, type=int,
                    help="Number of splits for the cross validation")


def get_average_features(filenames):
    """#Load and aggregate the resnet features by the average.

    #Args:
    #    filenames: list of filenames of length `num_patients` corresponding to resnet features

    #Returns:
    #    features: np.array of mean resnet features, shape `(num_patients, 2048)`
    #
    # Load numpy arrays"""
    
    features = []

    for f in filenames:
        patient_features = np.load(f)

      # Remove location features (but we could use them?)
        patient_features = patient_features[:, 3:]

        aggregated_features_1 = np.max(patient_features, axis=0)
        aggregated_features_2 = np.mean(patient_features, axis=0)
        aggregated_features_3 = np.min(patient_features, axis=0)
        aggregated_features = np.concatenate((aggregated_features_1,aggregated_features_2, aggregated_features_3))
        print(aggregated_features.shape)
        features.append(aggregated_features)

    features = np.stack(features, axis=0)
    print(features.shape)
    return features

if __name__ == "__main__":
    args = parser.parse_args()

    path_root = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
    path = Path(path_root)
    assert path.is_dir()

    # -------------------------------------------------------------------------
    # Load the data
    args.data_dir = path
    assert args.data_dir.is_dir()

    train_dir = args.data_dir / "train_input" / "resnet_features"
    test_dir = args.data_dir / "test_input" / "resnet_features"

    train_output_filename = args.data_dir / "train_output.csv"

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
    features_test = get_average_features(filenames_train)
    

    features_train_shuf, labels_train_shuf = shuffle(features_train, labels_train, random_state=0)
    
    # -------------------------------------------------------------------------
    # Use the average resnet features to predict the labels

    # Multiple cross validations on the training set
    #  aucs = []
    #  for seed in range(args.num_runs):
    #      # Use logistic regression with L2 penalty
        
    #     estimator = RandomForestRegression(
    #          best_estimator)
    # #     #estimator = sklearn.linear_model.LogisticRegression(
    # #        # penalty="l2", C=1.0, solver="liblinear")

    #      cv = sklearn.model_selection.StratifiedKFold(n_splits=args.num_splits, shuffle=True,
    #                                                   random_state=seed)

    # #     # Cross validation on the training set
    #      auc = sklearn.model_selection.cross_val_score(estimator, X=features_train, y=labels_train,
    # #                                                   cv=cv, scoring="roc_auc", verbose=0)

    #      aucs.append(auc)


    #rfr = RandomForestClassifier()
    
    #rfr = svm.SVC()
    rfr = MLPClassifier()

    # params = {
    #     "n_estimators": [10, 20, 50],
    #     "max_depth": [6,10,16,None]
    # }

    params = {
        "solver" :['lbfgs'],#'sgd','adam'], 
        "alpha":[1e-5],#,1e-6], #2e-5,5e-5
        "hidden_layer_sizes":[(100,50,50)],#(200,100)],#20,15,10,5)],#(7, 5),(5,2),(20,20,20,15,15,10,10)
        "random_state":[1],
        "activation" : ['relu'],#,'tanh'],#'logistic'
        "learning_rate_init":[1e-3],#,1e-2],#5e-1,1e-4
        "learning_rate":['adaptive'],#,'constant','invscaling'],
        "batch_size":[10],#5]#,50,100,200]
        "early_stopping" : [True]
    }

    clf = GridSearchCV(rfr, params, cv=5, verbose = 5, scoring = "roc_auc")

    clf.fit(features_train_shuf, labels_train_shuf)
    best_estimator = clf.best_estimator_
    
    print(clf.best_score_)
    print(clf.best_params_)
    
    # aucs = np.array(aucs)

    # print("Predicting weak labels by mean resnet")
    # print("AUC: mean {}, std {}".format(aucs.mean(), aucs.std()))

    # # -------------------------------------------------------------------------
    # # Prediction on the test set

    # # Train a final model on the full training set
    #  estimator = sklearn.linear_model.LogisticRegression(
    #     penalty="l2", C=1.0, solver="liblinear")
    best_estimator.fit(features_train, labels_train)

    preds_test = best_estimator.predict(features_test)

# # Check that predictions are in [0, 1]
    assert np.max(preds_test) <= 1.0
    assert np.min(preds_test) >= 0.0

# # -------------------------------------------------------------------------
# # Write the predictions in a csv file, to export them in the suitable format
# # to the data challenge platform
    ids_number_test = [i.split("ID_")[1] for i in ids_test]
    test_output = pd.DataFrame({"ID": ids_number_test, "Target": preds_test})
    test_output.set_index("ID", inplace=True)
    test_output.to_csv(args.data_dir / "preds_test_baseline.csv")
