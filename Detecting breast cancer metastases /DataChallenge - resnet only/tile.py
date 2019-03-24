import argparse
from pathlib import Path

import os
import numpy as np
import pandas as pd
import sklearn
import sklearn.linear_model
import sklearn.metrics
import sklearn.model_selection
import re
from   sklearn.ensemble        import RandomForestClassifier
from   sklearn.model_selection import GridSearchCV
from   sklearn.utils           import shuffle
from   sklearn                 import svm 
from   sklearn.neural_network  import MLPClassifier


parser = argparse.ArgumentParser()

def get_tile_features(filenames):
    cpt = 0
    for f in filenames:
        m =str(f)
        m = m.endswith("annotated.npy")
        if m == True:
            Xtrain_temp = np.load(f)
            Xtrain_temp = Xtrain_temp[:, 3:]
            if cpt == 0:
                Xtrain = Xtrain_temp
                cpt = cpt +1
            else:
                Xtrain = np.concatenate((Xtrain,Xtrain_temp),axis=0)
                print(Xtrain.shape)
    return Xtrain


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
    train_dir_Y = args.data_dir / "train_input"
    test_dir = args.data_dir / "test_input" / "resnet_features"

    train_output_filename = args.data_dir / "train_output.csv"

    train_output = pd.read_csv(train_output_filename)
    
    # stocker les labels des tiles
    Ytrain =[]
    Ytrain = pd.read_csv(train_dir_Y/"train_tile_annotations_2.csv")
    Ytrain = Ytrain.iloc[:,5] #prendre colonne 6 pour label 0 ou 1

    labels_train = Ytrain

    # Get the filenames for train
    filenames_train = [train_dir /
                       "{}.npy".format(idx) for idx in train_output["ID"]]
    for filename in filenames_train:
        assert filename.is_file(), filename

    features_train = get_tile_features(filenames_train)

    #print(Ytrain)
    #print(features_train[:,0])
    #print(features_train[:,1])

    features_train_shuf, labels_train_shuf = shuffle(features_train, labels_train, random_state=0)

    rfr = MLPClassifier()

    params = {
        "solver" :['lbfgs'],#'sgd','adam'], 
        "alpha":[1e-5],#,1e-6], #2e-5,5e-5
        "hidden_layer_sizes":[(200,100)],#(200,100)],#20,15,10,5)],#(7, 5),(5,2),(20,20,20,15,15,10,10)
        "random_state":[1],
        "activation" : ['relu'],#,'tanh'],#'logistic'
        "learning_rate_init":[1e-3],#,1e-2],#5e-1,1e-4
        "learning_rate":['adaptive'],#,'constant','invscaling'],
        "batch_size":[10],#5]#,50,100,200]
        "early_stopping" : [True]
    }

    """params = {
        "max_depth":[4],
        "n_estimators":[10,20,30]
    }"""
    clf = GridSearchCV(rfr, params, cv=5, verbose = 5, scoring = "roc_auc")

    clf.fit(features_train_shuf, labels_train_shuf)
    best_estimator = clf.best_estimator_
    print(clf.best_score_)
    print(clf.best_params_)