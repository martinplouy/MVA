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
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.utils import shuffle
from sklearn import svm
from sklearn.neural_network import MLPClassifier
# from baseline import computeGridSearchOfAllModels


parser = argparse.ArgumentParser()


def get_tile_features(filenames):
    cpt = 0
    for f in filenames:
        m = str(f)
        m = m.endswith("annotated.npy")
        if m == True:
            Xtrain_temp = np.load(f)
            Xtrain_temp = Xtrain_temp[:, 3:]
            if cpt == 0:
                Xtrain = Xtrain_temp
                cpt = cpt + 1
            else:
                Xtrain = np.concatenate((Xtrain, Xtrain_temp), axis=0)
    return Xtrain


def computeTilePredictorModel():
    path_root = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))

    path = Path(path_root)
    data_dir = path

    train_dir = data_dir / "train_input" / "resnet_features"
    train_input_dir = data_dir / "train_input"
    train_output_filename = data_dir / "train_output.csv"
    train_output = pd.read_csv(train_output_filename)

    # Get the filenames for train
    filenames_train = [train_dir /
                       "{}.npy".format(idx) for idx in train_output["ID"]]

    for filename in filenames_train:
        assert filename.is_file(), filename
    features_train = get_tile_features(filenames_train)

    # stocker les labels des tiles
    Ytrain = pd.read_csv(train_input_dir/"train_tile_annotations_2.csv")
    Ytrain = Ytrain.iloc[:, 5]  # prendre colonne 6 pour label 0 ou 1
    labels_train = Ytrain

    features_train_shuf, labels_train_shuf = shuffle(
        features_train, labels_train, random_state=0)

    print(np.sum(labels_train), len(labels_train))

    # mlp = MLPClassifier(solver="adam", early_stopping=True, alpha=1e-5,
    #                     hidden_layer_sizes=(100, 50, 20), activation='relu', learning_rate_init=0.0001)
    # mlp.fit(features_train_shuf, labels_train_shuf)

    bestLogClassifier = sklearn.linear_model.LogisticRegression(
        penalty="l2", C=0.05, solver="liblinear", tol=0.01)
    bestLogClassifier.fit(features_train, labels_train)

    # bestSVCClassifier = sklearn.svm.SVC(
    #     probability=True, gamma='scale', C=3.0, tol=0.1, kernel="rbf")
    # bestSVCClassifier.fit(features_train, labels_train)
    return bestLogClassifier


if __name__ == "__main__":
    path_root = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
    path = Path(path_root)
    data_dir = path
    train_dir = data_dir / "train_input" / "resnet_features"
    train_dir_Y = data_dir / "train_input"
    test_dir = data_dir / "test_input" / "resnet_features"

    train_output_filename = data_dir / "train_output.csv"

    train_output = pd.read_csv(train_output_filename)

    # stocker les labels des tiles
    Ytrain = pd.read_csv(train_dir_Y/"train_tile_annotations_2.csv")
    Ytrain = Ytrain.iloc[:, 5]  # prendre colonne 6 pour label 0 ou 1

    labels_train = Ytrain

    # Get the filenames for train
    filenames_train = [train_dir /
                       "{}.npy".format(idx) for idx in train_output["ID"]]

    for filename in filenames_train:
        assert filename.is_file(), filename

    features_train = get_tile_features(filenames_train)

    features_train_shuf, labels_train_shuf = shuffle(
        features_train, labels_train, random_state=0)
    computeGridSearchOfAllModels(features_train_shuf, labels_train_shuf)
