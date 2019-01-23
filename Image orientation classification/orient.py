#!/usr/bin/python 3

import sys
import time
from knn import *
from AdaBoost import *
from random_forest import *

mode = sys.argv[1] # test or train
data_file = sys.argv[2]
model_file = sys.argv[3]
model = sys.argv[4]
start_time = time.time()
# KNN model
if model == "nearest":
    if mode == "train":
        model_file = knn_train(data_file, model_file)
    if mode == "test":
        train_features, train_labels, test_features, test_labels, test_images = knn_readfiles(data_file, model_file)
        k = 40
        accuracy = 0
        with open('output.txt', 'w') as fp:
            for i in range(len(test_images)):
                predicted_label = knn(test_features[i], train_features, train_labels, k)
                fp.write(str(test_images[i]) + " " + str(predicted_label))
                fp.write("\n")
                accuracy += (1 if predicted_label == test_labels[i] else 0)
        accuracy = 100 * (float(accuracy)/len(test_images))
        print("accuracy: " + str(accuracy))

# Adaboost model
if model == "adaboost":
    if mode == "train":
        train_ADB(data_file, model_file)
    elif mode == "test":
        test_ADB(data_file, model_file)

# Random Forest
if model == "forest":
    if mode == "train":
        random_forest_train(data_file, model_file)
    if mode == "test":
        accuracy, predictions, test_images = random_forest_test(data_file, model_file)
        with open('output.txt', 'r') as fp1:
            for i in range(len(test_images)):
                fp.write(str(test_images[i]) + " " + str(predictions[i]))
                fp.write("\n")
        print("accuracy: " + str(accuracy))

# Best - Adaboost
if model == "best":
    if mode == "train":
        train_ADB(data_file, model_file)
    elif mode == "test":
        test_ADB(data_file, model_file)
