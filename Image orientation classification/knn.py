import sys
import operator
import numpy as np
from queue import PriorityQueue
import math
import time


def knn_train(train_file, model_file):

    with open(train_file, "r") as fp:
        with open(model_file, "w") as fp1:
            for line in fp.readlines():
                fp1.write(line)
    fp.close()
    fp1.close()

    return model_file

def knn_readfiles(test_file, model_file):
    training_features = []
    training_labels = []
    testing_features = []
    testing_labels = []
    testing_images = []
    with open(model_file, "r") as fp:
        for line in fp.readlines():
            image = line.split()
            training_labels.append(int(image[1]))
            pixels = image[2:]
            pixels = list(map(int, pixels))
            training_features.append(pixels)
    fp.close()

    with open(test_file, "r") as fp:
        for line in fp.readlines():
            image = line.split()
            testing_images.append(image[0])
            testing_labels.append(int(image[1]))
            pixels = image[2:]
            pixels = list(map(int, pixels))
            testing_features.append(pixels)
    fp.close()
    return np.array(training_features), np.array(training_labels), np.array(testing_features), np.array(testing_labels), np.array(testing_images)

def knn(test_features, train_features, train_labels, k):
    labels = {0:0, 90:0, 180:0, 270:0}
    knn_test = PriorityQueue()
    for i in range(len(train_labels)):
        train_pixels = train_features[i]
        train_orient = train_labels[i]
        distance = math.sqrt(np.sum(np.power(np.subtract(train_pixels, test_features), 2)))
        knn_test.put((distance, train_orient))
    for i in range(k):
        item = knn_test.get()
        labels[item[1]] += 1
    predicted_label = max(labels, key=labels.get)
    return predicted_label
