#!/usr/bin/python 3

import sys
import random
import math
import operator
import numpy as np

# Test function for Adaboost model
def test_ADB(test_file, model_file):

    match = 0
    mismatch = 0
    test_f = open(test_file, "r")
    result = open("output_adaboost.txt", "w+") # write image tag and predicted orientation to output_adaboost.txt
    model_f = open(model_file, "r")
    NT = sum(1 for ln in open(test_file))   # lines in training file
    NM = sum(1 for ln in open(model_file))  # lines in model file

    orient_model = np.zeros((NM, 1), dtype=np.int_)  # create orientation array of size 1xNM for model file
    fets_model = np.zeros((NM, 2), dtype=np.int_)  # create features array of size 2xNM for test file
    test = np.empty(NT, dtype='S100')          # string to be written to output.txt
    orient = np.zeros((NT, 1), dtype=np.int_)   # create orientation array of size 1xNT for test file
    fets = np.zeros((NT, 192), dtype=np.int_)   # create features array of size 192xNT for test file
    A_set = np.zeros((NM, 1), dtype=np.float_)  # a factor of each model entry

    # fetch details of test file data
    for n, r in enumerate(test_f):
        img = r.split(' ', 2)
        test[n] = img[0]
        orient[n] = int(img[1])
        fets[n] = np.array([int(i) for i in img[2].split(' ')])
    test_f.close()

    # fetch details of model file data generated from training using Adaboost
    for n, r in enumerate(model_f):
        img = r.split(' ')
        orient_model[n] = int(img[0])
        fets_model[n] = [int(img[1]), int(img[2])]
        A_set[n] = float(img[3])
    test_f.close()

    # Calculate the final value H() sign and perform voting
    for n, r in enumerate(orient):
        orients = {0: 0, 90: 0, 180: 0, 270: 0}
        for m, rc in enumerate(orient_model):
            h = hypothesis([fets[n][col] for col in fets_model[m]])
            if h == 1:
                orients[int(rc)] += float(A_set[m])
            else:
                orients[int(rc)] -= float(A_set[m])
        # select the angle with max votes
        max_angle = max(orients.items(), key=operator.itemgetter(1))[0]
        image_tag = str(test[n])[2:-1]
        result.write("%s %s\n" % (image_tag, str(max_angle))) # write contents to output file
        # if the predicted angle matches with the actual angle, increment match count
        if max_angle == orient[n]:
            match += 1
        else :
            mismatch += 1
    # print accuracy
    print("AdaBoost accuracy = " + str(100*(match/(match+mismatch))))

# Hypothesis decision for given features - 1/0
def hypothesis(Features):
    return 1 if Features[0] >= Features[1] else 0

# function to generate data subsets of size 80% of train data
def generate_data_subset(orientation, W) :

    global row_N
    global N
    global sample
    global weights
    global Feat
    global Ang
    global features
    global angle

    # randomly fetch 80% of train data
    row_N = np.random.choice([i for i in range(N)], int(((.80)*N)), replace=False, p=W.ravel())

    # initialize respective sizes with zeros
    Feat = np.zeros((len(row_N), len(sample)), dtype=np.int_)
    Ang = np.zeros((len(row_N), 1), dtype=np.int_)
    weights = np.zeros((len(row_N), 1), dtype=np.float_)

    # for each training data sample obtained above
    for ln, row in enumerate(row_N) :
        Ang[ln] = 1 if int(angle[row]) == int(orientation) else 0 # save angle to 1 if it matches with orientation , else save to 0
        Feat[ln] = [features[row][col] for col in sample] # populate features
        weights[ln] = W[row]    # populate weights

# Train function for Adaboost model
def train_ADB(train_file, model_file) :

    global N
    global row_N
    global features
    global angle
    global sample
    global weights

    N = sum(1 for r in open(train_file))    # calculate no.of lines in train_file
    W = np.ones((N, 1), dtype=np.float_) * (1.0 / N)    # initialize weights to 1/N
    modeled_data = open(model_file, "w+")   # write the model attributes to model text file, which is later used for accuracy calculation in testing
    train_d = open(train_file, "r")         # get train file
    features = np.zeros((N, 192), dtype=np.int_)    # create a features array of size 192xN and initilize with zeros
    angle = np.zeros((N, 1), dtype=np.int_) # create a angle(orientations) array of size 1xN and initilize with zeros

    # fill the features and angle array created above
    for n, r in enumerate(train_d) :
        img = r.split(' ', 2)
        angle[n] = int(img[1])
        features[n] = np.array([int(x) for x in img[2].split(' ')])
    train_d.close()

    # run the train data for 'K' no.of hypotheses or decision stubs
    K = 200
    for orientation in [0, 90, 180, 270] : # iterate for all labels
        for k in range(K): # iterate for all K
            sample = random.sample(range(192), 2)   # generate 2 random feature indexes to compare for hypothesis
            generate_data_subset(orientation, W)    # generate random 80% samples from train file
            Hypotheses = [0 if int(Ang[a]) != int(hypothesis(Feat[a])) else 1 for a in range(len(Ang))]
            A = (math.log(float(1+Hypotheses.count(1))/float(1+Hypotheses.count(0)))*0.5)    # adding 1 to prevent log 0 error and dividing A by 2 ( 1 vs All classifier )
            modeled_data.write("%s %s %s\n" %(str(orientation),' '.join(str(i) for i in sample),str(A)))
            weights = [math.exp(-A) if Hypotheses[i] == 1 else math.exp(A) for i in range(len(weights))] # assign weights as per return value of each hypothesis
            weights /= np.sum(weights)  # normalize weights
            for n, r in enumerate(row_N): # populate weights
                W[r] = weights[n]
            W /= np.sum(W)
    modeled_data.close()

"""

mode = sys.argv[1] # test or train
#train_file = sys.argv[2]
test_file = sys.argv[2]
model_file = sys.argv[3]
#model = "adaboost"

# Adaboost model
if model == "adaboost":
    if mode == "train":
        train_ADB(train_file, model_file)
    elif mode == "test":
        test_ADB(test_file, model_file)
"""
