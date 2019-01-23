from random import seed, randrange
from math import sqrt, log
import json

def str_col_to_float(dset, col):
    for row in dset:
        row[col] = float(row[col].strip())

def str_col_to_int(dset, col):
    class_values = [row[col] for row in dset]
    uniq = set(class_values)
    lookup = dict()
    for i, val in enumerate(uniq):
        lookup[val] = i
    for row in dset:
        row[col] = lookup[row[col]]
    return lookup

def test_split(col, val, dset):
    left, right = list(), list()
    for row in dset:
        if row[col] < val:
            left.append(row)
        else:
            right.append(row)
    return left, right

def accuracy_metric(actual, predicted):
    correct = 0.0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct/float(len(actual))*100.0

def gini_index(groups, class_values):
    size0 = float(len(groups[0]))
    size1 = float(len(groups[1]))
    gini = 0

    for group in groups:
        if len(group) == 0:
            continue
        gini_group = 0
        for class_val in class_values:
            proportion = [row[-1] for row in group].count(class_val)/float(len(group))
            gini_group += (proportion*(1.0 - proportion))
        gini += gini_group
    return gini

def get_split(dset, n_features):
    class_values = list(set([row[-1] for row in dset]))
    b_index, b_value, b_score, b_groups = 999, 999, 999, None
    feat_sub = list()
    while len(feat_sub) < n_features:
        index = randrange(len(dset[0]) - 1)
        if index in feat_sub:
            continue
        feat_sub.append(index)
    for feature in feat_sub:
        for row in dset:
            groups = test_split(feature, row[feature], dset)
            gini = gini_index(groups, class_values)
            if gini < b_score:
                b_index, b_value, b_score, b_groups = feature, row[feature], gini, groups
    return {'index':b_index, 'value':b_value, 'groups':b_groups}

def to_terminal(group):
    outcomes = [row[-1] for row in group]
    return max(set(outcomes), key=outcomes.count)

def split(node, max_depth, min_size, n_features, depth):
    left, right = node['groups']
    del(node['groups'])
    if not left or not right:
        node['left'] = node['right'] = to_terminal(left + right)
        return

    if depth >= max_depth:
        node['left'], node['right'] = to_terminal(left), to_terminal(right)
        return

    if len(left) <= min_size:
        node['left'] = to_terminal(left)

    else:
        node['left'] = get_split(left, n_features)
        split(node['left'], max_depth, min_size, n_features, depth+1)

    if len(right) <= min_size:
        node['right'] = to_terminal(right)

    else:
        node['right'] = get_split(right, n_features)
        split(node['right'], max_depth, min_size, n_features, depth+1)

def build_tree(train, max_depth, min_size, n_features):
    root = get_split(train, n_features)
    split(root, max_depth, min_size, n_features, 1)
    return root

def predict(node, row):
    if row[node['index']] < node['value']:
        if isinstance(node['left'], dict):
            return predict(node['left'], row)
        else:
            return node['left']
    else:
        if isinstance(node['right'], dict):
            return predict(node['right'], row)
        else:
            return node['right']

def subsample(dset, ratio):
    n_samp = round(len(dset)*ratio)
    sample = list()
    while len(sample) < n_samp:
        index = randrange(len(dset))
        sample.append(dset[index])
    return sample

def bagging_pred(trees, row):
    predictions = [predict(tree, row) for tree in trees]
    return max(set(predictions), key=predictions.count)

def random_forest(train_data, model_file, max_depth, min_size, sample_size, n_trees, n_features):
    trees = list()
    for i in range(n_trees):
        sample = subsample(train_data, sample_size)
        tree = build_tree(sample, max_depth, min_size, n_features)
        print(tree)
        trees.append(tree)
    json.dump(trees, open(model_file, 'w'))

def random_forest_test(data_file, model_file):
    test_data = []
    test_images = []
    with open(data_file, "r") as fp2:
        for line in fp2.readlines():
            image = line.split()
            test_data.append(image[1:])
            test_images.append(image[0])
    fp2.close()

    for i in range(0, len(test_data[0]) - 1):
        str_col_to_float(test_data, i)
    str_col_to_int(test_data, len(test_data[0]) - 1)

    trees = json.load(open(model_file, 'r'))
    predictions = [bagging_pred(trees, row) for row in test_data]

    actual = [row[-1] for row in test_data]
    accuracy = accuracy_metric(actual, predictions)
    return accuracy, predictions, test_images

def random_forest_train(data_file, model_file):
    train_data = []
    with open(data_file, "r") as fp1:
        for line in fp1.readlines():
            image = line.split()
            train_data.append(image[1:])
    fp1.close()
    for i in range(0, len(train_data[0]) - 1):
        str_col_to_float(train_data, i)

    str_col_to_int(train_data, len(train_data[0]) - 1)

    seed(1)
    max_depth = 10
    min_size = 100
    sample_size = 1.0
    n_features = int(sqrt(len(train_data[0]) - 1))
    n_trees = [100]
    for n_tree in n_trees:
        random_forest(train_data, model_file, max_depth, min_size, sample_size, n_tree, n_features)
