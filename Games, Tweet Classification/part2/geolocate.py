#!/bin/python3

# This program classifies tweets based on the location using naive bayes classifier.

import sys
import math
import string

# This function is used to find the accuracy of the classifier.
def accuracy(tweets_test):
    count = 0
    for tweet in tweets_test:
        if tweets_test[tweet]['actual_label'] == tweets_test[tweet]['predicted_label']:
            count += 1

    accuracy = (count / float(len(tweets_test))) * 100

# This function is used for the training of the classifier.
# We get the prior and the likelihood from this function.
def trainingNaiveBayes(train_file, vocabulary, cities, train_number_of_tweets, location, stopwords):

    with open(train_file, 'r', encoding="ISO-8859-1") as f:

        for line in f:
            train_number_of_tweets += 1
            blah = line.split()
            city = line.partition(' ')[0]
            tweet = blah[1:]
            if city in location:
                if city in cities:
                    cities[city]['tweets'] += 1
                else:
                    cities[city] = {}
                    cities[city]['words'] = 0
                    cities[city]['tweets'] = 1
                tokens = [w.lower() for w in tweet]  # convert to lower case
                table = str.maketrans('', '', string.punctuation)  # remove punctuation from each word
                stripped = [w.translate(table) for w in tokens]
                bag_of_words = [w for w in stripped if not w in stopwords] # filter out the stop words
                cities[city]['words'] += len(bag_of_words)

                for word in bag_of_words:
                    if word not in vocabulary:
                        vocabulary[word] = {}
                        vocabulary[word]['cities'] = {}
                        vocabulary[word]['cities'][city] = 1
                        vocabulary[word]['count'] = 1
                    else:
                        vocabulary[word]['count'] += 1
                        if city in vocabulary[word]['cities']:
                            vocabulary[word]['cities'][city] += 1
                        else:
                            vocabulary[word]['cities'][city] = 1

    for word in vocabulary:
        for city in cities:
            if city not in vocabulary[word]['cities']:
                vocabulary[word]['cities'][city] = 0

    train_bag_of_words = vocabulary
    for word in list(train_bag_of_words.keys()):
        if train_bag_of_words[word]['count'] == 1:
            del(train_bag_of_words[word])
    return train_bag_of_words, cities, train_number_of_tweets

# This function does the testing of the classifier.
# We calculate the posterior of each label given the word and choose the label with the max. posterior.
def testingNaiveBayes(test_file, cities, train_bag_of_words, test_tweet_number, train_number_of_tweets, location, stopwords):

    with open(test_file, 'r', encoding="ISO-8859-1") as f:
        tweets_test_data = {}
        for line in f:
            test_tweet_number += 1
            blah = line.split()
            tweet = blah[1:]
            city = line.partition(' ')[0]
            tokens = [w.lower() for w in tweet]  # convert to lower case
            table = str.maketrans('', '', string.punctuation)  # remove punctuation from each word
            stripped = [w.translate(table) for w in tokens]
            test_bag_of_words = [w for w in stripped if not w in stopwords] # filter out the stopwords
            for i in list(range(0, len(test_bag_of_words))):
                if len(test_bag_of_words[i]) > 5:
                    test_bag_of_words[i] = test_bag_of_words[:5]

            tweets_test_data[test_tweet_number] = {}
            tweets_test_data[test_tweet_number]['actual_label'] = city
            tweets_test_data[test_tweet_number]['predicted_label'] = ''
            tweets_test_data[test_tweet_number]['tweet'] = tweet
            tweets_test_data[test_tweet_number]['posterior'] = {}

            max_posterior = -float("inf")
            max_posterior_city = ''

            for city in cities:
                words_in_city = cities[city]['words']
                tweets_test_data[test_tweet_number]['posterior'][city] = math.log(cities[city]['tweets'])/float(train_number_of_tweets)
                for word in test_bag_of_words:
                    try:
                        tweets_test_data[test_tweet_number]['posterior'][city] += math.log((train_bag_of_words[word]['cities'][city] + 1)/float(words_in_city + len(train_bag_of_words)))
                    except (KeyError, TypeError):
                        continue
                if tweets_test_data[test_tweet_number]['posterior'][city] > max_posterior:
                    max_posterior = tweets_test_data[test_tweet_number]['posterior'][city]
                    max_posterior_city = city
                    tweets_test_data[test_tweet_number]['predicted_label'] = max_posterior_city
    return tweets_test_data

# This function is used to give us the top 5 common words for each location.
def top_words(cities, train_vocab):
    cities_common_words = {}
    for city in cities:
        words = []
        count = []
        common_words = []
        i = 1

        for word in train_vocab:
            words.append(word)
            count.append(train_vocab[word]['cities'][city])

        while i < 7:
            k = count.index(max(count))
            common_words.append(words.pop(k))
            del (count[k])
            i += 1

        cities_common_words[city] = common_words
    return cities_common_words

# This function is used to write the output in a file
def output(out_file, cities_common, data):

    f = open(out_file, 'w', encoding="ISO-8859-1")
    s = " "

    for tweet in data:
        s += str(data[tweet]['predicted_label']) + " " + str(data[tweet]['actual_label']) + " " + str(
            data[tweet]['tweet'])
        s += '\n'

    s += '\n'
    s += "The top 5 words for each city are:"
    s += "\n"

    for city in cities_common.keys():
        l = str(city) + " "
        for word in cities_common[city]:
            l += str(word) + " "
        l += '\n'
        s += l

    f.write(s)
    f.close()

# This is the main function.
def main():

    train_number_of_tweets = 0
    test_tweet_number = 0
    cities = {}
    vocabulary = {}

    train_file = sys.argv[1]
    test_file = sys.argv[2]
    output_file = sys.argv[3]

    location = ['Orlando,_FL', 'Boston,_MA', 'Manhattan,_NY', 'Chicago,_IL', 'Los_Angeles,_CA', 'San_Diego,_CA', 'Houston,_TX', 'Philadelphia,_PA', 'Toronto,_Ontario', 'San_Francisco,_CA', 'Atlanta,_GA', 'Washington,_DC']

    stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself',
                 'yourselves', 'he', 'him',
                 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'u', 'itself', 'they', 'them', 'their',
                 'theirs', 'themselves',
                 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were',
                 'be', 'been', 'being', 'have',
                 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or',
                 'because', 'as', 'until',
                 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during',
                 'before', 'after',
                 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
                 'further', 'then', 'once',
                 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'more', 'few', 'most',
                 'other', 'some', 'such',
                 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 'can', 'will', 'just', 'don',
                 's', 't', 'shoudl', 'now']

    # Reading the training data ...

    train_bag_of_words, cities, train_number_of_tweets = trainingNaiveBayes(train_file, vocabulary, cities, train_number_of_tweets, location, stopwords)

    # Reading the test data ...

    tweets_test_data = testingNaiveBayes(test_file, cities, train_bag_of_words, test_tweet_number, train_number_of_tweets, location, stopwords)

    # Finding the accuracy ...

    accuracy(tweets_test_data)

    # Finding the top 5 words ...

    cities_common_words = top_words(cities, train_bag_of_words)

    # Writting the output into a file ...

    output(output_file, cities_common_words, tweets_test_data)

main()
