#!/usr/bin/env python3
#
# ./ocr.py : Perform optical character recognition, usage:
# ./ocr.py train-image-file.png train-text.txt test-image-file.png
#
# Authors: (insert names here)
# (based on skeleton code by D. Crandall, Oct 2018)
# Maanvitha Gongalla
# Kriti Shree
# Murali Kammili

## Report - Part 2
#
# Simple Model
#      In our simple model implementation, first we tried to calculate posteriors for each given training image character
#      by finding out it's emission probability and multiplying it with the prior probability of each corresponding character by looping
#      Then we are find the character which has obtained maximum posterior and appending that character in the final predicted string.
#      With this approach we observed that the prior term in the expression of calculating posterior is influencing the whole expression even after factoring it down to a large number( say multipluying it with pow(10,-10))
#      Hence we have put the prior term in comments and calculated using just likelihood - P(O_i|L-i)
#
# Viterbi Model
#        In our Viterbi implementation, we are calculation transition probability instead of prior probability discussed as above in Simple model.
#        The transition probability is calculated by dividing no.of occurences of all possible pairs of characters using basic conditional probablility definition ( P(A|B) = P(A n B)/P(B) )
#        This is calculated by calling a function to clcute trnsition probabilities just once at beginning and each time we retrieve values from the saved dictionary values
#
#    The approach followed for calculating likelihood / emission probabilities and initial probabilities is same for both models.
#    To calculate initial probability we am obtained the most occuring character through which a word begins in a given training file
#    In case of emission probabilities we am matching each grid cells of each character with all train letters.
#    We have two terms where one has the count of all matched * characters and other one with matched space characters. I am return the sum of these two terms after giving them relative weights.
#    This relative weight has to be adjusted as per testing data. From the test files given, we have experimented and tried to approximate by finding average of weights.
#    We have assigned relatively lower weight to matchng space characters. This ensures to provide better results for noisy test files.
#    As a final answer we are printing the output obtained form Viterbi model since it is better compared to Simple model from the test cases observed and also because it takes into account of the transition probabilities in posterior calculation step.


import math
from PIL import Image, ImageDraw, ImageFont
import sys
from _operator import indexOf
from copy import deepcopy

CHARACTER_WIDTH=14
CHARACTER_HEIGHT=25
# list of all characters considered as per assignment
cset = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','0','1','2','3','4','5','6','7','8','9','(',')',',','.','-','!','?','"',' ','\'']

def load_letters(fname):
    im = Image.open(fname)
    px = im.load()
    (x_size, y_size) = im.size
    #print(im.size)
    #print(int(x_size / CHARACTER_WIDTH) * CHARACTER_WIDTH)
    result = []
    for x_beg in range(0, int(x_size / CHARACTER_WIDTH) * CHARACTER_WIDTH, CHARACTER_WIDTH):
        result += [ [ "".join([ '*' if px[x, y] < 1 else ' ' for x in range(x_beg, x_beg+CHARACTER_WIDTH) ]) for y in range(0, CHARACTER_HEIGHT) ], ]
    return result

def load_training_letters(fname):
    TRAIN_LETTERS="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789()-!?'\"., "
    letter_images = load_letters(fname)
    return { TRAIN_LETTERS[i]: letter_images[i] for i in range(0, len(TRAIN_LETTERS) ) }

# taken from label.py - part 1
def read_data(fname):
    exemplars = []
    file = open(fname, 'r');
    for line in file:
        data = tuple([w for w in line.split()])
        exemplars += [ (data[0::2]), ]      # considering alternate words to skip the tags
    return exemplars


## Prior probability for Simple model = P(l_i)
# called just once
def prior_P(data):
        # These are the prior probabilities.
        # This is calculated for all the character set from the training data

    prior_probs, tag_trans = {}, {}
    letters = []
    #constructs a single list of all file data
    for sentence in data :
        for i in sentence :
            for j in i :
                if j in cset :
                    """if j.isupper() :
                        letters.append(j.lower())
                    else :
                        letters.append(j)   """
                    letters.append(j)
            letters.append(' ')

    for c1 in letters:
        if c1 in tag_trans.keys():
            tag_trans[c1] += 1
        else:
            tag_trans[c1] = 1

    # if there is any character that doesn't occur, then assign zero
    for c in cset:
        if c not in tag_trans.keys():
            tag_trans[c] = 0

    for tag in tag_trans.keys():
        prior_probs[tag] = (tag_trans[tag])/(len(letters)+1)

    return prior_probs, len(letters)+1


## Transition Probability = P(l_i|l_i-1)
# called just once
def transition_P(data, prev_letter):
        # These are the transition probabilities.
        # The transition probabilities are calculated for the transitions between state.
        # The probability of a state S_i going into another state S_i+1 gives the transition probability P(S_i+1|S_i)
        # This is calculated for every possible transitions from the training data

    trans_probs, tag_trans = {}, {}

    for c1 in cset:
        for c2 in cset:
            if (c1,c2) in tag_trans.keys():
                tag_trans[(c1,c2)] += 1
            else:
                tag_trans[(c1,c2)] = 1

    count = 0
    # check if we can find the required tag pair and maintain count of how many times it occurred
    for (tag,next_tag) in tag_trans.keys():
        if tag == prev_letter:
            count +=1

    for (tag, next_tag) in tag_trans.keys():  # try to pass calculated priors
        trans_probs[(tag, next_tag)] = (tag_trans[(tag, next_tag)])/(count+1)

    return trans_probs


## Emission Probability = P(O_i|l_i)
# P(O_i n l_i) (conditional probability)
# heuristic used for calculating emission probability - no.of matching cells for both * and space characters with relative weights in varying proportions
def matching_heuristic(letter,train_letters):
    vset = {}
    for ch in cset :
        match = 0
        smatch = 0
        p = train_letters[ch]
        for h in range(CHARACTER_HEIGHT):
            for w in range(CHARACTER_WIDTH):
                if ( p[h][w]=='*' and letter[h][w]=='*' ) :
                    match += 1
                elif ( p[h][w]==' ' and letter[h][w]==' ' ) :
                    smatch += 1
        vset[ch] = .95*match + smatch*.25 # relatively lower weight to space matching when compared to stars

    return vset

## Initial Probability - P(l_1) - using first letters
# called just once
def initial_P(exemplars):
    start_tags, initial_tag_probs,  = {}, {}
    sentence = 0
    words = 0
    for line in exemplars:
        sentence += 1
        for word in line:
            words += 1
            if word[0] in start_tags.keys():
                start_tags[word[0]] += 1
            elif word[0] in cset:
                start_tags[word[0]] = 1

    for tag in start_tags.keys():
        initial_tag_probs[tag] = start_tags[tag]/(words+1)

    return initial_tag_probs

#####
# main program
#(train_img_fname, train_txt_fname, test_img_fname) = sys.argv[1:]
# usage:
#     ./ocr.py train-image-file.png train-text.txt test-image-file.png

(train_img_fname, train_txt_fname, test_img_fname) = sys.argv[1:]

train_letters = load_training_letters(train_img_fname)
test_letters = load_letters(test_img_fname)
train_data = read_data(train_txt_fname)  # list of tuples, where each tuple contains words that form a sentence

initial_letter_P = initial_P(train_data)
most_starter_letter = max(initial_letter_P, key=initial_letter_P.get)

# Simple Model
predicted_sentence = []
transition_letter_Ps, total_letters = prior_P(train_data)
for i in range(len(test_letters)) :
    posterior = {}
    e_p = matching_heuristic(test_letters[i],train_letters)
    for c in cset:
        if c in transition_letter_Ps.keys() :
            # here i am considering just likelihood.
            # when i am multiplying with the prior probability , the prior value is dominating the whole expression even after factoring it down to very less value
            posterior[c] = e_p[c]#/(CHARACTER_HEIGHT*CHARACTER_WIDTH) * transition_letter_Ps[c]
        else :
            posterior[c] = 0
    # considering maximum of all the calculated posteriors to determine which letter occurs next
    predicted_sentence.append(max(posterior,key=posterior.get))
print("Simple:    ", end='')
print(''.join(predicted_sentence))

# Viterbi
# P(l_i|O_i) = P(O_i|l_i) * P(l_i|l_i-1)
predicted_sentence = []
transition_letter_Ps = transition_P(train_data,most_starter_letter)
for i in range(len(test_letters)) :
    posterior = {}
    # calculate emission probability
    e_p = matching_heuristic(test_letters[i],train_letters)
    for c in cset :
        if i==0 :
            t_p = 1 # transition probability term for first letter / initial probability
        else :
            t_p = transition_letter_Ps[predicted_sentence[-1],c]    # each time consider the previous letter and current letter pairs
        # posterior calculation - emission probability * transition probability
        posterior[c] = (e_p[c]/(CHARACTER_HEIGHT*CHARACTER_WIDTH)) * t_p
    predicted_sentence.append(max(posterior,key=posterior.get)) # append predicted letter at each step
print("Viterbi:   ", end='')
print(''.join(predicted_sentence))

print("Final answer:    ")
print(''.join(predicted_sentence))
