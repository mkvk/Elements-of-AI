###################################
# CS B551 Fall 2018, Assignment #3
#
# Your names and user ids:
# Maanvitha Gongalla --- mgongall
# Murali Kishore --- mkammili
# Kriti Shree --- kshree
#
# (Based on skeleton code by D. Crandall)
#
#
####
# Put your report here!!

# The parts of speech tagging is implemented using three models:
# 1. Simple -- Naive Bayes classifier
# 2. HMM -- Hidden Markov Models using the Viterbi decoding algorithm
# 3. Complex -- Markov Chain Monte Carlo with Gibbs Sampling

# TRAINING :-
# Here, we used the train data file provided to derive all the probabilities required for us to predicts the POS tags.
# 1. The likelihood i.e. the probability of every word given every tag -- P(W_i|S_i)
# 2. The priors of each tag i.e. the probability of a tag occurring in the entire train data -- P(S_i)
# 3. The first order transition probabilities i.e. the probability of a tag given a previous tag -- P(S_i|S_i-1)
# 4. The second order transition probabilities i.e. the probability of tag given previous two tags -- P(S_i|S_i-1,S_i-2)
# 5. The initial state probabilities i.e. the probability that a tag is the first in a sentence -- P(S = S_i)
# All these probabilities are calculated using the training data file and then these are used to find the predict the POS tags.

# The models are implemented as follows:

# 1. Simple Model:-
    # In this model, we used simple Bayes rules to calculate the posteriors.
    # The model used here is the simple 1(b) where there are no transition probabilities.
    # So just multiplying the likelihood and the prior gives us the posterior.
    # We choose the maximum posterior and add that tag to the word.
    # Finally, we return all the predicted tags.

# 2. HMM Model:-
    # In this model, we used the 1(a) model where there are only first order transitions between the previous tag and the current tag.
    # We calculate the most probable sequence of tags for a sentence.
    # The optimal way is to use viterbi decoding algorithm with dynamic progamming.
    # This is calculated as below.
    # The initial tag probability + transition probability + emission probability make up the log posterior.
    # For every word in the sentence, we calculate for every tag.. the maximum transition probabilities for the previous tags.
    # The maximum value of all would be taken as the appropriate sequence for the sentence given.
    # To find the tags, we backtrack the path for every decision made as done in dynamic programming.
    # This final path backtracked gives us the final predicted tags.
    # Added a small probability of 0.00000000000000000001 for words unseen in test data while calculating emission.
    # Added a small probability of 0.00000000000000000001 for transitions unseen in test data while calculating transition probabilities.

# 3. Complex Model:-
    # In this model, we use the 1(c) where there are both first order and second order transitions.
    # First order transitions are between previous tag and current tag.
    # Second order transitions are between second previous tag and current tag.
    # Here we directly take in the test data sentences and perform gibbs sampling on it.
    # Initially, we assign "noun" to all the tags in the sentence. This will be our initial sample.
    # Now, we perform gibbs sampling on this initial sample and keep on updating the samples as well as the probabilities.
    # Initially, gibbs sampling gives us a poor performance so we do sampling for a while and then throw those samples away.
    # This is known as the burn-in period. In this case, we're throwing out 1000 samples.
    # Then we sample 5 results and print them.
    # Since we have to return only one list, we are returning the top result.
    # Initially, I took a 1000 iterations but the program takes around 4-5 minutes to run.
    # So I got it down to 500 check if it's converging early. It gives almost the same accuracy but then the run time got down to 2 minutes.
    # I got it down to 200 and the runtime decreased to 1 1/2 minute. The accuracy increased just a little bit - 0.5%.
    # Got it even down to 100 the runtime seems to stay around 1 minute but the accuracy decreased a little bit - 0.3%.
    # So, finally decided to leave it with 200 iterations.
    # Added a small probability of 0.00000001 for words unseen in test data while calculating emission.
    # Added a small probability of 0.00000001 for transitions unseen in test data while calculating transition probabilities.

# RESULTS :-
# 1. Simple:  words -- 93.92%  sentences -- 47.50%
# 2. HMM:  words -- 90.74%  sentences -- 35.65%
# 3. Complex:  words -- 92.70%  sentences -- 39.20%

####

import random
import math

# We've set up a suggested code structure, but feel free to change it. Just
# make sure your code still works with the label.py and pos_scorer.py code
# that we've supplied.
#

class Solver:
    # Calculate the log of the posterior probability of a given sentence
    #  with a given part-of-speech labeling. Right now just returns -999 -- fix this!

    train_data = 0
    # The training needs to be done only once at the start of the program to reduce the runtime of the program.
    # Hence, train_data is taken as a global variable for the class.
    # Made all the methods involved in training @staticmethod again, to reduce the runtime of the program.

    # This is the function which prints out the posterior for each model.
    def posterior(self, model, sentence, label):
        exemplars, prior_of_tags, tag_counts, word_counts, total_words, likelihood, trans_probs, initial_tag_probs, second_trans_probs = Solver.train_data

        # For the Simple model, we just calculate the posterior using likelihood and priro
        # Posterior = Likelihood * Prior
        if model == "Simple":
            posterior = 0.0
            for i in range(len(sentence)):
                if (sentence[i], label[i]) in likelihood.keys() and label[i] in initial_tag_probs.keys():
                    posterior += (math.log(likelihood[(sentence[i], label[i])]) + math.log(prior_of_tags[label[i]]))
            return posterior
            #return -999

        # For the Complex model, we need to use initial state probabilities, emission probabilities, first and second order transition probabilities.
        # There will be four cases for this purpose;
        # 1. For the first word -- we just calculate initial state probabilities * emission probabilities.
        # 2. For the second word -- we calculate initial state probabilities * emission probabilities * first order transition probabilities.
        # 3. For the last word -- we
        # 4. For the remaining word -- we calculate initial state probabilities * emission * first order transition * second order transition probabilities.
        # Added a small probability of 0.00000001 for words unseen in test data while calculating emission.
        # Added a small probability of 0.00000001 for transitions unseen in test data while calculating transition probabilities.
        elif model == "Complex":
            posterior = 0.0
            for i in range(len(sentence)):
                if i == 0:
                    emission = likelihood[(sentence[i], label[i])] if (sentence[i], label[i]) in likelihood.keys() else 0.00000001
                    if label[i] in initial_tag_probs.keys():
                        prob = math.log(initial_tag_probs[label[i]] * emission)
                        posterior += prob

                elif i == 1:
                    emission = likelihood[(sentence[i], label[i])] if (sentence[i], label[i]) in likelihood.keys() else 0.00000001
                    first_trans = trans_probs[(label[i - 1], label[i])] if (label[i - 1], label[i]) in trans_probs.keys() else 0.00000001
                    if label[i] in initial_tag_probs.keys():
                        prob = math.log(initial_tag_probs[label[i - 1]] * emission * first_trans)
                        posterior += prob

                else:
                    emission = likelihood[(sentence[i], label[i])] if (sentence[i], label[i]) in likelihood.keys() else 0.00000001
                    first_trans = trans_probs[(label[i - 1], label[i])] if (label[i - 1], label[i]) in trans_probs.keys() else 0.00000001
                    second_trans = second_trans_probs[(label[i - 2], label[i - 1], label[i])] if (label[i - 2], label[i - 1], label[i]) in trans_probs.keys() else 0.00000001
                    if label[i] in initial_tag_probs.keys():
                        prob = math.log(initial_tag_probs[label[i - 2]] * emission * first_trans * second_trans)
                        posterior += prob
            return posterior
            #return -999

        # For the HMM model, we just calculate the posterior using initial state probabilities, emission probabilities and first order transition probabilities.
        # We have two cases here:
        # 1. For the first word, we just calculate initial state probabilities * emission probabilities.
        # 2. For the remaining words, we calculate initial state probabilities * emission probabilities * first transition probabilities
        elif model == "HMM":
            posterior = 0.0
            for i in range(len(sentence)):
                if i == 0:
                    if (sentence[i], label[i]) in likelihood.keys() and label[i] in initial_tag_probs.keys():
                        prob = math.log(initial_tag_probs[label[i]] * likelihood[(sentence[i], label[i])])
                        posterior += prob
                else:
                    if (sentence[i], label[i]) in likelihood.keys():
                        emission = likelihood[(sentence[i], label[i])]
                        if (label[i - 1], label[i]) in trans_probs.keys() and label[i] in initial_tag_probs.keys():
                            trans = trans_probs[(label[i - 1], label[i])]
                            prob = math.log(initial_tag_probs[label[i - 1]] * emission * trans)
                            posterior += prob
            return posterior
            #return -999
        else:
            print("Unknown algo!")

    # Do the training!
    #
    ######### ---------- new method for prior added  --------- #######

    @staticmethod
    def prior(exemplars):
    #def priors(self, line, tag_counts, word_counts, priors_of_tags, total_number_of_words):
        # Need counts for each tag - P(S_i)
        # Need counts for each word - P(W_i)
        # Need total number of words - total_number_of_words
        # Prior(S_i) = P(S_i)/(total_number_of_words)

        tag_counts = {}
        word_counts = {}
        total_number_of_words = 0
        priors_of_tags = {}

        for line in exemplars:
            for word in line[0]:
                if word in word_counts.keys():
                    word_counts[word] += 1
                else:
                    word_counts[word] = 1
                total_number_of_words += 1
            for tag in line[1]:
                if tag in tag_counts.keys():
                    tag_counts[tag] += 1
                else:
                    tag_counts[tag] = 1
            for tag in tag_counts.keys():
                priors_of_tags[tag] = tag_counts[tag]/total_number_of_words

        return priors_of_tags, tag_counts, word_counts, total_number_of_words

    ######### ---------- new method for likelihood added  --------- #######

    @staticmethod
    def likelihood(exemplars, tag_counts, total_number_of_words):
    #def likelihood(self, line, likelihood):
        # Need counts for every word given a tag -- P(W|S_i)
        # Need the counts for every tag -- P(S_i)
        # Need total number of words -- total_number_of_words
        # likelihood = P(W|S_i)/(P(S_i) * total_number_of_words)

        likelihood = {}
        for line in exemplars:
            for word in range(len(line[0])):
                if (line[0][word], line[1][word]) in likelihood.keys():
                    likelihood[(line[0][word], line[1][word])] += 1
                else:
                    likelihood[(line[0][word], line[1][word])] = 1
        for word_tag in likelihood.keys():
                likelihood[word_tag] = float(likelihood[word_tag])/tag_counts[word_tag[1]]
        return likelihood

    ######### ---------- new method for posterior added  --------- #######

    @staticmethod
    def simple_posterior(word, tag_counts, likelihood):
        # Need to calculate the probability for every word given a tag -- P(S_i|W_i)
        # Need counts for every tag -- P(S_i)
        # Need likelihood -- P(W_i|S_i)
        # Posterior = Likelihood * prior
        # P(S_i|W_i) = P(W_i|S_i) * P(S_i)
        # Since the product of these probabilities could lead to very small numbers
        # We'll be taking the sum of logarithm of posteriors

        posterior = []
        for tag in tag_counts:
            if (word, tag) in likelihood:
                posterior.append((math.log(likelihood[(word, tag)] * tag_counts[tag]), tag))

            # Tried various combinations for this part but it only decreased the accuracy. So will just let it go.
            #else:
            #    posterior.append(((-float("inf")), tag))
        if not posterior:
            posterior.append((-1, 'noun'))
        return posterior

    ######### ---------- method taken from the label.py program  --------- #######

    @staticmethod
    def read_data(fname):
        exemplars = []
        file = open(fname, 'r');
        for line in file:
            data = tuple([w.lower() for w in line.split()])
            exemplars += [ (data[0::2], data[1::2]), ]
        return exemplars

    ######### ---------- new method for initial state probabilities added  --------- #######

    @staticmethod
    def initial_state_probs(exemplars):
        # This is number of times a tag begins a sentence divided by the total number of sentences
        # This is calculated for each tag
        # This is same as P(Q_0 = q_0)
        # The intial state probability -- initial_tag_probs
        # Need the number of times a tag starts a sentence -- start_tags
        # Need number of sentences -- sentence
        # P(Q_0 = q_0) = start_tags[q_0]/sentence

        start_tags, initial_tag_probs, last_tags = {}, {}, {}
        sentence = 0
        for line in exemplars:
            sentence += 1
            if line[1][0] in start_tags.keys():
                start_tags[line[1][0]] += 1
            else:
                start_tags[line[1][0]] = 1
        for tag in start_tags.keys():
            initial_tag_probs[tag] = float(start_tags[tag])/sentence
        return initial_tag_probs

    ######### ---------- new method for second order transition probabilities added ----- ######

    @staticmethod
    def second_transition_probabilities(data):
        # These are second order transition probabilities.
        # The second order transitions are between the second previous state and the current state.
        # The probability of a state S_i given state S_i-1 and S_i-2 -- P(S_i|S_i-1,S_i-2)
        # This if calculated for every possible tri pair (tag, previous_tag, second_previous_tag)

        second_trans_probs, tag_trans, tag_all_trans = {}, {}, {}
        for line in data:
            for i in range(len(line[1]) - 2):
                if(line[1][i], line[1][i+1], line[1][i+2]) in tag_trans.keys():
                    tag_trans[(line[1][i], line[1][i+1], line[1][i+2])] += 1
                else:
                    tag_trans[(line[1][i], line[1][i+1], line[1][i+2])] = 1
                if (line[1][i], line[1][i+1]) in tag_all_trans.keys():
                    tag_all_trans[(line[1][i], line[1][i+1])] += 1
                else:
                    tag_all_trans[(line[1][i], line[1][i+1])] = 1
        for (tag, next_tag, next_next_tag) in tag_trans.keys():
            second_trans_probs[(tag, next_tag, next_next_tag)] = float(tag_trans[(tag, next_tag, next_next_tag)])/tag_all_trans[(tag, next_tag)]
        return second_trans_probs

    ######### ---------- new method for transition probabilities added  --------- #######

    @staticmethod
    def transition_probabilities(data):
        # These are the first order transition probabilities.
        # The transition probabilities are calculated for the transitions between state.
        # The probability of a state S_i given state S_i-1 gives the transition probability P(S_i|S_i-1)
        # This is calculated for every possible transitions from the training data

        trans_probs, tag_trans, tag_all_trans = {}, {}, {}

        for line in data:
            for i in range(len(line[1]) - 1):
                if (line[1][i], line[1][i + 1]) in tag_trans.keys():
                    tag_trans[(line[1][i], line[1][i + 1])] += 1
                else:
                    tag_trans[(line[1][i], line[1][i + 1])] = 1
                if line[1][i] in tag_all_trans.keys():
                    tag_all_trans[line[1][i]] += 1
                else:
                    tag_all_trans[line[1][i]] = 1

        for (tag, next_tag) in tag_trans.keys():
            trans_probs[(tag, next_tag)] = float(tag_trans[(tag, next_tag)])/tag_all_trans[tag]

        return trans_probs

    ######### ---------- new method for gibbs sampling added  --------- #######

    def gibbs_sampling(self, sentence, sample):
        # This function is used to do gibbs sampling
        # Initially, the sample is taken as a list of tags all assigned to "noun"

        exemplars, prior_of_tags, tag_counts, word_counts, total_words, likelihood, trans_probs, initial_tag_probs, second_trans_probs = self.train('bc.train')
        all_tags = list(tag_counts.keys())
        for index in range(len(sentence)):
            word = sentence[index]
            probabilities = [0] * len(tag_counts)

            previous_tag = sample[index - 1] if index > 0 else ""
            pre_pre_tag = sample[index - 2] if index > 1 else ""

            for j in range(len(tag_counts)):
                current_tag = all_tags[j]

                # The beginning of the sentence is the first word which means there are no previous transitions
                # Only need emission, initial and next tranition probabilities.
                if index == 0:
                    emission = likelihood[(word, current_tag)] if (word, current_tag) in likelihood.keys() else 0.00000001
                    probabilities[j] = initial_tag_probs[current_tag] * emission

                # The second word in the sentence, we don't have any second order transitions.
                # So we just use initial state probabilities, first order transition and emission probabilities.
                elif index == 1:
                    emission = likelihood[(word, current_tag)] if (word, current_tag) in likelihood.keys() else 0.00000001
                    first_trans = trans_probs[(previous_tag, current_tag)] if (previous_tag, current_tag) in trans_probs.keys() else 0.00000001
                    probabilities[j] = initial_tag_probs[previous_tag] * emission * first_trans

                # For all the middle words in the sentence
                # Need emission, initial, previous and next transition probabilities
                else:
                    emission = likelihood[(word, current_tag)] if (word, current_tag) in likelihood.keys() else 0.00000001
                    first_trans = trans_probs[(previous_tag, current_tag)] if (previous_tag, current_tag) in trans_probs.keys() else 0.00000001
                    second_trans = second_trans_probs[(pre_pre_tag, previous_tag, current_tag)] if (pre_pre_tag, previous_tag, current_tag) in trans_probs.keys() else 0.00000001
                    probabilities[j] = initial_tag_probs[pre_pre_tag] * emission * first_trans * second_trans

            # Now, we normalize all the probabilities so that all of them equals 1
            prob_sum = sum(probabilities)
            probabilities = [p / prob_sum for p in probabilities]

            # Picking a random value between 0 and 1 and we update the probabilities accordingly
            random_value = random.random()

            # Based on the marginal distribution, randomly pick a part of speech for the word
            # We find the cumulative sum of the normalized probabilities
            # Using the random value picked between 0 and 1, get the index using the cumulative sum at each index
            cum_sum = 0
            for i in range(len(probabilities)):
                prob = probabilities[i]
                cum_sum += prob
                if random_value < cum_sum:
                    sample[index] = all_tags[i]
                    break

        return sample

    # This function calls all the functions required for training and returns all the necessary data.
    def train(self, data):
        #pass

        ##### -------- modified ------- #####
        if Solver.train_data != 0:
            return Solver.train_data
        exemplars = Solver.read_data(data)
        priors_of_tags, tag_counts, word_counts, total_words = Solver.prior(exemplars)
        likelihood = Solver.likelihood(exemplars, tag_counts, total_words)
        initial_tag_probs = Solver.initial_state_probs(exemplars)
        trans_probs = Solver.transition_probabilities(exemplars)
        second_trans_probs = Solver.second_transition_probabilities(exemplars)
        Solver.train_data = (exemplars, priors_of_tags, tag_counts, word_counts, total_words, likelihood, trans_probs, initial_tag_probs, second_trans_probs)
        return Solver.train_data

    # Functions for each algorithm. Right now this just returns nouns -- fix this!
    #
    def simplified(self, sentence):
        # This is the function where we calculate the posterior probabilities using the simplified HMM.
        # The transition probabilities are ignored since there are no transitions in the HMM model taken.
        # Hence, the posterior would just be --- Posterior = Likelihood * Prior

        predicted_tags = []
        exemplars, prior_of_tags, tag_counts, word_counts, total_words, likelihood, trans_probs, intital_tag_probs, second_trans_probs = self.train('bc.train')
        for word in sentence:
            word_posterior = Solver.simple_posterior(word, tag_counts, likelihood)
            word_posterior.sort(reverse = True)
            predicted_tags.append(word_posterior[0][1])
        return predicted_tags

    def complex_mcmc(self, sentence):
        # We use gibbs sampling to predict the tags for our test data
        # The first 1000 examples are thrown out known as the burn-in time
        # Then we sample for 5 examples and return the top most result
        # Initially, all the tags in the sample are labelled as "noun"

        sample = ["noun"] * len(sentence)
        for i in range(200):
            sample = self.gibbs_sampling(sentence, sample)
        samples = []
        for p in range(5):
            sample = self.gibbs_sampling(sentence, sample)
            samples.append(sample)
        print("The Top 5 sampled results")
        for i in samples:
            print(str(i))
        return samples[0]

    def hmm_viterbi(self, sentence):
        # This is the function to calculate the most probable sequence of tags for a sentence.
        # This is calculated using the second HMM model where transitions are present.
        # The optimal way is to use viterbi decoding algorithm with decoding.
        # This is calculated as below.
        # The initial tag probability + transition probability + emission probability make up the posterior.
        # For every word in the sentence, we calculate for every tag.. the maximum transition probabilities for the previous tags.
        # Th maximum value of all would be taken as the appropriate sequence for the sentence given.
        # To find the tags, we backtrack the path for every decision made as done in dynamic programming.
        # This final path backtracked gives us the final predicted tags.
        # Added a small probability of 0.00000000000000000001 for words unseen in test data while calculating emission.
        # Added a small probability of 0.00000000000000000001 for transitions unseen in test data while calculating transition probabilities.


        final_tags = []
        exemplars, prior_of_tags, tag_counts, word_counts, total_words, likelihood, trans_probs, initial_tag_probs, second_trans_probs = self.train('bc.train')

        # Populating the predicted_tags list with the initial posteriors, i.e. log(Posterior) = log(likelihood) + log(intial probability for that tag)
        predicted_tags = [[(math.log(likelihood[(sentence[0], pos)]) + math.log(initial_tag_probs[pos]), pos, pos) if (sentence[0], pos) in likelihood else (math.log(0.00000000000000000001) + math.log(initial_tag_probs[pos]), pos, pos) for pos in tag_counts]]

        # If the length of the sentence is one just return the maximum value of the above predicted_tags list
        if len(sentence) == 1:
            tag = max(predicted_tags[0], key=lambda x: x[0])[2]
            return [tag]

        # For every word in the sentence, we calculate for every tag.. the maximum transition probabilities for the previous tags.
        # And choose the maximum of these values. This just gives us our most probable sequence.
        for word in sentence[1:]:
            previous_tags = predicted_tags[-1]
            current_tag_post = []
            for tag in tag_counts:
                emission = math.log(likelihood[(word, tag)]) if (word, tag) in likelihood else math.log(0.00000000000000000001)
                previous_tag_post = []
                for i in range(len(previous_tags)):
                    if (previous_tags[i][1], tag) in trans_probs:
                        trans_prob = math.log(trans_probs[(previous_tags[i][1], tag)])
                    else:
                        trans_prob = math.log(0.00000000000000000001)
                    previous_tag_post.append((emission + previous_tags[i][0] + trans_prob, previous_tags[i][1], tag))
                current_tag_post.append(max(previous_tag_post))
            predicted_tags.append(current_tag_post)

        # Now, we backtrack the tags choosen and return the final tags predicted.
        for tag in predicted_tags:
            max_posterior = max(tag, key=lambda tup:tup[0])
            final_tags.append(max_posterior[2])
        return final_tags
        #return [ "noun" ] * len(sentence)

    # This solve() method is called by label.py, so you should keep the interface the
    #  same, but you can change the code itself.
    # It should return a list of part-of-speech labelings of the sentence, one
    #  part of speech per word.
    #
    def solve(self, model, sentence):
        print("solver")
        print(sentence)
        if model == "Simple":
            return self.simplified(sentence)
        elif model == "Complex":
            return self.complex_mcmc(sentence)
        elif model == "HMM":
            return self.hmm_viterbi(sentence)
        else:
            print("Unknown algo!")
