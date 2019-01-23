This program is to classify tweets based on the location.
There are 12 cities as the labels.

Initially, I implemented it using the following technique:
 1. I look for all the unique words in the training set, clean them and put them in a list.
 2. Then I calculate the prior
 3. Then the likelihood of every word given the city.
 4. Then I calculate the posterior = prior * likelihood.

 But this approach takes a lot of time for execution.

 So, now I'm doing it dynamically using multi-dimensional dictionaries.
 1. I read a line from the training file.
 2. Split the line into words. The first word being the city and the rest being the tweet.
 3. Do the smoothing:
    a. change all the words into lower case
    b. remove all kinds of punctuation from these words.
    c. filter out the stop words.
    these are done using basic string functions.
    I used the nltk tools earlier but later found out that they aren't in silo.
    So I used the stopwords list from nltk and manually filtered the stop words.
    The stop words are referenced from the nltk stopwords list.
 4.calculate the prior and likelihood simultaneously.

 Now, for the testing...
 5. Read the testing data line by line.
 6. Split into words.
 7. Clean the data.
 8. calculate the posterior.
 This is also done dynamically for every line read in the file.

 I also used laplace smoothing to prevent the posterior from becoming zero because of a new word in testing file.
 The laplace smoothing only adds a one in the numerator to the words that aren't in the training data while calculating frequency of the words
 and it adds the length of the city's bag of words in the denominator.

 Initially, I started with manually removing the special characters but then I found the string functions for removing the punctution.
 I've also tried lemmatizing but that resulted in overfitting the data.

 The output will be written in the file mentioned by the user.
 This file will also have the top 5 common word list for each location at the end of the file.
