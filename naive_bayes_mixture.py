# naive_bayes.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 09/28/2018
# Modified by Jaewook Yeom 02/02/2020

"""
This is the main entry point for Part 2 of this MP. You should only modify code
within this file for Part 2 -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""


import numpy as numpy
import math
from collections import Counter
import nltk





def naiveBayesMixture(train_set, train_labels, dev_set, imessages, bigram_lambda,unigram_smoothing_parameter, bigram_smoothing_parameter, pos_prior):
    """
    train_set - List of list of words corresponding with each movie review
    example: suppose I had two reviews 'like this movie' and 'i fall asleep' in my training set
    Then train_set := [['like','this','movie'], ['i','fall','asleep']]

    train_labels - List of labels corresponding with train_set
    example: Suppose I had two reviews, first one was positive and second one was negative.
    Then train_labels := [1, 0]

    dev_set - List of list of words corresponding with each review that we are testing on
              It follows the same format as train_set

    bigram_lambda - float between 0 and 1

    unigram_smoothing_parameter - Laplace smoothing parameter for unigram model (between 0 and 1)

    bigram_smoothing_parameter - Laplace smoothing parameter for bigram model (between 0 and 1)

    pos_prior - positive prior probability (between 0 and 1)
    """



    # TODO: Write your code here
    positive_counter = Counter()
    negative_counter = Counter()
    total_num_pos_reviews = 0
    total_num_neg_reviews = 0
    total_pos_words = 0
    total_neg_words = 0
    pos_bigram_counter = Counter()
    neg_bigram_counter = Counter()
    total_pos_bigrams = 0
    total_neg_bigrams = 0

    cross_idx = 0
    for review in train_set:
        wordlist = set()
        for word in review:
            if train_labels[cross_idx] == 1:
                positive_counter[word] += 1
                total_pos_words += 1
            elif train_labels[cross_idx] == 0:
                negative_counter[word] += 1
                total_neg_words += 1
            wordlist.add(word)
        for bigram in nltk.bigrams(review):
            if train_labels[cross_idx] == 1:
                pos_bigram_counter[bigram] += 1
                total_pos_bigrams += 1
            elif train_labels[cross_idx] == 0:
                neg_bigram_counter[bigram] += 1
                total_neg_bigrams += 1
        cross_idx += 1


    dev_labels = []
    pos_count = 0
    neg_count = 0
    for review in dev_set:
        positive_unig_prob = math.log(pos_prior)
        positive_unig_prob += math.log(1 - bigram_lambda)

        positive_bi_prob = math.log(bigram_lambda)
        positive_bi_prob += math.log(pos_prior)

        negative_unig_prob = math.log(1 - pos_prior)
        negative_unig_prob += math.log(1 - bigram_lambda)

        negative_bi_prob = math.log(bigram_lambda)
        negative_bi_prob += math.log(1 - pos_prior)

        negative_bi_prob += math.log(bigram_lambda)
        for word in review:
            positive_unig_prob += math.log((positive_counter[word] + unigram_smoothing_parameter)
                                      / (total_pos_words + (unigram_smoothing_parameter * len(positive_counter.keys()))))
            negative_unig_prob += math.log((negative_counter[word] + unigram_smoothing_parameter)
                                      / (total_neg_words + (unigram_smoothing_parameter * len(negative_counter.keys()))))
        for bigram in nltk.bigrams(review):
            positive_bi_prob += math.log((pos_bigram_counter[bigram] + bigram_smoothing_parameter)
                                         / (total_pos_bigrams + (bigram_smoothing_parameter * len(pos_bigram_counter.keys()))))
            negative_bi_prob += math.log((neg_bigram_counter[bigram] + bigram_smoothing_parameter)
                                         / (total_neg_bigrams + (bigram_smoothing_parameter * len(neg_bigram_counter.keys()))))
        if positive_unig_prob + positive_bi_prob > negative_unig_prob + negative_bi_prob:
            dev_labels.append(1)
            pos_count += 1
        else:
            dev_labels.append(0)
            neg_count += 1

    positive_texts = []
    negative_texts = []
    all_texts = []
    pos_count = 0
    neg_count = 0
    for message in imessages:
        positive_unig_prob = math.log(pos_prior)
        positive_unig_prob += math.log(1 - bigram_lambda)

        positive_bi_prob = math.log(bigram_lambda)
        positive_bi_prob += math.log(pos_prior)

        negative_unig_prob = math.log(1 - pos_prior)
        negative_unig_prob += math.log(1 - bigram_lambda)

        negative_bi_prob = math.log(bigram_lambda)
        negative_bi_prob += math.log(1 - pos_prior)

        negative_bi_prob += math.log(bigram_lambda)
        for word in message[0]:
            positive_unig_prob += math.log((positive_counter[word] + unigram_smoothing_parameter)
                                      / (total_pos_words + (unigram_smoothing_parameter * len(positive_counter.keys()))))
            negative_unig_prob += math.log((negative_counter[word] + unigram_smoothing_parameter)
                                      / (total_neg_words + (unigram_smoothing_parameter * len(negative_counter.keys()))))
        for bigram in nltk.bigrams(review):
            positive_bi_prob += math.log((pos_bigram_counter[bigram] + bigram_smoothing_parameter)
                                         / (total_pos_bigrams + (bigram_smoothing_parameter * len(pos_bigram_counter.keys()))))
            negative_bi_prob += math.log((neg_bigram_counter[bigram] + bigram_smoothing_parameter)
                                         / (total_neg_bigrams + (bigram_smoothing_parameter * len(neg_bigram_counter.keys()))))
        if positive_unig_prob + positive_bi_prob > negative_unig_prob + negative_bi_prob:
            positive_texts.append(message[0], message[1], message[2])
            pos_count += 1
        else:
            dev_labels.append(0)
            neg_count += 1
    # return predicted labels of development set (make sure it's a list, not a numpy array or similar)
    return dev_labels
