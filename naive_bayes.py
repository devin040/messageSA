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
This is the main entry point for Part 1 of MP3. You should only modify code
within this file for Part 1 -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""

import numpy as numpy
import math
from collections import Counter


def naiveBayes(train_set, train_labels, dev_set, smoothing_parameter, pos_prior):
    """
    train_set - List of list of words corresponding with each movie review
    example: suppose I had two reviews 'like this movie' and 'i fall asleep' in my training set
    Then train_set := [['like','this','movie'], ['i','fall','asleep']]

    train_labels - List of labels corresponding with train_set
    example: Suppose I had two reviews, first one was positive and second one was negative.
    Then train_labels := [1, 0]

    dev_set - List of list of words corresponding with each review that we are testing on
              It follows the same format as train_set

    smoothing_parameter - The smoothing parameter you provided with --laplace (1.0 by default)

    pos_prior - positive prior probability (between 0 and 1)
    """



    # TODO: Write your code here
    
    # positive_counter = Counter()
    # negative_counter = Counter()
    # total_num_pos_reviews = 0
    # total_num_neg_reviews = 0
    # distinct_pos_word_set = set()
    # distinct_neg_word_set = set()
    #
    # for x in train_labels:
    #     if x == 1:
    #         total_num_pos_reviews += 1
    #     else:
    #         total_num_neg_reviews += 1
    # cross_idx = 0
    # for review in train_set:
    #     wordlist = set()
    #     for word in review:
    #         if train_labels[cross_idx] == 1 and word not in wordlist:
    #             positive_counter[word] += 1
    #         elif train_labels[cross_idx] == 0 and word not in wordlist:
    #             negative_counter[word] += 1
    #         wordlist.add(word)
    #     cross_idx += 1
    #
    # dev_labels = []
    # for review in dev_set:
    #     positive_prob = math.log10(pos_prior)
    #     negative_prob = math.log10(1 - pos_prior)
    #     for word in review:
    #         positive_prob += math.log10(
    #             positive_counter[word] + smoothing_parameter / (total_num_pos_reviews + (smoothing_parameter * 2)))
    #         negative_prob += math.log10(
    #             negative_counter[word] + smoothing_parameter / (total_num_neg_reviews + (smoothing_parameter * 2)))
    #     if positive_prob > negative_prob:
    #         dev_labels.append(1)
    #     else:
    #         dev_labels.append(0)
    # return dev_labels

    positive_counter = Counter()
    negative_counter = Counter()
    total_num_pos_reviews = 0
    total_num_neg_reviews = 0
    total_pos_words = 0
    total_neg_words = 0

    for x in train_labels:
        if x == 1:
            total_num_pos_reviews += 1
        else:
            total_num_neg_reviews += 1
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
        cross_idx += 1
    #print(positive_counter)
    dev_labels = []
    pos_count = 0
    neg_count = 0
    for review in dev_set:
        positive_prob = math.log(pos_prior)
        negative_prob = math.log(1 - pos_prior)
        for word in review:
            positive_prob += math.log((positive_counter[word] + smoothing_parameter) / (total_pos_words + (smoothing_parameter * len(positive_counter.keys()))))
            pos = positive_counter[word] + smoothing_parameter / (total_pos_words + (smoothing_parameter * len(positive_counter.keys())))
            negative_prob += math.log((negative_counter[word] + smoothing_parameter) / (total_neg_words + (smoothing_parameter * len(negative_counter.keys()))))
        if positive_prob > negative_prob:
            dev_labels.append(1)
            pos_count += 1
        else:
            dev_labels.append(0)
            neg_count += 1
    return dev_labels

    # return predicted labels of development set (make sure it's a list, not a numpy array or similar)