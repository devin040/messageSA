# tf_idf_bayes.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 09/28/2018
# Modified by Jaewook Yeom 02/02/2020

"""
This is the main entry point for the Extra Credit Part of this MP. You should only modify code
within this file for the Extra Credit Part -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""

import numpy as np
import math
from collections import Counter
import time
import pandas as pd
from tqdm import tqdm
import spacy
#import pudb; pu.db
from sklearn.model_selection import train_test_split
from spacy.lang.en.stop_words import STOP_WORDS

nlp = spacy.load("en_core_web_sm")
stop_words = STOP_WORDS

def compute_tf_idf(train_set, train_labels, dev_set):
    """
    train_set - List of list of words corresponding with each movie review
    example: suppose I had two reviews 'like this movie' and 'i fall asleep' in my training set
    Then train_set := [['like','this','movie'], ['i','fall','asleep']]

    train_labels - List of labels corresponding with train_set
    example: Suppose I had two reviews, first one was positive and second one was negative.
    Then train_labels := [1, 0]

    dev_set - List of list of words corresponding with each review that we are testing on
              It follows the same format as train_set

    Return: A list containing words with the highest tf-idf value from the dev_set documents
            Returned list should have same size as dev_set (one word from each dev_set document)
    """

    word_counter = Counter()
    idf_array = []
    cross_idx = 0
    for review in train_set:
        wordlist = set()
        for word in review:
            if word not in wordlist:
                word_counter[word] += 1
            wordlist.add(word)

    for review in dev_set:
        rev_word_counter = Counter()
        total_words = 0
        max_idf = -1
        max_idf_word = ""
        for word in review:
            rev_word_counter[word] += 1
            total_words += 1
        for word in review:
            tf_idf = (rev_word_counter[word] / total_words) * math.log(len(train_set) / (1 + word_counter[word]))
            if tf_idf > max_idf:
                max_idf = tf_idf
                max_idf_word = word
        idf_array.append(max_idf_word)

    # return list of words (should return a list, not numpy array or similar)
    return idf_array

def prepareTweetsCorpusForPipeline():
    """
    Prepare the raw csv file as a list of strings (tweets).
    """
    tweets = pd.read_csv('training.1600000.processed.noemoticon.csv', index_col=False, skiprows=lambda x: x%2!=0, engine='python')
    tweets_arr = []
    X = tweets["tweet"]
    ylabels = tweets["sent"]
    train_tweets, test_tweets, train_labels, test_labels = train_test_split(X, ylabels, test_size=10000, train_size=2500)

    return train_tweets, train_labels, test_tweets, test_labels

def spacy_tokenize(text):
    tokenized = []
    for token in nlp(str(text)):
        #if token.is_punct:
        #    continue
        if token.is_space or token.text[0] == '@' or token.text[0] == '.':
            continue
        tokenized.append(token.text.lower())
    return tokenized
