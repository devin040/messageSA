# mp3_mixture.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 09/28/2018
# Modified by Jaewook Yeom 02/02/2020
import sys
import argparse
import configparser
import copy
import numpy as np
#import pudb; pu.db
import reader
import naive_bayes_mixture as nb
from sklearn.naive_bayes import CategoricalNB, MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn import metrics, preprocessing
from sklearn.linear_model import LogisticRegression
import tf_idf
from sklearn.base import TransformerMixin
from sklearn.preprocessing import FunctionTransformer
from sklearn.svm import LinearSVC, SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
"""
This file contains the main application that is run for Part 2 of this MP.
"""

def compute_accuracies(predicted_labels, dev_set, dev_labels):
    yhats = predicted_labels
    accuracy = np.mean(yhats == dev_labels)
    tp = np.sum([yhats[i] == dev_labels[i] and yhats[i] == 1 for i in range(len(yhats))])
    precision = tp / np.sum([yhats[i] == 1 for i in range(len(yhats))])
    recall = tp / (np.sum([yhats[i] != dev_labels[i] and yhats[i] == 0 for i in range(len(yhats))]) + tp)
    f1 = 2 * (precision * recall) / (precision + recall)
    return accuracy, f1, precision, recall

def compute_accuracies_sent140(predicted_labels, dev_set, dev_labels):
    yhats = predicted_labels
    accuracy = np.mean(yhats == dev_labels)
    tp = np.sum([yhats[i] == dev_labels[i] and yhats[i] == 4 for i in range(len(yhats))])
    precision = tp / np.sum([yhats[i] == 4 for i in range(len(yhats))])
    recall = tp / (np.sum([yhats[i] != dev_labels[i] and yhats[i] == 0 for i in range(len(yhats))]) + tp)
    f1 = 2 * (precision * recall) / (precision + recall)
    return accuracy, f1, precision, recall

class DenseTransformer(TransformerMixin):

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **fit_params):
        return X.todense()

def main(args):
    train_set, train_labels, dev_set, dev_labels = reader.load_dataset(args.training_dir,args.development_dir,args.stemming,args.lower_case)
    imessages, imessage_batches = reader.load_imessage_dataset(args.stemming, args.lower_case)
    predicted_labels = nb.naiveBayesMixture(train_set, train_labels, dev_set, imessages, imessage_batches, args.bigram_lambda, args.unigram_smoothing, args.bigram_smoothing, args.pos_prior)

    accuracy, f1, precision, recall = compute_accuracies(predicted_labels, dev_set, dev_labels)
    print("Accuracy:",accuracy)
    print("F1-Score:",f1)
    print("Precision:",precision)
    print("Recall:",recall)

    train_set, train_labels, dev_set, dev_labels = reader.loadTweets(args.stemming, args.lower_case)
    predicted_tweet_labels = nb.naiveBayesMixtureSent140(train_set, train_labels, dev_set, args.bigram_lambda, args.unigram_smoothing, args.bigram_smoothing, args.pos_prior)

    accuracy, f1, precision, recall = compute_accuracies_sent140(predicted_tweet_labels, dev_set, dev_labels)
    print("Accuracy:",accuracy)
    print("F1-Score:",f1)
    print("Precision:",precision)
    print("Recall:",recall)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CS440 MP3 Naive Bayes Mixture (Part 2)')

    parser.add_argument('--training', dest='training_dir', type=str, default = './imdb_dataset/train',
                        help='the directory of the training data')
    parser.add_argument('--development', dest='development_dir', type=str, default = './imdb_dataset/dev',
                        help='the directory of the development data')
    parser.add_argument('--stemming',dest="stemming", type=bool, default=False,
                        help='Use porter stemmer')
    parser.add_argument('--lower_case',dest="lower_case", type=bool, default=False,
                        help='Convert all word to lower case')
    parser.add_argument('--bigram_lambda',dest="bigram_lambda", type=float, default = 0.5,
                        help='Bigram Lambda Value - default 0.5')
    parser.add_argument('--unigram_smoothing',dest="unigram_smoothing", type=float, default = 1.0,
                        help='Unigram Laplace smoothing parameter - default 1.0')
    parser.add_argument('--bigram_smoothing',dest="bigram_smoothing", type=float, default = 1.0,
                        help='Bigram Laplace smoothing parameter - default 1.0')
    parser.add_argument('--pos_prior',dest="pos_prior", type=float, default = 0.8,
                        help='Positive prior, i.e. Num_positive_comments / Num_comments')
    args = parser.parse_args()
    main(args)

