# sent140SentimentAnalysis.py
#--------------------------------
# Created by Devin Tark (devin.tark@gmail.com) on 6/21/20

"""
This file parses and analyzes the sentiment140 dataset (help.sentiment140.com) to learn to determine
sentiment (pos/neg) in tweets. Uses spacy for tokenization and scikit Pipeline with Logisticregression
"""

import numpy as np
import math
import pandas as pd
from tqdm import tqdm
import spacy
import pudb; pu.db
from sklearn.model_selection import train_test_split
from spacy.lang.en.stop_words import STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn import metrics, preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.base import TransformerMixin
from sklearn.preprocessing import FunctionTransformer
from sklearn.svm import LinearSVC, SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
import reader

nlp = spacy.load("en_core_web_sm")
stop_words = STOP_WORDS

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

def analyze_imessages_to_file(pipe):
    """
    Classify imessage dataset from the pre-trained pipe passed as arg
    Note: Texts are unlabeled (for now)
    """
    imessage_batch = reader.loadiMessageBatchesPipeline()
    imessage_batch_pd = pd.Series(imessage_batch)
    predicted_labels = pipe.predict_proba(imessage_batch_pd)
    pos_texts = []
    neg_texts = []
    for i in range(len(predicted_labels)):
        if predicted_labels[i][0] < predicted_labels[i][1]:
            pos_texts.append((predicted_labels[i], imessage_batch[i]))
        else:
            neg_texts.append((predicted_labels[i], imessage_batch[i]))
    pos_texts = sorted(pos_texts, key=lambda x: x[0][1])
    neg_texts = sorted(neg_texts, key=lambda x: x[0][0])
    with open('lrpos.txt', 'w+', encoding='utf-8') as o:
        for i in pos_texts:
            o.write(str(i))
            o.write("\n")
        o.close()
    with open('lrneg.txt', 'w+', encoding='utf-8') as o:
        for i in neg_texts:
            o.write(str(i))
            o.write("\n")
        o.close()

def imdb_classify():
    train_set, train_labels, test_set, test_labels = reader.load_imdb_dataset_LR(
                                                                                './imdb_dataset/train',
                                                                                './imdb_dataset/dev')
    train_set = pd.Series(train_set)
    train_labels = pd.Series(train_labels)
    test_set = pd.Series(test_set)
    test_labels = pd.Series(test_labels)
    imdb_pipe = Pipeline([('vectorizer', bow_vector),
                     ('tfidf', tf_idf_vector),
                     ('classifier', classifier)])
    imdb_pipe.fit(train_set, train_labels)
    predicted_labels = imdb_pipe.predict(test_set)
    print("-----------LogisticRegression------------------")
    print("Accuracy:",metrics.accuracy_score(test_labels,predicted_labels))
    print("Precision",metrics.precision_score(test_labels,predicted_labels, pos_label=1))
    print("Recall",metrics.recall_score(test_labels,predicted_labels, pos_label=1)) 
    print("F1",metrics.f1_score(test_labels,predicted_labels, pos_label=1))
   # print(metrics.classification_report(test_labels, imdb_pipe.predict(test_set), digits=3))
    return imdb_pipe

def sent140classify():
    train_set, train_labels, test_set, test_labels = prepareTweetsCorpusForPipeline()
    bow_vector = CountVectorizer(tokenizer = spacy_tokenize, ngram_range=(1,1))
    tf_idf_vector = TfidfTransformer()

    classifier = LogisticRegression(max_iter=300, solver='saga')
    nb = MultinomialNB(alpha=.005, class_prior=[.2,.8])
    pipe = Pipeline([('vectorizer', bow_vector),
                     ('tfidf', tf_idf_vector),
                     ('classifier', classifier)])
    pipe.fit(train_set, train_labels)
    predicted_labels = pipe.predict(test_set)
    print("-----------LogisticRegression------------------")
    print("Accuracy:",metrics.accuracy_score(test_labels,predicted_labels))
    print("Precision",metrics.precision_score(test_labels,predicted_labels, pos_label=4))
    print("Recall",metrics.recall_score(test_labels,predicted_labels, pos_label=4))
    print("Recall",metrics.f1_score(test_labels,predicted_labels, pos_label=4))
    return pipe

def spacy_tokenize(text):
    tokenized = []
    for token in nlp(str(text)):
        #if token.is_punct:
        #    continue
        if token.is_space or token.text[0] == '@' or token.text[0] == '.':
            continue
        tokenized.append(token.text.lower())
    return tokenized

def tune_hyper_params(pipe):

    hyperparameters = {
        'vectorizer__ngram_range': [(1,1), (1,2), (2,2)],
        'tfidf__use_idf': (True, False),
        'tfidf__norm': ('l1', 'l2'),
        'classifier__max_iter': [50, 100, 300],
        'classifier__solver': ['lbfgs', 'sag', 'saga']
    }

    print("classification report")
    print()
    print(metrics.classification_report(test_labels, pipe.predict(test_set), digits=3))
    clf = GridSearchCV(pipe, hyperparameters,cv=10, scoring='f1_micro')
    clf.fit(train_set, train_labels)
    print("Best params:")
    print()
    print(clf.best_params_)
    print()
    print("Grid Scores on the Dev Set:")
    print()
    for mean, std, params in zip(clf.cv_results_['mean_test_score'],
                                 clf.cv_results_['std_test_score'],
                                 clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r" % (mean, std  * 2, params))
    print()
    print("Classification report:")
    print()
    print(metrics.classification_report(test_labels, clf.predict(test_set), digits=3))

def main():
    #train_set, train_labels, test_set, test_labels = prepareTweetsCorpusForPipeline()
    bow_vector = CountVectorizer(tokenizer = spacy_tokenize, ngram_range=(1,1))
    tf_idf_vector = TfidfTransformer()

    classifier = LogisticRegression(max_iter=300, solver='saga')
    nb = MultinomialNB(alpha=.005, class_prior=[.2,.8])
    """
    pipe = Pipeline([('vectorizer', bow_vector),
                     ('tfidf', tf_idf_vector),
                     ('classifier', classifier)])
    pipe.fit(train_set, train_labels)
    predicted_labels = pipe.predict(test_set)
    print("-----------LogisticRegression------------------")
    print("Accuracy:",metrics.accuracy_score(test_labels,predicted_labels))
    print("Precision",metrics.precision_score(test_labels,predicted_labels, pos_label=4))
    print("Recall",metrics.recall_score(test_labels,predicted_labels, pos_label=4)) 
    print("Recall",metrics.f1_score(test_labels,predicted_labels, pos_label=4))
    hyperparameters = {
        'vectorizer__ngram_range': [(1,1), (1,2), (2,2)],
        'tfidf__use_idf': (True, False),
        'tfidf__norm': ('l1', 'l2'),
        'classifier__max_iter': [50, 100, 300],
        'classifier__solver': ['lbfgs', 'sag', 'saga']
    }

    print("classification report")
    print()
    print(metrics.classification_report(test_labels, pipe.predict(test_set), digits=3))
    clf = GridSearchCV(pipe, hyperparameters,cv=10, scoring='f1_micro')
    clf.fit(train_set, train_labels)
    print("Best params:")
    print()
    print(clf.best_params_)
    print()
    print("Grid Scores on the Dev Set:")
    print()
    for mean, std, params in zip(clf.cv_results_['mean_test_score'],
                                 clf.cv_results_['std_test_score'],
                                 clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r" % (mean, std  * 2, params))
    print()
    print("Classification report:")
    print()
    print(classification_report(test_labels, clf.predict(test_set), digits=3))
    """

    train_set, train_labels, test_set, test_labels = reader.load_imdb_dataset_LR(
                                                                                './imdb_dataset/train',
                                                                                './imdb_dataset/dev')
    imessage_batch = reader.loadiMessageBatchesPipeline()
    imessage_batch_pd = pd.Series(imessage_batch)
    train_set = pd.Series(train_set)
    train_labels = pd.Series(train_labels)
    test_set = pd.Series(test_set)
    test_labels = pd.Series(test_labels)
    imdb_pipe = Pipeline([('vectorizer', bow_vector),
                     ('tfidf', tf_idf_vector),
                     ('classifier', classifier)])
    imdb_pipe.fit(train_set, train_labels)
    """
    predicted_labels = imdb_pipe.predict(test_set)
    print("-----------LogisticRegression------------------")
    print("Accuracy:",metrics.accuracy_score(test_labels,predicted_labels))
    print("Precision",metrics.precision_score(test_labels,predicted_labels, pos_label=1))
    print("Recall",metrics.recall_score(test_labels,predicted_labels, pos_label=1)) 
    print("F1",metrics.f1_score(test_labels,predicted_labels, pos_label=1))
    """
   # print(metrics.classification_report(test_labels, imdb_pipe.predict(test_set), digits=3))

if __name__ == "__main__":
    main()
