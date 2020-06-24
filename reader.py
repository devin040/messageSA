# reader.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 09/28/2018
"""
This file is responsible for providing functions for reading the files
"""
from os import listdir
import numpy as np
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from tqdm import tqdm
import json
import pandas as pd
#import pudb; pu.db
import spacy
import string
from spacy.lang.en.stop_words import STOP_WORDS

nlp = spacy.load("en_core_web_sm")
porter_stemmer = PorterStemmer()
tokenizer = RegexpTokenizer(r'\w+')
bad_words = {'aed','oed','eed'} # these words fail in nltk stemmer algorithm

def loadDir(name,stemming,lower_case):
    # Loads the files in the folder and returns a list of lists of words from
    # the text in each file
    X0 = []
    count = 0
    for f in tqdm(listdir(name)):
        fullname = name+f
        text = []
        with open(fullname, 'rb') as f:
            for line in f:
                if lower_case:
                    line = line.decode(errors='ignore').lower()
                    text += tokenizer.tokenize(line)
                else:
                    text += tokenizer.tokenize(line.decode(errors='ignore'))
        if stemming:
            for i in range(len(text)):
                if text[i] in bad_words:
                    continue
                text[i] = porter_stemmer.stem(text[i])
        X0.append(text)
        count = count + 1
    return X0

def loadiMessagesDir(name,numfiles,stemming,lower_case):
    texts = []
    ret = []
    with open("./message_data/zun_texts.json", "r") as f:
        texts = json.load(f)
    for i in tqdm(range(len(texts))):
        text = []
        if not isinstance(texts[i]['text'], str):
            continue
        if lower_case:
            lower_text = texts[i]['text'].lower()
            text = tokenizer.tokenize(lower_text)
        else:
            text = texts[i]['text']
            if not isinstance(text, str):
                continue
            text = tokenizer.tokenize(texts[i]['text'])
        if stemming: 
            for j in range(len(text)):
                if text[j] in bad_words:
                    continue
                text[j] = porter_stemmer.stem(text[j])
        ret.append((text, texts[i]['timestamp'], texts[i]['from_me']))
    return ret

def loadiMessageBatches(stemming, lower_case):
    texts = []
    ret = [] 
    with open("./message_data/zun_texts.json", "r") as f:
        texts = json.load(f)
    for i in range(0, len(texts)-6, 6):
        text = ""
        for j in range(6):
            if texts[i+j]['text'] is not None:
                text += texts[i+j]['text']
                text += " "
        if lower_case:
            text = text.lower()
        text = tokenizer.tokenize(text)
        if stemming:
            for k in range(len(text)):
                if text[k] in bad_words:
                    continue
                text[k] = porter_stemmer.stem(text[k])
        ret.append((text, texts[i]['timestamp']))
    return ret

def loadTweets(stemming, lower_case):
    tweets = pd.read_csv('training.1600000.processed.noemoticon.csv', engine='python')
    tweets_arr = []
    for row in tweets.itertuples():
        tweets_arr.append((int(row[1]), row[6]))
    tweets_arr = np.array(tweets_arr)
    train_tweets = []
    train_tweets_ret = []
    test_tweets = []
    test_tweets_ret = []
    for i in tqdm(range(tweets_arr.shape[0])):
        if i % 2 == 0:
            train_tweets.append(tweets_arr[i])
        else:
            test_tweets.append(tweets_arr[i])
    train_tweets = train_tweets[::80]
    test_tweets = test_tweets[::320]
    np.random.shuffle(train_tweets)
    np.random.shuffle(test_tweets)
    tokenized = []

    stop_words = STOP_WORDS
    train_tweets_ret = [train_tweets[i][1] for i in range(len(train_tweets))]
    train_tweets_labels = [int(train_tweets[i][0]) for i in range(len(train_tweets))]
    train_tweets = []
    print("Learning....")
    for i in tqdm(train_tweets_ret):
        tokenized = []
        for token in nlp(str(i)):
           # if token.is_punct:
            #    continue
            tokenized.append(token.text.lower())
        if lower_case:
            tokenized = [text.lower() for text in tokenized]
        if stemming:
            for i in range(len(tokenized)):
                if tokenized[i] in bad_words:
                    continue
                tokenized[i]  = porter_stemmer.stem(tokenized[i])
        train_tweets.append(tokenized)

    test_tweets_ret = [test_tweets[i][1] for i in range(len(test_tweets))]
    test_tweets_labels = [int(test_tweets[i][0]) for i in range(len(test_tweets))]
    test_tweets = []
    print("Evaluating.....")
    for i in tqdm(test_tweets_ret):
        tokenized = []
        for token in nlp(str(i)):
           # if token.is_punct:
            #    continue
            tokenized.append(token.text.lower())
        if lower_case:
            tokenized = [text.lower() for text in tokenized]
        if stemming:
            for i in range(len(tokenized)):
                if tokenized[i] in bad_words:
                    continue
                tokenized[i]  = porter_stemmer.stem(tokenized[i])
        test_tweets.append(tokenized)
    train_tweets_labels = np.array(train_tweets_labels)
    test_tweets_labels = np.array(test_tweets_labels)

    return train_tweets, train_tweets_labels, test_tweets, test_tweets_labels

def load_imdb_train_LR(name):
    X0 = []
    files = np.array(listdir(name))
    files = files[:10000:2]
    for f in tqdm(files):
        fullname = name+f
        text = ""
        with open(fullname, 'rb') as f:
            for line in f:
                line = line.decode(errors='ignore')
                text += line + " "
        X0.append(text)
    return X0

def load_imdb_test_LR(name):
    X0 = []
    files = np.array(listdir(name))
    files = files[:12000:6]
    for f in tqdm(files):
        fullname = name+f
        text = ""
        with open(fullname, 'rb') as f:
            for line in f:
                line = line.decode(errors='ignore')
                text += line + " "
        X0.append(text)
    return X0

def load_dataset(train_dir, dev_dir, stemming, lower_case):
    X0 = loadDir(train_dir + '/pos/',stemming, lower_case)
    X1 = loadDir(train_dir + '/neg/',stemming, lower_case)
    X = X0 + X1
    Y = len(X0) * [1] + len(X1) * [0]
    Y = np.array(Y)

    X_test0 = loadDir(dev_dir + '/pos/',stemming, lower_case)
    X_test1 = loadDir(dev_dir + '/neg/',stemming, lower_case)
    X_test = X_test0 + X_test1
    Y_test = len(X_test0) * [1] + len(X_test1) * [0]
    Y_test = np.array(Y_test)
    return X,Y,X_test,Y_test

def load_imessage_dataset(stemming, lower_case):
    X_imessage = loadiMessagesDir(None, None, stemming, lower_case)
    X_imessage_batch = loadiMessageBatches(stemming, lower_case)
    return X_imessage, X_imessage_batch

def load_imdb_dataset_LR(train_dir, dev_dir):
    X0 = load_imdb_train_LR('imdb_new_dataset/aclimdb/train/pos/')
    X1 = load_imdb_train_LR('imdb_new_dataset/aclimdb/train/neg/')
    X = X0 + X1
    Y = len(X0) * [1] + len(X1) * [0]
    X = np.array(X)
    Y = np.array(Y)

    X_test0 = load_imdb_test_LR('imdb_new_dataset/aclimdb/test/pos/')
    X_test1 = load_imdb_test_LR('imdb_new_dataset/aclimdb/test/neg/')
    X_test = X_test0 + X_test1
    Y_test = len(X_test0) * [1] + len(X_test1) * [0]
    X_test = np.array(X_test)
    Y_test = np.array(Y_test)
    return X,Y,X_test,Y_test

def loadiMessageBatchesPipeline():
    texts = []
    ret = []
    with open("./message_data/zun_texts.json", "r") as f:
        texts = json.load(f)
    for i in range(0, len(texts)-6, 6):
        text = ""
        for j in range(6):
            if texts[i+j]['text'] is not None:
                text += texts[i+j]['text']
                text += " "
        #ret.append((text, texts[i]['timestamp']))
        ret.append(text.strip())
    return ret
