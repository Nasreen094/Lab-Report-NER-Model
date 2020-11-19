import pandas as pd
import glob
import errno
import nltk
from itertools import chain
from itertools import groupby
from operator import itemgetter
import re
import json
import sklearn
import scipy.stats
from sklearn.metrics import make_scorer
import numpy as np
import random
random.seed(42)
import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics
from datetime import datetime
import pickle
import settings
import os
BASE_DIR = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
))

train_path1=settings.train_path1

train_path2=settings.train_path2



crf_module_vocab = settings.crf_vocab

range_category = crf_module_vocab['range_cat']


def is_name(word):
    if any([word.upper().strip().endswith(i) for i in crf_module_vocab['suffix']]):
        return True
    elif any([word.upper().strip().startswith(i) for i in crf_module_vocab['starts']]):
        return True
    elif any([i in word.upper().strip() for i in crf_module_vocab['sub_names']]):
        return True

    else:
        return False


def is_unit(word):
    #     if word in units_list:
    #         return True
    if word.upper().strip().endswith('/L'):
        return True
    elif re.match(r'(^(10)\s?\^\s?[1-9]\s?(/[Uu]?[lL])?$)|(^(10)?\s?~?\s?\d/[Uu]?[Ll]$)', word):
        return True
    else:
        return False


range_list = ['-', '>', '<', '-—', '—-', '=', '–', 'Up to 15', 'Up']


def is_range(word):
    if word in range_list:
        return True
    elif re.match(r'^\d*\.?\d*\s?[-|—|>|<|=|-—|–|-]\s?\d+\.?\d*$', word):
        return True
    else:
        return False


def is_range_cat(word):
    if word in range_category:
        return True

    else:
        return False


def word2features(sent, i):
    word = str(sent[i][0])
    # print(type(word))
    postag = sent[i][2]
    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'postag': postag,
        'postag[:2]': postag[:2],
        'word[+3:]': word[+3:],
        'word[+2:]': word[+2:],
        #   'is_unit()': is_unit(word),
        'is_range_cat()': is_range_cat(word),
        'is_name()': is_name(word),
        #'length':len(word)

    }
    if i > 0:
        word1 = str(sent[i-1][0])
        postag1 = sent[i-1][1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:word.isdigit()': word1.isdigit(),
            '-1:is_name()': is_name(word1),
            # '-1:is_range_cat()': is_range_cat(word1),
            #  '-1:length':len(word1)
        })
    else:
        features['BOS'] = True

    if i < len(sent)-1:
        word1 = str(sent[i+1][0])
        postag1 = sent[i+1][1]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:word.isdigit()': word1.isdigit(),
            '+1:is_name()': is_name(word1),
            #  '+1:is_range_cat()': is_range_cat(word1),
            #  '+1:length':len(word1)
        })
    else:
        features['EOS'] = True

    return features


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]


def sent2labels(sent):
    return [label for token, label, postag in sent]


def get_training_data():
    files = glob.glob(train_path1)
    train = []
    print(files)
    for name in files:
        try:
            df = pd.read_csv(name, index_col=0)
            df['w'] = df['w'].replace({'\n': '#n'})

            df['pos'] = df['w'].apply(lambda x: nltk.pos_tag([str(x)])[0][1])
            tuple1 = [tuple(x) for x in df.values]
            train.append(tuple1)

        except IOError as exc:
            if exc.errno != errno.EISDIR:
                raise
    print(len(train))
    files = glob.glob(train_path2)

    for name in files:
        try:
            df = pd.read_csv(name, index_col=0)
            df['w'] = df['w'].replace({'\n': '#n'})

            df['pos'] = df['w'].apply(lambda x: nltk.pos_tag([str(x)])[0][1])
            tuple1 = [tuple(x) for x in df.values]
            train.append(tuple1)

        except IOError as exc:
            if exc.errno != errno.EISDIR:
                raise
    print(len(train))
    return train


def retrain_model():
    try:
        train = get_training_data()
        X = [sent2features(s) for s in train]
        y = [sent2labels(s) for s in train]
        crf = sklearn_crfsuite.CRF(
            algorithm='ap',
            all_possible_states=True,
            all_possible_transitions=True,
            max_iterations=1000,


        )
        crf.fit(X, y)
        now = datetime.now()
        now = now.strftime("%Y-%m-%d::%H:%M:%S")
        modelfilename = os.path.join(
            BASE_DIR, 'model', now+'_crf_model.sav')
        pickle.dump(crf, open(modelfilename, 'wb'))
        
    except Exception as exc:
        print(exc)
        return False
    return True

  
