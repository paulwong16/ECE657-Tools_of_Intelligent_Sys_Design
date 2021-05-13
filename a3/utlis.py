import glob
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
import torch.nn as nn

import numpy as np


def get_data(path_neg, path_pos):
    neg_data = []
    pos_data = []
    files_neg = glob.glob(path_neg)
    files_pos = glob.glob(path_pos)
    for neg in files_neg:
        with open(neg, 'r', encoding='utf-8') as neg_f:
            neg_data.append(neg_f.readline())

    for pos in files_pos:
        with open(pos, 'r', encoding='utf-8') as pos_f:
            pos_data.append(pos_f.readline())

    neg_label = np.zeros(len(neg_data)).tolist()
    pos_label = np.ones(len(pos_data)).tolist()

    corpus = neg_data + pos_data
    labels = neg_label + pos_label

    return corpus, labels


def normalize(corpus):
    nltk.download('stopwords')
    normalized_corpus = []
    for text in corpus:
        text = text.lower().strip()
        tokenizer = RegexpTokenizer(r'\w+')
        tokens = tokenizer.tokenize(text)

        stopword = stopwords.words('english')
        filtered_tokens = [token for token in tokens if token not in stopword]

        filtered_text = ' '.join(filtered_tokens)
        normalized_corpus.append(filtered_text)

    return normalized_corpus


def tfidf_extractor(corpus, ngram_range=(1, 1)):
    vectorizer = TfidfVectorizer(min_df=1, norm='l2', smooth_idf=True, use_idf=True, ngram_range=ngram_range)
    features = vectorizer.fit_transform(corpus)
    return vectorizer, features


class MLP(nn.Module):
    def __init__(self, input_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 2)
        self.input_size = input_size

    def forward(self, x):
        x = x.view(-1, self.input_size)
        x = nn.functional.dropout(torch.relu(self.fc1(x)))
        x = nn.functional.dropout(torch.relu(self.fc2(x)))
        x = self.fc3(x)
        return x