#!/usr/bin/python3
# -*- coding: utf-8 -*-

import sys
import string
import urllib.request
import pandas as pd
import nltk
import matplotlib.pyplot as plt
from nltk import FreqDist
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
from wordcloud import WordCloud
from optparse import OptionParser
from ast import literal_eval
nltk.download('stopwords')

dropbox_url = input('Please provide dropbox URL: ')

# Loading English stopwords and extending it a bit

stopwords = nltk.corpus.stopwords.words('english')
for word in [
    'would',
    'don\xe2\x80\x99t',
    'it\xe2\x80\x99s',
    'get',
    'real',
    'need',
    ]:
    stopwords.append(word)


def read_df(dropbox_url):
    """Reads dataset from Dropbox URL. Returns pandas dataframe."""

    df = pd.read_csv(dropbox_url + '?dl=1')
    return df


def word_tokenize(text):
    """Tokenizes string."""

    text = ''.join([ch for ch in text if ch not in string.punctuation])
    tokens = nltk.word_tokenize(text)
    return tokens


def stem_tokens(tokens, stemmer):
    """Stemmes tokens."""

    stemmer = PorterStemmer()
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed


def sent_tokenize(text):
    """Splits text into sentences."""

    sent_tokens = nltk.sent_tokenize(text)
    return sent_tokens


def process_tokens(tokens):
    """Process tokens - remove stopwords, converts to lowercase, stemmes."""

    stemmer = PorterStemmer()
    proc_tokens = [w.lower() for w in tokens]
    proc_tokens = [word for word in proc_tokens if word
                   not in stopwords]
    proc_tokens = stem_tokens(proc_tokens, stemmer)
    return proc_tokens


def calculate_top_words(proc_tokens, n):
    """Calculate top N words from given tokens."""

    fdist = FreqDist(proc_tokens)
    return fdist.most_common(n)


def plot_world_cloud(tokens):
    """Plots and saves wordcloud from tokens to current directory."""

    wordcloud = WordCloud(width=800, height=600, max_words=100,
                          background_color='white'
                          ).generate(' '.join(tokens))
    fig = plt.figure(figsize=(15, 5))
    plt.imshow(wordcloud)
    plt.axis('off')
    fig.savefig('worldcloud.png')
    plt.close()


def process_df(data):
    """Calculate simple dataset stats."""

    data['word_tokens'] = data['letter_text'].apply(lambda row: \
            word_tokenize(row))
    data['num_of_words'] = data['word_tokens'].apply(lambda row: \
            len(row))
    data['num_of_unique_words'] = data['word_tokens'].apply(lambda x: \
            len(set(x)))
    data['num_of_sent'] = data['letter_text'].apply(lambda row: \
            len(sent_tokenize(row)))
    data['avg_sent_len'] = round(data['num_of_words']
                                 / data['num_of_sent'], 0)
    data['voc_density'] = round(data['num_of_unique_words']
                                / data['num_of_words'], 2)
    return data


def main():
    data = read_df(dropbox_url)
    df = process_df(data)
    print('Dataset ' + dropbox_url)
    print('Average number of words: ' + str(round(df['num_of_words'
            ].mean(), 2)))
    print('Average average sentence length: ' \
        + str(round(df['avg_sent_len'].mean(), 2)))
    print('Average vocabularity density: ' + str(round(df['voc_density'
            ].mean(), 2)))
    all_tokens = [y for x in df['word_tokens'].tolist() for y in x]
    proc_tokens = process_tokens(all_tokens)
    print('Top words in dataset: ' \
        + str(calculate_top_words(proc_tokens, 7)))
    plot_world_cloud(proc_tokens)


if __name__ == '__main__':
    main()
