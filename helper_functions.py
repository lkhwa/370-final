import numpy as np
import pandas as pd
import seaborn as sns # for visualiation
import matplotlib.pyplot as plt # plotting
import matplotlib
import re 
from nltk.corpus import stopwords, words
from nltk.tokenize import word_tokenize
from collections import Counter
from textblob import TextBlob
from nltk import FreqDist

# Load data
business_data = pd.read_csv('./data/yelp_business.csv')
tip_data = pd.read_csv('./data/yelp_tip.csv')

# Select businesses under 'Cafes' and 'Coffee & Tea' categories
def filter_category():
    df = business_data[['business_id', 'city', 'state', 'stars', 'categories']]
    return df[df['categories'].str.contains('Cafes|Coffee & Tea')]

# Merge business ids (with text of tips) from tips dataframe
def merge_id():
    df = filter_category()
    return pd.merge(df, tip_data[['business_id','text']], on='business_id')

# Read sentiment words line by line from .txt file
def readwords(filename):
    f = open(filename)
    words = [line.rstrip() for line in f.readlines()]
    return words

# Process string to remove punctuation, stopwords and numerical characters
def process_string(s):
    string = re.split("\r\n", s)
    result = ""
    for substring in string:
        result += substring + " "
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(result.lower())
    remove_stopwords = [word for word in word_tokens if word not in stop_words]
    return [word for word in remove_stopwords if word.isalpha()]

# Count the number of positive words in a string
def count_positive(row):
    text = row['text']
    positive = readwords('./positive-words.txt')
    string = process_string(text)
    count = Counter(string)

    pos = 0
    for key, val in count.items():
        if key in positive:
            pos += val
    return pos

# Count the number of negative words in a string
def count_negative(row):
    text = row['text']
    negative = readwords('./negative-words.txt')
    string = process_string(text)
    count = Counter(string)

    neg = 0
    for key, val in count.items():
        if key in negative:
            neg += val
    return neg

# Returns the overall sentiment (polarity score) of a string
# Polarity score is a float within the range -1.0 and 1.0
def overall_sentiment(row):
    text = row['text']
    tips = TextBlob(text)
    return round(tips.sentiment.polarity, 2)

# Returns a dataframe which consists columns such as:
# 'business_id', 'city', 'state', 'stars', 'categories', 'text',
# 'num_positive_words', 'num_negative_words', 'overall_sentiment'
def get_final_df():
    df = merge_id()
    df['num_positive_words'] = df.apply(count_positive, axis=1)
    df['num_negative_words'] = df.apply(count_negative, axis=1)
    df['overall_sentiment'] = df.apply(overall_sentiment, axis=1)
    return df

def aggregate_df(df):
    df1 = df[['business_id', 'stars', 'num_positive_words', 'num_negative_words', 'overall_sentiment']]
    df1 = df1.groupby('business_id').agg({'stars': 'first', 'num_positive_words':'sum', 'num_negative_words': 'sum', 'overall_sentiment': 'sum'}).reset_index()
    return df1

def top_hundred(df):
    string = (' '.join(df['text']))
    fdist = FreqDist(process_string(string))
    return fdist.most_common(100)

def get_tips(list, data):
    pd.set_option('display.max_colwidth', -1)
    new = pd.DataFrame()
    for word in list:
        df1 = data[['business_id', 'text', 'stars']][data['text'].str.contains(word)]
        df1 = df1.groupby('stars').agg({'business_id': 'first', 'text': 'first'}).reset_index()
        df1['word'] = word
        new = new.append(df1, ignore_index=True)
    return new

def top_businesses_by_tip(data, word_list):
    pd.set_option('max_colwidth',40)
    df1 = data[['business_id', 'text','state', 'stars']]
    df1 = df1.groupby('business_id').agg({'state': len, 'text': lambda x: ' '.join(x), 'stars': 'mean'}).reset_index().sort_values(by=['state'], ascending=False)
    df1.rename(columns={'state': 'tip_count', 'stars': 'rating'}, inplace=True)
    for word in word_list:
        df1[word] = df1['text'].apply(lambda text: text.count(word))
    return df1.head(n=100)

