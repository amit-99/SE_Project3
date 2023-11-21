import re
import string
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

linkPattern = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'


def removeLinks(text):
    return re.sub(linkPattern, '', text)

def stripEmojis(text):
    return text.encode('ascii', 'ignore').decode('ascii')


def stripPunctuations(text):
    return text.translate(str.maketrans('', '', string.punctuation))

def stripExtraWhiteSpaces(text):
    return text.strip()

def removeSpecialChar(text):
    return re.sub(r'\W+ ', '', text)


def sentiment_scores(sentence):

    sid_obj = SentimentIntensityAnalyzer()
    sentiment_dict = sid_obj.polarity_scores(sentence)
    return sentiment_dict
