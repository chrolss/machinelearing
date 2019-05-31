import requests as req
from bs4 import BeautifulSoup as bs
from nltk import word_tokenize
import nltk
# nltk.download()
from collections import Counter
from nltk.corpus import stopwords

url = 'https://en.wikipedia.org/wiki/Data_science'

article = req.get(url)

html = bs(article.text, 'html.parser')
title = html.select('#firstHeading')[0].text
paragraphs = html.select("p")
text = ""
for para in paragraphs:
    text = text + para.text

word_tokens = word_tokenize(text)
lower_tokens = [t.lower() for t in word_tokens]


# Retain alphabetic words: alpha_only
alpha_only = [t for t in lower_tokens if t.isalpha()]

# Remove all stop words: no_stops
stops = stopwords.words('english')
no_stops = [t for t in alpha_only if t not in stops]


word_counter = Counter(no_stops)