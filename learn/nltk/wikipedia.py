import requests as req
from bs4 import BeautifulSoup as bs
from nltk import word_tokenize
import nltk
# nltk.download()
from nltk.stem import WordNetLemmatizer
from collections import Counter
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from nltk.stem import SnowballStemmer

# url = 'https://en.wikipedia.org/wiki/Data_science'
# url = 'https://en.wikipedia.org/wiki/finland'
# url = 'https://www.socialdemokraterna.se/vart-parti/politiker/stefan-lofven/tal3/2019/forsta-maj-tal/'
url = 'https://www.presidentti.fi/uutinen/tasavallan-presidentti-sauli-niiniston-uudenvuodenpuhe-1-1-2019/'

article = req.get(url)

html = bs(article.text, 'html.parser')
paragraphs = html.select("p")
text = ""
for para in paragraphs:
    text = text + para.text

word_tokens = word_tokenize(text)
lower_tokens = [t.lower() for t in word_tokens]


# Retain alphabetic words: alpha_only
alpha_only = [t for t in lower_tokens if t.isalpha()]

# Remove all stop words: no_stops
stops = stopwords.words('finnish')
no_stops = [t for t in alpha_only if t not in stops]

# Instantiate the WordNetLemmatizer
# wordnet_lemmatizer = WordNetLemmatizer()
stemmer = SnowballStemmer("finnish")

# Lemmatize all tokens into a new list: lemmatized
# lemmatized = [wordnet_lemmatizer.lemmatize(t) for t in no_stops]
lemmatized = [stemmer.stem(t) for t in no_stops]
word_counter = Counter(no_stops)



# Plotting section

# lists = sorted(word_counter.most_common(10))
# x, y = zip(*lists)
x, y = zip(*word_counter.most_common(20))
_ = plt.xticks(rotation=45)
_ = plt.barh(x, y)
_ = plt.show()
