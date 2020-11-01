import lyricsgenius as lg
import joblib
import pandas as pd
import re

# genius APIのtoken設定用
from pit import Pit
import os
if not os.environ.get('EDITOR'):
    os.environ['EDITOR'] = 'vi'

# fetching ozzy data
genius = lg.Genius(Pit.get('genius')['token'])
artist = genius.search_artist("Ozzy Osbourne", sort="title")
joblib.dump(artist, "./data/ozzy.jb", compress=3)

# preprocessing
d = pd.DataFrame(data={
    'title':
    [song.title for song in artist.songs]
})
d['title_f'] = d['title'].apply(lambda x: re.sub(r"( *\[.+\]| *\(.+\))", "", x))
d = d.reset_index()
d['index'] = d['index'] + 1
d['cnt_duplication'] = d.groupby("title_f")["index"].rank(method="first")
d = d.query('cnt_duplication == 1')[['index', 'title']]
d['lyrics'] = [artist.songs[i-1].lyrics for i in d['index']]
d = d.dropna()

import nltk
from nltk.tokenize import PunktSentenceTokenizer
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
# ストップワードのリストを初回だけダウンロード
nltk.download('stopwords')
nltk.download('punkt')

stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

d['lyrics_f'] = d.lyrics\
    .apply(lambda x: re.sub(r"\[.+\]", " ", x))\
    .apply(lambda x: re.sub(r"\n", " ", x))\
    .apply(lambda x: re.sub(r" +", " ", x))\
    .apply(lambda x: " ".join([ps.stem(token) for token in word_tokenize(x.lower()) if token != "" if token not in stop_words]))

# saving preprocessed dataframe
d.to_csv("./data/df_ozzy.csv", index=False)