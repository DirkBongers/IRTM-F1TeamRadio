
"""
Created on Thu Apr 12 21:13:10 2018

@author: dbn
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, AgglomerativeClustering
#from wordcloud import WordCloud
#PATH = 'C:\\Users\\LucBl\\OneDrive\\Bureaublad\\Uni Master\\IRTM\\Assignment\\IRTM-F1TeamRadio\\F1 Teamradio\\'
PATH = 'C:\\Users\\dbn\\Desktop\\IRTM-F1TeamRadio\\F1 Teamradio\\'
filelist = os.listdir(PATH + 'Transcripts')
data = pd.read_excel(PATH+'Transcripts\\'+ filelist[0])
racename = filelist[0].replace('2017-','')
racename = racename.replace('-Race.xlsx','')
Track = [racename]*len(data.Driver)
data['Track'] = Track
for i in range(1,19):
    df = pd.read_excel(PATH+'Transcripts\\'+ filelist[i])
    racename = filelist[i].replace('2017-','')
    racename = racename.replace('-Race.xlsx','')
    Track = [racename]*len(df.Driver)
    df['Track'] = Track
    #print(df.columns.values)
    #print(filelist[i])
    data = pd.concat([data,df])
data = data.dropna(axis=0,how='any')
data.index = range(0,len(data.Driver))

filelistR = os.listdir(PATH + 'Results')
dataR = pd.read_excel(PATH+'Results\\'+ filelistR[0])
racenameR = filelistR[0].replace('2017-','')
racenameR = racenameR.replace('-Results.xlsx','')
TrackR = [racenameR]*len(dataR.DRIVER)
dataR['Track'] = TrackR
for i in range(1,19):
    df = pd.read_excel(PATH+'Results\\'+ filelistR[i])
    racenameR = filelistR[i].replace('2017-','')
    racenameR = racenameR.replace('-Results.xlsx','')
    TrackR = [racenameR]*len(df.DRIVER)
    df['Track'] = TrackR
    #print(df.columns.values)
    #print(filelist[i])
    dataR = pd.concat([dataR,df])

To = [None]*len(data.Driver)
From = [None]*len(data.Driver)
data['To'] = pd.Series(To,index = data.index)
data['From'] = pd.Series(From,index = data.index)
for i in range(0,len(data.Driver)):
    if 'To' in data.Driver[i]:
        data.To[i] = data.Driver[i].replace('To\xa0','')
        data.To[i] = data.To[i].replace('To ','')
        data.To[i] = data.To[i].replace('To','')
    elif 'From' in data.Driver[i]:
        data.From[i] = data.Driver[i].replace('From\xa0','')
        data.From[i] = data.From[i].replace('From ','')
        data.From[i] = data.From[i].replace('From','')
    else:
        data.To[i] = data.Driver[i]
        data.From[i] = data.Driver[i]

#dataFromMax = data[data.From == 'Max Verstappen']
#dataFromMax.index = range(0,len(dataFromMax.Driver))
#from wordcloud import WordCloud
#wc = WordCloud().generate(" ".join(dataFromMax.Message))
#import matplotlib.pyplot as plt
#plt.imshow(wc, interpolation='bilinear')
#plt.axis("off")
#
#dataFromKimi = data[data.From == 'Kimi Raikkonen']
#dataFromKimi.index = range(0,len(dataFromKimi.Driver))
#wc = WordCloud().generate(" ".join(dataFromKimi.Message))
#plt.imshow(wc, interpolation='bilinear')
#plt.axis("off")
import string
from collections import Counter

allText = " ".join([str(data.Message[j]).lower().translate(str.maketrans('','',string.punctuation)) for j in range(0,len(data.Message))])
allwords = Counter()
allwords.update(allText.split())

DriverFromDictionary = pd.DataFrame(columns = data.From.unique())
DriverFromDictionary['All'] = pd.Series(dict(allwords.most_common()))
for i in range(0,len(data.From.unique())):
    DriverName = data.From.unique()[i]
    if DriverName != None and DriverName != '':
        dataFromDriver = data[data.From == DriverName]
        dataFromDriver.index = range(0,len(dataFromDriver.Driver))
        dataFromDriver.Message = [str(dataFromDriver.Message[j]).lower().translate(str.maketrans('','',string.punctuation)) for j in range(0,len(dataFromDriver.Message))]
        
        TotalText = " ".join(dataFromDriver.Message)
        words = Counter()
        words.update(TotalText.split())
        DriverFromDictionary[DriverName] = pd.Series(dict(words.most_common()))
    else:
        continue
DriverFromDictionary = DriverFromDictionary.fillna(0)
DriverToDictionary = pd.DataFrame(columns = data.To.unique())
DriverToDictionary['All'] = pd.Series(dict(allwords.most_common()))
for i in range(0,len(data.To.unique())):
    DriverName = data.To.unique()[i]
    if DriverName != None and DriverName != '':
        dataToDriver = data[data.To == DriverName]
        dataToDriver.index = range(0,len(dataToDriver.Driver))
        dataToDriver.Message = [str(dataToDriver.Message[j]).lower().translate(str.maketrans('','',string.punctuation)) for j in range(0,len(dataToDriver.Message))]
        
        TotalText = " ".join(dataToDriver.Message)
        words = Counter()
        words.update(TotalText.split())
        DriverToDictionary[DriverName] = pd.Series(dict(words.most_common()))
    else:
        continue
DriverToDictionary = DriverToDictionary.fillna(0)        


#def generateWordCloud(nameDriver,DriverDictionary):
#    plt.figure()
#    wc = WordCloud().generate_from_frequencies(DriverDictionary[nameDriver])
#    plt.imshow(wc,interpolation = 'bilinear')
#    plt.axis('off')
#
#generateWordCloud('Romain Grosjean',DriverFromDictionary)
#generateWordCloud('Max Verstappen',DriverFromDictionary)
#generateWordCloud('Lewis Hamilton',DriverFromDictionary)















import re
import string
import sys
import warnings
from random import Random

import nltk
import numpy as np
from nltk import ngrams
from nltk.stem.snowball import SnowballStemmer

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)


class CleanMachine:
    def __init__(self):
        """
        Initializes a SentenceCleaner object
        """
        return

    def clean(self, documents: list, sample: int = -1, seed=1, remove_punctuation: bool = True, language='german',
              stopwords=nltk.corpus.stopwords.words('german'), stemmer=SnowballStemmer('german'),
              min_n_words: int = 1, max_n_words: int = sys.maxsize):
        """
        Overall function which cleans a list of documents by applying the following pipeline:
        - removes leading, trailing and multiple whitespaces
        - removes sentences containing offensive language
        - remove sentences that are too short or too long
        - sample
        - set to lowercase
        - deal with accents
        - remove punctuation
        - remove stopwords
        - stemming


        :param documents: list of documents to clean
        :param sample: sampling rate
        :param seed: seed used for sampling rate
        :param remove_punctuation: whether punctuation should be removed
        :param language: language of the text
        :param stopwords: stopwords to remove
        :param stemmer: stemmer to use
        :param min_n_words: minimum size of a sentence
        :param max_n_words: maximum size of a sentence
        :return: the sentences in their cleaned and original format
        """

        docs = documents

        # removes leading, trailing and multiple whitespaces
        # removes sentences containing offensive language

        offensive_words = []
        docs = self.sanitize(documents=docs, forbidden_words=offensive_words)

        # remove sentences that are too short or too long

        docs = self.remove_sentences(documents=docs, min_n_words=min_n_words, max_n_words=max_n_words)
        original_docs = docs

        # sample
        docs = self.sample(documents=docs, sample=sample, seed=seed)
        original_docs = self.sample(documents=original_docs, sample=sample, seed=seed)

        # set to lowercase
        docs = self.to_lowercase(docs)

        # deal with accents
        docs = self.clean_accents(docs)

        if remove_punctuation:
            # remove punctuation
            docs = self.remove_punctuation(docs)
        # remove stopwords
        docs = self.remove_stopwords(docs, stopwords=stopwords)
        # stemming
        docs = self.stem(docs, stemmer=stemmer)
        return docs, original_docs

    def sanitize(self, documents: list, forbidden_words: list):
        """
        Removes leading and trailing white spaces and transform multiple whitespaces into one
        Remove sentences containing forbidden words
        Removes empty sentences
        :param documents: list of documents to sanitize
        :param forbidden_words: list of forbidden words
        :return: sanitized documents
        """
        return [re.sub(r'\s+', ' ', str(x).strip()) for x in documents
                if not any(substring in str(x) for substring in forbidden_words) and str(x) != 'nan']

    def get_gensim_dictionary(self, documents: list, ngrams_min=1, ngrams_max=1):
        """
        Generates a gensim dictionary object of the ngrams of a list of documents
        Gensim dictionary are needed when dealing with gensim topic modelling

        :param documents: list of documents
        :param ngrams_min: minimum ngram
        :param ngrams_max: maximum ngram
        :return: gensim Dictionary
        """

        docs_ngrams = [[] for doc in documents]

        for i in range(0, len(documents)):
            for ngram in range(ngrams_min, ngrams_max + 1):
                list_of_ngrams = ngrams(documents[i].split(), ngram)
                to_add = [list(item) for item in list_of_ngrams]
                to_add = [" ".join(word) for word in to_add]
                docs_ngrams[i].extend(to_add)

        return corpora.Dictionary(docs_ngrams)

    def sample(self, documents: list, sample: int = -1, seed=1):
        """
        sample sentences
        if the sample is between 0 and 1, it will sample a percentage, otherwise, it will
        sample that number
        :param documents: list of documents to sample
        :param sample:  sampling rate/number
        :param seed: seed used for the sampling
        :return:
        """

        rnd = Random(seed)

        # Keep sample between acceptable bounds
        if sample < 0 or sample > len(documents):
            sample = len(documents)

        indices = [True] * sample
        indices.extend([False] * (len(documents) - sample))
        indices = np.array(indices)

        # Use shuffled indices to make sure both lists are sampled the same way
        rnd.shuffle(indices)
        return list(np.array(documents)[indices])

    def remove_sentences(self, documents: list, min_n_words: int = 1, max_n_words: int = sys.maxsize):
        """
        Removes documents based on the number of words they contained
        :param documents: list of documents to be inspected
        :param min_n_words: minimum number of words allowed in a sentence
        :param max_n_words: maximum number of words allowed in a sentence
        :return: sentences which satisfy these conditions
        """
        docs = self.tokenize(documents)
        # Computes the indices that need to be kept
        to_keep = np.array([len(tokens) <= max_n_words and len(tokens) >= min_n_words for tokens in docs])

        return list(np.array(documents)[to_keep])

    def to_lowercase(self, documents: list):
        """
        sets all documents to lowercase
        :param documents: documents to be inspected
        :return: list of documents in lowercase
        """
        docs = [x.lower() for x in documents]
        return docs

    def tokenize(self, documents: list, language: str = 'german'):
        """
        Tokenizes a list of documents (i.e. extract words from it)
        Only tokenizes if it hasn't been done yet
        :param documents: list of documents to be tokenized
        :param language: specific language used for tokenization
        :return: tokenized documents
        """

        docs = []
        for sent in documents:
            if type(sent) is str:
                docs.append(nltk.word_tokenize(sent, language=language))
            elif type(sent) is list:
                docs.append(sent)
            else:
                raise ValueError('Expected either a string or a list of tokenized values')

        return docs

    def remove_punctuation(self, documents: list):
        """
        Remove punctuation from a list of documents
        :param documents: list of documents to clean
        :return: cleaned list of documents
        """

        # Special characters to also be removed
        special_characters = '😡😠👚👢•👡„👚�💲👜◾“👛️️👕👕?👑👑³💳👚👡👡´💳💲'

        documents = [sent.translate(str.maketrans('', '', string.punctuation + special_characters))
                     for sent in documents]
        return documents

    def remove_stopwords(self, documents: list, stopwords: list = nltk.corpus.stopwords.words('german')):
        """
        Removes stopwords from a list of documents
        The documents are assumed to be in lowercase
        :param documents: list of documents to be inspected
        :param stopwords: list of stopwords to remove
        :return: list of documents without stopwords
        """

        # Add custom stopwords to remove
        specific_stopwords = ['tinka', 'hallo', 'bitte', 'hey', 'hello', 'gern', 'mochte']
        stopwords.extend(specific_stopwords)

        temp = []
        documents = self.tokenize(documents)
        for doc in documents:
            _temp = []
            for word in doc:
                if not (word in stopwords):
                    _temp.append(word)
            temp.append(_temp)

        # Patch the tokens back together
        return [" ".join(tokens) for tokens in temp]

    def stem(self, documents: list, stemmer=SnowballStemmer("german")):
        """
        Stem a list of documents

        :param documents: list of documents to be stemmed
        :param stemmer: specific stemmer to be used
        :return: stemmed tokenized sentences
        """

        stemmed = []
        documents = self.tokenize(documents)
        for sent in documents:
            _stemmed = []
            for item in sent:
                _stemmed.append(stemmer.stem(item))
            stemmed.append(_stemmed)

        # Patch the stemmed tokens back together
        stemmed_docs = [" ".join(sent) for sent in stemmed]
        return stemmed_docs

    def term_frequency(self, documents: list, word: str):
        """
        Computes the frequency of a term in a list of documents
        :param documents: list of documents to be inspected
        :param word: target word
        :return: the frequency at which a term appears in all documents
        """
        if word == None:
            return 0

        frequency = sum([self.tokenize(x).count(word) for x in docs])
        return frequency

    def num_docs_containing(self, documents: list, word: str):
        """
        finds the number of documents containing a specific word/string
        :param documents: list of documents to be inspected
        :param word: target word to find in documents
        :return: number of documents containing the target word
        """

        count = 0
        for document in documents:
            if self.term_frequency(word, document) > 0:
                count += 1
        return count

    def word_count(self, documents: list):
        """
        counts the number of words in each document and returns it as a list
        :param documents: list of documents to be inspected
        :return: list of number of words in each document
        """

        return [len(self.tokenize(x)) for x in documents]

    def clean_accents(self, documents: list):
        """
        Normalizes umlauts, ß, and other accents
        :param documents: list of documents to be cleaned
        :return: cleaned documents (in the same order)
        """

        docs = documents

        # Defines the desired mapping
        umlaut_dictionary = {u'ä': 'a', u'ö': 'o', u'ü': 'u', u'ß': 'ss', u'ì': 'i', u'ı': 'i', u'é': 'e'}
        umap = {ord(key): val for key, val in umlaut_dictionary.items()}

        # Executes the mapping
        cleaned_docs = [d.translate(umap) for d in docs]

        return cleaned_docs  # Script of typical usage


if __name__ == '__main__':
    docs = data.Message
    pr = CleanMachine()
    cleaned_docs, original_docs = pr.clean(documents=docs, sample=10000)

    print('Number of documents = ' + str(len(original_docs)))
    print('Number of cleaned documents = ' + str(len(cleaned_docs)))

    data['CleanedData'] = cleaned_docs
#    for i in range(0, len(original_docs)):
#        print('Original document = ' + original_docs[i])
#        print("Cleaned document = " + cleaned_docs[i])
    
    tf = TfidfVectorizer(analyzer='word', ngram_range=(1,3), min_df = 0, stop_words = 'english')
    tfidf_matrix =  tf.fit_transform(cleaned_docs)
    km = KMeans(n_clusters=30, init='k-means++', max_iter=100, n_init=1,verbose=True)
    ac = AgglomerativeClustering(n_clusters = 30)
    km.fit(tfidf_matrix)
    ac.fit(tfidf_matrix.toarray())
    data['KmeansClusterNums'] = km.labels_
    data['AggloClusterNums'] = ac.labels_


from nltk.sentiment.vader import SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()
sentences = data.Message[data.Driver == 'Max Verstappen']
pos = []
neg = []
for sentence in sentences:
    print(sentence)
    ss = sid.polarity_scores(sentence)
    for k in sorted(ss):
        pos.append(ss['pos'])
        neg.append(ss['neg'])
        print('{0}: {1}, '.format(k, ss[k]), end='')
        print()
        
def MatchTeamRadioWithResultData(Radio,Results):
    Radio.index = range(0,len(Radio.Driver))
    Results.index = range(0,len(Results.DRIVER))
    racenames = []
    filelistR = os.listdir(PATH + 'Results')
    for i in range(0,len(filelistR)):
        racenameR = filelistR[i].replace('2017-','')
        racenameR = racenameR.replace('-Results.xlsx','')
        racenames.append(racenameR)
        
    drivernames = []
    drivernames.append(Radio.To.unique().tolist())
    drivernames.append(Radio.From.unique().tolist())
    drivernames = list(set().union(drivernames[0],drivernames[1]))
    Radio['Team'] = [None]*len(Radio.Driver)
    for i in range(0,len(drivernames)):
        team = None
        for j in range(0,len(Results.DRIVER)):
            if Results.DRIVER[j] == drivernames[i]:
                team = Results.CAR[j]
                j = len(Results.Driver)
                for k in range(0,len(Radio.Driver)):
                    if Radio.Driver[k] == drivernames[i]:
                        Radio.Team[k] = team
            
    
#plt.plot(neg)
#plt.plot(pos)
<<<<<<< HEAD
import time
   

=======
#import time
#   
#
>>>>>>> 55871e18902c45c12ef9232263da62c263a6db54
#for i in range(0,len(pos)):
#    N = 2
#    ind =np.arange(N)  # the x locations for the groups
#    width = 0.27       # the width of the bars
#    fig = plt.figure()
#    
#    ax = fig.add_subplot(111)
#    yvals = [pos[i],neg[i]]
#    rects1 = ax.bar(ind, yvals, width, color='r')
#    ax.set_ylabel('probability')
#    ax.set_xticks(ind)
#    ax.set_xticklabels( ('Positive', 'Negative') )
#    ax.set_ylim([0,0.8])
#    def autolabel(rects):
#        for rect in rects:
#            h = rect.get_height()
#            ax.text(rect.get_x()+rect.get_width()/2., 1.05*h, '%d'%int(h),
#                    ha='center', va='bottom')
#    autolabel(rects1)  
#    plt.show()
<<<<<<< HEAD
#      time.sleep(0.3)  
=======
#    time.sleep(0.3)  
>>>>>>> 55871e18902c45c12ef9232263da62c263a6db54












