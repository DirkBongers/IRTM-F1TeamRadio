# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 21:13:10 2018

@author: dbn
"""
import os
import pandas as pd
PATH = 'C:\\Users\\dbn\\Desktop\\IRTM-F1TeamRadio\\F1 Teamradio\\'
filelist = os.listdir(PATH + 'Transcripts')
data = pd.read_excel(PATH+'Transcripts\\'+ filelist[0])
for i in range(1,19):
    df = pd.read_excel(PATH+'Transcripts\\'+ filelist[i])
    print(df.columns.values)
    print(filelist[i])
    data = pd.concat([data,df])
data = data.dropna(axis=0,how='any')
data.index = range(0,len(data.Driver))

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


def generateWordCloud(nameDriver,DriverDictionary):
    plt.figure()
    wc = WordCloud().generate_from_frequencies(DriverDictionary[nameDriver])
    plt.imshow(wc,interpolation = 'bilinear')
    plt.axis('off')

generateWordCloud('Romain Grosjean',DriverFromDictionary)
generateWordCloud('Max Verstappen',DriverFromDictionary)
generateWordCloud('Lewis Hamilton',DriverFromDictionary)
