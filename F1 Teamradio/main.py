# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 21:13:10 2018

@author: dbn
"""
import os
import pandas as pd
PATH = 'C:\\Users\\dbn\\Desktop\\F1 Teamradio\\'
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