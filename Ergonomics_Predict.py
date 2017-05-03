# -*- coding: utf-8 -*-
"""
Created on Mon May  1 18:00:42 2017

@author: amarendra
"""
#Approach: We tokenize the sentences and create unigram/biograms etc and develop a tfidf matrix
#For each document we also have a label. Each record is a document and each column is a word present
#in the entire corpus. Each cell is a tfidf value showing the significance of the word to that document
import pandas as pd
import os
from time import time
from sklearn.feature_extraction.text import TfidfVectorizer
os.getcwd()
dataframe=pd.read_csv("Corpus1.csv", delimiter = ",", header = None) 
print(dataframe)
print (dataframe.shape)
dataframe
dataframe.columns=['Text'] #Changing Column names 
type(dataframe) #Type is a data frame
dataframe.head(5) #Displays first five records
Text_data = dataframe['Text']
print(Text_data)
print(len(Text_data))
t0 = time()
for x in range(len(Text_data.index)):
    print (Text_data.iloc[x])
print("Extracting tf-idf features...")
#First we initiate an empty tfidf object with specific conditions
tfidf_vectorizer = TfidfVectorizer(ngram_range=(1,3))#max_df=0.95, min_df=2, stop_words='english' #USE HELP TO SEE WHAT EACH DOES)
t0 = time()
#Next we give the data for processing
tfidf = tfidf_vectorizer.fit_transform(Text_data)
print("done in %0.3fs." % (time() - t0))
dense = tfidf.todense()
dense.shape
feature_names = tfidf_vectorizer.get_feature_names()
print(len(feature_names))
feature_names[13:20]
def search():
    new = input('Enter Your Question:')
    response = tfidf_vectorizer.transform([new])
    from sklearn.metrics.pairwise import cosine_similarity
    map(lambda x: cosine_similarity(response, x), dense)
    from sklearn.neighbors import KNeighborsClassifier
    model = KNeighborsClassifier(n_neighbors=1, algorithm='brute')
    
    #This is how an ML algorithm is instantiated.
    from sklearn.neighbors import KNeighborsClassifier
    model = KNeighborsClassifier(n_neighbors=1, algorithm='brute')
    #This is how any ML algorithm is trained.
    model.fit(dense,Text_data)
    #This is how such an ML algorithm is used for prediction
    return(model.predict(response))

choice = ""
stop_code = 'No'
while choice != stop_code:
    result = search()
    print(result)
    choice =  input("You want to Continue with next Question Yes/No :")     