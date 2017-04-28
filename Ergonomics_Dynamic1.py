# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 20:23:56 2017

@author: amarendra
"""
import os
import nltk
import urllib.request
from bs4 import BeautifulSoup
os.getcwd()
url = "http://newergonomics.enlightus.com/ergonomics-home.html"
html = urllib.request.urlopen(url).read()
soup = BeautifulSoup(html)
print(soup)
# Removing  All types of Elements.
for script in soup(["script", "style"]):
    script.extract()    # rip it out
# Extacting the Text From the Page
text = soup.get_text()
print(text)
# Breaking in to lines and Removing Unnecessary Spaces
lines = (line.strip() for line in text.splitlines())
# Breaking Multiple Headlines into a line each
chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
# Removing Blank Lines
text = '\n'.join(chunk for chunk in chunks if chunk)
print(text)
# Dividing into Sentences
sentences = nltk.tokenize.sent_tokenize(text)
#Printing Sentences in the file
print("***Sentences of the File***")
for sent in sentences:
    print('\n')
    print(sent),
#Tokenizing
tokens = nltk.tokenize.word_tokenize(text)
print("***Tokens of File***")
print(tokens)
# Lower Casing
lower_tokens = [token.lower() for token in tokens]
print("***After Lower Casing the Tokens***")
print(lower_tokens)
#removal stop words
from nltk.corpus import stopwords
stop = stopwords.words('english')
tokens = [token for token in lower_tokens if token not in stop]
print("***After Removal of Stop Words***")
print(tokens)
#Lemmatizing
lmtzr = nltk.stem.WordNetLemmatizer()
tokens = [lmtzr.lemmatize(token) for token in tokens]
print("***After Lemmatizing the Tokes***")
print(tokens)
#Removing Digits from the tokens
new_tokens = []
for x in tokens:
    if not x.isdigit():
       new_tokens.append(x)
print(new_tokens)
# Writing preprocessed data to file
file = open("homepage.txt","w")
file.writelines(new_tokens)
file.close()

