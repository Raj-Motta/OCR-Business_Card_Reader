import numpy as np
import pandas as pd
import string
import re
import random
import pickle

#Opening BUSINESS CARD file which contains the information to be used to train the algorithm
with open("businessCard.txt", mode = "r", encoding = "utf8", errors = "ignore") as f:
    text = f.read()
#Creating Data List
dataList = list(map(lambda x:x.split("\t"),text.split("\n")))

#Creating Data Frame and deleting all empty rows
dataFrame = pd.DataFrame(dataList[1:], columns = dataList[0]) 
dataFrame.dropna(inplace=True)

#Code to clean the text in our dataFrame
whiteSpace = string.whitespace 
punctuation = '!#$%&\'()*+:;<=>?[\\]^`{|}~'
tableWhiteSpace = str.maketrans('','', whiteSpace)
tablePunctuation = str.maketrans('','', punctuation)
#Function to perform the cleaning process
def cleanText(text):
    text = str(text)
    text = text.lower()
    removeWhiteSpace = text.translate(tableWhiteSpace)
    removePunctuation = removeWhiteSpace.translate(tablePunctuation)

    return str(removePunctuation)

#Cleaning text
dataFrame["text"] = dataFrame["text"].apply(cleanText)
#Removing all empty rows
dataClean = dataFrame.query("text != ''")
dataClean.dropna(inplace = True)

#Grouping all information by ID (file name)
allCardsData = []
group = dataClean.groupby(by = "id")
#iterating through all files with the help of cards
#Using for loop to create a Content:Annotation format and store it in all cards data
cards = group.groups.keys()
for card in cards:
    cardData = []
    groupArray = group.get_group(card)[["text","tag"]].values
    content = ""
    annotations = {"entities": []}
    start = 0
    end = 0

    for text, label in groupArray:
        text = str(text)
        stringLen = len(text) + 1

        start = end
        end = start + stringLen

        if label != "O":
            annot = (start, end-1,label)
            annotations["entities"].append(annot)

        content = content + text + " "

    cardData = (content, annotations)
    allCardsData.append(cardData)

#Shuffling cards to a random format
random.shuffle(allCardsData)
#Dividing data into training and testing data with 90:10 ratio
TrainData = allCardsData[:240]
TestData = allCardsData[240:]

#Creating the pickle files
pickle.dump(TrainData, open("./data/TrainData.pickle", mode = "wb"))
pickle.dump(TestData, open("./data/TestData.pickle", mode = "wb"))
