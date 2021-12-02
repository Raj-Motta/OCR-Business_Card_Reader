import numpy as np
import pandas as pd
import cv2
import pytesseract
from glob import glob
import spacy
from spacy import displacy
import re
import string
import warnings
warnings.filterwarnings("ignore")

#Loading spacy model created after training
model_ner = spacy.load("./output/model-best")

#Function to clean the text from our images
def cleanText(text):
    whiteSpace = string.whitespace
    punctuation = '!#$%&\'()*+:;<=>?[\\]^`{|}~'
    tableWhiteSpace = str.maketrans('','', whiteSpace)
    tablePunctuation = str.maketrans('','', punctuation)
    text = str(text)
    text = text.lower()
    removeWhiteSpace = text.translate(tableWhiteSpace)
    removePunctuation = removeWhiteSpace.translate(tablePunctuation)

    return str(removePunctuation)

#Function to group sets of texts on their BIO-distribution
class groupgen():
    def __init__(self):
        self.id = 0
        self.text = 0

    def getgroup(self, text):
        if self.text == text:
            return self.id
        else:
            self.id += 1
            self.text = text
            return self.id

#Parser function to clean the text for a final time before display
def parser(text, label):
    if label == "PHONE":
        text = re.sub(r"\D\-","", text)

    elif label == "EMAIL":
        text = text.lower()
        allow_special_chars = "@_.\-"
        text = re.sub(r"[^A-Za-z0-9{}]".format(allow_special_chars), "", text)
    
    elif label == "WEB":
        text = text.lower()
        allow_special_chars = "_.\-:/%#"
        text = re.sub(r"[^A-Za-z0-9{}]".format(allow_special_chars), "", text)
    
    elif label in ("NAME", "DES"):
        text = text.lower()
        text = re.sub(r"[^a-z ]", "", text)
        text = text.title()

    elif label == "ORG":
        text = text.lower()
        text = re.sub(r"[^a-z0-9 ]", "", text)
        text = text.title()
        
    return text

#Creating groupgen class object
grp_gen = groupgen()

#function to create and display the predictions after processing an image
def getPredictions(image):
    #Converting image to data
    tessdata = pytesseract.image_to_data(image)
    #Converting data to data list
    dataList = list(map(lambda x: x.split("\t"), tessdata.split("\n")))
    
    #Creating a data frame for easier iteration, readability and simpler code, and deleting empty rows
    dataFrame = pd.DataFrame(dataList[1:], columns = dataList[0])
    dataFrame.dropna(inplace=True)
    #Passing all dataframe text to cleanText() for cleaning
    dataFrame["text"] = dataFrame["text"].apply(cleanText)
    #Removing all empty rows created after cleaning
    dataClean = dataFrame.query("text != ''")
    #Creating content variable that stores all the text provided by the image in a readable string format
    content = " ".join([w for w in dataClean["text"]])
    #processing content through the spacy model for recognition and classification
    doc = model_ner(content)

    #Converting recognised text into a JSON format
    doc_json = doc.to_json()
    doc_text = doc_json["text"]
    #Creating dataFrame tokens for better readability and easier coding
    dataFrame_token = pd.DataFrame(doc_json["tokens"])
    dataFrame_token["token"] = dataFrame_token[["start", "end"]].apply(lambda x:doc_text[x[0]:x[1]], axis = 1)
    right_table = pd.DataFrame(doc_json["ents"])[["start", "label"]]
    dataFrame_token = pd.merge(dataFrame_token, right_table, how = "left", on = "start")
    dataFrame_token.fillna("O", inplace = True)

    #Spotting the starting and ending points of every word
    dataClean["end"] = dataClean['text'].apply(lambda x: len(x)+1).cumsum() - 1
    dataClean["start"]=dataClean[['text','end']].apply(lambda x: x[1]-len(x[0]),axis=1)

    #Creating common dataframe to store all information about the image
    dataFrame_info = pd.merge(dataClean, dataFrame_token[["start", "token", "label"]], how = "inner", on = "start")

    #Taking information from dataFrame_info and storing it in boundingBoxFrame, condition is to remove all rows that have the label "O" as they're not needed
    boundingBoxFrame = dataFrame_info.query("label != 'O'")

    #Creating dummy image
    #img = image.copy()

    #for x, y, w, h, label in boundingBoxFrame[["left", "top", "width", "height", "label"]].values:
    #    x = int(x)
    #    y = int(y)
    #    w = int(w)
    #    h = int(h)

    #   cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 3)
    #    cv2.putText(img, str(label), (x,y), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 2)

    #Combining all of the information to form single text pieces
    boundingBoxFrame["label"] = boundingBoxFrame["label"].apply(lambda x: x[2:])
    boundingBoxFrame["group"] = boundingBoxFrame["label"].apply(grp_gen.getgroup)
    boundingBoxFrame[["left", "top", "width", "height"]] = boundingBoxFrame[["left", "top", "width", "height"]].astype(int)
    boundingBoxFrame["right"] = boundingBoxFrame["left"] + boundingBoxFrame["width"]
    boundingBoxFrame["bottom"] = boundingBoxFrame["top"] + boundingBoxFrame["height"]
    
    col_group = ["left", "top", "right", "bottom", "label", "token", "group"]
    group_tag_img = boundingBoxFrame[col_group].groupby(by = "group")
    group_tag_img
    img_tagging = group_tag_img.agg({
        "left" : min,
        "right": max,
        "top" : min,
        "bottom": max,
        "label": np.unique,
        "token":lambda x: " ".join(x)
    })
    
    #Creating dummy image
    img_bounded = image.copy()
    #For output
    for l,r,t,b,label,token in img_tagging.values:
        cv2.rectangle(img_bounded, (l,t), (r,b), (255, 0, 0), 3)
        cv2.putText(img_bounded, label,(l,t), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255))

    #Storing all information in a format that can easily be stored in a database
    info_array = dataFrame_info[["token", "label"]].values
    entities = dict(NAME=[], ORG=[], DES=[], PHONE=[], EMAIL=[], WEB = [])
    previous = "O"

    for token, label in info_array:
        bio_tag = label[0]
        label_tag = label[2:]

        text = parser(token, label_tag)

        if bio_tag in ("B", "I"):
            if previous != label_tag:
                entities[label_tag].append(text)
            else:
                if bio_tag == "B":
                    entities[label_tag].append(text)
                else:
                    if label_tag in ("NAME", "ORG", "DES"):
                        entities[label_tag][-1] = entities[label_tag][-1] + " " + text
                    else:
                        entities[label_tag][-1] = entities[label_tag][-1] + text

        previous = label_tag

    #Returning information for user to use as they wish
    return img_bounded, entities
