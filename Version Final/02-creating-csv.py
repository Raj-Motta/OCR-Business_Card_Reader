import numpy as np
import pandas as pd
import cv2
import pytesseract

import os
from glob import glob
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")

#Creating a list of all cards in the data folder
imgPaths = glob("./Selected/*.jpeg")
#Creating a frame with id, text format
allBusinessCard = pd.DataFrame(columns=['id','text'])

#Entering values into allBusinessCard data frame
for imgPath in  tqdm(imgPaths,desc='BusinessCard'):
    
    #imgPath = imgPaths[0]
    _, filename = os.path.split(imgPath)
    # extract data and text 
    image = cv2.imread(imgPath)
    data = pytesseract.image_to_data(image)
    dataList = list(map(lambda x: x.split('\t'),data.split('\n')))
    dataFrame = pd.DataFrame(dataList[1:],columns=dataList[0])
    dataFrame.dropna(inplace=True)
    dataFrame['conf'] = dataFrame['conf'].astype(int)

    useFulData = dataFrame.query('conf >= 30')

    # Dataframe
    businessCard = pd.DataFrame()
    businessCard['text'] = useFulData['text']
    businessCard['id'] = filename
    
    # concatenation
    allBusinessCard = pd.concat((allBusinessCard,businessCard))

#Creating CSV file of allBusinessCard data frame
allBusinessCard.to_csv('businessCard.csv',index=False)
