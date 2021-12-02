import numpy as np
import pandas as pd
import cv2
import pytesseract

#Opening an image of our choice
img = cv2.imread("./Selected/052.jpeg")

#cv2.imshow("Business Card", img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

#Converting the image to a data format
data = pytesseract.image_to_data(img)
#Creating data List
dataList = list(map(lambda x: x.split("\t"), data.split("\n")))

#Creating a data frame for easier iteration, readability and simpler code, and deleting empty rows
dataFrame = pd.DataFrame(dataList[1:], columns = dataList[0])
dataFrame.dropna(inplace=True)
col_int = ['level','page_num','block_num','par_num','line_num','word_num','left','top','width','height','conf']
dataFrame[col_int] = dataFrame[col_int].astype(int)
print(dataFrame)

#Creating a dummy image
image = img.copy()
#Showing that we can collect text from an image by making rectangles around them
level = "word"
for l,x,y,w,h,c,txt in dataFrame[['level','left','top','width','height','conf','text']].values:
    if level == "page":
        if l == 1:
            cv2.rectangle(image, (x,y), (x+w,y+h), (0,0,0), 5)
        else:
            continue
    elif level == "block":
        if l == 2:
            cv2.rectangle(image, (x,y), (x+w,y+h), (255,0,0), 2)
        else:
            continue
    elif level == "para":
        if l == 3:
            cv2.rectangle(image, (x,y), (x+w,y+h), (0,255,0), 2)
        else:
            continue
    elif level == "line":
        if l == 4:
            cv2.rectangle(image, (x,y), (x+w,y+h), (0,0,255), 2)
        else:
            continue
    elif level == "word":
        if l == 5:
            cv2.rectangle(image, (x,y), (x+w,y+h), (0,0,0), 2)
            cv2.putText(image,txt,(x,y),cv2.FONT_HERSHEY_PLAIN,1,(255,0,0),2)
        else:
            continue

#Displaying final        
cv2.imshow("Bounded Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
