import cv2
import prediction as pred

#Opening an image, this is where the user selects the image to be captured
image = cv2.imread("./Selected/054.jpeg")
#cv2.namedWindow("Business", cv2.WINDOW_NORMAL)    
#cv2.imshow("Business", image)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

#Executing prediction code (prediction.py#getPredictions) to collect all information from an image
img_result,entities = pred.getPredictions(image)

#Displaying the final result of the info collected (This is where we take the data from and put it into the database)
cv2.namedWindow("Results", cv2.WINDOW_NORMAL)    
cv2.imshow("Results", img_result)
cv2.waitKey(0)
cv2.destroyAllWindows()
print(entities)
