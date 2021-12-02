import numpy as np
import cv2
import matplotlib.pyplot as plt
import imutils
from imutils.perspective import four_point_transform
import prediction as pred
import pytesseract

def resizer(img, width=500):
    h,w,c = img.shape
    
    height = int((h/w)*width)
    size = (width, height)
    img = cv2.resize(img, size)
    return img, size

def apply_brightness_contast(img, brightness = 0, contrast = 0):
    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255+brightness
        alpha_b = (highlight - shadow)/255
        gamma_b = shadow
        buf = cv2.addWeighted(img, alpha_b, img, 0, gamma_b)
    else:
        buf = img.copy()
    
    if contrast != 0:
        f = 131*(contrast + 127)/(127*(131-contrast))
        alpha_c = f
        gamma_c = 127*(1-f)

        buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)

    return buf

def document_scanner(original_img):
    resized_img, size = resizer(original_img)

    enhanced_img = cv2.detailEnhance(resized_img, sigma_s = 35, sigma_r = 0.15)
    grey_scale = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2GRAY)
    blurred_img = cv2.GaussianBlur(grey_scale, (5,5), 0)
    edge_detection = cv2.Canny(blurred_img, 75, 200)
    kernel = np.ones((5,5), np.uint8)
    dilated_img = cv2.dilate(edge_detection, kernel, iterations=1)
    closing = cv2.morphologyEx(dilated_img, cv2.MORPH_CLOSE, kernel)
    img_contours , hire = cv2.findContours(closing, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(img_contours, key = cv2.contourArea, reverse = True)
    for contour in contours:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02*peri, True)

        if len(approx) == 4:
            four_points = np.squeeze(approx)
            break
    
    marked_img = resized_img.copy()
    cv2.drawContours(marked_img, [four_points], -1, (255,0,0), 3)


    multiplier = original_img.shape[1] / size[0]
    four_points_original = four_points * multiplier
    four_points_original = four_points_original.astype(int)

    wrapped_img = four_point_transform(original_img, four_points_original)
    
    final_img = apply_brightness_contast(wrapped_img, 20, 30)
    return original_img, wrapped_img, final_img

img = cv2.imread("./Selected/052.jpeg")
original, wrapped, final= document_scanner(img)
origtext = pytesseract.image_to_string(original)
wraptext = pytesseract.image_to_string(wrapped)
finaltext = pytesseract.image_to_string(final)
if len(origtext) > len(finaltext):
    bounded_img, ent = pred.getPredictions(original)
elif len(wrapped) > len(finaltext):
    bounded_img, ent = pred.getPredictions(wrapped)
else:
    bounded_img, ent = pred.getPredictions(final)
print(ent)
cv2.imshow("Bounded", bounded_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
