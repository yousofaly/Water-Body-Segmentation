import cv2
import os 
import keras.backend as K
#define cropping function 
def crop(path, mode = 'dims'):

    # Read the image, convert it into grayscale, and make in binary image for threshold value of 1.
    img = cv2.imread(path,0)

    # use binary threshold, all pixel that are beyond 3 are made white
    _, thresh = cv2.threshold(img, 3, 255, cv2.THRESH_BINARY)

    # Now find contours in it.
    contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    # get contours with highest height
    lst_contours = []
    for cnt in contours:
        ctr = cv2.boundingRect(cnt)
        lst_contours.append(ctr)
    x,y,w,h = sorted(lst_contours, key=lambda coef: coef[3])[-1]
    #get new image shape
    if mode == 'dims':
        return (x,y,w,h)
    #get cropped image
    else:
        return(cv2.imread(path)[y:y+h,x:x+w,:])
