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

#define function to get paths from data folder
def get_paths():
  image_paths = []
  mask_paths = []
  for p in sorted(os.listdir('Data/Images')):
    image_paths.append(os.path.join('Data','Images',p))
  for p in sorted(os.listdir('Data/Masks'):
    mask_paths.append(os.path.join('Data','Masks',p))
  
  return (image_paths, mask_paths)
                  
#custom metric and loss function 
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
  return (1-dice_coef(y_true, y_pred))
