import cv2
import numpy as np
#image generator from files df
def generator(images, masks, ih = 1024, iw = 1024,
              start = 0, stop = 4, bs = 32, aug = None,
              interp = cv2.INTER_NEAREST, rs = 255):
  image_array = []
  mask_array = []
  while True:
    for i in range(start,stop):
      image_array.append(cv2.resize(cv2.imread(images[i]), (ih,iw), interpolation = interp) / rs)
      mask_array.append(cv2.resize(cv2.imread(masks[i]), (ih,iw), interpolation = interp) / rs)
      if len(mask_array) == bs:
        if aug is not None:
          #to do implement augmentation
          pass
          
        yield np.array(image_array), np.array(mask_array)
        image_array = []
        mask_array = [
