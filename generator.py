import cv2

#image generator from files df
def generator(df, ih = 1024, iw = 1024, start = 0, stop = 4, bs = 32, 
              aug = None, interp = cv2.INTER_NEAREST, rs = 255):
    images = []
    masks = []
    h,w = df['height'], df['width']
    while True:
        for i in range(start,stop):
            images.append(cv2.resize(cv2.imread(df['image'][i]), (ih,iw), interpolation = interp) / rs)
            masks.append(cv2.resize(cv2.imread(df['mask'][i]), (ih,iw), interpolation = interp) / rs)
            
            if len(masks) == bs:
                if aug is not None:
                    #to do implement augmentation
                    pass
                yield np.array(images), np.array(masks)
            
                images = []
                masks = []
