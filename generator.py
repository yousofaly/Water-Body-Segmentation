import cv2

#image generator from files df
def generator(df, ih = 1024, iw = 1024, start = 0, stop = 4, bs = 32, 
              aug = None, interp = cv2.INTER_NEAREST):
    images = []
    masks = []
    h,w = df['height'], df['width']
    while True:
        for i in range(start,stop):
            images.append(cv2.resize(cv2.imread(df['image'][i]), (ih,iw), interpolation = interp))
            masks.append(cv2.resize(cv2.imread(df['mask'][i]), (ih,iw), interpolation = interp))
            
            if len(masks) == bs:
                if aug is not None:
                    #to do add in aug
                    pass
                yield np.array(images), np.array(masks)
            
            images = []
            masks = []
