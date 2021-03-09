from scipy.ndimage import zoom
import cv2

#image generator from files df
def generator(df, ih = 1024, iw = 1024, start = 0, stop = 4, bs = 32, aug = None):
    images = []
    masks = []
    h,w = df['height'], df['width']
    while True:
        for i in range(start,stop):
            images.append(zoom(cv2.imread(df['image'][i]), (ih/h[i], iw/w[i],1)))
            masks.append(zoom(cv2.imread(df['mask'][i]), (ih/h[i], iw/w[i],1)))
        
            if len(masks) == bs:
                if aug is not None:
                    pass
                yield np.array(images), np.array(masks)
            
            images = []
            masks = []
