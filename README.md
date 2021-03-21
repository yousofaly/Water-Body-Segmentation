# Water Body Segmentation
## Introduction
In this project, I train and evaluate a U-Net (implemented in Keras) to segment water bodies from satellite images. A baseline U-Net acheived a dice-score of 0.85. I have yet to add an augmentation step. 
## Data 
Data was downloaded from https://www.kaggle.com/franciscoescobar/satellite-images-of-water-bodies. It consisits of two folder (Images and Masks), each containing 2841 JPEG images (with corresponding names). They are of various image shapes. This means I needed a resizing step in my image generator.
## Training and Required Packages
I used Google Colab (and a GPU runtime) to train on 1988 images, which took about 2 seconds/batch of 3 images. Required libraries
* numpy 
* os
* keras
* cv2
* matplotlib
## Files
dice.py
* dice coefficient (metric) and dice loss (custom loss) 

generator.py
 * image generator which reads images and masks from Data/Images and Data/Masks
 * resizes to desired shape (I used 256,256)
 * yields batches upon reaching desired batch size

models.py
* here I define a U-Net and a smaller segmentation model (U-Net skinny)

train.py
 * creates.compiles model from models.py
 * train/validaton/test split (70/20/10) assuming all one folder each for images and masks 
 * train with callbacks (EarylStopper, CSVLogger, and checkpointer)

show_prediction.py
 * display image, true mask and predicted mask 
 * OR
 * display image and predicted mask if there is no correspoinding ground truth mask 

evaluate.py
 * loads desired model from results folder
 * evaluates on validation data

## Results
Baseline model (unet from models.py) acheived a dice score of 0.85. Below are some examples of the images (from the validation set), the true mask and the predicted mask.

![image](https://user-images.githubusercontent.com/56979366/111925366-9ecde600-8a76-11eb-90bd-b5a4a0ad96cb.png)

![image](https://user-images.githubusercontent.com/56979366/111925388-b5743d00-8a76-11eb-92b5-32e89d64cde2.png)

![image](https://user-images.githubusercontent.com/56979366/111925400-be650e80-8a76-11eb-9408-ec536364bc90.png)


