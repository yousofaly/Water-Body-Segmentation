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
## Files
- dice.py
- 
