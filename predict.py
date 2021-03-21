import os 
import matplotlib.pyplot as plt
import cv2

#function to open image, predict and display the original image, true mask and the predicted mask
image_folder = os.path.join('Data','Images')
mask_folder = os.path.join('Data','Masks')
def plot(array, title):
  plt.figure()
  plt.imshow(array)
  plt.title(title)
  plt.axis('off')

def evaluate_prediction(image_name, model, ih = 256, iw = 256,
                        mask_folder = mask_folder, image_folder = image_folder):
  
  #define image array to predict and true mask
  test_image = cv2.resize(cv2.imread(os.path.join(image_folder, image_name)), (ih,iw), interpolation = cv2.INTER_NEAREST)
  test_mask = cv2.imread(os.path.join(mask_folder, image_name))[:,:,0]
  output_shape = test_mask.shape[::-1]

  #predict on image
  prediction = model.predict(np.expand_dims(test_image, axis = 0))

  #display image, true mask and prediction
  plot(cv2.imread(os.path.join(image_folder, image_name)), 'Image')
  plot(test_mask, 'True Mask')
  plot(cv2.resize(prediction[0,:,:,0], (output_shape)), 'Prediction'
