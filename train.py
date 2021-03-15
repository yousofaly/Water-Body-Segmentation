from datetime import date
import processing 
import generator
from models import unet
from datetime import date
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping

image_paths, mask_pahts = processing.get_paths()

#train,validate,test(evaluate) split
t,v,e = 0.7,0.2,0.1
tstart, tstop = 0, int(len(image_paths)*t)
vstart, vstop = tstop, int(len(image_paths)*(t+v))
estart, estop = vstop, len(image_paths)

#define train and val data generator 
n_classes = 1
(ih,iw) = (256,256) #define desired input_shape
batch_size = 16
n_epochs = 100
train_gen = generator(image_paths, mask_paths, ih,iw, start = tstart, stop = tstop, bs = batch_size)
val_gen = generator(image_paths, mask_paths, ih,iw, start = vstart, stop = vstop, bs = batch_size)

#define model parameters (new)
opt = Adam(lr = 1e-5)
loss = processing.dice_coef_loss
metrics = processing.dice_coef

#make a directory to save result
d = str(date.today())
experiment_name = 'experiment name' #set experiment name
save_folder = os.path.join('results',d,experiment_name)
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

#deifne call backs
checkpointer = ModelCheckpoint(os.path.join(save_folder,'best_model.h5'),
                               monitor = 'val_loss', verbose = 1, save_best_only = True)

csv_logger = CSVLogger(os.path.join(save_folder,'log.csv'), append=True, separator=';')

early_stopper = EarlyStopping(monitor = 'val_loss', patience = 10)

callbacks_list = [checkpointer, csv_logger, early_stopper]

#build model and compile
model = unet(input_shape=(ih,iw,3))
model.compile(loss = loss, optimizer = opt)

history = model.fit(train_gen, epochs = n_epochs,
                    steps_per_epoch = (tstop // batch_size),
                    validation_data = val_gen,
                    validation_steps = ((vstop - vstart) // batch_size),
                    callbacks = callbacks_list, verbose = 1)
