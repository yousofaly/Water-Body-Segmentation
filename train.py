import generator
#!pip install keras-segmentation #, from https://github.com/divamgupta/image-segmentation-keras
from keras_segmentation.models import unet
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from tensorflow.keras.metrics import MeanIoU
from tensorflow.keras.optimizers import Adam, SGD

#train,validate,test(evaluate) split
t,v,e = 0.7,0.2,0.1
tstart, tstop = 0, int(len(files)*t)
vstart, vstop = tstop, int(len(files)*(t+v))
estart, estop = vstop, len(files)
#print (tstart,tstop,vstart,vstop,estart,estop)

#define train and val data generator 
n_classes = 1
ih,iw = 512,512
batch_size = 4
n_epochs = 100
train_gen = generator(files, start = tstart, stop = tstop, bs = batch_size)
val_gen = generator(files, start = vstart, stop = vstop, bs = batch_size)

#define model parameters
opt = Adam(lr = 1e-5)
if n_classes == 1:
    loss = 'binary_crossentropy' # only 1 class
else:
    loss = 'categorical_crossentropy'
metrics = MeanIoU(num_classes = n_classes)

#make a directory to save results
save_folder = os.path.join('results')
if not os.path.exists(save_folder):
    os.makedirs(save_folder)
    
#deifne call backs
checkpointer = ModelCheckpoint(os.path.join(save_folder,'best_model.h5'),
                               monitor = 'val_loss', verbose = 1, save_best_only = True)

csv_logger = CSVLogger(os.path.join(save_folder,'log.out'), append=True, separator=';')

early_stopper = EarlyStopping(monitor = 'val_loss', patience = 10)

callbacks_list = [checkpointer, csv_logger, early_stopper]

#build model and compile
model = unet.unet(n_classes, input_height = ih, input_width = iw)
model.compile(loss = loss, optimizer = opt, metrics = [metrics])
#model.summary()

history = model.fit(train_gen, epochs = n_epochs,
                   steps_per_epoch = (tstop // batch_size),
                   validation_data = val_gen,
                   validation_steps = ((vstop - vstart) // batch_size),
                   callbacks = callbacks_list, verbose = 1)
