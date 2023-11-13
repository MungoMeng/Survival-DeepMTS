# python imports
import os
import glob
import sys
import random
from argparse import ArgumentParser
import matplotlib
matplotlib.use('Agg')

# third-party imports
import tensorflow as tf
import numpy as np
from keras.backend.tensorflow_backend import set_session
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping, LearningRateScheduler
from keras.utils import multi_gpu_model

# project imports
import datagenerators
import networks
import losses
import metrics

def lr_scheduler(epoch):

    if epoch < 50:
        lr = 1e-4
    elif epoch < 100:
        lr = 5e-5
    elif epoch < 200:
        lr = 1e-5
    else:
        lr = 1e-6
    print('lr: %f' % lr)
    return lr


class ParallelModelCheckpoint(ModelCheckpoint):
    def __init__(self,model,filepath, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1):
        self.single_model = model
        super(ParallelModelCheckpoint,self).__init__(filepath, monitor, verbose,save_best_only, save_weights_only,mode, period)

    def set_model(self, model):
        super(ParallelModelCheckpoint,self).set_model(self.single_model)


def train(train_dir,
          valid_dir,
          model_dir,
          device,
          lr,
          nb_epochs,
          steps_per_epoch,
          validation_steps,
          batch_size,
          validation_size,
          load_model_file,
          initial_epoch):
    
 
    # image size
    vol_size = [128,128,128]
    # Clinical feature size
    Clinic_size = 1
    
 
    # prepare data files
    train_vol_names = glob.glob(os.path.join(train_dir, '*.npz'))
    assert len(train_vol_names) > 0, "Could not find any training data"
    random.shuffle(train_vol_names)  # shuffle volume list
    
    valid_vol_names = glob.glob(os.path.join(valid_dir, '*.npz'))
    assert len(valid_vol_names) > 0, "Could not find any validation data"
    random.shuffle(valid_vol_names)  # shuffle volume list
    
    
    # prepare model folder
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)

    
    # device handling
    if device == 'gpu0':
        device = '/gpu:0'
    if device == 'gpu1':
        device = '/gpu:1'
    if device == 'multi-gpu':
        device = '/cpu:0'
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    set_session(tf.Session(config=config))  
        
        
    # prepare the model
    with tf.device(device):
        model = networks.DeepMTS(vol_size, Clinic_size)

        # load initial weights
        if load_model_file != './':
            print('loading', load_model_file)
            model.load_weights(load_model_file, by_name=True) 
         
    # Multiple GPUs used       
    if device == '/cpu:0':
        Parallelmodel = multi_gpu_model(model, gpus=2)  
            
            
    # data generator
    train_example_gen = datagenerators.example_gen(train_vol_names, batch_size=batch_size, balance_class=True)
    data_gen_train = datagenerators.gen_train(train_example_gen, Sort=True)
    
    valid_example_gen = datagenerators.example_gen(valid_vol_names, batch_size=validation_size, balance_class=False)
    data_gen_valid = datagenerators.gen_valid(valid_example_gen, Sort=True)
 
    
    # callback settings
    save_file_name = os.path.join(model_dir, '{epoch:02d}-{val_Survival_Cindex:.3f}-{val_Segmentation_Dice:.3f}.h5')
    if device == '/cpu:0': # Multiple GPUs used   
        save_callback = ParallelModelCheckpoint(model, save_file_name, save_best_only=False, save_weights_only=True, monitor='val_Survival_Cindex', mode='max')
    else:
        save_callback = ModelCheckpoint(save_file_name, save_best_only=False, save_weights_only=True, monitor='val_Survival_Cindex', mode='max')
    
    save_log_name = os.path.join(model_dir, 'log.csv')
    csv_logger = CSVLogger(save_log_name, append=True)
    
    early_stopping = EarlyStopping(monitor='val_Survival_Cindex', patience=50, mode='max')
    scheduler = LearningRateScheduler(lr_scheduler)
    
    
    # compile settings and fit
    if device == '/cpu:0': # Multiple GPUs used   
        Parallelmodel.compile(optimizer=Adam(lr=lr), 
                              metrics={'Segmentation':metrics.Dice,'Survival':metrics.Cindex},
                              loss=[losses.Dice_loss, losses.Cox_loss],
                              loss_weights=[1.0, 1.0])
            
        Parallelmodel.fit_generator(data_gen_train,
                                    validation_data=data_gen_valid,
                                    validation_steps=validation_steps,
                                    initial_epoch=initial_epoch,
                                    epochs=nb_epochs,
                                    callbacks=[save_callback, csv_logger, early_stopping, scheduler],
                                    steps_per_epoch=steps_per_epoch,
                                    verbose=1)
    else:
        model.compile(optimizer=Adam(lr=lr), 
                      metrics={'Segmentation':metrics.Dice,'Survival':metrics.Cindex},
                      loss=[losses.Dice_loss, losses.Cox_loss],
                      loss_weights=[1.0,1.0])
            
        model.fit_generator(data_gen_train,
                            validation_data=data_gen_valid,
                            validation_steps=validation_steps,
                            initial_epoch=initial_epoch,
                            epochs=nb_epochs,
                            callbacks=[save_callback, csv_logger, early_stopping, scheduler],
                            steps_per_epoch=steps_per_epoch,
                            verbose=1)

        
if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--train_dir", type=str,
                        dest="train_dir", default='./',
                        help="training data folder")
    parser.add_argument("--valid_dir", type=str,
                        dest="valid_dir", default='./',
                        help="validation data folder")
    parser.add_argument("--model_dir", type=str,
                        dest="model_dir", default='./models/',
                        help="models folder")
    parser.add_argument("--device", type=str, default='multi-gpu',
                        dest="device", 
                        help="gpuN or multi-gpu")
    parser.add_argument("--lr", type=float,
                        dest="lr", default=1e-4, 
                        help="learning rate")
    parser.add_argument("--epochs", type=int,
                        dest="nb_epochs", default=200,
                        help="number of epoch")
    parser.add_argument("--steps_per_epoch", type=int,
                        dest="steps_per_epoch", default=50,
                        help="iterations of each epoch")
    parser.add_argument("--validation_steps", type=int,
                        dest="validation_steps", default=10,
                        help="iterations for validation")
    parser.add_argument("--initial_epoch", type=int,
                        dest="initial_epoch", default=0,
                        help="initial epoch")
    parser.add_argument("--batch_size", type=int,
                        dest="batch_size", default=8,
                        help="batch size")
    parser.add_argument("--validation_size", type=int,
                        dest="validation_size", default=20,
                        help="validation size")
    parser.add_argument("--load_model_file", type=str,
                        dest="load_model_file", default='./',
                        help="optional h5 model file to initialize with")

    args = parser.parse_args()
    train(**vars(args))
