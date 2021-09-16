# py imports
import os
import sys
import glob
from argparse import ArgumentParser

# third party
import tensorflow as tf
import scipy.io as sio
import numpy as np
import cv2
from keras.backend.tensorflow_backend import set_session
from scipy.interpolate import interpn
from sklearn import metrics
from lifelines.utils import concordance_index

import networks
import datagenerators


def dice(vol1, vol2, labels=None, nargout=1):
    
    if labels is None:
        labels = np.unique(np.concatenate((vol1, vol2)))
        labels = np.delete(labels, np.where(labels == 0))  # remove background

    dicem = np.zeros(len(labels))
    for idx, lab in enumerate(labels):
        vol1l = vol1 == lab
        vol2l = vol2 == lab
        top = 2 * np.sum(np.logical_and(vol1l, vol2l))
        bottom = np.sum(vol1l) + np.sum(vol2l)
        bottom = np.maximum(bottom, np.finfo(float).eps)  # add epsilon.
        dicem[idx] = top / bottom

    if nargout == 1:
        return dicem
    else:
        return (dicem, labels)


def test(train_dir,
         test_dir,
         device, 
         load_model_file
         ):
    
    # prepare data files
    # inside the folder are npz files with the 'vol' and 'label'.
    train_vol_names = glob.glob(os.path.join(train_dir, '*.npz'))
    assert len(train_vol_names) > 0, "Could not find any training data"
    test_vol_names = glob.glob(os.path.join(test_dir, '*.npz'))
    assert len(test_vol_names) > 0, "Could not find any testing data"

    
    # image size
    vol_size = [96,128,144]
    # Clinical feature size
    Clinic_size = 4
    
    
    # device handling
    if 'gpu' in device:
        if '0' in device:
            device = '/gpu:0'
        if '1' in device:
            device = '/gpu:1'
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        set_session(tf.Session(config=config))
    else:
        device = '/cpu:0'
    
    
    # load weights of model
    with tf.device(device):
        net = networks.Multitask_framework(vol_size, Clinic_size)
        #net = networks.Multitask_cascaded_DenseNet(vol_size, Clinic_size)
        #net = networks.Multitask_no_DenseNet(vol_size, Clinic_size)
        #net = networks.Singletask_Segmentation(vol_size, Clinic_size)
        #net = networks.Singletask_Survival(vol_size, Clinic_size)
        #net = networks.Singletask_DenseNet(vol_size, Clinic_size)
        net.load_weights(load_model_file)

        
    # evaluate on training set
    Dice_train_result = []
    Survival_train_result = []
    Survival_train_label = []
    for train_image in train_vol_names:
        
        # load subject
        PT, CT, Seg, Label, Clinic = datagenerators.load_example_by_name(train_image)
        
        PT = PT[:,26:122,10:138,0:144,:]
        CT = CT[:,26:122,10:138,0:144,:]
        Seg = Seg[26:122,10:138,0:144]
        Survival_train_label.append(Label)
        
        with tf.device(device):
            pred = net.predict([PT,CT,Clinic])
            
            Seg_pred = pred[0][0,...,0]
            Survival_pred = -pred[1][0,0]
            
            #Seg_pred = pred[0,...,0]
            #Survival_pred = 0
            
            #Seg_pred = np.zeros(vol_size)
            #Survival_pred = -pred[0,0]
        
        _, Seg_pred = cv2.threshold(Seg_pred,0.5,1,cv2.THRESH_BINARY)
        Dice_vals = dice(Seg_pred, Seg, labels=[1])
        Dice_train_result.append(Dice_vals)
        Survival_train_result.append(Survival_pred)
        
    
    # evaluate on testing set     
    Dice_test_result = []
    Survival_test_result = []
    Survival_test_label = []
    for test_image in test_vol_names:
        
        # load subject
        PT, CT, Seg, Label, Clinic = datagenerators.load_example_by_name(test_image)
        
        PT = PT[:,26:122,10:138,0:144,:]
        CT = CT[:,26:122,10:138,0:144,:]
        Seg = Seg[26:122,10:138,0:144]
        Survival_test_label.append(Label)
        
        with tf.device(device):
            pred = net.predict([PT,CT,Clinic])
            
            Seg_pred = pred[0][0,...,0]
            Survival_pred = -pred[1][0,0]
            
            #Seg_pred = pred[0,...,0]
            #Survival_pred = 0
            
            #Seg_pred = np.zeros(vol_size)
            #Survival_pred = -pred[0,0]
        
        _, Seg_pred = cv2.threshold(Seg_pred,0.5,1,cv2.THRESH_BINARY)
        Dice_vals = dice(Seg_pred, Seg, labels=[1])
        Dice_test_result.append(Dice_vals)
        Survival_test_result.append(Survival_pred)

        
    # calculat the mean results
    print('Dice: {:.3f} ({:.3f})'.format(np.mean(Dice_train_result), np.mean(Dice_test_result)))
    
    Survival_train_label = np.array(Survival_train_label)
    Survival_test_label = np.array(Survival_test_label)
    train_cindex = concordance_index(Survival_train_label[:,0], Survival_train_result, Survival_train_label[:,1])
    test_cindex = concordance_index(Survival_test_label[:,0], Survival_test_result, Survival_test_label[:,1])
    print('C-index: {:.3f} ({:.3f})'.format(train_cindex, test_cindex))

    

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--train_dir", type=str,
                        dest="train_dir", default='./',
                        help="training folder")
    parser.add_argument("--test_dir", type=str,
                        dest="test_dir", default='./',
                        help="training folder")
    parser.add_argument("--device", type=str, default='gpu0',
                        dest="device", help="cpu or gpuN")
    parser.add_argument("--load_model_file", type=str,
                        dest="load_model_file", default='./',
                        help="optional h5 model file to initialize with")

    args = parser.parse_args()
    test(**vars(args))
