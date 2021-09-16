import os, sys
import random
import numpy as np
import cv2
import random

import imgaug as ia
from imgaug import augmenters as iaa
        
def gen_train(gen, Sort=True):
    while True:
        X = next(gen)
        PT = X[0]
        CT = X[1]
        Seg = X[2]
        Label = X[3]
        Clinic = X[4]
        
        #data augmentation
        PT, CT, Seg = Data_augmentation(PT, CT, Seg)
        
        PT = PT[:,26:122,10:138,0:144,:]
        CT = CT[:,26:122,10:138,0:144,:]
        Seg = Seg[:,26:122,10:138,0:144,:]
        
        # Sort samples by survival time
        if Sort == True:
            PT = Sort_by_time(PT, Label[:,0])
            CT = Sort_by_time(CT, Label[:,0])
            Seg = Sort_by_time(Seg, Label[:,0])
            Event = Sort_by_time(Label[:,1:], Label[:,0])
            Clinic = Sort_by_time(Clinic, Label[:,0])
        else:
            Event = Label[:,1:]
        
        yield ([PT, CT, Clinic], [Seg, Event])
        #yield ([PT, CT, Clinic], [Seg])
        #yield ([PT, CT, Clinic], [Event])
            
            
def gen_valid(gen, Sort=True):
    while True:
        X = next(gen)
        PT = X[0]
        CT = X[1]
        Seg = X[2]
        Label = X[3]
        Clinic = X[4]
        
        PT = PT[:,26:122,10:138,0:144,:]
        CT = CT[:,26:122,10:138,0:144,:]
        Seg = Seg[:,26:122,10:138,0:144,:]
        
        # Sort samples by survival time
        if Sort == True:
            PT = Sort_by_time(PT, Label[:,0])
            CT = Sort_by_time(CT, Label[:,0])
            Seg = Sort_by_time(Seg, Label[:,0])
            Event = Sort_by_time(Label[:,1:], Label[:,0])
            Clinic = Sort_by_time(Clinic, Label[:,0])
        else:
            Event = Label[:,1:]
        
        yield ([PT, CT, Clinic], [Seg, Event])
        #yield ([PT, CT, Clinic], [Seg])
        #yield ([PT, CT, Clinic], [Event])

        
def load_valid_examples(vol_names, Sort=True):
    
    # preload all data
    PT, CT, Seg, Label, Clinic = load_examples_by_names(vol_names)
    
    PT = PT[:,26:122,10:138,0:144,:]
    CT = CT[:,26:122,10:138,0:144,:]
    Seg = Seg[:,26:122,10:138,0:144,:]
        
    # Sort samples by survival time
    if Sort == True:
        PT = Sort_by_time(PT, Label[:,0])
        CT = Sort_by_time(CT, Label[:,0])
        Seg = Sort_by_time(Seg, Label[:,0])
        Event = Sort_by_time(Label[:,1:], Label[:,0])
        Clinic = Sort_by_time(Clinic, Label[:,0])
    else:
        Event = Label[:,1:]
    
    return PT, CT, Clinic, Seg, Event
    
    
def example_gen_rtload(vol_names, batch_size=1, balance_class=True):

    # Load data in real time
    while True:
        
        if balance_class == True:
            # manually balance class
            idxes = []
            num_pos = num_neg = 0
            while num_pos<batch_size/2 or num_neg<batch_size/2:
                idx = np.random.randint(len(vol_names))
                idx_PFS = load_volfile(vol_names[idx], np_var='PFS')
                if idx_PFS==0 and num_neg<batch_size/2:
                    idxes.append(idx)
                    num_neg = num_neg+1
                if idx_PFS==1 and num_pos<batch_size/2:
                    idxes.append(idx)
                    num_pos = num_pos+1
        else:
            idxes = np.random.randint(len(vol_names), size=batch_size)

            
        # load the selected data
        npz_data = []
        for idx in idxes:
            X = load_volfile(vol_names[idx], np_var='all')
            npz_data.append(X)
            
        X_data = []
        for i in range(batch_size):
            X = npz_data[i]['PT']
            X = X[np.newaxis, ..., np.newaxis]
            X_data.append(X)
        if batch_size > 1:
            return_vals = [np.concatenate(X_data, 0)]
        else:
            return_vals = [X_data[0]]
                           
        X_data = []
        for i in range(batch_size):
            X = npz_data[i]['CT']
            X = X[np.newaxis, ..., np.newaxis]
            X_data.append(X)
        if batch_size > 1:
            return_vals.append(np.concatenate(X_data, 0))
        else:
            return_vals.append(X_data[0])

        X_data = []
        for i in range(batch_size):
            X = npz_data[i]['Seg']
            X = X[np.newaxis, ..., np.newaxis]
            X_data.append(X)
        if batch_size > 1:
            return_vals.append(np.concatenate(X_data, 0))
        else:
            return_vals.append(X_data[0])
                
        X_data = []
        for i in range(batch_size):
            Time = npz_data[i]['Time']
            PFS = npz_data[i]['PFS']
            X = np.array([Time,PFS])
            X = X[np.newaxis, ...]
            X_data.append(X)
        if batch_size > 1:
            return_vals.append(np.concatenate(X_data, 0))
        else:
            return_vals.append(X_data[0])
            
        X_data = []
        for i in range(batch_size):
            X = npz_data[i]['Clinic']
            X = X[np.newaxis, ...]
            X_data.append(X)
        if batch_size > 1:
            return_vals.append(np.concatenate(X_data, 0))
        else:
            return_vals.append(X_data[0])
            
        yield tuple(return_vals)
    
            
def example_gen_preload(vol_names, batch_size=1, balance_class=True):
    
    # preload all data
    PT, CT, Seg, Label, Clinic = load_examples_by_names(vol_names)

    while True:
        
        if balance_class == True:
            # manually balance class
            idxes = []
            num_pos = num_neg = 0
            while num_pos<batch_size/2 or num_neg<batch_size/2:
                idx = np.random.randint(len(vol_names))
                idx_PFS = Label[idx,1]
                if idx_PFS==0 and num_neg<batch_size/2:
                    idxes.append(idx)
                    num_neg = num_neg+1
                if idx_PFS==1 and num_pos<batch_size/2:
                    idxes.append(idx)
                    num_pos = num_pos+1
        else:
            idxes = np.random.randint(len(vol_names), size=batch_size)
            
        X_data = []
        for idx in idxes:
            X = PT[idx]
            X = X[np.newaxis, ...]
            X_data.append(X)
        if batch_size > 1:
            return_vals = [np.concatenate(X_data, 0)]
        else:
            return_vals = [X_data[0]]
                              
        X_data = []
        for idx in idxes:
            X = CT[idx]
            X = X[np.newaxis, ...]
            X_data.append(X)
        if batch_size > 1:
            return_vals.append(np.concatenate(X_data, 0))
        else:
            return_vals.append(X_data[0])
       
        X_data = []
        for idx in idxes:
            X = Seg[idx]
            X = X[np.newaxis, ...]
            X_data.append(X)  
        if batch_size > 1:
            return_vals.append(np.concatenate(X_data, 0))
        else:
            return_vals.append(X_data[0])
                       
        X_data = []
        for idx in idxes:
            X = Label[idx]
            X = X[np.newaxis, ...]
            X_data.append(X)  
        if batch_size > 1:
            return_vals.append(np.concatenate(X_data, 0))
        else:
            return_vals.append(X_data[0])
              
        X_data = []
        for idx in idxes:
            X = Clinic[idx]
            X = X[np.newaxis, ...]
            X_data.append(X)   
        if batch_size > 1:
            return_vals.append(np.concatenate(X_data, 0))
        else:
            return_vals.append(X_data[0])
              
        yield tuple(return_vals)
         
            
def load_examples_by_names(vol_names):            
    
    random.shuffle(vol_names)
    
    X_data = []
    for vol_name in vol_names:
        X = load_volfile(vol_name, np_var='PT')
        X = X[np.newaxis, ..., np.newaxis]
        X_data.append(X)
    return_vals = [np.concatenate(X_data, 0)]                    
                    
    X_data = []
    for vol_name in vol_names:
        X = load_volfile(vol_name, np_var='CT')
        X = X[np.newaxis, ..., np.newaxis]
        X_data.append(X)
    return_vals.append(np.concatenate(X_data, 0))
    
    X_data = []
    for vol_name in vol_names:
        X = load_volfile(vol_name, np_var='Seg')
        X = X[np.newaxis, ..., np.newaxis]
        X_data.append(X)
    return_vals.append(np.concatenate(X_data, 0))
    
    X_data = []
    for vol_name in vol_names:
        Time = load_volfile(vol_name, np_var='Time')
        PFS = load_volfile(vol_name, np_var='PFS')
        X = np.array([Time,PFS])
        X = X[np.newaxis, ...]
        X_data.append(X)            
    return_vals.append(np.concatenate(X_data, 0))
    
    X_data = []
    for vol_name in vol_names:
        X = load_volfile(vol_name, np_var='Clinic')
        X = X[np.newaxis, ...]
        X_data.append(X)            
    return_vals.append(np.concatenate(X_data, 0))
    
    return tuple(return_vals)
    
    
def load_example_by_name(vol_name):
    """
    np_var: specify the name of the variable in numpy files, if your data is stored in 
        npz files. default to 'vol_data'
    """
    X = load_volfile(vol_name, np_var='PT')
    X = X[np.newaxis, ..., np.newaxis]
    return_vals = [X]
    
    X = load_volfile(vol_name, np_var='CT')
    X = X[np.newaxis, ..., np.newaxis]
    return_vals.append(X)
    
    X = load_volfile(vol_name, np_var='Seg')
    #X = X[np.newaxis, ..., np.newaxis]
    return_vals.append(X)
    
    Time = load_volfile(vol_name, np_var='Time')
    PFS = load_volfile(vol_name, np_var='PFS')
    X = np.array([Time,PFS])
    return_vals.append(X)
    
    X = load_volfile(vol_name, np_var='Clinic')
    X = X[np.newaxis, ...]
    return_vals.append(X)
    
    return tuple(return_vals)


def load_testing_example_by_name(vol_name):
    """
    np_var: specify the name of the variable in numpy files, if your data is stored in 
        npz files. default to 'vol_data'
    """
    X = load_volfile(vol_name, np_var='PT')
    X = X[np.newaxis, ..., np.newaxis]
    return_vals = [X]
    
    X = load_volfile(vol_name, np_var='CT')
    X = X[np.newaxis, ..., np.newaxis]
    return_vals.append(X)
    
    X = load_volfile(vol_name, np_var='Clinic')
    X = X[np.newaxis, ...]
    return_vals.append(X)
    
    return tuple(return_vals)


#--------------------------------------------------------------------------------------
# Util Functions

def load_volfile(datafile, np_var):
    """
    load volume file
    formats: nii, nii.gz, mgz, npz
    if it's a npz (compressed numpy), variable names innp_var (default: 'vol_data')
    """
    assert datafile.endswith(('.nii', '.nii.gz', '.mgz', '.npz')), 'Unknown data file'

    if datafile.endswith(('.nii', '.nii.gz', '.mgz')):
        # import nibabel
        if 'nibabel' not in sys.modules:
            try :
                import nibabel as nib  
            except:
                print('Failed to import nibabel. need nibabel library for these data file types.')

        X = nib.load(datafile).get_data()
        
    else: # npz
        if np_var == 'all':
            X = X = np.load(datafile)
        else:
            X = np.load(datafile)[np_var]

    return X


def Sort_by_time(data, time):
    '''
    Sort samples by survival time
    Designed for Cox loss function.
    '''
    
    sorted_arg = np.argsort(time)
    sorted_data = np.zeros(data.shape)
    
    for i in range(len(time)):
        sorted_data[i] = data[sorted_arg[i]]
        
    return sorted_data


#--------------------------------------------------------------------------------------
# Function for data argumentation

def Data_augmentation(PT, CT, Seg):
    
    # define augmentation sequence
    aug_seq = iaa.Sequential([
        # horizontal flips
        iaa.Fliplr(0.5), 
        # translate/move them and rotate them.
        iaa.Affine(translate_px={"x": [-10, 10], "y": [0, 0]},rotate=(-5, 5))
        ],random_order=True) # apply augmenters in random order
    
    aug_seq_no_flip = iaa.Sequential([
        # translate/move them and rotate them.
        iaa.Affine(translate_px={"x": [-10, 10], "y": [0, 0]},rotate=(-5, 5))
        ],random_order=False)
    
    # pre-process data shape
    PT = PT[..., 0]
    CT = CT[..., 0]
    Seg = Seg[..., 0]
    
    # flip/translate in x axls, rotate along z axls
    images = np.concatenate((PT,CT,Seg), -1)
    
    images_aug = aug_seq_no_flip(images=images) 
    
    PT = images_aug[..., 0:int(images_aug.shape[3]/3)]    
    CT = images_aug[..., int(images_aug.shape[3]/3):int(images_aug.shape[3]/3*2)]
    Seg = images_aug[..., int(images_aug.shape[3]/3*2):int(images_aug.shape[3])]
    
    # translate in z axls, rotate along y axls
    PT = np.transpose(PT,(0,3,1,2))
    CT = np.transpose(CT,(0,3,1,2))
    Seg = np.transpose(Seg,(0,3,1,2))
    images = np.concatenate((PT,CT,Seg), -1)
    
    images_aug = aug_seq(images=images) 
    
    PT = images_aug[..., 0:int(images_aug.shape[3]/3)]    
    CT = images_aug[..., int(images_aug.shape[3]/3):int(images_aug.shape[3]/3*2)]
    Seg = images_aug[..., int(images_aug.shape[3]/3*2):int(images_aug.shape[3])]
    
    # translate in y axls, rotate along x axls
    PT = np.transpose(PT,(0,3,1,2))
    CT = np.transpose(CT,(0,3,1,2))
    Seg = np.transpose(Seg,(0,3,1,2))
    images = np.concatenate((PT,CT,Seg), -1)
    
    images_aug = aug_seq_no_flip(images=images) 
    
    PT = images_aug[..., 0:int(images_aug.shape[3]/3)]    
    CT = images_aug[..., int(images_aug.shape[3]/3):int(images_aug.shape[3]/3*2)]
    Seg = images_aug[..., int(images_aug.shape[3]/3*2):int(images_aug.shape[3])]
    
    # recover axls
    PT = np.transpose(PT,(0,3,1,2))
    CT = np.transpose(CT,(0,3,1,2))
    Seg = np.transpose(Seg,(0,3,1,2))
    
    # reset Seg mask to 1/0
    for i in range(Seg.shape[0]):
        _, Seg[i] = cv2.threshold(Seg[i],0.2,1,cv2.THRESH_BINARY)
    
    # post-process data shape
    PT_aug = PT[..., np.newaxis]
    CT_aug = CT[..., np.newaxis]
    Seg_aug = Seg[..., np.newaxis]
    
    return PT_aug, CT_aug, Seg_aug
    
    
    
    
    