# main imports
import sys

# third party
import numpy as np
import keras.backend as K
from keras.models import Model
import keras.layers as KL
from keras.layers import Layer
from keras.layers import Conv3D, Activation, Input, UpSampling3D, concatenate, Conv3DTranspose, ZeroPadding3D, AveragePooling3D, BatchNormalization, MaxPooling3D, GlobalAveragePooling3D, Dense, Flatten, multiply
from keras.layers import LeakyReLU, Reshape, Lambda, PReLU, add, Dropout
from keras.initializers import RandomNormal
import keras.initializers
from tensorflow.keras import regularizers
import tensorflow as tf
from tensorflow.keras import layers


def DeepMTS(vol_size, Clinic_size):
    
    # inputs
    PET = Input(shape=[*vol_size, 1])
    CT = Input(shape=[*vol_size, 1]) 
    Clinic = Input(shape=[Clinic_size]) 
    
    Seg_pred, x_out1, x_out2, x_out3, x_out4, x_out5 = Seg_net([PET, CT])
    #Seg_pred, x_out1, x_out2, x_out3, x_out4, x_out5 = U_net([PET, CT])
    
    x_out6, x_out7, x_out8 = DenseNet([PET, CT, Seg_pred])
    
    x_in1 = [x_out1, x_out2, x_out3, x_out4, x_out5]
    x_in2 = [x_out6, x_out7, x_out8]
    Survival_pred = classifier(x_in1, x_in2, Clinic, L2_reg=0.1, num_node=64)
    
    return Model(inputs=[PET, CT, Clinic], outputs=[Seg_pred, Survival_pred])


#--------------------------------------------------------------------------------------
#Detailed architecture functions
#--------------------------------------------------------------------------------------

def classifier(x_in1, x_in2, Clinic, L2_reg, num_node, droprate=0.5):

    x_concat1 = []
    for i in range(len(x_in1)):
        x = GlobalAveragePooling3D()(x_in1[i])
        x_concat1.append(x)
    x_concat1 = concatenate(x_concat1)
    
    x_concat2 = []
    for i in range(len(x_in2)):
        x = GlobalAveragePooling3D()(x_in2[i])
        x_concat2.append(x)
    x_concat2 = concatenate(x_concat2)
    
    x_1 = Dropout(droprate)(x_concat1)
    x_1 = Dense(num_node, activation="relu", kernel_regularizer=regularizers.l2(L2_reg))(x_1)
    
    x_2 = Dropout(droprate)(x_concat2)
    x_2 = Dense(num_node, activation="relu", kernel_regularizer=regularizers.l2(L2_reg))(x_2)
    
    x = concatenate([x_1, x_2, Clinic])
    
    x = Dropout(droprate)(x)
    Survival_pred = Dense(1, activation="linear", name="Survival", kernel_regularizer=regularizers.l2(L2_reg))(x)
    
    return Survival_pred


#--------------------------------------------------------------------------------------

def DenseNet(x_in, droprate=0.05):
    
    x_in = concatenate(x_in)
       
    x = Conv3D(24, (3, 3, 3), strides=(2,2,2), kernel_initializer="he_uniform", name="initial_conv3D")(x_in)
    x = Dropout(droprate)(x)
    x = MaxPooling3D((3, 3, 3), strides=(2, 2, 2))(x)
     
    x = dense_block(x, 4)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv3D(16, (1, 1, 1), kernel_initializer="he_uniform", padding="same")(x) 
    x = Dropout(droprate)(x)
    
    x_out1 = BatchNormalization()(x)
    x_out1 = Activation("relu")(x_out1)
    
    x = AveragePooling3D((2, 2, 2), strides=(2, 2, 2))(x)
    
    x = dense_block(x, 8)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv3D(32, (1, 1, 1), kernel_initializer="he_uniform", padding="same")(x) 
    x = Dropout(droprate)(x)
    
    x_out2 = BatchNormalization()(x)
    x_out2 = Activation("relu")(x_out2)
    
    x = AveragePooling3D((2, 2, 2), strides=(2, 2, 2))(x)  
    
    x = dense_block(x, 16)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv3D(64, (1, 1, 1), kernel_initializer="he_uniform", padding="same")(x) 
    x = Dropout(droprate)(x) 
    
    x_out3 = BatchNormalization()(x)
    x_out3 = Activation("relu")(x_out3)
    
    return x_out1, x_out2, x_out3

    
def dense_block(inputs, numlayers):

    concatenated_inputs = inputs
    for i in range(numlayers):
        x = dense_factor(concatenated_inputs)
        concatenated_inputs = concatenate([concatenated_inputs, x], axis=-1)

    return concatenated_inputs


def dense_factor(inputs, kernel_num1=64,kernel_num2=16,droprate=0.05):
    
    x = BatchNormalization()(inputs)
    x = Activation("relu")(x)
    x = Conv3D(kernel_num1, (1,1,1), kernel_initializer='he_uniform',padding='same')(x)
    x = Dropout(droprate)(x)
    
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv3D(kernel_num2, (3,3,3), kernel_initializer='he_uniform',padding='same')(x)
    outputs = Dropout(droprate)(x)    
    
    return outputs

#--------------------------------------------------------------------------------------

def Seg_net(x_in):
    
    x_in = concatenate(x_in)
    
    res_1 = Conv3D(8, 1, strides=1, padding="same", kernel_initializer="he_uniform")(x_in)
    res_1 = BatchNormalization()(res_1)
    res_1 = Activation("relu")(res_1)
    
    x = Conv3D(8, 3, strides=1, padding="same", kernel_initializer="he_uniform")(x_in)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    res_2 = add([x,res_1])
    
    x = Conv3D(8, 3, strides=1, padding="same", kernel_initializer="he_uniform")(res_2)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x_1 = add([x,res_2])

    
    # downsampling 1
    x_in = MaxPooling3D()(x_1)

    res_1 = Conv3D(16, 1, strides=1, padding="same", kernel_initializer="he_uniform")(x_in)
    res_1 = BatchNormalization()(res_1)
    res_1 = Activation("relu")(res_1)
    
    x = Conv3D(16, 3, strides=1, padding="same", kernel_initializer="he_uniform")(x_in)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    res_2 = add([x,res_1])
    
    x = Conv3D(16, 3, strides=1, padding="same", kernel_initializer="he_uniform")(res_2)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x_2 = add([x,res_2])
    
    
    # downsampling 2
    x_in = MaxPooling3D()(x_2)

    res_1 = Conv3D(32, 1, strides=1, padding="same", kernel_initializer="he_uniform")(x_in)
    res_1 = BatchNormalization()(res_1)
    res_1 = Activation("relu")(res_1)
    
    x = Conv3D(32, 3, strides=1, padding="same", kernel_initializer="he_uniform")(x_in)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    res_2 = add([x,res_1])
    
    x = Conv3D(32, 3, strides=1, padding="same", kernel_initializer="he_uniform")(res_2)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    res_3 = add([x,res_2])
    
    x = Conv3D(32, 3, strides=1, padding="same", kernel_initializer="he_uniform")(res_3)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x_3 = add([x,res_3])
    
    
    # downsampling 3
    x_in = MaxPooling3D()(x_3)

    res_1 = Conv3D(64, 1, strides=1, padding="same", kernel_initializer="he_uniform")(x_in)
    res_1 = BatchNormalization()(res_1)
    res_1 = Activation("relu")(res_1)
    
    x = Conv3D(64, 3, strides=1, padding="same", kernel_initializer="he_uniform")(x_in)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    res_2 = add([x,res_1])
    
    x = Conv3D(64, 3, strides=1, padding="same", kernel_initializer="he_uniform")(res_2)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    res_3 = add([x,res_2])
    
    x = Conv3D(64, 3, strides=1, padding="same", kernel_initializer="he_uniform")(res_3)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x_4 = add([x,res_3])
    
    
    # downsamping 4
    x_in = MaxPooling3D()(x_4)

    res_1 = Conv3D(128, 1, strides=1, padding="same", kernel_initializer="he_uniform")(x_in)
    res_1 = BatchNormalization()(res_1)
    res_1 = Activation("relu")(res_1)
    
    x = Conv3D(128, 3, strides=1, padding="same", kernel_initializer="he_uniform")(x_in)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    res_2 = add([x,res_1])
    
    x = Conv3D(128, 3, strides=1, padding="same", kernel_initializer="he_uniform")(res_2)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    res_3 = add([x,res_2])
    
    x = Conv3D(128, 3, strides=1, padding="same", kernel_initializer="he_uniform")(res_3)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    res_4 = add([x,res_3])
    
    x = Conv3D(128, 3, strides=1, padding="same", kernel_initializer="he_uniform")(res_4)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x_5 = add([x,res_4])
    
    
    # upsampling 1
    x = UpSampling3D()(x_5)
    x_in = concatenate([x, x_4])
    
    res_1 = Conv3D(64, 1, strides=1, padding="same", kernel_initializer="he_uniform")(x_in)
    res_1 = BatchNormalization()(res_1)
    res_1 = Activation("relu")(res_1)
    
    x = Conv3D(64, 3, strides=1, padding="same", kernel_initializer="he_uniform")(x_in)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    res_2 = add([x,res_1])
    
    x = Conv3D(64, 3, strides=1, padding="same", kernel_initializer="he_uniform")(res_2)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    res_3 = add([x,res_2])
    
    x = Conv3D(64, 3, strides=1, padding="same", kernel_initializer="he_uniform")(res_3)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = add([x,res_3])
    
    
    # upsampling 2
    x = UpSampling3D()(x)
    x_in = concatenate([x, x_3])
    
    res_1 = Conv3D(32, 1, strides=1, padding="same", kernel_initializer="he_uniform")(x_in)
    res_1 = BatchNormalization()(res_1)
    res_1 = Activation("relu")(res_1)
    
    x = Conv3D(32, 3, strides=1, padding="same", kernel_initializer="he_uniform")(x_in)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    res_2 = add([x,res_1])
    
    x = Conv3D(32, 3, strides=1, padding="same", kernel_initializer="he_uniform")(res_2)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    res_3 = add([x,res_2])
    
    x = Conv3D(32, 3, strides=1, padding="same", kernel_initializer="he_uniform")(res_3)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = add([x,res_3])
    
    
    # upsampling 3
    x = UpSampling3D()(x)
    x_in = concatenate([x, x_2])
    
    res_1 = Conv3D(16, 1, strides=1, padding="same", kernel_initializer="he_uniform")(x_in)
    res_1 = BatchNormalization()(res_1)
    res_1 = Activation("relu")(res_1)
    
    x = Conv3D(16, 3, strides=1, padding="same", kernel_initializer="he_uniform")(x_in)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    res_2 = add([x,res_1])
    
    x = Conv3D(16, 3, strides=1, padding="same", kernel_initializer="he_uniform")(res_2)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = add([x,res_2])
    
    
    # upsampling 4
    x = UpSampling3D()(x)
    x_in = concatenate([x, x_1])
    
    res_1 = Conv3D(8, 1, strides=1, padding="same", kernel_initializer="he_uniform")(x_in)
    res_1 = BatchNormalization()(res_1)
    res_1 = Activation("relu")(res_1)
    
    x = Conv3D(8, 3, strides=1, padding="same", kernel_initializer="he_uniform")(x_in)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    res_2 = add([x,res_1])
    
    x = Conv3D(8, 3, strides=1, padding="same", kernel_initializer="he_uniform")(res_2)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = add([x,res_2])
    
    
    # Segmentation output
    x = Conv3D(2, 1, strides=1, padding="same", kernel_initializer="he_uniform")(x)
    x = Activation("softmax")(x) 
    Seg_pred = Lambda(lambda x: x[...,0:1], name="Segmentation")(x)
    
    
    # Radiomics output
    x_out1 = Conv3D(4, 1, kernel_initializer="he_uniform", padding="same")(x_1) 
    x_out1 = BatchNormalization()(x_out1)
    x_out1 = Activation("relu")(x_out1)
    
    x_out2 = Conv3D(8, 1, kernel_initializer="he_uniform", padding="same")(x_2) 
    x_out2 = BatchNormalization()(x_out2)
    x_out2 = Activation("relu")(x_out2)
    
    x_out3 = Conv3D(16, 1, kernel_initializer="he_uniform", padding="same")(x_3) 
    x_out3 = BatchNormalization()(x_out3)
    x_out3 = Activation("relu")(x_out3)
    
    x_out4 = Conv3D(32, 1, kernel_initializer="he_uniform", padding="same")(x_4) 
    x_out4 = BatchNormalization()(x_out4)
    x_out4 = Activation("relu")(x_out4)
    
    x_out5 = Conv3D(64, 1, kernel_initializer="he_uniform", padding="same")(x_5) 
    x_out5 = BatchNormalization()(x_out5)
    x_out5 = Activation("relu")(x_out5)

    return Seg_pred, x_out1, x_out2, x_out3, x_out4, x_out5 

#--------------------------------------------------------------------------------------

def Unet(x_in):
    
    x_in = concatenate(x_in)
    
    x = Conv3D(8, 3, strides=1, padding="same", kernel_initializer="he_uniform")(x_in)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    
    x = Conv3D(8, 3, strides=1, padding="same", kernel_initializer="he_uniform")(x)
    x = BatchNormalization()(x)
    x_out1 = Activation("relu")(x)
    
    
    # downsampling 1
    x = MaxPooling3D()(x_out1)

    x = Conv3D(16, 3, strides=1, padding="same", kernel_initializer="he_uniform")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    
    x = Conv3D(16, 3, strides=1, padding="same", kernel_initializer="he_uniform")(x)
    x = BatchNormalization()(x)
    x_out2 = Activation("relu")(x)
    
    
    # downsampling 2
    x = MaxPooling3D()(x_out2)
    
    x = Conv3D(32, 3, strides=1, padding="same", kernel_initializer="he_uniform")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    
    x = Conv3D(32, 3, strides=1, padding="same", kernel_initializer="he_uniform")(x)
    x = BatchNormalization()(x)
    x_out3 = Activation("relu")(x)
    
    
    # downsampling 3
    x = MaxPooling3D()(x_out3)

    x = Conv3D(64, 3, strides=1, padding="same", kernel_initializer="he_uniform")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    
    x = Conv3D(64, 3, strides=1, padding="same", kernel_initializer="he_uniform")(x)
    x = BatchNormalization()(x)
    x_out4 = Activation("relu")(x)
    
    
    # downsamping 4
    x = MaxPooling3D()(x_out4)

    x = Conv3D(128, 3, strides=1, padding="same", kernel_initializer="he_uniform")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    
    x = Conv3D(128, 3, strides=1, padding="same", kernel_initializer="he_uniform")(x)
    x = BatchNormalization()(x)
    x_out5 = Activation("relu")(x)
    
    
    # upsampling 1
    x = UpSampling3D()(x_out5)
    x = concatenate([x, x_out4])
    
    x = Conv3D(64, 3, strides=1, padding="same", kernel_initializer="he_uniform")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    
    x = Conv3D(64, 3, strides=1, padding="same", kernel_initializer="he_uniform")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    
    
    # upsampling 2
    x = UpSampling3D()(x)
    x = concatenate([x, x_out3])
    
    x = Conv3D(32, 3, strides=1, padding="same", kernel_initializer="he_uniform")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    
    x = Conv3D(32, 3, strides=1, padding="same", kernel_initializer="he_uniform")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    
    
    # upsampling 3
    x = UpSampling3D()(x)
    x = concatenate([x, x_out2])
    
    x = Conv3D(16, 3, strides=1, padding="same", kernel_initializer="he_uniform")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    
    x = Conv3D(16, 3, strides=1, padding="same", kernel_initializer="he_uniform")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    
    
    # upsampling 4
    x = UpSampling3D()(x)
    x = concatenate([x, x_out1])
    
    x = Conv3D(8, 3, strides=1, padding="same", kernel_initializer="he_uniform")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    
    x = Conv3D(8, 3, strides=1, padding="same", kernel_initializer="he_uniform")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    
    
    # Segmentation outpur
    x = Conv3D(2, 1, strides=1, padding="same", kernel_initializer="he_uniform")(x)
    x = Activation("softmax")(x) 
    Seg_pred = Lambda(lambda x: x[...,0:1], name="Segmentation")(x)

    return Seg_pred, x_out1, x_out2, x_out3, x_out4, x_out5 

