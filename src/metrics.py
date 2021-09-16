# Third party inports
import tensorflow as tf
import keras.backend as K
import numpy as np


def Dice(y_true, y_pred):
    """
    Dice score for binary segmentation
    """
    
    ndims = len(y_pred.get_shape().as_list()) - 2
    vol_axes = list(range(1, ndims+1))

    y_pred = tf.cast(y_pred > 0.5, y_pred.dtype)
    top = 2 * tf.reduce_sum(y_true * y_pred, vol_axes)
    bottom = tf.maximum(tf.reduce_sum(y_true + y_pred, vol_axes), 1e-5)
    
    dice = tf.reduce_mean(top/bottom)
    return dice


def Cindex(y_true, y_pred):
    '''
    C-index score for risk prediction.
    y_pred is the predicted risk from trained model.
    y_true is event indicator, event=0 means censored
    Survival time is not requied as input
    Samples should be sorted with increasing survial time 
    '''
    
    risk = y_pred
    event = tf.cast(y_true, risk.dtype)
    
    g = tf.subtract(risk, risk[:,0])
    g = tf.cast(g == 0.0, risk.dtype) * 0.5 + tf.cast(g > 0.0, risk.dtype)

    
    f = tf.matmul(event, tf.cast(tf.transpose(event)>-1, risk.dtype)) 
    f = tf.matrix_band_part(f, 0, -1) - tf.matrix_band_part(f, 0, 0)

    top = tf.reduce_sum(tf.multiply(g, f))
    bottom = tf.reduce_sum(f)
    
    cindex = top/bottom

    return cindex

