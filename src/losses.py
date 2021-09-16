# Third party inports
import tensorflow as tf
import keras.backend as K
import numpy as np


def Dice_loss(y_true, y_pred):
    """
    N-D dice for binary segmentation
    """
    
    ndims = len(y_pred.get_shape().as_list()) - 2
    vol_axes = list(range(1, ndims+1))

    top = 2 * tf.reduce_sum(y_true * y_pred, vol_axes)
    bottom = tf.maximum(tf.reduce_sum(y_true + y_pred, vol_axes), 1e-5)
    dice = tf.reduce_mean(top/bottom)
    return -dice
    

def Cox_loss(y_true, y_pred):
    '''
    Calculate the average Cox negative partial log-likelihood.
    y_pred is the predicted risk from trained model.
    y_true is event indicator, event=0 means censored
    Survival time is not requied as input
    Samples should be sorted with increasing survial time 
    '''
    
    risk = y_pred
    event = tf.cast(y_true, dtype=risk.dtype)
    
    risk_exp = tf.exp(risk)
    risk_exp_cumsum = tf.cumsum(risk_exp, reverse=True)
    likelihood = risk - tf.log(risk_exp_cumsum)
    uncensored_likelihood = tf.multiply(likelihood,event)
    
    n_observed = tf.reduce_sum(event)
    cox_loss = -tf.reduce_sum(uncensored_likelihood)/n_observed

    return cox_loss


