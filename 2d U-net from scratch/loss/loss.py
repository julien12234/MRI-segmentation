import numpy as np
import math

import torch.nn.functional as F
import torch.nn as nn
import torch
from scipy.ndimage import distance_transform_edt

def sigmoid(x):
    return 1/(1+np.exp(-x))

def dice_coefficient(pred, target):
    """
    Arguments:
        pred: our prediction 
        target: ground truth values
    -------------------------------------------------------------------------
        epsilon (float): small constant added to numerator and denominator to
                        avoid divide by 0 errors.
    Returns:
        dice_coefficient (float): computed value of dice coefficient.     
    """
    
    epsilon = 0.0001
    den = 2*np.sum(pred * target) + epsilon
    nom = np.sum(pred) + np.sum(target) + epsilon
    return round(den/nom,3)

def accuracy(pred,target,normalize=True, sample_weight=None):
    '''Compute the accuracy between a prediction and a target, which is equivalent to (TP+TN)/total
    
    Args:
        pred (np.array): predicted value
        target (np.array): targeted value, usually the label

    Returns:
        double: accuracy between pred and target
    '''
    return np.mean(pred==target)

def recall(pred,target):
    """Compute the recall between a prediction and a target, which is equivalent to (TP)/(TP+FN)

    Args:
        pred (np.array): predicted value
        target (np.array): targeted value, usually the label

    Returns:
        double: Recall between pred and target
    """
    TP = np.sum(np.logical_and(pred == 1, target == 1))
    FN = np.sum(np.logical_and(pred == 0, target == 1))
    return TP/(TP + FN)

def precision(pred,target):
    TP = np.sum(np.logical_and(pred == 1, target == 1))
    FP = np.sum(np.logical_and(pred == 1, target == 0))
    if (TP + FP) == 0:
        return 0
    else:
        return TP/(TP + FP)

