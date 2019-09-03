import math
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from tensorflow.keras import backend as K

def step_decay(epoch):
    initial_lrate = 1e-2
    drop = 0.921688
    epochs_drop = 10.0
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lrate

def step_decay_25_864065(epoch):
    initial_lrate = 1e-2
    drop = 0.864065
    epochs_drop = 25
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lrate

def step_decay_100_599484(epoch):
    initial_lrate = 5e-2
    drop = 0.599484
    epochs_drop = 100
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lrate

def exp_decay(epoch):
    initial_lrate = 1e-2
    k = 0.0081548
    lrate = initial_lrate * np.exp(-k*epoch)
    return lrate

class SkipLoggerCallback(tf.keras.callbacks.Callback):
    
    def __init__(self, n):
        self.n = n
        
    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.n == 0:
            loss = logs.get('loss')
            accuracy = logs.get('accuracy')
            val_loss = logs.get('val_loss')
            val_accuracy = logs.get('val_accuracy')
            try:
                print("epoch = %4d    loss = %0.6f    accuracy = %0.2f    val_loss = %0.6f    val_accuracy = %0.2f" % (epoch, loss, accuracy, val_loss, val_accuracy))
            except:
                print(epoch,loss,accuracy,val_loss,val_accuracy)
            
            
class LearningRateHistoryCallback(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.lr = []
        
    def on_epoch_end(self, epoch, logs={}):      
        lr = self.model.optimizer.lr
        decay = self.model.optimizer.decay
        iterations = self.model.optimizer.iterations
        lr_with_decay = lr / (1. + decay * K.cast(iterations, K.dtype(decay)))
        self.lr.append(lr_with_decay)
        #print('iteration # %d, learning rate: %.2e' % (iterations, lr_with_decay))
        
        
class StepDecayHistoryCallback(tf.keras.callbacks.Callback):
    
    def on_train_begin(self, logs={}):
        self.losses = []
        self.lr = []
 
    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.lr.append(step_decay_25_864065(len(self.losses)))
        
class ExpDecayHistoryCallback(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.lr = []
        
    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.lr.append(exp_decay(len(self.losses)))