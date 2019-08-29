import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from tensorflow.keras import backend as K


class SkipLoggerCallback(tf.keras.callbacks.Callback):
    
    def __init__(self, n):
        self.n = n
        
    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.n == 0:
            loss = logs.get('loss')
            accuracy = logs.get('accuracy')
            val_loss = logs.get('val_loss')
            val_accuracy = logs.get('val_accuracy')
            print("epoch = %4d    loss = %0.6f    accuracy = %0.2f    val_loss = %0.6f    val_accuracy = %0.2f" % (epoch, loss, accuracy, val_loss, val_accuracy))
            
            
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