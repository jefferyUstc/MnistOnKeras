"""
callbacks summary:

# usually used callback function
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, TensorBoard
# other callback
from keras.callbacks import ProgbarLogger, RemoteMonitor, ReduceLROnPlateau, CSVLogger, LambdaCallback, TerminateOnNaN
# automatic used callbacks
from keras.callbacks import BaseLogger, History
# base call
from keras.callbacks import Callback
"""
from settings import epochs, lr_power
from keras.callbacks import Callback

class my_callback(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))


def lr_scheduler(epoch, lr, mode='power_decay'):
    """
    function for keras.callbacks.LearningRateScheduler
    """
    if mode is 'power_decay':
        # original lr scheduler
        lr = lr * ((1 - float(epoch) / epochs) ** lr_power)
    if mode is 'exp_decay':
        # exponential decay
        lr = (float(lr) ** float(lr_power)) ** float(epoch + 1)
    # adam default lr
    if mode is 'adam':
        lr = 0.001

    if mode is 'progressive_drops':
        # drops as progression proceeds, good for sgd
        if epoch > 0.9 * epochs:
            lr = 0.0001
        elif epoch > 0.75 * epochs:
            lr = 0.001
        elif epoch > 0.5 * epochs:
            lr = 0.01
        else:
            lr = 0.1
    return lr
