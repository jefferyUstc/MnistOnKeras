"""
implemention of lightweight CNN Model on Mnist Dataset,
try to include as more as possible features that I learn from keras, by the way,compare to tensorflow
hope to give a good learning example

backend:tensorflow
"""

from __future__ import print_function
import keras
from get_data import x_train, x_test, y_train, y_test
from model import keras_model
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, TensorBoard
from callback_func import my_callback, lr_scheduler
from settings import batch_size, epochs


# some hypeparams
model = keras_model()
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(lr=0.1),
              metrics=['accuracy'])
# --------------------------callbacks------------------------------
my_cb = my_callback()
callbacks = [
    # Interrupt training if `val_loss` stops improving for over 2 epochs
    EarlyStopping(patience=2, monitor='val_loss'),
    # Write TensorBoard logs to `./logs` directory
    TensorBoard(log_dir='./logs'),
    # save checkpoint each-epoch if val-loss drop
    ModelCheckpoint(filepath='./model_data/weights.hdf5',
                    verbose=1, save_best_only=True),
    # dynamic learning 
    LearningRateScheduler(lr_scheduler),
    # custom callback
    my_cb,
]
# -----------------------------------------------------------------
history = model.fit(x_train, y_train, batch_size=batch_size,
                    epochs=epochs, verbose=1,
                    validation_data=(x_test, y_test),
                    callbacks=callbacks)
score = model.evaluate(x_test, y_test, verbose=0)
print('history:', history)
print('my_cb.losses:', my_cb.losses)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

