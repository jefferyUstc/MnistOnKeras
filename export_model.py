#! /usr/bin/python
# _*_ coding: utf-8 _*_
__author__ = 'Jeffery'
__date__ = '2019/1/9 23:20'

import tensorflow as tf
from model import keras_model
import os
import keras.backend as K
from keras.losses import categorical_crossentropy
from keras.optimizers import Adadelta


def export_model(model,
                 export_model_dir,
                 model_version
                 ):
    """
    :param export_model_dir: type string, save dir for exported model
    :param model_version: type int best
    :return:no return
    """
    with tf.get_default_graph().as_default():
        # prediction_signature
        tensor_info_input = tf.saved_model.utils.build_tensor_info(model.input)
        tensor_info_output = tf.saved_model.utils.build_tensor_info(model.output)
        print(model.output.shape, '**', tensor_info_output)
        prediction_signature = (
            tf.saved_model.signature_def_utils.build_signature_def(
                inputs={'images': tensor_info_input}, # Tensorflow.TensorInfo
                outputs={'result': tensor_info_output},
                method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))
        print('step1 => prediction_signature created successfully')
        # set-up a builder
        export_path_base = export_model_dir
        export_path = os.path.join(
            tf.compat.as_bytes(export_path_base),
            tf.compat.as_bytes(str(model_version)))
        builder = tf.saved_model.builder.SavedModelBuilder(export_path)
        builder.add_meta_graph_and_variables(
            # tags:SERVING,TRAINING,EVAL,GPU,TPU
            sess=K.get_session(),
            tags=[tf.saved_model.tag_constants.SERVING],
            signature_def_map={'prediction_signature': prediction_signature,},
            )
        print('step2 => Export path(%s) ready to export trained model' % export_path, '\n starting to export model...')
        builder.save(as_text=True)
        print('Done exporting!')

if __name__ == '__main__':
    model = keras_model()
    model.compile(loss=categorical_crossentropy,
              optimizer=Adadelta(lr=0.1),
              metrics=['accuracy'])
    model.load_weights('./model_data/weights.hdf5')
    model.summary()
    export_model(
        model,
        './export_model',
        1
    )