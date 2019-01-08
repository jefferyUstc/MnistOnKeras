#! /usr/bin/python
# _*_ coding: utf-8 _*_
__author__ = 'Jeffery'
__date__ = '2019/1/9 23:21'

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
import tensorflow as tf
import numpy as np
import grpc
from get_data import x_test, y_test

def request_server(img_np,
                   server_url,
                   model_name,
                   signature_name,
                   input_name,
                   output_name
                   ):
    """
    below info about model
    :param model_name:
    :param signature_name:
    :param output_name:
    :param input_name:

    :param img_np: processed img , numpy.ndarray type [h,w,c]
    :param server_url: TensorFlow Serving url,str type,e.g.'0.0.0.0:8500'
    :return: type numpy array
    """
    # connect channel
    channel = grpc.insecure_channel(server_url)
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    # set up request
    request = predict_pb2.PredictRequest()
    request.model_spec.name = model_name  # request.model_spec.version = "1"
    request.model_spec.signature_name = signature_name
    request.inputs[input_name].CopyFrom(
        tf.contrib.util.make_tensor_proto(img_np, shape=list(img_np.shape)))
    # get response
    response = stub.Predict(request, 5.0)
    # res_from_server_np = np.asarray(response.outputs[output_name].float_val)
    res_from_server_np = tf.make_ndarray(response.outputs[output_name])  # [[....]]
    res_from_server_np = res_from_server_np.flatten()
    return np.where(res_from_server_np==np.max(res_from_server_np))


if __name__ == '__main__':
    # print(x_test.shape, y_test.shape)  # (10000, 28, 28, 1) (10000, 10)
    x_input = np.expand_dims(x_test[0], 0)
    print(x_input.shape)
    res_from_server_np = request_server(x_input,
                    '0.0.0.0:8500',
                    'mymodel', 
                    'prediction_signature',
                    "images",
                    "result")
    print('predict:', res_from_server_np, '\ntruth:',
                                        np.where(y_test[0]==np.max(y_test[0])))
