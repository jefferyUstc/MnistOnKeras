"""
some kind of functions to save model
"""
import os

def save_weights_only(model, path='./ur_model_data', save_format=None):
    """
    Save weights to a TensorFlow Checkpoint file
    To testore the model's state, model.load_weights('./my_model')
    notably, this requires a model with the same architecture.

    Parameters:
    save_format: None | h5, when None path should be a dir;when h5 path shoud be path_to/*.h5
    """
    model.save_weights(path, save_format=save_format)


def save_model_only(model, format='json'):
    """
    Recreate the model (freshly initialized)
    fresh_model = keras.models.from_json(json_string)
    Serializes a model to YAML format
    Recreate the model
    fresh_model = keras.models.from_yaml(yaml_string)

    notbly: subclass model can not be Serialized
    """
    if format == 'json':
        return model.to_json()
    elif format == 'yaml':
        return model.to_yaml()
    else:
        raise ValueError('only json or yaml should be passed in ')


def save_all(model, dir='./model'):
    """
    Recreate the model:model = keras.models.load_model('my_model.h5')
    """
    if not os.path.exists(dir):
        os.mkdir(dir)
    model.save(os.path.join(dir, 'model_final.h5'))
