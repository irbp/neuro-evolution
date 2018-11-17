from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import backend as K

import tensorflow as tf
import os

early_stopper = EarlyStopping(patience=5)

def check_tpu():
    try:
        tpu_address = 'grpc://' + os.environ['COLAB_TPU_ADDR']
        print ('TPU address is: ', tpu_address)

        with tf.Session(tpu_address) as session:
          devices = session.list_devices()
    
        return True, tpu_address
    except Exception:
        return False, ""

def compile_model(network, input_shape):
    """

    :param network dict: dictionary with network parameters
    :param input_shape tuple: tuple with tradin data shape
    :return: compiled model
    """

    nb_layers = network.get('n_layers', 2)
    nb_neurons = network.get('n_neurons', 10)
    activation = network.get('activations', 'sigmoid')
    optimizer = network.get('optimizers', 'adam')

    model = Sequential()

    model.add(Dense(nb_neurons, activation=activation, input_shape=input_shape))
    model.add(Dropout(network.get('dropout', 1)))
    for i in range(nb_layers - 1):
        model.add(Dense(nb_neurons, activation=activation))
        model.add(Dropout(network.get('dropout', 1)))

    model.add(Dense(
        network.get('last_layer_neurons', 1),
        activation=network.get('last_layer_activations', 'sigmoid'),
    ))

    has_tpu, tpu_address = check_tpu()
    print(has_tpu)
    if has_tpu:
        tpu_model = tf.contrib.tpu.keras_to_tpu_model(
            model,
            strategy=tf.contrib.tpu.TPUDistributionStrategy(
                tf.contrib.cluster_resolver.TPUClusterResolver(tpu_address)
            ))

        tpu_model.compile(loss=network.get('losses', 'binary_crossentropy'), optimizer=optimizer,
                    metrics=[network.get('metrics', 'accuracy')])
        return tpu_model

    else:
        model.compile(loss=network.get('losses', 'binary_crossentropy'), optimizer=optimizer,
                    metrics=[network.get('metrics', 'accuracy')])
        return model


def train_and_score(network, x_train, y_train, x_test, y_test):
    """

    :param network dict: dictionary with network parameters
    :param x_train array: numpy array with features for traning
    :param y_train array: numpy array with labels for traning
    :param x_test array: numpy array with labels for test
    :param y_test array: numpy array with labels for test
    :return float: score
    """

    model = compile_model(network, (x_train.shape[1],))

    model.fit(x_train, y_train,
              batch_size=network.get('batch_size', 128),
              epochs=network.get('epochs', 10000),  # using early stopping, so no real limit
              verbose=network.get('verbose', 0),
              validation_data=(x_test, y_test),
              callbacks=[early_stopper])

    score = model.evaluate(x_test, y_test, verbose=0)

    K.clear_session()
    print("------ Accuracy: {}".format(score[1]))

    return score[1]  # 1 is accuracy. 0 is loss.
