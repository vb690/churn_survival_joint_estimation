import time

import numpy as np

from keras.layers import Dense, Embedding, TimeDistributed, LSTM
from keras.layers import Activation
from keras.layers import Flatten
from keras.layers import Dropout, BatchNormalization


class _AbstractEstimator:
    '''
    Protected Class storing common methods inherited by actual estimator
    '''
    def _generate_embedding_block(self, input_tensor, input_dim, output_dim,
                                  input_length, tag):
        '''
        Protected method generating an embedding block, composed by an
        embedding and flatten layer.

        Args:
            - input_tensor:a keras tensor, input of the EB
            - input_dim: an integer, number of unique categories
            - output_dim: an integer, number of values composing the dense
                          rappresentation of each category
            - input_length: an integer, lenght of the sequcne of categorical
                            features
            - tag: a string, identifier for the block

        Returns:
            - embedding: is a keras tensor, portion of graph generated by EB
        '''
        embedding = Embedding(
            input_dim=input_dim,
            output_dim=output_dim,
            input_length=input_length,
            name='embedding_layer_{}'.format(tag)
            )(input_tensor)
        embedding = Flatten(
            name='flatten_embedding_layer_{}'.format(tag)
            )(embedding)
        return embedding

    def _generate_fully_connected_block(self, input_tensor, layers, activation,
                                        dropout, tag, prob):
        '''
        Protected method genearting a block of fuly connected layers (FCB)
        Each block can have an arbitrary number of layer each one having an
        arbitrary number of units

        Batch normalization and dropout are applied between each layer

        Args:
            - input_tensor: a keras tensor, input of the FCB
            - layers:       a list, specifying number of layers and units

              example: [100, 100, 100] <-- 3 layers 100 units each

            - activation:   a string, activation function applied to each unit
            - dropout:      a float, probability of a unit being dropped
            - tag:          a string, identifier for the block
            - prob:         a bolean, wheather to apply dropout at test time
                            allowing the model to be probabilistic

        Returns:
            - fully_connected: a keras tesnor, portion of graph generated
                               by FCB
        '''
        if len(layers) == 0:
            return input_tensor
        for layer, units in enumerate(layers):

            if layer == 0:
                fully_connected = Dense(
                    units=units,
                    name='{}_dense_layer_{}'.format(layer, tag)
                    )(input_tensor)
                fully_connected = BatchNormalization(
                    name='{}_bn_dense_layer_{}'.format(layer, tag)
                    )(fully_connected)
                fully_connected = Activation(
                    activation,
                    name='{}_activation_dense_layer_{}'.format(layer, tag)
                    )(fully_connected)
                fully_connected = Dropout(
                    rate=dropout,
                    name='{}_dropout_dense_layer_{}'.format(layer, tag)
                    )(fully_connected,  training=prob)
            else:
                fully_connected = Dense(
                    units=units,
                    name='{}_dense_layer_{}'.format(layer, tag)
                    )(fully_connected)
                fully_connected = BatchNormalization(
                    name='{}_bn_dense_layer_{}'.format(layer, tag)
                    )(fully_connected)
                fully_connected = Activation(
                    activation,
                    name='{}_activation_dense_layer_{}'.format(layer, tag)
                    )(fully_connected)
                fully_connected = Dropout(
                    rate=dropout,
                    name='{}_dropout_dense_layer_{}'.format(layer, tag)
                    )(fully_connected, training=prob)

        return fully_connected

    def _generate_time_distriuted_block(self, input_tensor, layers, activation,
                                        dropout, tag, prob):
        '''
        Protected method genearting a block of time distributed fully connected
        layers (TDB). Each block can have an arbitrary number of layer
        each one having an arbitrary number of units. Time distributed layers
        apply a distinct fully connected layer to each step in  a time series

        example: input --> [[x11, x12, x13], [x21, x22, x23], [x31, x32, x33]]
                 TDB --> [2]
                 output -->[[z11, z12], [z21, z22], [z31, z32]]

        Dropout is applied between each layer

        Args:
            - input_tensor: a keras tensor, input of the FCB
            - layers:       a list, specifying number of layers and units

              example: [100, 100, 100] <-- 3 layers 100 units each

            - activation:   a string, activation function applied to each unit
            - dropout:      a float, probability of a unit being dropped
            - tag:          a string, identifier for the block
            - prob:         a bolean, wheather to apply dropout at test time
                            allowing the model to be probabilistic

        Returns:
            - time_distributed: a keras tesnor, portion of graph generated
                                by TDB
        '''
        if len(layers) == 0:
            return input_tensor
        for layer, units in enumerate(layers):

            if layer == 0:
                time_distributed = TimeDistributed(
                    layer=Dense(
                        units=units,
                        activation='relu'
                        ),
                    name='{}_time_distributed_dense_{}'.format(layer, tag)
                    )(input_tensor)
                time_distributed = TimeDistributed(
                    layer=Dropout(
                        rate=dropout
                        ),
                    name='{}_distributed_dropout_{}'.format(layer, tag)
                    )(time_distributed, training=prob)
            else:
                time_distributed = TimeDistributed(
                    layer=Dense(
                        units=units,
                        activation='relu'
                        ),
                    name='{}_time_distributed_dense_{}'.format(layer, tag)
                    )(time_distributed)
                time_distributed = TimeDistributed(
                    layer=Dropout(
                        rate=dropout
                        ),
                    name='{}_distributed_dropout_{}'.format(layer, tag)
                    )(time_distributed, training=prob)

        return time_distributed

    def _generate_recurrent_block(self, input_tensor, layers, tag):
        '''
        Protected method genearting a block of reccurrent layers (LSTM)
        .Each block can have an arbitrary number of layers
        each one having an arbitrary number of LSTM cells. Recurrent layers
        apply their trasformations sequentially over the input
        time-steps taking this way temporality into account

        Args:
            - input_tensor: a keras tensor, input of the FCB
            - layers:       a list, specifying number of layers and units

              example: [100, 100, 100] <-- 3 layers 100 LSTM cells each
            - tag:          a string, identifier for the block

        Returns:
            - recurrnt: a keras tesnor, portion of graph generated by
                        recurrent block
        '''
        if len(layers) == 0:
            return input_tensor
        return_sequences = len(layers) > 1
        for layer, units in enumerate(layers):

            if layer == 0:
                recurrent = LSTM(
                    units=units,
                    return_sequences=return_sequences,
                    name='{}_LSTM_layer_{}'.format(layer, tag)
                    )(input_tensor)
            elif layer == len(layers) - 1:
                recurrent = LSTM(
                    units=units,
                    name='{}_LSTM_layer_{}'.format(layer, tag)
                    )(recurrent)
            else:
                recurrent = LSTM(
                    units=units,
                    return_sequences=return_sequences,
                    name='{}_LSTM_layer_{}'.format(layer, tag)
                    )(recurrent)

        return recurrent

    def get_para_count(self):
        '''
        Method wrapping Keras count_params method so that is not mandatory
        to access the underlying Keras model

        Returns:
            -num_parameters: an integer specifying the number of parameters
                             for the model
        '''
        num_parameters = self.n_parameters
        return num_parameters

    def get_model_tag(self):
        '''
        Method for getting the model tag (identifier)

        Returns:
            -model_tag: a string specifying the model tag
        '''
        model_tag = self.model_tag
        return model_tag

    def get_fitting_time(self):
        '''
        Method for getting the time (on second) taken by the model to fit
        the data (untill convergence)

        Returns:
            -fitting_time: integer specifying the fitting time in seconds
        '''
        fitting_time = self.fitting_time
        return fitting_time

    def fit(self, **kwargs):
        '''
        Method wrapping Keras fit method so that is not mandatory to access
        the underlying Keras model

        Args:
         - **kwargs: keyword arguments usually passed to Keras fit method

        '''
        start = time.time()
        self._model.fit(**kwargs)
        end = time.time()
        setattr(self, 'fitting_time', end - start)

    def predict(self, **kwargs):
        '''
        Method wrapping Keras fit method so that is not mandatory to access
        the underlying Keras model

        Args:
          - **kwargs: keyword arguments usually passed to Keras fit method

        Returns:
          - prediction: array or list of arrays
                        storing the model predictions

        '''
        prediction = self._model.predict(**kwargs)
        return prediction

    def predict_with_uncertainty(self, X_test, y_test, n_iter, batch_size,
                                 verbose=1):
        '''
        Method for perfroming predictions while including uncertainty
        The reference paper is: http://proceedings.mlr.press/v48/gal16.pdf

        In practice, this method will call the predict method of the model
        n_iter times and will 'glue' the reuslts in a numpy array of shape
        (number of targets, number of iteration, shape of the output)

        example with 3 targets, 3 iterations and 3 dimensional output
        (softmax):

        [[[0.6, 0.3, 0.1],
          [0.3, 0.6, 0.1],
          [0.4, 0.4, 0.2]],

         [[0.2, 0.7, 0.1],
          [0.2, 0.6, 0.2],
          [0.1, 0.8, 0.1]],

         [[0.3, 0.5, 0.2],
          [0.1, 0.3, 0.6],
          [0.1, 0.1, 0.8]]]

        For having an estimate of uncertainty we can then compute statistics
        (mu and sigma) over the number of iterations axis or directly visualize
        the distribution of values. It is likely that
        more iterations ---> more time and more precise estimates

        Args:
            - X_test: numpy array, input features for prediction
            - y_test: numpy array, input target from the train set
            - n_iter: integer, number of iterations perfromed
            - batch_size: integer, size of the input batch
            - verbose: integer, ammount of verbosity fro the predict method

        Returns:
            - predictions: numpy array of shape
                           (n_targets, n_iter, target_shape)
                           predictions perfromed by the model
        '''
        if self.prob is not True or self.hp_schema['dropout'] == 0:
            raise ValueError('Trying to predict with uncertainty using \
                non-probabilistic model.')
        predictions = np.empty(
            shape=(y_test.shape[0], n_iter, y_test.shape[1])
            )
        for iter in range(n_iter):

            prediction = self._model.predict(
                X_test,
                batch_size=batch_size,
                verbose=verbose
                )
            predictions[:, iter, :] = prediction

        return predictions
