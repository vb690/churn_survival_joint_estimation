import numpy as np

from keras.models import Model

from keras.layers import Input, Dense, Activation
from keras.layers import Masking, Lambda
from keras.layers import Reshape, RepeatVector
from keras.layers import Dropout

import keras.backend as K

from modules.utilities.model_utilities import _AbstractEstimator


class MeanModel(_AbstractEstimator):
    '''
    Class implementing a baseline mean model where:

    y_hat = mean(y_train)
    '''
    def __init__(self, X_train, y_train):
        '''
        Method called when instatiating a MeanModel object, it is kept for
        consistency with the other models

        Args:
            - X_train: numpy array, input features
            - y_train: numpy array, input target
        '''
        self.model_tag = 'mean_model'
        self.n_parameters = 1
        self.fitting_time = 0

    def generate_model(self):
        '''
        Void method kept for consistency with the other models
        '''
        pass

    def fit(self, x, y, **kwrags):
        '''
        Method for 'fitting' the MeanModel, it only requires y but
        X is maintained for consistency

        Args:
            - X: numpy array, input features
            - y: numpy array, input target
        '''
        setattr(self, 'mean_estimation', np.mean(y))

    def predict(self, x, **kwrags):
        '''
        Method for performing 'prediction' with the MeanModel, it only
        requires the shape of the expected ground truth array but x is
        kept for consistency.
        '''
        prediction = np.repeat(self.mean_estimation, len(x))
        return prediction


class LinearModel(_AbstractEstimator):
    '''
    Class implementing a simple linear model, where:

    y_hat = b + w1 * x1 + ... + wn * xn

    It does not support predict_with_uncertainty, prediction has to be
    perfromed acessing the underlying keras model object once the
    generate_model() method is called.
    '''
    def __init__(self, X_train, y_train):
        '''
        Method called when instatiating a LinearModel object

        Args:
            - X_train: numpy array, input features
            - y_train: numpy array, input target

        Returns:
            -None
        '''
        self.__prob = False
        self.X_train = X_train.shape
        self.y_train = y_train.shape

    def generate_model(self, hp_schema, comp_schema, regression=True,
                       model_tag=None):
        '''
        Method for generating the tensorflow graph employed as linear model

        Args:
         - hp_schema: a dictionary, it stores the schema for
                      the hyperparameters

                      Example:
                      {'dropout' : float specifying the ammount of input
                                   features masked to zero,

                       'regularizer' : keras regularizer object, specify the
                                       type regularization applied to the
                                       weights matrix this allows to have lasso
                                       , ridge or elasticnet models
                        }
         - comp_schema: a dictionary, it stores the schema for
                        compiling the model

                        Example:
                        {optimizer: string or keras optimizer, optimization
                                    algorithm employed,

                         loss: string or keras loss, loss minimized by
                               the optimizer,

                         metrics: list of strings or keras metrics, additional
                                  metrics computed for monitoring convergence
                        }
         - regression: a bolean, it specify wether the model is target to a
                       regression or classification task
         - tag: a string, specify the model identifier applied to each layer

        Returns:
         - None
        '''
        if model_tag is None:
            model_tag = 'reg' if regression else 'clas'
        setattr(self, 'hp_schema', hp_schema)
        setattr(self, 'model_tag', model_tag)

        input = Input(
            shape=(self.X_train[1], ),
            name='input_{}'.format(model_tag)
        )
        dropout_input = Dropout(
            rate=self.hp_schema['dropout'],
            name='dopout_input_{}'.format(model_tag)
        )(input)
        matmul = Dense(
            units=self.y_train[1],
            name='matmul_{}'.format(model_tag),
            kernel_regularizer=self.hp_schema['regularizer']
        )(dropout_input)
        if regression:
            link = Activation(
                'linear',
                name='identity_link_{}'.format(model_tag)
            )(matmul)
        elif not regression and self.y_train[1] > 1:
            link = Activation(
                'softmax',
                name='softmax_link_{}'.format(model_tag)
             )(matmul)
        else:
            link = Activation(
                'sigmoid',
                name='sigmoid_link_{}'.format(model_tag)
            )(matmul)
        model = Model(inputs=input, outputs=link)
        model.compile(
            optimizer=comp_schema['optimizer'],
            loss=comp_schema['loss'],
            metrics=comp_schema['metrics']
        )

        setattr(self, '_model', model)
        setattr(self, 'n_parameters', model.count_params())


class MultilayerPerceptron(_AbstractEstimator):
    '''
    Class implementing a multilayer perceptron NN

    It does support predict_with_uncertainty, conventional prediction
    has to be perfromed acessing the underlying keras model object once
    the generate_model() method is called
    '''
    def __init__(self, X_train, y_train):
        '''
        Method called when instatiating a MultilayerPerceptron object

        Args:
            - X_train: numpy array, input features
            - y_train: numpy array, input target

        Returns:
            -None
        '''
        self.X_train = X_train.shape
        self.y_train = y_train.shape

    def generate_model(self, hp_schema, comp_schema, regression=True,
                       prob=False, model_tag=None):
        '''
        Method for generating the tensorflow graph employed as
        multilayer perceptron

        Args:
         - hp_schema: a dictionary, it stores the schema for
                      the hyperparameters

                      Example:
                      {layers : list of integers specifying number of layers
                                and hidden units ([100, 100, 100])
                       , 'dropout' : float specifying the ammount of hidden
                                     units masked to zero
                       , 'activation' : string sepcifying tha activation
                                        function applied between each layer
                                }
         - comp_schema: a dictionary, it stores the schema for
                        compiling the model
                        Example:
                        {optimizer: string or keras optimizer, optimization
                                    algorithm employed
                         , loss: string or keras loss, loss minimized by
                                 the optimizer
                         , metrics: list of strings or keras metrics,
                                    additional metrics computed for monitoring
                                    convergence
                                 }
         - regression: a bolean, it specify wether the model is target to a
                       regression or classification task
         - prob: a bolean, whether the model will employ dropout at estimation
                 time allowing for uncertainty estimation
         - tag: a string, specify the model identifier applied to each layer

        Returns:
         - None
        '''
        setattr(self, 'hp_schema', hp_schema)
        setattr(self, 'prob', prob)
        if model_tag is None:
            model_tag = 'reg' if regression else 'clas'
        setattr(self, 'model_tag', model_tag)

        input = Input(
            shape=(self.X_train[1],),
            name='input_{}'.format(model_tag)
        )
        fc_block = self._generate_fully_connected_block(
            input,
            self.hp_schema['layers'],
            self.hp_schema['activation'],
            self.hp_schema['dropout'],
            model_tag,
            prob,
        )
        act = Dense(
            units=self.y_train[1],
            name='act_{}'.format(model_tag)
        )(fc_block)
        if regression:
            act = Activation(
                'linear',
                name='identity_activation_{}'.format(model_tag)
            )(act)
        elif not regression and self.y_train[1] > 1:
            act = Activation(
                'softmax',
                name='softmax_activation_{}'.format(model_tag)
            )(act)
        else:
            act = Activation(
                'sigmoid',
                name='sigmoid_activation_{}'.format(model_tag)
            )(act)
        model = Model(inputs=input, outputs=act)
        model.compile(
            optimizer=comp_schema['optimizer'],
            loss=comp_schema['loss'],
            metrics=comp_schema['metrics']
        )

        setattr(self, '_model', model)
        setattr(self, 'n_parameters', model.count_params())


class BifurcatingModel(_AbstractEstimator):
    '''
    Class implementing a NN for joint estimation of
    regression and classification targets. The underlying assumption, however,
    is that the features are suitable for representing both the
    classification and regression targets (e.g. churn and survival time)

    It does support predict_with_uncertainty, conventional prediction
    has to be perfromed acessing the underlying keras model object once
    the generate_model() method is called
    '''
    def __init__(self, X_train, y_train):
        '''
        Method called when instatiating a MultilayerPerceptron object

        Args:
            - X_feat_train: numpy array, input features
            - X_cont_train: numpy array, input context
            - y_reg_train: numpy array, input regression target
            - y_clas_train: numpy array, input classification target

        Returns:
            -None
        '''
        self.X_feat_train = X_train[0].shape
        self.X_cont_train = X_train[1].shape
        self.y_reg_train = y_train[0].shape
        self.y_clas_train = y_train[1].shape

    def __apply_special_masking(self, tensors):
        '''
        Private method for concatenating a temporal tensor and a static tensor
        mantaining the masking in the temporal tensor. This method is employed
        by a lambda layer in the construction of the bifurcating model.

        Args
            - tensors: a list of keras tensors masked and not masked

        Return:
            - a masked concatenation of the two tesnors
        '''
        masked_tensor = tensors[0]
        non_masked_tensor = tensors[1]

        concat = K.concatenate([masked_tensor, non_masked_tensor])

        mask = K.any(K.not_equal(masked_tensor, self.masked), axis=2)
        mask = K.cast(mask, K.dtype(concat))
        mask = K.stack([mask], axis=2)
        return concat * mask

    def generate_model(self, hp_schema, comp_schema, sequence_len,
                       masked=0.0, prob=False, model_tag=None):
        '''
        Method for generating the tensorflow graph employed as
        bifurcating model

        Args:
         - hp_schema: a dictionary of dictionaries, it stores the schemas for
                      the hyperparameters for each compontet of the model
         - comp_schema: a dictionary, it stores the schema
                        for compiling the model
                        Example:
                            {optimizer: string or keras optimizer, optimization
                                        algorithm employed
                             , loss: string or keras loss,
                                     loss minimized by the optimizer
                             , metrics: list of strings or keras metrics,
                                        additional metrics computed for
                                        monitoring convergence
                             , loos_weights: list of floats, weights aplied
                                             to the two lossess of the model
                            }
         - sequence_len: is an integer specifying the maximum length of the
                         input sequence
         - masked: is a float, it specifies the value that will be masked
         - prob: a bolean, whether the model will employ dropout at estimation
                 time allowing for uncertainty estimation

        Returns:
         - None
        '''
        if model_tag is None:
            model_tag = 'bifurcating'
        setattr(self, 'masked', masked)
        setattr(self, 'prob', prob)
        setattr(self, 'em_schema', hp_schema['em_schema'])
        setattr(self, 'td_schema', hp_schema['td_schema'])
        setattr(self, 're_schema', hp_schema['re_schema'])
        setattr(self, 'fc_schema', hp_schema['fc_schema'])
        setattr(self, 'model_tag', model_tag)

        # inputs
        features_input = Input(
            shape=(self.X_feat_train[1], ),
            name='features_input'
        )
        features_reshape = Reshape(
            target_shape=(sequence_len, self.X_feat_train[1] // sequence_len),
            name='reshape_layer'
        )(features_input)

        context_input = Input(
            shape=(self.X_cont_train[1], ),
            name='context_input'
        )

        # embedding creation
        embedding = self._generate_embedding_block(
            input_tensor=context_input,
            input_dim=self.em_schema['input_dim'],
            output_dim=self.em_schema['output_dim'],
            input_length=self.X_cont_train[1],
            tag='shared'
        )

        # temporal fusion embedding and raw-features
        # repeat the result of the embedding for each timestep
        embedding = RepeatVector(
            sequence_len,
            name='embedding_temporalization_layer'
        )(embedding)
        if masked is not None:
            features_padded = Lambda(
                self.__apply_special_masking,
                name='re_padding_layer',
            )([features_reshape, embedding])
            features_padded = Masking(
                mask_value=masked,
                name='masking_layer'
            )(features_padded)
        else:
            features_padded = K.concatenate([features_reshape, embedding])
        time_distributed = self._generate_time_distriuted_block(
            input_tensor=features_padded,
            layers=self.td_schema['layers'],
            activation=self.td_schema['activation'],
            dropout=self.td_schema['dropout'],
            tag='shared',
            prob=prob
        )

        recurrent = self._generate_recurrent_block(
            input_tensor=time_distributed,
            layers=self.re_schema['layers'],
            tag='shared'
        )

        # regression head
        dense_reg = self._generate_fully_connected_block(
            input_tensor=recurrent,
            layers=self.fc_schema['layers'],
            activation=self.fc_schema['activation'],
            dropout=self.fc_schema['dropout'],
            tag='reg',
            prob=prob,
        )
        output_reg = Dense(
            units=self.y_reg_train[1],
            name='output_dense_reg'
        )(dense_reg)
        output_reg = Activation(
            'linear',
            name='output_linear_reg'
        )(output_reg)

        # classification head
        dense_clas = self._generate_fully_connected_block(
            input_tensor=recurrent,
            layers=self.fc_schema['layers'],
            activation=self.fc_schema['activation'],
            dropout=self.fc_schema['dropout'],
            tag='clas',
            prob=prob
        )

        output_clas = Dense(
            units=self.y_clas_train[1],
            name='output_dense_clas'
        )(dense_clas)
        output_clas = Activation(
            'sigmoid',
            name='output_sigmoid_clas'
        )(output_clas)

        # build and compile the model
        model = Model(
            inputs=[features_input, context_input],
            outputs=[output_reg, output_clas]
        )
        model.compile(
            optimizer=comp_schema['optimizer'],
            loss=comp_schema['losses'],
            metrics=comp_schema['metrics'],
            loss_weights=comp_schema['loss_weights']
        )

        setattr(self, '_model', model)
        setattr(self, 'n_parameters', model.count_params())
