import numpy as np

import pandas as pd

from sklearn.metrics import f1_score

from keras.regularizers import l1, l1_l2
from keras.callbacks import EarlyStopping
from keras_contrib.callbacks.cyclical_learning_rate import CyclicLR

from modules.utilities.general_utilities import smape_np, smape_k, load_arrays
from modules.models import MeanModel, LinearModel
from modules.models import MultilayerPerceptron, BifurcatingModel

###############################################################################
#                           hyperparameters                                   #
###############################################################################

# the hp are a slight adaptation of those employed in
# https://arxiv.org/abs/1905.10998 we wanted to test if the better perfomance
# of the BM was caused by it having more parameters various tested
# configurations suggest that is the architecteure providing the better
# perfromance for consistency all the models here are developed in Keras
enet_schemas = {'hp_schema': {'dropout': 0.0,
                              'regularizer': l1_l2(l1=0.5, l2=0.5)
                              },

                'comp_schema': {'optimizer': 'adam',
                                'loss': smape_k,
                                'metrics': ['mean_absolute_error']
                                }
                }

logistic_schemas = {'hp_schema': {'dropout': 0.0,
                                  'regularizer': l1(0.01)
                                  },

                    'comp_schema': {'optimizer': 'adam',
                                    'loss': 'binary_crossentropy',
                                    'metrics': ['acc']
                                    }

                    }

perceptron_r_schemas = {'hp_schema': {'layers': [90, 90, 90],
                                      'activation': 'relu',
                                      'dropout': 0.0
                                      },

                        'comp_schema': {'optimizer': 'adam',
                                        'loss': smape_k,
                                        'metrics': ['mean_absolute_error']
                                        }
                        }

perceptron_c_schemas = {'hp_schema': {'layers': [90, 90, 90],
                                      'activation': 'relu',
                                      'dropout': 0.0
                                      },

                        'comp_schema': {'optimizer': 'adam',
                                        'loss': 'binary_crossentropy',
                                        'metrics': ['acc']
                                        }
                        }

bifurcating_schemas = {'hp_schema': {'em_schema': {'input_dim': 10,
                                                   'output_dim': 10
                                                   },

                                     'td_schema': {'layers': [25],
                                                   'activation': 'relu',
                                                   'dropout': 0.2
                                                   },

                                     're_schema': {'layers': [50]},

                                     'fc_schema': {'layers': [100],
                                                   'activation': 'relu',
                                                   'dropout': 0.0
                                                   }
                                     },

                       'comp_schema': {'optimizer': 'adam',
                                       'losses': [smape_k,
                                                  'binary_crossentropy'
                                                  ],
                                       'metrics': ['mean_absolute_error',
                                                   'acc'
                                                   ],
                                       'loss_weights': [1.0, 1.0]
                                       }
                       }

###############################################################################
#                                  test pipeline                              #
###############################################################################

results = pd.DataFrame(
    columns=[
        'estimator',
        'n_parameters',
        'context',
        'metric',
        'score',
        'fitting_time'
    ]
)
index = 0

###############################################################################
#                        collapsed + unrolled data                            #
###############################################################################

for data_format in ['collapsed', 'unrolled']:

    loaded_arrays = load_arrays(arrays=['X_tr', 'X_ts',
                                        'y_r_tr', 'y_r_ts',
                                        'y_c_tr', 'y_c_ts',
                                        'context_tr', 'context_ts'
                                        ],
                                dir_name=data_format
                                )

###############################################################################
#                             survival estimation                             #
###############################################################################

    enet = LinearModel(
        X_train=loaded_arrays['X_tr'],
        y_train=loaded_arrays['y_r_tr']
        )
    enet.generate_model(
        hp_schema=enet_schemas['hp_schema'],
        comp_schema=enet_schemas['comp_schema'],
        regression=True,
        model_tag='enet_{}'.format(data_format)
        )

    perceptron_r = MultilayerPerceptron(
        X_train=loaded_arrays['X_tr'],
        y_train=loaded_arrays['y_r_tr']
        )
    perceptron_r.generate_model(
        hp_schema=perceptron_r_schemas['hp_schema'],
        comp_schema=perceptron_r_schemas['comp_schema'],
        regression=True,
        model_tag='mlp_r_{}'.format(data_format)
        )

    models = [enet, perceptron_r]
    for model in models:

        print('Testing model {}'.format(model.get_model_tag()))
        lr_scheduler = CyclicLR(mode='triangular2')
        early_stopping = EarlyStopping(
            monitor='val_loss',
            min_delta=0.001,
            patience=4,
            restore_best_weights=True
            )
        model.fit(
            x=loaded_arrays['X_tr'],
            y=loaded_arrays['y_r_tr'],
            batch_size=256,
            epochs=1,
            verbose=2,
            validation_split=0.2,
            callbacks=[lr_scheduler, early_stopping]
        )

        for context in np.unique(loaded_arrays['context_ts']):

            indices = np.argwhere(
                loaded_arrays['context_ts'] == context
                ).flatten()
            prediction = model.predict(
                x=loaded_arrays['X_ts'][indices]
                )
            score = smape_np(
                y_true=loaded_arrays['y_r_ts'][indices],
                y_pred=prediction
                )
            results.loc[index] = [
                model.get_model_tag(),
                model.get_para_count(),
                context,
                'smape',
                score,
                model.get_fitting_time()
            ]
            index += 1

###############################################################################
#                      churn probability estimation                           #
###############################################################################

    logistic = LinearModel(
        X_train=loaded_arrays['X_tr'],
        y_train=loaded_arrays['y_c_tr']
        )
    logistic.generate_model(
        hp_schema=logistic_schemas['hp_schema'],
        comp_schema=logistic_schemas['comp_schema'],
        regression=False,
        model_tag='logistic_{}'.format(data_format),
        )

    perceptron_c = MultilayerPerceptron(
        X_train=loaded_arrays['X_tr'],
        y_train=loaded_arrays['y_c_tr']
        )
    perceptron_c.generate_model(
        hp_schema=perceptron_c_schemas['hp_schema'],
        comp_schema=perceptron_c_schemas['comp_schema'],
        regression=False,
        model_tag='mlp_c_{}'.format(data_format)
        )

    models = [logistic, perceptron_c]
    for model in models:

        print('Testing model {}'.format(model.get_model_tag()))
        lr_scheduler = CyclicLR(mode='triangular2')
        early_stopping = EarlyStopping(monitor='val_loss',
                                       min_delta=0.001,
                                       patience=4,
                                       restore_best_weights=True
                                       )
        model.fit(
            x=loaded_arrays['X_tr'],
            y=loaded_arrays['y_c_tr'],
            batch_size=256,
            epochs=1,
            verbose=2,
            validation_split=0.2,
            callbacks=[lr_scheduler, early_stopping]
        )

        for context in np.unique(loaded_arrays['context_ts']):

            indices = np.argwhere(
                loaded_arrays['context_ts'] == context
                ).flatten()
            prediction = model.predict(
                x=loaded_arrays['X_ts'][indices]
                )
            score = f1_score(
                y_true=loaded_arrays['y_c_ts'][indices],
                y_pred=np.around(prediction.flatten()),
                average='macro'
                )
            results.loc[index] = [
                model.get_model_tag(),
                model.get_para_count(),
                context,
                'f1',
                score,
                model.get_fitting_time()
            ]
            index += 1

###############################################################################
#                               temporal data                                 #
###############################################################################

loaded_arrays = load_arrays(arrays=['X_feat_tr', 'X_feat_ts',
                                    'X_cont_tr', 'X_cont_ts',
                                    'y_r_tr', 'y_r_ts',
                                    'y_c_tr', 'y_c_ts',
                                    'context_tr', 'context_ts',
                                    ],
                            dir_name='temporal'
                            )

###############################################################################
#                                 baseline                                    #
###############################################################################

# baseline model does not care about the data format
baseline_r = MeanModel(
    X_train=loaded_arrays['X_feat_tr'],
    y_train=loaded_arrays['y_r_tr']
)
baseline_r.generate_model()

baseline_c = MeanModel(
    X_train=loaded_arrays['X_feat_tr'],
    y_train=loaded_arrays['y_c_tr']
)
baseline_c.generate_model()
print('Testing baseline models')
# we 'fit' the baseline models context wise for making a fair comparison
for context in np.unique(loaded_arrays['context_ts']):

    # fitting the model context-wise
    indices = np.argwhere(
        loaded_arrays['context_ts'] == context
        ).flatten()

    baseline_r.fit(
        x=loaded_arrays['X_feat_tr'][indices],
        y=loaded_arrays['y_r_tr'][indices]
    )
    baseline_c.fit(
        x=loaded_arrays['X_feat_tr'][indices],
        y=loaded_arrays['y_c_tr'][indices]
    )

    # predict context-wise
    prediction = baseline_r.predict(
        x=loaded_arrays['X_feat_tr'][indices]
    )
    score = smape_np(
        y_true=loaded_arrays['y_r_ts'][indices],
        y_pred=prediction
        )
    results.loc[index] = [
        baseline_r.get_model_tag(),
        baseline_r.get_para_count(),
        context,
        'smape',
        score,
        baseline_r.get_fitting_time()
    ]
    index += 1

    prediction = baseline_c.predict(
        x=loaded_arrays['X_feat_tr'][indices]
    )
    print(prediction)
    score = f1_score(
        y_true=loaded_arrays['y_c_ts'][indices],
        y_pred=np.around(prediction),
        average='macro'
        )
    results.loc[index] = [
        baseline_c.get_model_tag(),
        baseline_c.get_para_count(),
        context,
        'f1',
        score,
        baseline_c.get_fitting_time()
    ]
    index += 1

###############################################################################
#                             bifurcating model                               #
###############################################################################

bifurcating = BifurcatingModel(
    X_train=[
        loaded_arrays['X_feat_tr'],
        loaded_arrays['X_cont_tr']
        ],
    y_train=[
        loaded_arrays['y_r_tr'],
        loaded_arrays['y_c_tr']
        ]
    )
bifurcating.generate_model(
    hp_schema=bifurcating_schemas['hp_schema'],
    comp_schema=bifurcating_schemas['comp_schema'],
    sequence_len=20,
    masked=0.0,
    model_tag='bifurcating_temporal',
    )

print('Testing model {}'.format(bifurcating.get_model_tag()))
bifurcating.fit(x=[
            loaded_arrays['X_feat_tr'],
            loaded_arrays['X_cont_tr']
            ],
            y=[
            loaded_arrays['y_r_tr'],
            loaded_arrays['y_c_tr']
            ],
            batch_size=256,
            epochs=1,
            verbose=2,
            validation_split=0.2,
            callbacks=[lr_scheduler, early_stopping]
            )
for context in np.unique(loaded_arrays['context_ts']):

    indices = np.argwhere(
        loaded_arrays['context_ts'] == context
        ).flatten()
    prediction = bifurcating.predict(x=[
        loaded_arrays['X_feat_ts'][indices],
        loaded_arrays['X_cont_ts'][indices]
        ]
    )
    score = smape_np(
        y_true=loaded_arrays['y_r_ts'][indices],
        y_pred=prediction[0]
        )
    results.loc[index] = [
        bifurcating.get_model_tag(),
        bifurcating.get_para_count(),
        context,
        'smape',
        score,
        model.get_fitting_time()
    ]
    index += 1
    score = f1_score(
        y_true=loaded_arrays['y_c_ts'][indices],
        y_pred=np.around(prediction[1].flatten()),
        average='macro'
        )
    results.loc[index] = [
        bifurcating.get_model_tag(),
        bifurcating.get_para_count(),
        context,
        'f1',
        score,
        bifurcating.get_fitting_time()
    ]
    index += 1

results.to_csv('results\\models_comparison.csv')
print(results)
