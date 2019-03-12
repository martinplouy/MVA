from AttentionWithContext import AttentionWithContext
import sys
import json
import numpy as np
import pandas as pd
import os

from keras.callbacks import EarlyStopping, ModelCheckpoint

from keras.models import Model
from keras.layers import Input, Embedding, Dropout, Bidirectional, GRU, CuDNNGRU, TimeDistributed, Dense

from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasRegressor

# = = = = = = = = = = = = = = =

is_GPU = True
save_weights = True
save_history = True

path_root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
path_to_code = path_root + '/code/'
path_to_data = path_root + '/data/'

sys.path.insert(0, path_to_code)

# = = = = = = = = = = = = = = =


def bidir_gru(my_seq, n_units, is_GPU):
    '''
    just a convenient wrapper for bidirectional RNN with GRU units
    enables CUDA acceleration on GPU
    # regardless of whether training is done on GPU, model can be loaded on CPU
    # see: https://github.com/keras-team/keras/pull/9112
    '''
    if is_GPU:
        return Bidirectional(CuDNNGRU(units=n_units,
                                      return_sequences=True),
                             merge_mode='concat', weights=None)(my_seq)
    else:
        return Bidirectional(GRU(units=n_units,
                                 activation='tanh',
                                 dropout=0.0,
                                 recurrent_dropout=0.0,
                                 implementation=1,
                                 return_sequences=True,
                                 reset_after=True,
                                 recurrent_activation='sigmoid'),
                             merge_mode='concat', weights=None)(my_seq)


def create_model(n_units_sentence=10, n_units_doc=10, drop_rate=0.5, my_optimizer='nadam'):
    sent_ints = Input(shape=(docs_train.shape[2],))
    sent_wv = Embedding(input_dim=embeddings.shape[0],
                        output_dim=embeddings.shape[1],
                        weights=[embeddings],
                        input_length=docs_train.shape[2],
                        trainable=False,
                        )(sent_ints)

    sent_wv_dr = Dropout(drop_rate)(sent_wv)
    sent_wa = bidir_gru(sent_wv_dr, n_units_sentence, is_GPU)
    sent_att_vec, word_att_coeffs = AttentionWithContext(
        return_coefficients=True)(sent_wa)
    sent_att_vec_dr = Dropout(drop_rate)(sent_att_vec)
    sent_encoder = Model(sent_ints, sent_att_vec_dr)

    doc_ints = Input(shape=(docs_train.shape[1], docs_train.shape[2],))
    sent_att_vecs_dr = TimeDistributed(sent_encoder)(doc_ints)
    doc_sa = bidir_gru(sent_att_vecs_dr, n_units_doc, is_GPU)
    doc_att_vec, sent_att_coeffs = AttentionWithContext(
        return_coefficients=True)(doc_sa)
    doc_att_vec_dr = Dropout(drop_rate)(doc_att_vec)

    preds = Dense(units=1)(doc_att_vec_dr)
    model = Model(doc_ints, preds)

    model.compile(loss='mean_squared_error',
                  optimizer=my_optimizer,
                  metrics=['mae'])
    return model


# = = = = = hyper-parameters = = = = =

#n_units = 10
#drop_rate = 0.5
#batch_size = 96
#my_optimizer = 'adam'
my_patience = 3

parameters = {
    'n_units_sentence': [10, 50],
    'n_units_doc': [10, 50],
    'drop_rate': [0.3, 0.5],
    'my_optimizer': ['nadam'],
    "batch_size": [16, 96, 256],
    "epochs": [5, 10]
}

# = = = = = data loading = = = = =

docs = np.load(path_to_data + 'documents.npy')
embeddings = np.load(path_to_data + 'embeddings.npy')

with open(path_to_data + 'train_idxs.txt', 'r') as file:
    train_idxs = file.read().splitlines()

train_idxs = [int(elt) for elt in train_idxs]

# create validation set
np.random.seed(12219)
idxs_select_train = np.random.choice(
    range(len(train_idxs)), size=int(len(train_idxs)*0.80), replace=False)
idxs_select_val = np.setdiff1d(range(len(train_idxs)), idxs_select_train)

idxs_select_train = idxs_select_train[0:20000]
idxs_select_val = idxs_select_val[0:20000]

train_idxs_new = [train_idxs[elt] for elt in idxs_select_train]
val_idxs = [train_idxs[elt] for elt in idxs_select_val]

docs_train = docs[train_idxs_new, :, :]
docs_val = docs[val_idxs, :, :]

best_params = []
for tgt in range(3, 4):

    with open(path_to_data + 'targets/train/target_' + str(tgt) + '.txt', 'r') as file:
        target = file.read().splitlines()

    target_train = np.array([target[elt]
                             for elt in idxs_select_train]).astype('float')
    target_val = np.array([target[elt]
                           for elt in idxs_select_val]).astype('float')

    print('data loaded')

    # = = = = = defining architecture = = = = =

    model = KerasRegressor(build_fn=create_model)

    print('model compiled')

    grid = GridSearchCV(
        estimator=model, param_grid=parameters, verbose=10, cv=2)

    # = = = = = training = = = = =

    early_stopping = EarlyStopping(monitor='val_loss',
                                   patience=my_patience,
                                   mode='min')

    # save model corresponding to best epoch
    checkpointer = ModelCheckpoint(filepath=path_to_data + 'model_' + str(tgt),
                                   verbose=1,
                                   save_best_only=True,
                                   save_weights_only=True)

    if save_weights:
        my_callbacks = [early_stopping, checkpointer]
    else:
        my_callbacks = [early_stopping]

    grid.fit(docs_train,
             target_train,
             validation_data=(docs_val, target_val),
             callbacks=my_callbacks)

    df = pd.DataFrame(grid.cv_results_)
    df.to_csv(path_to_data + "cv_results_target_" + str(tgt) + ".csv")
    best_params.append(grid.best_params_)

    # hist = model.history.history

    # if save_history:
    #     with open(path_to_data + 'model_history_' + str(tgt) + '.json', 'w') as file:
    #         json.dump(hist, file, sort_keys=False, indent=4)

    print('* * * * * * * target', tgt, 'done * * * * * * *')

for tgt in range(len(best_params)):
    print("the best params for target ", str(tgt), " are ", best_params[tgt])
