from AttentionWithContext import AttentionWithContext
import sys
import json
import numpy as np
from sklearn.metrics import mean_squared_error
import os

from keras.callbacks import EarlyStopping, ModelCheckpoint

from keras.models import Model
from keras.layers import Input, Embedding, Dropout, Bidirectional, GRU, CuDNNGRU, TimeDistributed, Dense

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


# = = = = = hyper-parameters = = = = =

n_units = 10
drop_rate = 0.3
batch_size = 96
nb_epochs = 5
my_optimizer = 'nadam'
my_patience = 4

# this flag is meant to preprocess only a subsample of the data so that we can test quickly what works and what doesn't work
quick_run = False
# = = = = = data loading = = = = =

docs = np.load(path_to_data + 'documents.npy')
embeddings = np.load(path_to_data + 'embeddings.npy')

with open(path_to_data + 'train_idxs.txt', 'r') as file:
    train_idxs = file.read().splitlines()

train_idxs = [int(elt) for elt in train_idxs]

train_idxs_map = {}
for (idx, elt) in enumerate(train_idxs):
    train_idxs_map[elt] = idx


if quick_run:
    with open(path_to_data + 'graphlists.txt', 'r') as file:
        graphlists = file.read().splitlines()
    graphlists = [int(elt) for elt in graphlists]
    train_idxs = graphlists

    # create validation set
    np.random.seed(12219)
    idxs_select_train = np.random.choice(
        range(len(train_idxs)), size=int(len(train_idxs)*0.8), replace=False)
    idxs_select_val_test = np.setdiff1d(
        range(len(train_idxs)), idxs_select_train)

    idxs_select_val = np.random.choice(
        range(len(idxs_select_val_test)), size=int(len(idxs_select_val_test)*0.5), replace=False)

    idxs_select_test = np.setdiff1d(
        range(len(idxs_select_val_test)), idxs_select_val)

    train_idxs_new = [train_idxs[elt] for elt in idxs_select_train]
    val_idxs = [train_idxs[elt] for elt in idxs_select_val]
    test_idxs = [train_idxs[elt] for elt in idxs_select_test]

    docs_train = docs[idxs_select_train, :, :]
    docs_val = docs[idxs_select_val, :, :]
    docs_test = docs[idxs_select_test, :, :]
else:
    # create validation set
    np.random.seed(12219)
    idxs_select_train = np.random.choice(
        range(len(train_idxs)), size=int(len(train_idxs)*0.80), replace=False)
    idxs_select_val = np.setdiff1d(range(len(train_idxs)), idxs_select_train)
    idxs_select_val_test = np.setdiff1d(
        range(len(train_idxs)), idxs_select_train)

    idxs_select_val = np.random.choice(
        range(len(idxs_select_val_test)), size=int(len(idxs_select_val_test)*0.5), replace=False)

    idxs_select_test = np.setdiff1d(
        range(len(idxs_select_val_test)), idxs_select_val)

    train_idxs_new = [train_idxs[elt] for elt in idxs_select_train]
    val_idxs = [train_idxs[elt] for elt in idxs_select_val]
    test_idxs = [train_idxs[elt] for elt in idxs_select_test]

    docs_train = docs[train_idxs_new, :, :]
    docs_val = docs[val_idxs, :, :]
    docs_test = docs[test_idxs, :, :]


mse_arr = []

n_units_arr_doc = [50, 50, 50, 50]
n_units_arr_sentence = [50, 50, 50, 10]
optimizer = ["nadam", 'adam', 'adam', 'nadam']
drop_rate = [0.3, 0.3, 0.3, 0.3]
batch_sizes = [96, 96, 96, 256]

for tgt in range(3, 4):

    print('* * * * * * * training model ', tgt, ' * * * * * * *')

    with open(path_to_data + 'targets/train/target_' + str(tgt) + '.txt', 'r') as file:
        target = file.read().splitlines()

    target_train = np.array([target[train_idxs_map.get(elt)]
                             for elt in train_idxs_new]).astype('float')
    target_val = np.array([target[train_idxs_map.get(elt)]
                           for elt in val_idxs]).astype('float')

    # = = = = = defining architecture = = = = =

    sent_ints = Input(shape=(docs_train.shape[2],))

    sent_wv = Embedding(input_dim=embeddings.shape[0],
                        output_dim=embeddings.shape[1],
                        weights=[embeddings],
                        input_length=docs_train.shape[2],
                        trainable=False,
                        )(sent_ints)

    sent_wv_dr = Dropout(drop_rate[tgt])(sent_wv)
    sent_wa = bidir_gru(sent_wv_dr, n_units_arr_sentence[tgt], is_GPU)
    sent_att_vec, word_att_coeffs = AttentionWithContext(
        return_coefficients=True)(sent_wa)
    sent_att_vec_dr = Dropout(drop_rate[tgt])(sent_att_vec)
    sent_encoder = Model(sent_ints, sent_att_vec_dr)

    doc_ints = Input(shape=(docs_train.shape[1], docs_train.shape[2],))
    sent_att_vecs_dr = TimeDistributed(sent_encoder)(doc_ints)
    doc_sa = bidir_gru(sent_att_vecs_dr, n_units_arr_doc[tgt], is_GPU)
    doc_att_vec, sent_att_coeffs = AttentionWithContext(
        return_coefficients=True)(doc_sa)
    doc_att_vec_dr = Dropout(drop_rate[tgt])(doc_att_vec)

    preds = Dense(units=1)(doc_att_vec_dr)
    model = Model(doc_ints, preds)

    model.compile(loss='mean_squared_error',
                  optimizer=optimizer[tgt],
                  metrics=['mae'])

    print('model compiled')

    # = = = = = training = = = = =

    early_stopping = EarlyStopping(monitor='val_loss',
                                   patience=my_patience,
                                   mode='min')

    # save model corresponding to best epoch
    checkpointer = ModelCheckpoint(filepath=path_to_data + 'model_' + str(tgt),
                                   verbose=1,
                                   save_best_only=True,
                                   save_weights_only=True)

    if save_weights and not quick_run:
        my_callbacks = [early_stopping, checkpointer]
    else:
        my_callbacks = [early_stopping]

    model.fit(docs_train,
              target_train,
              batch_size=batch_sizes[tgt],
              epochs=nb_epochs,
              validation_data=(docs_val, target_val),
              callbacks=my_callbacks)

    predict = model.predict(docs_test).tolist()
    target_test = np.array([target[train_idxs_map.get(elt)]
                            for elt in test_idxs]).astype('float')

    mse = mean_squared_error(target_test, predict)
    mse_arr.append(mse)

    hist = model.history.history

    if save_history:
        with open(path_to_data + 'model_history_' + str(tgt) + '.json', 'w') as file:
            json.dump(hist, file, sort_keys=False, indent=4)

    print('* * * * * * * target', tgt, 'done * * * * * * *')


print('* * * * * * * MSE for all targets is : ', np.mean(mse_arr))
