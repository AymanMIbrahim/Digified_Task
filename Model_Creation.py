from tensorflow import keras
from tensorflow.keras import layers,models
from sklearn.model_selection import train_test_split
from tensorflow.keras.regularizers import L1L2
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.callbacks import ReduceLROnPlateau
import tensorflow as tf
import pickle


def create_model():
    with open('Dataset_Data.pkl', 'rb') as f:
        X = pickle.load(f)

    with open('Dataset_Target.pkl', 'rb') as f:
        Y = pickle.load(f)

    print("Data Shape: ",X.shape)
    print("Target Shape: ", Y.shape)

    input_ = layers.Input(shape=[X.shape[1], X.shape[2]], name='input')
    x = layers.Bidirectional(layers.LSTM(32, dropout=0.2, return_sequences=True), name='bidirectional-lstm_1')(
        input_)
    x = layers.Dropout(0.2, name='dropout1')(x)
    x = layers.Bidirectional(layers.LSTM(16, dropout=0.2, kernel_regularizer=L1L2(l1=0.01, l2=0.0)),
                             name='bidirectional-lstm_4')(x)
    x = layers.Dense(64, activation='relu', name='dense')(x)
    output = layers.Dense(2, activation='softmax', name='classification')(x)
    model = models.Model(input_, output)
    opt = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9)
    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    print(model.summary())

    return model

def train(model,base_path,version,patience=50,factor=0.1):

    with open('Dataset_Data.pkl', 'rb') as f:
        X = pickle.load(f)

    with open('Dataset_Target.pkl', 'rb') as f:
        Y = pickle.load(f)

    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, shuffle=True)

    early_stop = EarlyStopping('val_accuracy', patience=patience)
    reduce_lr = ReduceLROnPlateau('val_accuracy', factor=factor,
                                  patience=int(patience / 4), verbose=1)

    trained_models_path = base_path + 'BERT_BI-LSTM-' + version
    model_names = trained_models_path + "_" + "Digified_Text_Classification" + '_{epoch:02d}_{val_accuracy:.2f}.hdf5'
    model_checkpoint = ModelCheckpoint(model_names, 'val_accuracy', verbose=1,
                                       save_best_only=True)
    callbacks = [model_checkpoint, early_stop, reduce_lr]


    model.fit(x=X_train, y=Y_train, validation_data=(X_val, Y_val), epochs=2000, batch_size=8
              , verbose=1, shuffle=True, callbacks=callbacks)