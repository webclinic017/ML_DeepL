#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  2 18:51:37 2018

@author: yoelvis
"""
import numpy as np
#np.set_printoptions(threshold=np.nan)
np.random.seed(seed=7)
from tensorflow import set_random_seed, Print
set_random_seed(9)

from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint, Callback
from keras import Sequential
from keras.optimizers import Adam
from keras.layers import Convolution2D, Activation, regularizers, Reshape, LSTM, Dropout, TimeDistributed, Lambda, Dense

from keras.utils import to_categorical

from sklearn.preprocessing import MinMaxScaler
from pandas import DataFrame
from keras.models import model_from_yaml
from keras import backend as K

class Fitting_DeepL():
    def __init__(self,
                 ts,
                 putmodel, 
                 lags,
                 start_date, 
                 end_date, 
                 model,
                 ticker,
                 percent_factor,
                 skip,
                 nConv,
                 nLSTM,
                 learning,
                 batch_size,
                 conv_nodes,
                 lstm_nodes,
                 kernel_size
                 ):
        
        self.model = model
        self.modelfile = putmodel + ticker + ".pkl"
        self.putmodel = putmodel
        self.lags = lags
        self.percent_factor = percent_factor
        self.skip = skip
        self.nConv = nConv
        self.nLSTM = nLSTM
        self.learning = learning
        self.batch_size = batch_size
        self.conv_nodes = conv_nodes
        self.lstm_nodes = lstm_nodes
        self.kernel_size = kernel_size
        
        self.history = DataFrame(columns=('Skip', 'cConv', 'nLSTM', 'learning', 
                                 'batch_size', 'conv_nodes', 'lstm_nodes', 
                                 'loss_train', 'acc_train', 'loss_test', 'acc_test'))
       
        print("Creating train/test split of data...")
       
        if end_date is None: 
            end_date = ts.tail(1).index
            
#    https://keras.io/getting-started/sequential-model-guide/
#    El reverso se necesita para que la secuancia de cada sample sea del pasado al presente
        self.X_train = ts[ts.index < end_date][["Lookback%s" % str(i) for i in reversed(range(lags))]]
        self.X_test = ts[ts.index >= end_date][["Lookback%s" % str(i) for i in reversed(range(lags))]]

        if self.model == "ConvLSTM_Clasifier":
            self.y_train = ts[ts.index < end_date]["UpDown"]
            self.y_test = ts[ts.index >= end_date]["UpDown"]
            self.Lookback0 = ts[ts.index < end_date]["Lookback0"]

        self.ticker = ticker

    def fitting(self):   
   
        timesteps = self.lags   # tiempo
        features = 1    # features or chanels (Volume)
        num_classes = 3  # 3 for categorical
        
        
        #data = np.random.random((1000, dim_row, dim_col))
        #clas = np.random.randint(3, size=(1000, 1))
        ##print(clas)
        #clas = to_categorical(clas)
        ##print(clas)
        data = self.X_train
        data_test = self.X_test
        print(data)
                
        data = data.values.reshape(len(data), timesteps, 1)
        data_test = data_test.values.reshape(len(data_test), timesteps, 1)
        print(data)
        
        clas = self.y_train
        clas_test = self.y_test 
        clas = to_categorical(clas)
        clas_test = to_categorical(clas_test)

        cat0 = self.y_train.tolist().count(0)
        cat1 = self.y_train.tolist().count(1)
        cat2 = self.y_train.tolist().count(2)
        
        print("may: ", cat1, "  ", "menor: ", cat2, " ", "neutro: ", cat0)
        
        n_samples_0 = cat0
        n_samples_1 = (cat1 + cat2)/2.0
        n_samples_2 = (cat1 + cat2)/2.0

        class_weight={
                0: 1.0,
                1: n_samples_0/n_samples_1,
                2: n_samples_0/n_samples_2}            
        
        def class_1_accuracy(y_true, y_pred):
        # cojido de: http://www.deepideas.net/unbalanced-classes-machine-learning/
            class_id_true = K.argmax(y_true, axis=-1)
            class_id_preds = K.argmax(y_pred, axis=-1)
            
            accuracy_mask = K.cast(K.equal(class_id_preds, 1), 'int32')
            class_acc_tensor = K.cast(K.equal(class_id_true, class_id_preds), 'int32') * accuracy_mask
            
            class_acc = K.sum(class_acc_tensor) / K.maximum(K.sum(accuracy_mask), 1)
            return class_acc
        
        
        class SecondOpinion(Callback):
            def __init__(self, model, x_test, y_test, N):
                self.model = model
                self.x_test = x_test
                self.y_test = y_test
                self.N = N
                self.epoch = 1
        
            def on_epoch_end(self, epoch, logs={}):
                if self.epoch % self.N == 0:
                    y_pred = self.model.predict(self.x_test)
                    pred_T = 0
                    pred_F = 0
                    for i in range(len(y_pred)):
                        if np.argmax(y_pred[i]) == 1 and np.argmax(self.y_test[i]) == 1:
                            pred_T += 1
                        if np.argmax(y_pred[i]) == 1 and np.argmax(self.y_test[i]) != 1:
                            pred_F += 1
                    if pred_T + pred_F > 0:
                        Pr_pos = pred_T/(pred_T + pred_F)
                        print("Yoe: epoch, Probabilidad pos: ", self.epoch, Pr_pos)
                    else:
                        print("Yoe Probabilidad pos: 0")
                self.epoch += 1
        
        
        
        
        
#################################################################################################################        
        model = Sequential()
        if self.nConv == 0:
            model.add(LSTM(units=self.lstm_nodes, return_sequences=True, activation='tanh', input_shape=(timesteps, features)))
        for i in range(self.nLSTM - 2):
            model.add(LSTM(units=self.lstm_nodes, return_sequences=True, activation='tanh'))
        model.add(LSTM(units=self.lstm_nodes, activation='tanh'))
        model.add(Dropout(0.5))
        model.add(Dense(num_classes, activation='softmax')) # the dimension of index one will be considered to be the temporal dimension
        #model.add(Activation('sigmoid'))  # for loss = 'binary_crossentropy'
        
        # haciendo x: x[:, -1, :], la segunda dimension desaparece quedando solo 
        # los ULTIMOS elementos (-1) de dicha dimension:
        # Try this to see:
        # data = np.random.random((5, 3, 4))
        # print(data)
        # print(data[:, -1, :])  
        
#        model.add(Lambda(lambda x: x[:, -1, :], output_shape = [output_dim]))
        print(model.summary())
        
        tensorboard_active = False
        val_loss = False
        second_opinion = True
        callbacks = []
        if tensorboard_active:
            callbacks.append(TensorBoard(
                log_dir=self.putmodel + "Tensor_board_data",
                histogram_freq=0,
                write_graph=True,
                write_images=True))
        if val_loss:
            callbacks.append(EarlyStopping(
                monitor='val_loss', 
                patience=5))
        if second_opinion:
            callbacks.append(SecondOpinion(model, data_test, clas_test, 10))
        #model.compile(loss = 'categorical_crossentropy', optimizer='Adam', metrics = ['categorical_accuracy'])
        #model.compile(loss = 'binary_crossentropy', optimizer=Adam(lr=self.learning), metrics = ['categorical_accuracy'])
        model.compile(loss = 'categorical_crossentropy', optimizer='Adam', metrics = [class_1_accuracy])
                
        model.fit(x=data, 
                  y=clas,
                  batch_size=self.batch_size, epochs=800, verbose=2, 
                  callbacks = callbacks,
                  class_weight = class_weight)
                  #validation_data=(data_test, clas_test))
        
#####################################################################################################################
        
        # serialize model to YAML
        model_yaml = model.to_yaml()
        with open("model.yaml", "w") as yaml_file:
            yaml_file.write(model_yaml)
        # serialize weights to HDF5
        model.save_weights("model.h5")
        print("Saved model to disk")
        
#        # load YAML and create model
#        yaml_file = open('model.yaml', 'r')
#        loaded_model_yaml = yaml_file.read()
#        yaml_file.close()
#        loaded_model = model_from_yaml(loaded_model_yaml)
#        # load weights into new model
#        loaded_model.load_weights("model.h5")
#        print("Loaded model from disk")
#        loaded_model.compile(loss = 'categorical_crossentropy', optimizer='Adam', metrics = [class_1_accuracy])
#        
        print("Computing prediction ...")
        y_pred = model.predict_proba(data_test)
        
        model.reset_states()
        print("Computing train evaluation ...")
        score_train = model.evaluate(data, clas, verbose=2)
        print('Train loss:', score_train[0])
        print('Train accuracy:', score_train[1])

        model.reset_states()
#        score_train_loaded = loaded_model.evaluate(data, clas, verbose=2)
#        loaded_model.reset_states()
#        print('Train loss loaded:', score_train[0])
#        print('Train accuracy loaded:', score_train[1])

        print("Computing test evaluation ...")
        score_test = model.evaluate(data_test, clas_test, verbose=2)
        print('Test loss:', score_test[0])
        print('Test accuracy:', score_test[1])

        model.reset_states()
#        score_test_loaded = loaded_model.evaluate(data_test, clas_test, verbose=2)
#        loaded_model.reset_states()
#        print('Test loss loaded:', score_test[0])
#        print('Test accuracy loaded:', score_test[1])

        
        pred_T = 0
        pred_F = 0        
        for i in range(len(y_pred)):
            if np.argmax(y_pred[i]) == 1 and np.argmax(clas_test[i]) == 1:
                pred_T += 1
#                print(y_pred[i])
            if np.argmax(y_pred[i]) == 1 and np.argmax(clas_test[i]) != 1:
                pred_F += 1
        if pred_T + pred_F > 0:
            Pr_pos = pred_T/(pred_T + pred_F)
            print("Yoe Probabilidad pos: ", Pr_pos)
        else:
            print("Yoe Probabilidad pos: 0")
        
        history = DataFrame([[self.skip, self.nConv, self.nLSTM, 
                    self.learning, self.batch_size, 
                    self.conv_nodes, self.lstm_nodes, 
                    score_train[0], score_train[1], 
                    score_test[0], score_test[1]]], columns = ('Skip', 'cConv', 'nLSTM', 'learning', 
                                 'batch_size', 'conv_nodes', 'lstm_nodes', 
                                 'loss_train', 'acc_train', 'loss_test', 'acc_test'))
        self.history = self.history.append(history)
                
        


    def print_to_disk(self, path, line):
        try:
            f = open(path, "r")
            f.close()
            f = open(path, "a")
            f.write(line)
            f.close
        except IOError:
            f = open(path, "w+")
            f.write(line)    
            f.close
        
# Future
    #model.add(BatchNormalization())
    #bias_regularizer=reg
    #kernel_regularizer=reg
    #recurrent_regularizer=reg



