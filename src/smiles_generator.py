#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (C) 2019 Vincent Lunot. All Rights Reserved.

"""SMILES generator module.

SMILESGenerator is a class containing all the functions for building and 
training the model, as well as generating the smiles.
    
@author: Vincent Lunot
"""


import math
import numpy as np
from keras.layers import Input, Dense, Lambda, Concatenate, Embedding, CuDNNLSTM
from keras.models import Model, Sequential
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras import losses
from keras.optimizers import Adam

from preprocess import read_vocabulary_mappings


class SMILESGenerator():
    
    def __init__(self, model_name, global_parameters, model_parameters, vocabulary_size, max_length):
        self.model_name = model_name
        self.max_length = max_length
        self.target_size = model_parameters['target_size']
        self.latent_size = model_parameters['latent_size']
        self.embedding_size = model_parameters['embedding_size']
        self.lstm_sizes = model_parameters['lstm_sizes']
        self.learning_rate = model_parameters['learning_rate']
        self.vocabulary_size = vocabulary_size
        self.int_to_char, self.char_to_int = read_vocabulary_mappings()
        self.start_char = global_parameters['start_char']
        self._create_autoencoder()
        self._create_generator()
        
        
    def _create_autoencoder(self):   
        smiles_input = Input(shape=(self.max_length, ))
        targets_input = Input(shape=(self.target_size, ))
        self._embedding = Embedding(self.vocabulary_size, self.embedding_size)
        embedded_smiles = self._embedding(smiles_input)
        encoder_inputs = Lambda(self._repeat_and_concatenate)([embedded_smiles, targets_input])
        encoder_lstm_seq = Sequential([CuDNNLSTM(d, return_sequences=True) for d in self.lstm_sizes[:-1]] 
                                      + [CuDNNLSTM(self.lstm_sizes[-1])])
        h = encoder_lstm_seq(encoder_inputs)        
        z_mean = Dense(self.latent_size)(h)
        z_log_var = Dense(self.latent_size)(h)
        
        def sampling(args):
            z_mean, z_log_var = args
            epsilon = K.random_normal(shape=K.shape(z_mean),
                                      mean=0.0, stddev=1.0)
            return z_mean + K.exp(0.5 * z_log_var) * epsilon
         
        z = Lambda(sampling, output_shape=(self.latent_size,))([z_mean, z_log_var])
        z_target = Concatenate()([z, targets_input])        
        self._decoder_lstms = [CuDNNLSTM(d, return_sequences=True, return_state=True) for d in reversed(self.lstm_sizes)]
        self._decoder_mean = Dense(self.vocabulary_size, activation='softmax')
        x = Lambda(self._repeat_and_concatenate)([embedded_smiles, z_target])
        for lstm in self._decoder_lstms:
            x, _, _ = lstm(x)
        x_decoded_mean = self._decoder_mean(x)        
        self.autoencoder = Model([smiles_input, targets_input], x_decoded_mean)
               
        def custom_loss(x, x_decoded_mean):
            crossentropy = K.mean(losses.categorical_crossentropy(x, x_decoded_mean))
            kl_loss = - 0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
            return crossentropy + kl_loss
      
        self.autoencoder.compile(optimizer=Adam(lr=self.learning_rate), loss=custom_loss)
    
    
    def _repeat_and_concatenate(self, inputs):
        input_3D, input_2D = inputs
        input_2D_repeat = K.tile(K.expand_dims(input_2D, 1), [1, K.shape(input_3D)[1], 1])
        return K.concatenate([input_3D, input_2D_repeat], axis=-1)
 
       
    def _create_generator(self):        
        smiles_input = Input(shape=(1, ))
        targets_input = Input(shape=(self.target_size, ))
        z_input = Input(shape=(self.latent_size, ))
        state_inputs = [Input(shape=(lstm_size, )) for lstm_size in self.lstm_sizes for _ in range(2)]
        embedded_smiles = self._embedding(smiles_input)
        z_target = Concatenate()([z_input, targets_input])
        lstms_states = []
        x = Lambda(self._repeat_and_concatenate)([embedded_smiles, z_target])
        for k, lstm in enumerate(self._decoder_lstms):
            x, state1, state2 = lstm(x, initial_state=[state_inputs[2*k], state_inputs[2*k+1]])
            lstms_states += [state1, state2]
        generator_output = self._decoder_mean(x)
        get_output = K.function([z_input, targets_input, smiles_input] + state_inputs, 
                                [generator_output] + lstms_states)
        self.run_generator = lambda Z, P, X, S: get_output([Z, P, X] + S)
        
                    
    def train(self, x, y, targets, validation_x, validation_y, validation_targets, batch_size, epochs):
        model_checkpoint = ModelCheckpoint(f'models/{self.model_name}-weights.h5', monitor='val_loss', 
                                           save_best_only=True, save_weights_only=True)
        if validation_x is not None:
            validation_data = self._batch_generator(validation_x, validation_y, validation_targets, batch_size)
            validation_steps = math.ceil(len(validation_x) / batch_size)
        else:
            validation_data = None
            validation_steps = None
        
        self.autoencoder.fit_generator(self._batch_generator(x, y, targets, batch_size), 
                                       steps_per_epoch=math.ceil(len(x) / batch_size),
                                       validation_data=validation_data,
                                       validation_steps=validation_steps,
                                       epochs=epochs,
                                       callbacks=[model_checkpoint])
        
        
    def _batch_generator(self, x, y, targets, batch_size):
        while True:
            permutation = np.random.permutation(len(x))
            for k in range(math.ceil(len(x) / batch_size)):
                batch_indices = permutation[k * batch_size : (k + 1) * batch_size]
                x_batch = x[batch_indices]
                y_batch = np.zeros((len(batch_indices), y.shape[1], self.vocabulary_size), np.float32)
                for i, index in enumerate(batch_indices):
                    for j in range(y.shape[1]):
                        y_batch[i, j, y[index, j]] = 1
                targets_batch = targets[batch_indices].astype(np.float32)
                yield [x_batch, targets_batch], y_batch
        
        
    def load_weights(self):
        self.autoencoder.load_weights(f'models/{self.model_name}-weights.h5')
        
        
    def generate_smiles(self, batch_size, target, latent_vector):
        x = np.zeros((batch_size, 1), np.uint8)
        x[:] = self.char_to_int[self.start_char]
        preds = []
        states = [np.zeros((batch_size, lstm_size), np.float32) for lstm_size in self.lstm_sizes for _ in range(2)]
        for _ in range(self.max_length):
            g = self.run_generator(latent_vector, target, x, states)
            x = g[0]
            states = g[1:]
            r = np.random.rand(batch_size, 1,  1)
            x = np.argmax(r < np.cumsum(x, axis=2), axis=2)
            preds.append(x)
        preds = np.concatenate(preds, 1)
        smiles = [self._convert_to_smiles(row) for row in preds]
        return smiles
    
    
    def _convert_to_smiles(self, integer_row):
        return ''.join([self.int_to_char[int(x)] for x in integer_row])
