#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (C) 2019 Vincent Lunot. All Rights Reserved.

"""Training script.

Functions for preparing the train and validation data and for training the 
models.
    
@author: Vincent Lunot
"""


import time
import numpy as np

from preprocess import (read_global_parameters,
                        read_models_parameters, 
                        read_fingerprints, 
                        read_rho, 
                        read_train_test_indices, 
                        read_encoded_input_output_smiles,
                        read_vocabulary_mappings)
from smiles_generator import SMILESGenerator


def transform_fingerprints(rho, fingerprints):
    targets = rho.transform(fingerprints)
    return targets


def train(model_name, global_parameters, model_parameters, rho, fingerprints, 
          encoded_input, encoded_output, vocabulary_size, max_length, 
          batch_size):   
    print('Preparing data and model...')
    train_bool, test_bool = read_train_test_indices(model_name)
    
    train_encoded_input = encoded_input[train_bool]
    test_encoded_input = encoded_input[test_bool]    
    train_encoded_output = encoded_output[train_bool]
    test_encoded_output = encoded_output[test_bool]
    
    targets = transform_fingerprints(rho, fingerprints)
    train_targets = targets[train_bool]
    test_targets = targets[test_bool]
    
    model = SMILESGenerator(model_name, global_parameters, model_parameters, 
                            vocabulary_size, max_length)    
    print('Training...')
    model.train(train_encoded_input, train_encoded_output, train_targets, 
                test_encoded_input, test_encoded_output, test_targets, 
                batch_size, model_parameters['num_epochs'])
    
    
if __name__ == '__main__':
    np.random.seed(12321)
    print('Training models...')
    starting_time = time.time()
    global_parameters = read_global_parameters()
    models_parameters = read_models_parameters()
    fingerprints = read_fingerprints()
    encoded_input, encoded_output = read_encoded_input_output_smiles()
    int_to_char, _ = read_vocabulary_mappings()
    vocabulary_size = len(int_to_char)
    batch_size = global_parameters['batch_size']
    max_length = global_parameters['max_length']
    for model_name in models_parameters['names']:
        model_parameters = models_parameters[model_name]
        rho = read_rho(model_name)        
        train(model_name, global_parameters, model_parameters, rho, fingerprints, 
              encoded_input, encoded_output, vocabulary_size, max_length, batch_size)
    print(f'Training all models took {round(time.time() - starting_time)}s.')
    