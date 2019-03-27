#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (C) 2019 Vincent Lunot. All Rights Reserved.

"""Preprocessing script.

Functions for preprocessing the datasets, and for reading the original data as
well as the preprocessed data.

Preprocessing consists in:
    - creating fingerprints associated to the datasets' smiles,
    - creating dimension reduction functions,
    - splitting the datasets into train and validation sets,
    - building vocabulary from smiles,
    - encoding input and output smiles using the computed vocabulary.
    
@author: Vincent Lunot
"""

import json
import os
import time
import numpy as np
import pandas as pd
from sklearn.decomposition import IncrementalPCA
from sklearn.externals import joblib
from rdkit import Chem
from rdkit.Chem import MACCSkeys


def read_global_parameters():
    with open('specs/global.json', 'r') as f:
        global_parameters = json.load(f)
    return global_parameters


def read_models_parameters():
    with open('specs/models.json', 'r') as f:
        models_parameters = json.load(f)
    return models_parameters


def read_train_data():
    train_data = pd.read_csv('data/molhack_train.data')
    return train_data


def read_moses_data():
    moses_data = pd.read_csv('data/dataset_v1.csv')
    return moses_data


def compute_fingerprints(smiles):
    print("\tComputing fingerprints from SMILES...")
    smiles_to_fingerprint = lambda x: MACCSkeys.GenMACCSKeys(Chem.MolFromSmiles(x))
    fingerprints = np.array(list(map(smiles_to_fingerprint, smiles)), np.int8)
    return fingerprints

def create_fingerprints(train_data):
    fingerprints = compute_fingerprints(train_data['SMILES'].values)
    np.save('preprocessed/fingerprints.npy', fingerprints)

def read_fingerprints():
    fingerprints = np.load('preprocessed/fingerprints.npy')
    return fingerprints


def create_reductions(models_parameters, fingerprints):
    print('\tCreating transformations...')
    models_names = models_parameters['names']
    for name in models_names:
        rho = IncrementalPCA(n_components=models_parameters[name]['target_size'], batch_size=8192)
        rho.fit(fingerprints)
        joblib.dump(rho, f'models/{name}-rho.joblib')

def read_rho(model_name):
    rho = joblib.load(f'models/{model_name}-rho.joblib')
    return rho


def create_indices(models_names, train_data, moses_data):
    print('\tCreating train/test indices...')
    train_data_indices = moses_data['SMILES'].isin(train_data['SMILES'])
    delta = len(moses_data) // len(models_names)
    for i, name in enumerate(models_names):
        part = np.zeros(len(moses_data), np.bool)
        part[i * delta : (i + 1) * delta] = True
        test_bool = np.logical_and(train_data_indices, part)
        np.save(f'models/{name}-test_bool.npy', test_bool)
                
def read_train_test_indices(model_name):
    test_indices = np.load(f'models/{model_name}-test_bool.npy')
    train_indices = ~test_indices
    return train_indices, test_indices
         

def create_vocabulary_mappings(smiles, start_char, end_char):
    print('\tCreating vocabulary dictionaries...')
    chars = sorted(set(''.join(smiles)))
    chars += [start_char, end_char]
    char_to_int = {c: i for i, c in enumerate(chars)}
    joblib.dump(chars, 'preprocessed/int_to_char.joblib')
    joblib.dump(char_to_int, 'preprocessed/char_to_int.joblib')
    
def read_vocabulary_mappings():
    int_to_char = joblib.load('preprocessed/int_to_char.joblib')
    char_to_int = joblib.load('preprocessed/char_to_int.joblib')
    return int_to_char, char_to_int
          

def compute_encoded_input_output_smiles(smiles, global_parameters, char_to_int, max_length):
    print('\tEncoding SMILES...')
    start_char = global_parameters['start_char']
    end_char = global_parameters['end_char']
    padded_input_smiles = [(start_char + s).ljust(max_length, end_char) for s in smiles]
    encoded_input_smiles = np.array([[char_to_int[c] for c in s] for s in padded_input_smiles], np.uint8)
    padded_output_smiles = [s.ljust(max_length, end_char) for s in smiles]
    encoded_output_smiles = np.array([[char_to_int[c] for c in s] for s in padded_output_smiles], np.uint8)
    return encoded_input_smiles, encoded_output_smiles
    
def create_encoded_input_output_smiles(smiles, global_parameters, char_to_int, max_length):
    encoded_input_smiles, encoded_output_smiles = compute_encoded_input_output_smiles(smiles, global_parameters, char_to_int, max_length)
    np.save('preprocessed/encoded_input_smiles.npy', encoded_input_smiles)
    np.save('preprocessed/encoded_output_smiles.npy', encoded_output_smiles)
    
def read_encoded_input_output_smiles():
    encoded_input_smiles = np.load('preprocessed/encoded_input_smiles.npy')
    encoded_output_smiles = np.load('preprocessed/encoded_output_smiles.npy')
    return encoded_input_smiles, encoded_output_smiles
    
    
if __name__ == '__main__':
    np.random.seed(12321)
    print('Preprocessing data...')
    starting_time = time.time()
    global_parameters = read_global_parameters()
    models_parameters = read_models_parameters()
    train_data = read_train_data()
    moses_data = read_moses_data()
    if not os.path.isfile('preprocessed/fingerprints.npy'):
        create_fingerprints(moses_data)
    fingerprints = read_fingerprints()
    create_reductions(models_parameters, fingerprints)
    create_indices(models_parameters['names'], train_data, moses_data)
    create_vocabulary_mappings(moses_data['SMILES'], global_parameters['start_char'], global_parameters['end_char'])
    _, char_to_int = read_vocabulary_mappings()
    max_length = global_parameters['max_length']
    create_encoded_input_output_smiles(moses_data['SMILES'], global_parameters, char_to_int, max_length)
    print(f'Preprocessing took {round(time.time() - starting_time)}s.')
