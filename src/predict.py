#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (C) 2019 Vincent Lunot. All Rights Reserved.

"""Predicting script.

The script can be run with the following arguments:
    --input-file
    --output-file
    --max-length
    --seed
    --evolve
    --no-evolve
    --reduced-set
    --full-set
    --num-epochs
    --evolve-model-num
    --num-iterations
Since it is a prototype, this list may not be exhaustive. It is recommended to 
check the list of arguments with the command: `predict.py --help`.

@author: Vincent Lunot
"""


import time
import argparse
import os
import glob
import zipfile
from shutil import copyfile
import numpy as np
import pandas as pd
import keras.backend as K
from warnings import warn

from rdkit import rdBase
rdBase.DisableLog('rdApp.error')
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import MACCSkeys

from smiles_generator import SMILESGenerator
from preprocess import (read_train_data, 
                        read_models_parameters, 
                        read_rho, 
                        read_vocabulary_mappings, 
                        read_global_parameters,
                        compute_fingerprints,
                        compute_encoded_input_output_smiles)
from train import transform_fingerprints


def extended_similarity_mol(m, ref_maccs):
    if m:
        return DataStructs.FingerprintSimilarity(MACCSkeys.GenMACCSKeys(m), ref_maccs)
    return -1

def extended_similarity_smiles(s, ref_maccs):
    m = Chem.MolFromSmiles(s)
    return extended_similarity_mol(m, ref_maccs)


def get_next_backup_int(f):
    backups = glob.glob(f + '.*')
    backups_exts = [b.split('.')[-1] for b in backups]
    backups_numbers = sorted([int(b) for b in backups_exts if b.isdigit()])
    if backups_numbers:
        new_int = backups_numbers[-1] + 1
    else:
        new_int = 0
    return new_int


def predict(global_parameters, models_parameters, test_solution, 
            training_smiles, prediction_file, max_length,
            evolve=False, reduced_set=False, num_epochs=None,
            evolve_model_num=-1, num_iterations=None):
    print('Preparing model...')
    scores = []
    all_smiles = ['SMILES']
    starting_time = time.time()
    int_to_char, char_to_int = read_vocabulary_mappings() 
    end_char = global_parameters['end_char']
    vocabulary_size = len(int_to_char)
    batch_size = global_parameters['batch_size']
    
    models = {}
    for model_name in models_parameters['names']:
        model_parameters = models_parameters[model_name]
        model = SMILESGenerator(model_name, global_parameters, model_parameters, vocabulary_size, max_length)
        model.load_weights()
        models[model_name] = model
        
    pred_models_names = models_parameters['names']

    if os.path.isfile(prediction_file):
        pred_smiles = pd.read_csv(prediction_file)['SMILES'].values
        
    if evolve:
        if os.path.isfile(prediction_file):
            evolve_model_name = sorted(pred_models_names)[evolve_model_num]
            if reduced_set:
                pred_smiles_100_slices = [
                        pred_smiles[k * 1000 : k * 1000 + 100] for k in range(len(test_solution))
                        if extended_similarity_mol(Chem.MolFromSmiles(pred_smiles[k * 1000 + 999]), 
                                                   DataStructs.CreateFromBitString(test_solution['fingerprints_center'].iloc[k])) < 1.0]
            else:
                pred_smiles_100_slices = [pred_smiles[k * 1000 : k * 1000 + 100] for k in range(len(test_solution))]
            pred_smiles_100 = np.concatenate(pred_smiles_100_slices)
            pred_smiles_reduced = pred_smiles_100[[all(c in char_to_int for c in p) and 
                                                   (len(p) < max_length - 1) for p in pred_smiles_100]]
            fingerprints = compute_fingerprints(pred_smiles_reduced)
            rho = read_rho(evolve_model_name)
            targets = transform_fingerprints(rho, fingerprints)
            encoded_input, encoded_output = compute_encoded_input_output_smiles(pred_smiles_reduced, global_parameters, char_to_int, max_length)         
            evolve_model = models[evolve_model_name]
            K.set_value(evolve_model.autoencoder.optimizer.lr, models_parameters[evolve_model_name]['learning_rate'] / 2)
            if num_epochs is None:
                num_epochs = int(20 * (100 * len(test_solution) / len(pred_smiles_100)) ** 0.3)
            evolve_model.train(encoded_input, encoded_output, targets, None, None, None, batch_size, num_epochs)
            pred_models_names = [evolve_model_name]
        else:
            warn('Predictions are missing. Simply computing predictions.')
            evolve = False
               
    print('Predicting...')
    for ref_index in range(len(test_solution)):
        print('*' * 60)
        print('Step', ref_index, flush=True)
        ref_maccs = DataStructs.CreateFromBitString(test_solution['fingerprints_center'].iloc[ref_index])
        
        begin_step_time = time.time()
        n_valid = 0
        candidates = []
        
        if os.path.isfile(prediction_file):
            candidates = list(pred_smiles[ref_index * 1000 : (ref_index + 1) * 1000])
            worst_score = extended_similarity_mol(Chem.MolFromSmiles(candidates[-1]), ref_maccs)
            if worst_score == 1.0:
                print('score: 1.0')
                scores.append(1.0)
                print('current overall score:', np.mean(scores))
                all_smiles += candidates
                continue      
            
        while n_valid < global_parameters['min_valid_mols']:
            for model_name in pred_models_names:
                model_parameters = models_parameters[model_name]
                rho = read_rho(model_name)
                targets = np.repeat(rho.transform(np.array([ref_maccs])), batch_size, axis=0)
                if num_iterations is None:
                    num_iterations = model_parameters['num_iterations']
                for _ in range(num_iterations):
                    latent_vector = np.random.normal(0.0, 1.0, (batch_size, model_parameters['latent_size']))
                    candidates += models[model_name].generate_smiles(batch_size, targets, latent_vector)
               
            candidates = list(set([s.split(end_char)[0] for s in candidates]))
            ms = [Chem.MolFromSmiles(s) for s in candidates]
            ms = [m for m in ms if m is not None]
            n_valid = len(ms)
            print('Number of candidates:', n_valid, flush=True)
        
        ms = sorted(ms, key=lambda m:extended_similarity_mol(m, ref_maccs), reverse=True)
        
        new_smiles = []
        n_selected = 0
        while len(new_smiles) < 1000:
            n_selected += 1500
            selected_ms = ms[:n_selected]
            ss = [Chem.MolToSmiles(m) for m in selected_ms]
            new_smiles = np.setdiff1d(ss, training_smiles)
            
        smiles_sims = [(s, extended_similarity_smiles(s, ref_maccs)) for s in new_smiles] 
        ordered_smiles_sims = sorted(smiles_sims, key=lambda c: c[1], reverse=True)
        ordered_sims = [ss[1] for ss in ordered_smiles_sims]
        print('best:' , ordered_sims[0])
        print('100th:', ordered_sims[99])
        print('1000th:', ordered_sims[999])
        score = 0.7 * np.mean(ordered_sims[:100]) + 0.3 * np.mean(ordered_sims[:1000])
        print('score:', score)
        scores.append(score)
        print('current overall score:', np.mean(scores))
        ordered_smiles = [ss[0] for ss in ordered_smiles_sims]
        with open(f'tmp/molhack_test.predict.{ref_index}', 'w') as f:
            for s in ordered_smiles[:1000]:
                f.write(s + "\n")
        all_smiles += ordered_smiles[:1000]
        current_time = time.time()
        print(f'time: step={round(current_time - begin_step_time)}s, total={round(current_time - starting_time)}s')
        
    if os.path.isfile(prediction_file):
        new_int = get_next_backup_int(prediction_file)
        copyfile(prediction_file, f'{prediction_file}.{new_int}')
    if evolve:
        weights_file = 'models/evolve-weights.h5'
        new_int = get_next_backup_int(weights_file)
        evolve_model.autoencoder.save_weights(f'{weights_file}.{new_int}')
        
    with open(prediction_file, 'w') as f:
        for s in all_smiles:
            f.write(s + "\n")
            
    with zipfile.ZipFile(prediction_file + '.zip', 'w', zipfile.ZIP_DEFLATED) as f:
        f.write(prediction_file)
        
            
def parse_arguments():
    """Define and parse the script arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input-file', 
                        help='file containing the target fingerprints (.solution)')
    parser.add_argument('-o', '--output-file', 
                        help='submission file (.predict)')
    parser.add_argument('--max-length', type=int, 
                        help='maximum_length')
    parser.add_argument('-s', '--seed', type=int, 
                        help='seed value')
    parser.add_argument('--reduced-set', dest='reduced_set', action='store_true', 
                        help='evolution uses a reduced set')
    parser.add_argument('--full-set', dest='reduced_set', action='store_false', 
                        help='evolution uses full set')
    parser.set_defaults(reduced_set=False)
    parser.add_argument('--evolve', dest='evolve', action='store_true', 
                        help='preprocessed model is updated w.r.t. latest predictions')
    parser.add_argument('--no-evolve', dest='evolve', action='store_false', 
                        help='uses preprocessed model')
    parser.set_defaults(evolve=True)
    parser.add_argument('--num-epochs', type=int, 
                        help='number of evolution epochs')
    parser.add_argument('--evolve-model-num', type=int, default=-1, 
                        help='model position')
    parser.add_argument('--num-iterations', type=int, 
                        help='number of iterations')
    args = parser.parse_args()
    return (args.input_file, args.output_file, args.max_length, args.seed, 
            args.reduced_set, args.evolve, args.num_epochs, 
            args.evolve_model_num, args.num_iterations)

    
if __name__ == '__main__':
    train_data = read_train_data()
    training_smiles = train_data['SMILES'].values.tolist()
    global_parameters = read_global_parameters()
    solution_file, prediction_file, max_length, seed, reduced_set, evolve, num_epochs, evolve_model_num, num_iterations = parse_arguments()
    if seed is None:
        seed = 12321
    np.random.seed(seed)
    print('seed:', seed)
    if not solution_file:
        solution_file = global_parameters['solution_file']
    print('solution file:', solution_file)
    if not prediction_file:
        prediction_file = global_parameters['prediction_file']
    print('prediction file:', prediction_file)
    if not max_length:
        max_length = global_parameters['max_length']
    print('max_length:', max_length)
    test_solution = pd.read_csv(solution_file)
    models_parameters = read_models_parameters()
    predict(global_parameters, models_parameters, test_solution, 
            training_smiles, prediction_file, max_length,
            evolve=evolve, reduced_set=reduced_set, num_epochs=num_epochs,
            evolve_model_num=evolve_model_num, num_iterations=num_iterations)
