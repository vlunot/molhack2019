# Vincent Lunot's MolHack 2019 prototype

This Docker image contains the solution developed by Vincent Lunot for the MolHack 2019 competition. This competition was organized by Insilico Medicine and held on CodaLab from February 25th to March 31st 2019.  
More details are available at: https://competitions.codalab.org/competitions/21716

## Run the image

Run this image with:
```
docker run --runtime=nvidia -it vlunot/mh2019:v1
```

## Available commands

Once the image is started, you can type:

- `./run_all.sh` to run the entire generation process,
- `./preprocess.sh` to run the preprocessing step,
- `./train.sh` to train the model (requires the preprocessing files),
- `./generate.sh` to generate the SMILES (requires the training files),
- `./clean_all.sh` to delete all computed results.

Intermediate computations were kept for further analysis. The `run_all.sh` script doesn't delete these results. If you want a full fresh start, you should use the `clean_all.sh` script first. 

## File hierarchy

The code for generating SMILES is in the `src` directory.  
The problem and model specifications are in the `specs` directory.  
The data used for training and testing is in the `data` directory. Please note that the dataset from MOSES is also included and used for training.  
The trained weights of the model are in the `models` directory.  
The generated SMILES are stored in the `res` directory.

## Computation time

This Docker image has been tested on the following configuration:

- Intel Core i5-4460
- 16GB RAM
- NVIDIA GeForce GTX 1070
- OS: Ubuntu 18.04

The final result, with a score of 0.998, required approximately 21 hours of computation. A score greater than 0.95 was obtained in less than 5 hours (preprocessing + training + generation time).  


Copyright (C) 2019 Vincent Lunot