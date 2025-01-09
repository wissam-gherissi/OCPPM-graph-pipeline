# OCPPM-graph-pipeline

# Configuration
## Creating a virtual environment with interpreter Python 3.10
For compatibility reasons, you need to create a new Python virtual environment using python version 3.10

Example:

    python310 -m venv env

## Karateclub
To resolve compatibility issues, you first need to install Karateclub library before installing the rest of required libraries.

    pip install karateclub
## OCPA Installation
This repository uses OCPA library for the preprocessing part. Please use the following instructions to ensure the ocpa library installation.

    git clone https://github.com/ocpm/ocpa.git
    cd ocpa
    pip install .
    
## Last configuration steps

Finally, we install the required libraries (torch, torch_geometric) and upgrade the following (pm4py, networkx)

    pip install torch, torch-geometric
    pip install pm4py --upgrade
    pip install networkx --upgrade

## Possible encounter
You may encounter an error caused by a compatibility issue between ocpa and pm4py:

To resolve this issue please refer to this solution:

In the ocpa/algo/enhancement/token_replay_based_performance/util.py

change the code:
    
    import pm4py.evaluation.replay_fitness.variants.token_replay as tb_replay 
To

    import pm4py.algo.evaluation.replay_fitness.variants.token_replay as tb_replay 

# Pipeline usage

The code is implemented into different blocks that are put together by a main_function file where the pipeline to be used is defined.
You can create your own graph embedding layer or prediction layer and add its usage in the main_function easily.

