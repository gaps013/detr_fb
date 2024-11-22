#!/bin/bash

# Define the Python script to run
PYTHON_SCRIPT="main.py"

# Define the different sets of named arguments

ARGS_LIST=(
    "--batch_size 15 --num_classes 6 --apply_transforms True --num_queries 100 --exp_name NC_6_NQ_100_TF"
    "--batch_size 15 --num_classes 6 --apply_transforms False --num_queries 100 --exp_name NC_6_NQ_100_NTF"
    "--batch_size 15 --num_classes 6 --apply_transforms True --num_queries 6 --exp_name NC_6_NQ_6_TF"
    "--batch_size 15 --num_classes 6 --apply_transforms False --num_queries 6 --exp_name NC_6_NQ_6_NTF"
    "--batch_size 15 --num_classes 91 --apply_transforms True --num_queries 6 --exp_name NC_91_NQ_6_TF"
    "--batch_size 15 --num_classes 91 --apply_transforms False --num_queries 6 --exp_name NC_91_NQ_6_NTF"
    "--batch_size 15 --num_classes 91 --apply_transforms True --num_queries 100 --exp_name NC_91_NQ_100_TF"
    "--batch_size 15 --num_classes 91 --apply_transforms False --num_queries 100 --exp_name NC_91_NQ_100_TNF"
    
)

# Loop through the arguments and run the Python script with each set of arguments
for ARGS in "${ARGS_LIST[@]}"; 
do
    python3 $PYTHON_SCRIPT $ARGS
done