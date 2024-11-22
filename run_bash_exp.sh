#!/bin/bash

# Define the Python script to run
PYTHON_SCRIPT="main.py"

# Define the different sets of named arguments

ARGS_LIST=(

    "--batch_size 15 --num_classes 6 --apply_transforms True --num_queries 100 --lr_scheduler step --exp_name NC_6_NQ_100_DTF_Step"
    "--batch_size 15 --num_classes 6 --apply_transforms False --num_queries 100 --lr_scheduler step --exp_name NC_6_NQ_100_NDTF_Step"
    "--batch_size 15 --num_classes 6 --apply_transforms True --num_queries 6 --lr_scheduler step --exp_name NC_6_NQ_6_DTF_Step"
    "--batch_size 15 --num_classes 6 --apply_transforms False --num_queries 6 --lr_scheduler step --exp_name NC_6_NQ_6_NDTF_Step"
    "--batch_size 15 --num_classes 91 --apply_transforms True --num_queries 6 --lr_scheduler step --exp_name NC_91_NQ_6_DTF_Step"
    "--batch_size 15 --num_classes 91 --apply_transforms False --num_queries 6 --lr_scheduler step --exp_name NC_91_NQ_6_NDTF_Step"
    "--batch_size 15 --num_classes 91 --apply_transforms True --num_queries 100 --lr_scheduler step --exp_name NC_91_NQ_100_DTF_Step"
    "--batch_size 15 --num_classes 91 --apply_transforms False --num_queries 100 --lr_scheduler step --exp_name NC_91_NQ_100_NDTF_Step"

    "--batch_size 15 --num_classes 6 --apply_transforms True --num_queries 100 --lr_scheduler cosine --exp_name NC_6_NQ_100_DTF_Cosine"
    "--batch_size 15 --num_classes 6 --apply_transforms False --num_queries 100 --lr_scheduler cosine --exp_name NC_6_NQ_100_NDTF_Cosine"
    "--batch_size 15 --num_classes 6 --apply_transforms True --num_queries 6 --lr_scheduler cosine --exp_name NC_6_NQ_6_DTF_Cosine"
    "--batch_size 15 --num_classes 6 --apply_transforms False --num_queries 6 --lr_scheduler cosine --exp_name NC_6_NQ_6_NDTF_Cosine"
    "--batch_size 15 --num_classes 91 --apply_transforms True --num_queries 6 --lr_scheduler cosine --exp_name NC_91_NQ_6_DTF_Cosine"
    "--batch_size 15 --num_classes 91 --apply_transforms False --num_queries 6 --lr_scheduler cosine --exp_name NC_91_NQ_6_NDTF_Cosine"
    "--batch_size 15 --num_classes 91 --apply_transforms True --num_queries 100 --lr_scheduler cosine --exp_name NC_91_NQ_100_DTF_Cosine"
    "--batch_size 15 --num_classes 91 --apply_transforms False --num_queries 100 --lr_scheduler cosine --exp_name NC_91_NQ_100_NDTF_Cosine"
    
)

# Loop through the arguments and run the Python script with each set of arguments
for ARGS in "${ARGS_LIST[@]}"; 
do
    python -m torch.distributed.launch --nproc_per_node=6 --use_env  $PYTHON_SCRIPT $ARGS
done