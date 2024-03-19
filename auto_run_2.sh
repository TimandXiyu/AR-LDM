#!/bin/bash

run_name=(
    "flintstones_contrast_weight_0.1"
    "flintstones_contrast_weight_0.5"
    "flintstones_contrast_weight_1.0"
    "flintstones_contrast_weight_1.5"
    "flintstones_contrast_weight_2.0"
)
contrast_weight=(
    0.1
    0.5
    1.0
    1.5
    2.0
)

# Loop through each test model file path and corresponding output directory
for i in "${!run_name[@]}"
do
    name="${run_name[i]}"
    w="${contrast_weight[i]}"
    echo "run_name=$name"
    echo "contrastive_weight=$w"
    python main.py "run_name=$name" "contrastive_weight=$w"
done