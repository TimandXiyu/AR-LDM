#!/bin/bash

run_name=(
    "flintstones_us10_source_free_contrast_0.2_distill_0.5"
    "flintstones_us10_source_free_contrast_0.1_distill_1.5"
)
contrast_weight=(
    0.2
    0.1
)
distill_weight=(
    0.5
    1.5
)

# Loop through each test model file path and corresponding output directory
for i in "${!run_name[@]}"
do
    name="${run_name[i]}"
    cw="${contrast_weight[i]}"
    dw="${distill_weight[i]}"
    echo "run_name=$name", "contrastive_weight=$cw", "distillation_weight=$dw"
    python main.py "run_name=$name" "contrastive_weight=$cw" "distillation_weight=$dw"
done