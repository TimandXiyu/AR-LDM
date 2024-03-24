#!/bin/bash

test_models=(
    "/home/xiyu/projects/AR-LDM/ckpts/flintstones_us10_contrast_weight_0.15/"
    "/home/xiyu/projects/AR-LDM/ckpts/flintstones_us10_contrast_weight_0.2/"
)
output_dirs=(
    "/home/xiyu/projects/AR-LDM/ckpts/output_images_us10_contrast_weight_0.15/"
    "/home/xiyu/projects/AR-LDM/ckpts/output_images_us10_contrast_weight_0.2/"
)

# Loop through each test model file path and corresponding output directory
for i in "${!test_models[@]}"
do
    test_model="${test_models[i]}"
    output_dir="${output_dirs[i]}"
    python main.py -cn config-test.yaml "test_model_file=$test_model" "sample_output_dir=$output_dir"
done