#!/bin/bash

test_models=(
    "/home/xiyu/projects/AR-LDM/ckpts/flintstones_contrast_weight_0.1/"
    "/home/xiyu/projects/AR-LDM/ckpts/flintstones_contrast_weight_0.5/"
    "/home/xiyu/projects/AR-LDM/ckpts/flintstones_contrast_weight_1.0/"
    "/home/xiyu/projects/AR-LDM/ckpts/flintstones_contrast_weight_1.5/"
    "/home/xiyu/projects/AR-LDM/ckpts/flintstones_contrast_weight_2.0/"
)
output_dirs=(
    "/home/xiyu/projects/AR-LDM/ckpts/output_images_contrast_weight_0.1/"
    "/home/xiyu/projects/AR-LDM/ckpts/output_images_contrast_weight_0.5/"
    "/home/xiyu/projects/AR-LDM/ckpts/output_images_contrast_weight_1.0/"
    "/home/xiyu/projects/AR-LDM/ckpts/output_images_contrast_weight_1.5/"
    "/home/xiyu/projects/AR-LDM/ckpts/output_images_contrast_weight_2.0/"
)

# Loop through each test model file path and corresponding output directory
for i in "${!test_models[@]}"
do
    test_model="${test_models[i]}"
    output_dir="${output_dirs[i]}"
    python main.py -cn config-test.yaml "test_model_file=$test_model" "sample_output_dir=$output_dir"
done