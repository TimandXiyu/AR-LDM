#!/bin/bash

# Array of test model file paths and corresponding output directories
test_model_files=(
    "/home/xiyu/projects/AR-LDM/ckpts/flintstones_train_10_unseen_256_distill/ 25"
    "/home/xiyu/projects/AR-LDM/ckpts/flintstones_us10_more_ref/ 25"
    "/home/xiyu/projects/AR-LDM/ckpts/flintstones_us10_no_ref/ 67"
    "/home/xiyu/projects/AR-LDM/ckpts/flintstones_us10_simple_distill/ 26"
)

output_directories=(
    "./ckpts/output_images_train_10_unseen_256_distill_seen"
    "./ckpts/output_images_us10_more_ref_seen"
    "./ckpts/output_images_us10_no_ref_seen"
    "./ckpts/output_images_us10_simple_distill_seen"
)

# Loop through each test model file path and corresponding output directory
for i in "${!test_model_files[@]}"
do
    test_model_file="${test_model_files[i]}"
    output_dir="${output_directories[i]}"

    echo "Running evaluation for: $test_model_file"
    echo "Output directory: $output_dir"
    python main.py test_model_file="$test_model_file" sample_output_dir="$output_dir"
done