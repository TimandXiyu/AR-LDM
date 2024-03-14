#!/bin/bash

# Define the range of distillation weights to search
distillation_weights=(0.5 1.0 1.5 2.0, 2.5)

# Train models with different distillation weights
for weight in "${distillation_weights[@]}"
do
    # Update the run name and distillation weight in the config file
    sed -i "s/run_name: .*/run_name: flintstones_grid_search_distill_$weight/" config.yaml
    sed -i "s/distillation_weight: .*/distillation_weight: $weight/" config.yaml

    # Train the model
    python main.py -cn config.yaml
done

# Test the saved checkpoints
for weight in "${distillation_weights[@]}"
do
    # Update the test_model_file and sample_output_dir in the config-test.yaml file
    sed -i "s|test_model_file: .*|test_model_file: \"./ckpts/flintstones_grid_search_distill_$weight/epoch=99.ckpt\"|" config-test.yaml
    sed -i "s|sample_output_dir: .*|sample_output_dir: ./ckpts/output_images_distill_$weight|" config-test.yaml

    # Test the model
    python main.py -cn config-test.yaml
done