#!/bin/bash

# Define the learning rates to grid search over
learning_rates=(0.00001 0.0001 0.001 0.01)

# Path to the base YAML configuration file
base_config="config.yaml"

# The command used to run the training. This will depend on your specific setup.
train_command="python main.py"

# Loop over the learning rates
for lr in "${learning_rates[@]}"
do
  # Create a new config filename based on the learning rate
  new_config="config_lr_${lr}.yaml"

  # Copy the base config file to a new file
  cp $base_config $new_config

  # Modify the learning rate in the new configuration file
  sed -i "s/init_lr: [^ ]*/init_lr: $lr/" $new_config

  # Run the training command with the new configuration file
  echo "Running training with learning rate: $lr"
  $train_command --config $new_config

  # Wait for the command to finish before starting the next one
  wait

  # Optionally, clean up by removing the temp config file
  rm $new_config
done
