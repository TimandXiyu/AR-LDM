#!/bin/bash

# Define the learning rates to grid search over
learning_rates=(0.00001 0.0001 0.001 0.01)

# Path to the YAML configuration file
config_file="config.yaml"

# Backup the original configuration file
cp $config_file "${config_file}.bak"

# The command used to run the training. This will depend on your specific setup.
train_command="python main.py"

# Loop over the learning rates
for lr in "${learning_rates[@]}"
do
  # Update the learning rate in the configuration file
  sed -i "s/^init_lr: .*/init_lr: $lr/" $config_file

  # Run the training command with the modified configuration file
  echo "Running training with learning rate: $lr"
  $train_command

  # Wait for the command to finish before starting the next one
  wait
done

# Restore the original configuration file
mv "${config_file}.bak" $config_file

# End of the script
