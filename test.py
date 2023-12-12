import torch
import torch.nn as nn
import loralib as lora  # Replace with your actual import

# Parameters for the test
in_features = 128
out_features = 64
r = 4  # Assumed parameter for LoraLayer

# Initialize the LoraLayer
lora_layer = lora.Linear(in_features, out_features, r)

# Create a dummy input tensor
input_tensor = torch.randn(1, in_features)  # Batch size of 1

# Perform a forward pass through the layer
output = lora_layer(input_tensor)

# Check if the output shape is correct
assert output.shape == (1, out_features), f"Output shape is incorrect: {output.shape}"

# Perform a backward pass to ensure gradients can be computed
output.sum().backward()  # We sum the output to get a scalar for .backward()

# Check if gradients are assigned
for param in lora_layer.parameters():
    assert param.grad is not None, "Gradients were not computed"

# Optionally, check if the gradients are of the correct shape
for param in lora_layer.parameters():
    assert param.grad.shape == param.shape, "Gradient shape mismatch"

# Print out a simple message if tests pass
print("All tests passed!")