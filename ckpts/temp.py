import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

# Set the root path for the images
root_path = './adapt_output_img/'  # Replace with your images' directory path

# Set the number of images
num_images = 5

# Create a figure and a set of subplots
fig, axs = plt.subplots(1, num_images, figsize=(20, 4))

# Loop through the image files
for i in range(num_images):
    # Construct the image filename with the full path
    filename = os.path.join(root_path, f'{5+i:04d}.png')

    # Read the image file
    img = mpimg.imread(filename)

    # Display the image
    axs[i].imshow(img)
    axs[i].axis('off')  # Hide the axis

# Adjust the subplot layout
plt.subplots_adjust(wspace=0.05, hspace=0)

# Show the plot
plt.show()
