import os
import matplotlib.pyplot as plt
from PIL import Image


def display_images_with_caption(folder1, folder2, caption):
    # Collect image file paths from both folders
    images1 = sorted([file for file in os.listdir(folder1) if file.endswith(('png', 'jpg', 'jpeg'))])[:5]
    images2 = sorted([file for file in os.listdir(folder2) if file.endswith(('png', 'jpg', 'jpeg'))])[:5]

    images1 = [Image.open(os.path.join(folder1, file)) for file in images1]
    images2 = [Image.open(os.path.join(folder2, file)) for file in images2]

    # Create a subplot with 2 rows
    fig, axs = plt.subplots(2, 5, figsize=(15, 10))

    # Display images from folder1 in row 1 and folder2 in row 2
    for i in range(5):
        axs[0, i].imshow(images1[i])
        axs[0, i].axis('off')
        axs[1, i].imshow(images2[i])
        axs[1, i].axis('off')

    # plt.subplots_adjust(wspace=0.1, hspace=0.1)
    fig.tight_layout()

    # Add a caption with a blue background box
    plt.figtext(0.5, 0.01, caption, ha="center", fontsize=15, bbox={"alpha": 0.1, "pad": 10})

    plt.show()


# Example usage
folder1 = '../ckpts/ori_pinkhat/0007'
folder2 = '../ckpts/adapt_output_img/pinkhat_output/0007'
display_images_with_caption(folder1,
                            folder2,
                            'Mr slate at hotel desk\nFred and Wilma are in a room. They are looking to the left. They both turn and look to the right at the same time.\nA man wearing spectacles is sitting in his office.  He holds his right finger up while talking to someone.\nman with glasses laughing is behind the desk in his office.\nMr Slate is in the office. He is standing behind a desk while laughing,')