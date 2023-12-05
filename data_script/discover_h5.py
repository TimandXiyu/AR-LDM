import h5py
import matplotlib.pyplot as plt
import numpy as np
import cv2
from matplotlib import rcParams


class HDF5ImageBatchDisplayer:
    def __init__(self, hdf5_file):
        self.hdf5_file = hdf5_file

    def display_image_batch(self, batch_index):
        with h5py.File(self.hdf5_file, 'r') as hdf:
            fig, axs = plt.subplots(1, 5, figsize=(15, 3))
            for i in range(5):
                image_data = hdf[f'train/image{i}'][batch_index]
                image_array = np.frombuffer(image_data.tobytes(), dtype=np.uint8)
                image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
                image = image[:128, :]
                axs[i].imshow(image)
                axs[i].axis('off')
            plt.show()

    def display_image_text_pair(self, batch_index):

        with h5py.File(self.hdf5_file, 'r') as hdf:
            fig, axs = plt.subplots(1, 5, figsize=(15, 6))
            for i in range(5):
                image_data = hdf[f'train/image{i}'][batch_index]
                image_array = np.frombuffer(image_data.tobytes(), dtype=np.uint8)
                image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
                image = image[:128, :]
                axs[i].imshow(image)
                axs[i].axis('off')
            text = self.read_text(batch_index)
            text = text.split('|')
            text = '\n'.join(text)
            plt.figtext(0.5, 0.01, text, ha='center', fontsize=15, bbox={"alpha":0.1, "pad":10})
            plt.tight_layout()
            plt.show()

    def display_all_batches(self, num_data_points):
        for batch_index in range(num_data_points):
            self.display_image_text_pair(batch_index)

    def read_text(self, batch_index):
        with h5py.File(self.hdf5_file, 'r') as hdf:
            text_data = hdf['train/text'][batch_index]
            return text_data.decode('utf-8')


if __name__ == '__main__':
    hdf5_path = "/media/mldadmin/home/s123mdg35_05/ar-ldm/data/custom_hatlady_N=50.hdf5"
    H5Display = HDF5ImageBatchDisplayer(hdf5_path)
    H5Display.display_all_batches(1)

