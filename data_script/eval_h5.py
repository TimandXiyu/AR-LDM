"""
This script aims to eval an existing h5 file to make sure the data is properly generated.
"""
import os
import h5py


import h5py
import matplotlib.pyplot as plt
import numpy as np
import cv2


class H5Evaluator:
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

    def read_text(self, batch_index=None):
        with h5py.File(self.hdf5_file, 'r') as hdf:
            print()
            if batch_index is None:
                text_data = hdf['train/text'][:]
            else:
                text_data = hdf['train/text'][batch_index]

            if isinstance(text_data, np.ndarray):
                # Convert numpy array to list
                annotations = text_data.tolist()
            else:
                # Convert single text data to a list with one element
                annotations = [text_data]
            # Convert bytes to string
            annotations = [str(annotation, 'utf-8') for annotation in annotations]

            for anno in annotations:
                # lower the anno
                anno = anno.lower()
                if "hoppy" in anno:
                    raise ValueError("the annotation contains hoppy")
                if "pebbles" in anno:
                    raise ValueError("the annotation contains pebbles")
                if "bamm bamm" in anno:
                    raise ValueError("the annotation contains bamm bamm")
            return annotations


if __name__ == '__main__':
    hdf5_path = "/scratch/users/ntu/xiyu004/ar-ldm/data/flintstones_rare-char_rmeoved.h5"
    H5Display = H5Evaluator(hdf5_path)
    texts = H5Display.read_text()
