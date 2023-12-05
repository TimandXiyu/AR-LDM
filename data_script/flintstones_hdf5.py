import argparse
import json
import os
import pickle

import cv2
import h5py
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import random


def main(args):
    splits = json.load(open(os.path.join(args.data_dir, 'train-val-test_split.json'), 'r'))
    train_ids, val_ids, test_ids = splits["train"], splits["val"], splits["test"]
    followings = pickle.load(open(os.path.join(args.data_dir, 'following_cache4.pkl'), 'rb'))
    annotations = json.load(open(os.path.join(args.data_dir, 'flintstones_annotations_v1-0.json')))
    descriptions = dict()
    for sample in annotations:
        descriptions[sample["globalID"]] = sample["description"]

    f = h5py.File(args.save_path, "w")
    for subset, ids in {'train': train_ids, 'val': val_ids, 'test': test_ids}.items():
        ids = [i for i in ids if i in followings and len(followings[i]) == 4]
        length = len(ids)
        # check file exist in ids
        for id in ids:
            if not os.path.exists(os.path.join(args.data_dir, 'video_frames_sampled', '{}.npy'.format(id))):
                print("file not exist: {}".format(id))

        group = f.create_group(subset)
        images = list()
        for i in range(5):
            images.append(
                group.create_dataset('image{}'.format(i), (length,), dtype=h5py.vlen_dtype(np.dtype('uint8'))))
        text = group.create_dataset('text', (length,), dtype=h5py.string_dtype(encoding='utf-8'))
        for i, item in enumerate(tqdm(ids, leave=True, desc="saveh5")):
            globalIDs = [item] + followings[item]
            txt = list()
            for j, globalID in enumerate(globalIDs):
                img = np.load(os.path.join(args.data_dir, 'video_frames_sampled', '{}.npy'.format(globalID)))
                img = np.concatenate(img, axis=0).astype(np.uint8)
                img = cv2.imencode('.png', img)[1].tobytes()
                img = np.frombuffer(img, np.uint8)
                images[j][i] = img
                txt.append(descriptions[globalID])
            text[i] = '|'.join([t.replace('\n', '').replace('\t', '').strip() for t in txt])
    f.close()


def main_adaptation(args, N):
    sentences = ['A man hiding his face with long pink hat sits on the roof of a car.']
    splits = json.load(open(os.path.join(args.data_dir, 'train-val-test_split.json'), 'r'))
    train_ids, val_ids, test_ids = splits["train"], splits["val"], splits["test"]
    all_ids = train_ids + val_ids + test_ids
    followings = pickle.load(open(os.path.join(args.data_dir, 'following_cache4.pkl'), 'rb'))
    annotations = json.load(open(os.path.join(args.data_dir, 'flintstones_annotations_v1-0.json')))
    descriptions = dict()
    for sample in annotations:
        descriptions[sample["globalID"]] = sample["description"]

    f = h5py.File(args.save_path, "w")
    ids = [i for i in all_ids if i in followings and len(followings[i]) == 4]
    filtered_ids = [i for i in ids if any(sentence in descriptions[i] for sentence in sentences)]

    N = min(N, len(ids) - len(filtered_ids))  # Ensure N is not larger than available samples
    random_ids = random.sample([i for i in ids if i not in filtered_ids], N)

    ids = filtered_ids + random_ids
    length = len(ids)

    # check file exist in ids
    for id in ids:
        if not os.path.exists(os.path.join(args.data_dir, 'video_frames_sampled', '{}.npy'.format(id))):
            print("file not exist: {}".format(id))

    group = f.create_group('train')
    images = list()
    for i in range(5):
        images.append(
            group.create_dataset('image{}'.format(i), (length,), dtype=h5py.vlen_dtype(np.dtype('uint8'))))
    text = group.create_dataset('text', (length,), dtype=h5py.string_dtype(encoding='utf-8'))
    for i, item in enumerate(tqdm(ids, leave=True, desc="saveh5")):
        globalIDs = [item] + followings[item]
        txt = list()

        for j, globalID in enumerate(globalIDs):
            img = np.load(os.path.join(args.data_dir, 'video_frames_sampled', '{}.npy'.format(globalID)))
            img = np.concatenate(img, axis=0).astype(np.uint8)
            img = cv2.imencode('.png', img)[1].tobytes()
            img = np.frombuffer(img, np.uint8)
            images[j][i] = img
            txt.append(descriptions[globalID])

        annotation = '|'.join([t.replace('\n', '').replace('\t', '').strip() for t in txt])

        if item in filtered_ids:
            # Display images and prompt for updated annotation
            fig, axs = plt.subplots(1, 5, figsize=(15, 3))  # Adjust the figure size if needed
            for j in range(5):
                full_img = cv2.imdecode(images[j][i], cv2.IMREAD_COLOR)
                cropped_img = full_img[:128, :]  # Modify this line if a different cropping is needed
                axs[j].imshow(cropped_img)
                axs[j].axis('off')
            plt.show()

            # Request user input for updated annotation
            updated_annotation = input(
                f"Original annotation: {annotation}\nType the updated annotation (or press Enter to keep the original): ")
            if updated_annotation:
                annotation = updated_annotation

            text[i] = annotation
        else:
            text[i] = annotation

    f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='arguments for flintstones hdf5 file saving')
    parser.add_argument('--data_dir', type=str, required=True, help='flintstones data directory')
    parser.add_argument('--save_path', type=str, required=True, help='path to save hdf5')
    args = parser.parse_args()
    main_adaptation(args, 50)
