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
from PIL import Image
from pathlib import Path


def main(args):
    """
    Code to create the refined flintstones dataset.
    All singles frames are read and checked if they have the target character based on multiple criteria.
    Then the frames are removed from the main list.
    The remaining frames are then used to create the hdf5 file.
    If one frame no longer exists in the main list, related stories will not appear in the hdf5 file.
    """
    splits = json.load(open(os.path.join(args.data_dir, 'train-val-test_split.json'), 'r'))
    train_ids, val_ids, test_ids = splits["train"], splits["val"], splits["test"]
    followings = pickle.load(open(os.path.join(args.data_dir, 'following_cache4.pkl'), 'rb'))
    annotations = json.load(open(os.path.join(args.data_dir, 'flintstones_annotations_v1-0.json')))

    rm_char = ["pebbles", "hoppy", "bamm bamm"]  # remove characters by mention of names
    target_root = r"D:\AR-LDM\data\target_chars"
    target_ids = list(Path(target_root).glob("**/*.jpg"))
    target_ids = [i.stem for i in target_ids]

    raw_chars = dict()
    characters = dict()
    descriptions = dict()
    for sample in annotations:
        descriptions[sample["globalID"]] = sample["description"]
        raw_chars[sample["globalID"]] = sample["characters"]
    for k, v in raw_chars.items():
        characters[k] = [single_char["labelNPC"] for single_char in v]

    for k, v in descriptions.items():
        for char in rm_char:
            char = char.lower()
            npc_ls = [ele.lower() for ele in characters[k]]
            if char in v.lower():
                target_ids.append(k)
            if char in npc_ls:
                target_ids.append(k)
    target_ids = list(set(target_ids))

    f = h5py.File(args.save_path, "w")
    for subset, ids in {"train": train_ids, "val": val_ids, "test": test_ids}.items():
        ids = [i for i in ids if i in followings and len(followings[i]) == 4]
        filtered_ids = []
        for first_frame, sub_frames in followings.items():
            all_frame = [first_frame] + sub_frames  # all frame id of a story
            if any(id in all_frame for id in target_ids):
                filtered_ids.append(first_frame)

        filtered_ids = list(set(filtered_ids))
        # check file exist in ids
        for id in ids:
            if not os.path.exists(os.path.join(args.data_dir, 'video_frames_sampled', '{}.npy'.format(id))):
                print("file not exist: {}".format(id))

        # ids = [[i] + followings[i] for i in ids if i not in filtered_ids]
        _ids = []
        for id in ids:
            if id not in filtered_ids:
                _ids.append([id] + followings[id])

        ids = _ids
        length = len(ids)
        group = f.create_group(subset)
        images = list()
        for i in range(5):
            images.append(
                group.create_dataset('image{}'.format(i), (length,), dtype=h5py.vlen_dtype(np.dtype('uint8'))))
        text = group.create_dataset('text', (length,), dtype=h5py.string_dtype(encoding='utf-8'))
        for i, item in enumerate(tqdm(ids, leave=True, desc="saveh5")):
            globalIDs = item
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='arguments for flintstones hdf5 file saving')
    parser.add_argument('--data_dir', type=str, required=True, help='flintstones data directory')
    parser.add_argument('--save_path', type=str, required=True, help='path to save hdf5')
    args = parser.parse_args()
    main(args)
