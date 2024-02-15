import random
import hydra
import pytorch_lightning as pl

import cv2
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import CLIPTokenizer
from tqdm import tqdm
from torch.utils.data import DataLoader
import os
import json
import pickle

import argparse


class CustomAnnotation(object):
    """
    Identify the image that has the unseen characters and save the annotations and images to dir and a txt file
    """

    def __init__(self, data_dir):
        super(CustomAnnotation, self).__init__()

        self.data_dir = data_dir
        self.named_unseen = ["pebbles", "hoppy", "bamm bamm"]

        self.hand_picked_unseen = os.listdir(os.path.join(self.data_dir, 'target_chars'))
        # going into each dir in hand_picked_unseen and get the file names
        self.hand_picked_unseen = {i: os.listdir(os.path.join(self.data_dir, 'target_chars', i)) for i in self.hand_picked_unseen}

        self.hand_picked_unseen = {k: [i.split(".")[0] for i in v] for k, v in self.hand_picked_unseen.items()}

        # save the hand_picked_unseen to a json file
        with open(os.path.join(self.data_dir, "hand_picked_unseen.json"), "w") as f:
            # dump with new lines for each entry
            json.dump(self.hand_picked_unseen, f, indent=4)

        splits = json.load(open(os.path.join(self.data_dir, 'train-val-test_split.json'), 'r'))
        self.train_ids, self.val_ids, self.test_ids = splits["train"], splits["val"], splits["test"]
        self.followings = pickle.load(open(os.path.join(self.data_dir, 'following_cache4.pkl'), 'rb'))
        self._followings = self.followings.copy()
        self.annotations = json.load(open(os.path.join(self.data_dir, 'flintstones_annotations_v1-0.json')))

        char_anno = dict()
        characters = dict()
        descriptions = dict()
        named_unseen_ids = list()
        for sample in self.annotations:
            descriptions[sample["globalID"]] = sample["description"]
            char_anno[sample["globalID"]] = sample["characters"]
        for k, v in char_anno.items():
            characters[k] = [single_char["labelNPC"] for single_char in v]

        for k, v in descriptions.items():
            for char in self.named_unseen:
                char = char.lower()
                npc_ls = [ele.lower() for ele in characters[k]]
                if char in v.lower():
                    named_unseen_ids.append(k)
                if char in npc_ls:
                    named_unseen_ids.append(k)
        named_unseen_ids = list(set(named_unseen_ids))

        self._hand_picked_unseen = [v for k, v in self.hand_picked_unseen.items()]
        # flatten the list
        self._hand_picked_unseen = [item for sublist in self._hand_picked_unseen for item in sublist]

        self.img_text = dict()
        for frame_id in self._hand_picked_unseen + named_unseen_ids:
            # get the description of the frame in description
            self.img_text[frame_id] = descriptions[frame_id]

        _ = {}
        for k, v in self.img_text.items():
            flag = False
            for char in self.named_unseen:
                char = char.lower()
                v = v.lower()
                if char in v:
                    flag = True
                    new_k = f"{char}:{k}"
                    _[new_k] = v
            if not flag:
                _[k] = v

        self.img_text = _

        # open hand_picked_unseen.json
        with open(os.path.join(self.data_dir, "hand_picked_unseen.json"), "r") as f:
            self.hand_picked_unseen = json.load(f)

        _ = {}
        for k, v in self.img_text.items():
            _k = k.split(":")[-1]
            flag = False
            for char, des in self.hand_picked_unseen.items():
                if _k in des:
                    new_k = f"{char}:{k}"
                    _[new_k] = v
                    flag = True
            if not flag:
                _[k] = v

        self.img_text = _

        # save the img_text to a json file
        with open(os.path.join(self.data_dir, "img_text.json"), "w") as f:
            # dump with new lines for each entry
            json.dump(self.img_text, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='arguments for flintstones hdf5 file saving')
    parser.add_argument('--data_dir', type=str, required=True, help='flintstones data directory')
    args = parser.parse_args()
    main = CustomAnnotation(data_dir=args.data_dir)