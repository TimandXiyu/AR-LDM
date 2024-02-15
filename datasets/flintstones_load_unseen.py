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

from models.blip_override.blip import init_tokenizer


class StoryDataset(Dataset):
    """
    A custom subset class for the LRW (includes train, val, test) subset
    """

    def __init__(self, subset, args):
        super(StoryDataset, self).__init__()
        self.args = args
        self.subset = subset

        self.target_chars = args.get(args.dataset).target_chars  # target_chars is a list of strings
        self.target_dir = args.get(args.dataset).target_dir
        self.data_dir = args.get(args.dataset).data_dir

        splits = json.load(open(os.path.join(self.data_dir, 'train-val-test_split.json'), 'r'))
        self.train_ids, self.val_ids, self.test_ids = splits["train"], splits["val"], splits["test"]
        self.followings = pickle.load(open(os.path.join(self.data_dir, 'following_cache4.pkl'), 'rb'))
        self._followings = self.followings.copy()
        self.annotations = json.load(open(os.path.join(self.data_dir, 'flintstones_annotations_v1-0.json')))

        self.unseen_char = dict()

        raw_chars = dict()
        characters = dict()
        descriptions = dict()
        target_ids = list()
        for sample in self.annotations:
            descriptions[sample["globalID"]] = sample["description"]
            raw_chars[sample["globalID"]] = sample["characters"]
        for k, v in raw_chars.items():
            characters[k] = [single_char["labelNPC"] for single_char in v]

        self.unseen_with_dir = os.listdir(self.target_dir)
        for unseen_char in self.target_chars:
            if unseen_char not in self.unseen_with_dir:
                for k, v in descriptions.items():
                    npc_ls = [ele.lower() for ele in characters[k]]
                    if unseen_char in v.lower():
                        target_ids.append(k)
                    if unseen_char in npc_ls:
                        target_ids.append(k)
            else:
                target_ids = os.listdir(os.path.join(self.target_dir, unseen_char))
                # remove .jpg extension
                target_ids = [i.split('.')[0] for i in target_ids]
                pass
            target_ids = list(set(target_ids))

            unseen_samples = []
            for k, v in self.followings.items():
                if k in target_ids or any(id in v for id in target_ids):
                    unseen_samples.append(k)

            for k, v in self._followings.items():
                # remove
                if k in unseen_samples:
                    try:
                        del self.followings[k]
                    except KeyError:
                        pass  # skip if the sample is already deleted

            unseen_samples = {starting_id: [starting_id] + self._followings[starting_id] for starting_id in unseen_samples}

            self.unseen_char[unseen_char] = unseen_samples

        _train_ids = self.train_ids.copy()
        _test_ids = self.test_ids.copy()
        for ids in _train_ids:
            if ids not in self.followings:
                self.train_ids.remove(ids)
        for ids in _test_ids:
            if ids not in self.followings:
                self.test_ids.remove(ids)

        random.seed(42)
        self.train_ids = random.sample(self.train_ids, len(self.train_ids) // 10)
        self.test_ids = random.sample(self.test_ids, len(self.test_ids) // 10)
        self.train_ids = [i for i in self.train_ids if i in self.followings and len(self.followings[i]) == 4]
        self.test_ids = [i for i in self.test_ids if i in self.followings and len(self.followings[i]) == 4]
        self.seen_char_train = [[k] + self.followings[k] for k in self.train_ids]
        self.seen_char_test = [[k] + self.followings[k] for k in self.test_ids]

        # split the unseen char into train and test
        self.unseen_char_train = dict()
        for k, v in self.unseen_char.items():
            self.unseen_char_train[k] = random.sample(list(v.keys()), int(len(v) * 0.2))
        self.unseen_char_test = dict()
        for k, v in self.unseen_char.items():
            self.unseen_char_test[k] = list(set(v.keys()) - set(self.unseen_char_train[k]))
        unseen_char = args.get(args.dataset).target_chars[args.get(args.dataset).round]
        self.unseen_char_train = self.unseen_char_train[unseen_char]
        self.unseen_char_test = self.unseen_char_test[unseen_char]

        # get the followings from the original followings because the
        self.unseen_char_train = [[k] + self._followings[k] for k in self.unseen_char_train]
        self.unseen_char_test = [[k] + self._followings[k] for k in self.unseen_char_test]

        self.train_ids = self.seen_char_train + self.unseen_char_train
        self.test_ids = self.unseen_char_test + self.unseen_char_test

        self.dataset = args.dataset
        self.max_length = args.get(args.dataset).max_length
        self.clip_tokenizer = CLIPTokenizer.from_pretrained('runwayml/stable-diffusion-v1-5', subfolder="tokenizer")
        self.blip_tokenizer = init_tokenizer()
        msg = self.clip_tokenizer.add_tokens(list(args.get(args.dataset).new_tokens), special_tokens=True)
        print("clip {} new tokens added".format(msg))
        msg = self.blip_tokenizer.add_tokens(list(args.get(args.dataset).new_tokens), special_tokens=True)
        print("blip {} new tokens added".format(msg))

        self.augment = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize([256, 256]),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

        self.blip_image_processor = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])
        ])

    def __getitem__(self, index):
        if self.subset == 'train':
            ids = self.train_ids
        elif self.subset == 'test_seen':
            ids = self.seen_char_test
        elif self.subset == 'test_unseen':
            ids = self.unseen_char_test
        else:
            raise ValueError("subset must be either train, test_seen, or test_unseen")
        story = ids[index]
        imgs = [os.path.join(self.data_dir, 'video_frames_sampled', '{}.npy'.format(path)) for path in story]
        imgs = [np.load(img) for img in imgs]
        # select a random frame on the first axis
        images = [img[np.random.randint(0, img.shape[0])] for img in imgs]

        source_images = torch.stack([self.blip_image_processor(im) for im in images])
        images = images[1:] if self.args.task == 'continuation' else images
        images = torch.stack([self.augment(im) for im in images]) \
            if self.subset in ['train', 'val'] else torch.from_numpy(np.array(images))

        texts = self.h5['text'][index].decode('utf-8').split('|')

        # tokenize caption using default tokenizer
        tokenized = self.clip_tokenizer(
            texts[1:] if self.args.task == 'continuation' else texts,
            padding="max_length",
            max_length=self.max_length,
            truncation=False,
            return_tensors="pt",
        )
        captions, attention_mask = tokenized['input_ids'], tokenized['attention_mask']

        tokenized = self.blip_tokenizer(
            texts,
            padding="max_length",
            max_length=self.max_length,
            truncation=False,
            return_tensors="pt",
        )
        source_caption, source_attention_mask = tokenized['input_ids'], tokenized['attention_mask']
        # for img_idx in range(len(images)):
        #     caption = captions[img_idx].tolist()
        #     caption_text = self.clip_tokenizer.decode(caption, skip_special_tokens=True)
        #     print(caption_text)
        return images, captions, attention_mask, source_images, source_caption, source_attention_mask

    """
    images: [5, 3, 256, 256]
    captions: [5, 91]
    attention_mask: [5, 91]
    source_images: [5, 3, 224, 224]
    source_caption: [5, 91]
    source_attention_mask: [5, 91]

    """

    def __len__(self):
        length = len(self.train_ids) if self.subset == 'train' else len(self.test_ids)
        return length


@hydra.main(config_path="..", config_name="config")
def test_case(args):
    pl.seed_everything(args.seed)

    story_dataset = StoryDataset('train', args=args)
    story_dataloader = DataLoader(story_dataset, batch_size=1, shuffle=False, num_workers=8)

    for batch in tqdm(story_dataloader):
        images, captions, attention_mask, source_images, source_caption, source_attention_mask = batch
        pass


if __name__ == "__main__":
    test_case()

