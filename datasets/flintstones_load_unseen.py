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
        self.use_handpick = args.get(args.dataset).use_handpick # using handpick followings, overwrite random sampling
        if subset == "test_unseen" or subset == "test_seen":
            self.early_stop = args.stop_sample_early if args.stop_sample_early else False
        else:
            self.early_stop = False
        self.cur_char = args.cur_char # suppose to be ONE name of the character

        self.target_chars = args.get(args.dataset).target_chars  # target_chars is a list of strings
        self.target_dir = args.get(args.dataset).target_dir
        self.data_dir = args.get(args.dataset).data_dir

        splits = json.load(open(os.path.join(self.data_dir, 'train-val-test_split.json'), 'r'))
        self.train_ids, self.val_ids, self.test_ids = splits["train"], splits["val"], splits["test"]
        self.followings = pickle.load(open(os.path.join(self.data_dir, 'following_cache4.pkl'), 'rb'))
        self._followings = self.followings.copy()
        self.annotations = json.load(open(os.path.join(self.data_dir, 'flintstones_annotations_v1-0.json')))
        self.nominal_name_mapping = json.load(open(os.path.join(self.data_dir, 'char_name_mapping.json'), 'r'))
        self.unseen_char_anno = json.load(open(os.path.join(self.data_dir, 'flintstones_unseen_anno.json'), 'r'))
        self.h5file = h5py.File(args.get(args.dataset).hdf5_file, "r")
        self.new_followings = json.load(open(os.path.join(self.data_dir, 'new_followings.json'), 'r'))
        self.seen_len = {"train": len(self.h5file['train']['text']), "test": len(self.h5file['test']['text'])}
        # get 10% random samples from train and test split of the h5 file
        self.seen_train_indexes = random.sample(range(self.seen_len["train"]), 0)
        self.seen_test_indexes = random.sample(range(self.seen_len["test"]), 400)

        self.unseen_char = dict()

        raw_chars = dict()
        characters = dict()
        self.descriptions = dict()
        for sample in self.annotations:
            self.descriptions[sample["globalID"]] = sample["description"]
            raw_chars[sample["globalID"]] = sample["characters"]
        for k, v in raw_chars.items():
            characters[k] = [single_char["labelNPC"] for single_char in v]

        print("parsing data started...")
        self.unseen_with_dir = os.listdir(self.target_dir)

        target_ids = []
        if self.cur_char is None:
            print("warning, you are using a placeholder for current character, make sure you are testing seen examples!")
            self.cur_char = 'slaghoople' # just a placeholder, should only be used when testing for seen examples!
        if self.cur_char not in self.unseen_with_dir:
            for k, v in self.descriptions.items():
                npc_ls = [ele.lower() for ele in characters[k]]
                if self.cur_char in v.lower():
                    target_ids.append(k)
                if self.cur_char in npc_ls:
                    target_ids.append(k)
        else:
            target_ids = os.listdir(os.path.join(self.target_dir, self.cur_char))
            target_ids = [i.split('.')[0] for i in target_ids]

        target_ids = list(set(target_ids))
        target_ids.sort()
        if self.use_handpick and self.cur_char in list(self.new_followings.keys()):
            unseen_train_ids = list(self.new_followings[self.cur_char].values())
            unseen_train_ids = [item for sublist in unseen_train_ids for item in sublist]
        else:
            rs = np.random.RandomState(56)
            unseen_train_ids = rs.choice(target_ids, size=10, replace=False)
            unseen_train_ids = unseen_train_ids.tolist()
        unseen_test_ids = [i for i in target_ids if i not in unseen_train_ids]
        unseen_train_story = []
        unseen_test_story = []
        for k, v in self.followings.copy().items():
            v = [k] + v
            if any(id in unseen_train_ids for id in v) and all(id not in unseen_test_ids for id in v):
                unseen_train_story.append(k)
            if any(id in unseen_test_ids for id in v) and all(id not in unseen_train_ids for id in v):
                unseen_test_story.append(k)
        # handpicked stories come with complete list of followings, just load them
        if self.use_handpick and self.cur_char in list(self.new_followings.keys()):
            unseen_train_story = self.new_followings[self.cur_char]
        else:
            unseen_train_story = {starting_id: [starting_id] + self._followings[starting_id] for starting_id in unseen_train_story}
            unseen_train_story = {k: v for k, v in unseen_train_story.items() if len(v) == 5}

        unseen_test_story = {starting_id: [starting_id] + self._followings[starting_id] for starting_id in unseen_test_story}
        unseen_test_story = {k: v for k, v in unseen_test_story.items() if len(v) == 5}

        print('parsing data finished')

        self.cur_char_anno = {}
        for k, v in self.unseen_char_anno.items():
            if k.split(":")[0] == self.cur_char:
                self.cur_char_anno[k.split(":")[1]] = v

        self.unseen_train = unseen_train_story
        self.unseen_test = unseen_test_story

        # convert dict to list
        self.unseen_train = list(self.unseen_train.values())
        self.unseen_test = list(self.unseen_test.values())

        self.dataset = args.dataset
        self.max_length = args.get(args.dataset).max_length
        self.clip_tokenizer = CLIPTokenizer.from_pretrained('runwayml/stable-diffusion-v1-5', subfolder="tokenizer")
        self.blip_tokenizer = init_tokenizer()
        msg = self.clip_tokenizer.add_tokens(list(args.get(args.dataset).new_tokens), special_tokens=True)
        print("clip {} new tokens added".format(msg))
        msg = self.blip_tokenizer.add_tokens(list(args.get(args.dataset).new_tokens), special_tokens=True)
        print("blip {} new tokens added".format(msg))

        nominal_names = []
        for char in self.target_chars:
            char_nominal_name = self.nominal_name_mapping[char][1] # 1st ele is the nominal name, 2nd is the base token
            nominal_names.append(char_nominal_name)
        msg = self.clip_tokenizer.add_tokens(nominal_names, special_tokens=True)
        print("clip {} new tokens added".format(msg))
        msg = self.blip_tokenizer.add_tokens(nominal_names, special_tokens=True)
        print("blip {} new tokens added".format(msg))
        print(f"In dataloader clip tokenizer, normal names: {nominal_names} added to tokenizer with ids: "
              f"{self.clip_tokenizer.convert_tokens_to_ids(nominal_names)}")
        print(f"In dataloader blip tokenizer, normal names: {nominal_names} added to tokenizer with ids: "
              f"{self.blip_tokenizer.convert_tokens_to_ids(nominal_names)}")

        self.augment = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize([512, 512]),
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
            # check if index within len of seen_train_indexes
            if index < len(self.seen_train_indexes):
                # reading from h5, convert index
                index = self.seen_train_indexes[index]
                images = list()
                for i in range(5):
                    im = self.h5file["train"]['image{}'.format(i)][index]
                    im = cv2.imdecode(im, cv2.IMREAD_COLOR)
                    idx = random.randint(0, 4)
                    images.append(im[idx * 128: (idx + 1) * 128])
                texts = self.h5file["train"]['text'][index].decode('utf-8').split('|')
            else:
                story = self.unseen_train[index - len(self.seen_train_indexes)]
                images = [os.path.join(self.data_dir, 'video_frames_sampled', '{}.npy'.format(path)) for path in story]
                images = [np.load(img) for img in images]
                images = [img[np.random.randint(0, img.shape[0])] for img in images]
                placeholder_name = self.nominal_name_mapping[self.cur_char][0]
                special_token = self.nominal_name_mapping[self.cur_char][1]
                if self.cur_char not in self.unseen_with_dir:
                    texts = [self.descriptions[i] for i in story]
                    texts = [text.lower() for text in texts]
                    texts = [text.replace(self.cur_char, special_token) for text in texts]
                else:
                    texts = []
                    for id in story:
                        try:
                            anno = self.cur_char_anno[id]
                            anno = anno.replace(placeholder_name, special_token)
                            texts.append(anno)
                        except KeyError:
                            texts.append(self.descriptions[id])

        elif self.subset == 'test_seen':
            index = self.seen_test_indexes[index]
            images = list()
            for i in range(5):
                im = self.h5file["train"]['image{}'.format(i)][index]
                im = cv2.imdecode(im, cv2.IMREAD_COLOR)
                idx = random.randint(0, 4)
                images.append(im[idx * 128: (idx + 1) * 128])
            texts = self.h5file["test"]['text'][index].decode('utf-8').split('|')
        elif self.subset == 'test_unseen':
            story = self.unseen_test[index]
            images = [os.path.join(self.data_dir, 'video_frames_sampled', '{}.npy'.format(path)) for path in story]
            images = [np.load(img) for img in images]
            images = [img[np.random.randint(0, img.shape[0])] for img in images]
            placeholder_name = self.nominal_name_mapping[self.cur_char][0]
            special_token = self.nominal_name_mapping[self.cur_char][1]
            if self.cur_char not in self.unseen_with_dir:
                texts = [self.descriptions[i] for i in story]
                texts = [text.lower() for text in texts]
                texts = [text.replace(self.cur_char, special_token) for text in texts]
            else:
                texts = []
                for id in story:
                    try:
                        anno = self.cur_char_anno[id]
                        anno = anno.replace(placeholder_name, special_token)
                        texts.append(anno)
                    except KeyError:
                        texts.append(self.descriptions[id])
        else:
            raise ValueError("subset must be either train, test_seen, or test_unseen")

        source_images = torch.stack([self.blip_image_processor(im) for im in images])
        images = images[1:] if self.args.task == 'continuation' else images
        images = torch.stack([self.augment(im) for im in images]) \
            if self.subset in ['train', 'val','train_unseen'] else torch.from_numpy(np.array(images))

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

        return images, captions, attention_mask, source_images, source_caption, source_attention_mask, texts

    def __len__(self):
        seen_train_len = len(self.seen_train_indexes)
        seen_test_len = len(self.seen_test_indexes)
        unseen_train_len = len(self.unseen_train)
        unseen_test_len = len(self.unseen_test)
        if self.subset == 'train':
            return seen_train_len + unseen_train_len
        elif self.subset == 'test_unseen':
            return self.early_stop if self.early_stop else unseen_test_len
        elif self.subset == 'test_seen':
            return self.early_stop if self.early_stop else seen_test_len


@hydra.main(config_path="..", config_name="config")
def test_case(args):
    pl.seed_everything(args.seed)

    story_dataset = StoryDataset('train', args=args)
    story_dataloader = DataLoader(story_dataset, batch_size=1, shuffle=False, num_workers=0)

    for batch in tqdm(story_dataloader):
        _ = batch


if __name__ == "__main__":
    test_case()

