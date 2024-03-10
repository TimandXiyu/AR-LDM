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
    def __init__(self, subset, args):
        super(StoryDataset, self).__init__()
        self.args = args
        self.subset = subset
        self.use_handpick = args.get(args.dataset).use_handpick
        self.early_stop = args.stop_sample_early if subset in ["test_unseen", "test_seen"] and args.stop_sample_early else False
        self.cur_char = args.cur_char

        self.data_dir = args.get(args.dataset).data_dir
        self.target_dir = args.get(args.dataset).target_dir
        self.target_chars = args.get(args.dataset).target_chars

        self.annotations = json.load(open(os.path.join(self.data_dir, 'flintstones_annotations_v1-0.json')))
        self.nominal_name_mapping = json.load(open(os.path.join(self.data_dir, 'char_name_mapping.json'), 'r'))
        self.unseen_char_anno = json.load(open(os.path.join(self.data_dir, 'flintstones_unseen_anno.json'), 'r'))
        self.new_followings = json.load(open(os.path.join(self.data_dir, 'new_followings.json'), 'r'))

        self.h5file = h5py.File(args.get(args.dataset).hdf5_file, "r")
        self.seen_len = {"train": len(self.h5file['train']['text']), "test": len(self.h5file['test']['text'])}
        self.seen_train_indexes = random.sample(range(self.seen_len["train"]), 10)
        self.seen_test_indexes = random.sample(range(self.seen_len["test"]), 400)

        self.followings = pickle.load(open(os.path.join(self.data_dir, 'following_cache4.pkl'), 'rb'))
        self._followings = self.followings.copy()

        self.descriptions = {sample["globalID"]: sample["description"] for sample in self.annotations}
        characters = {
            k: [single_char["labelNPC"] for single_char in v["characters"]]
            for k, v in self.annotations.items()
        }

        self.unseen_with_dir = os.listdir(self.target_dir)
        self.cur_char = self.cur_char or 'slaghoople'

        target_ids = self.get_target_ids(characters)
        unseen_train_ids, unseen_test_ids = self.split_unseen_ids(target_ids)
        unseen_train_story, unseen_test_story = self.get_unseen_stories(unseen_train_ids, unseen_test_ids)

        self.unseen_train = list(unseen_train_story.values())
        self.unseen_test = list(unseen_test_story.values())

        self.cur_char_anno = {k.split(":")[1]: v for k, v in self.unseen_char_anno.items() if k.split(":")[0] == self.cur_char}

        self.dataset = args.dataset
        self.max_length = args.get(args.dataset).max_length
        self.clip_tokenizer = CLIPTokenizer.from_pretrained('runwayml/stable-diffusion-v1-5', subfolder="tokenizer")
        self.blip_tokenizer = init_tokenizer()

        self.add_special_tokens(args)

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

    def get_target_ids(self, characters):
        target_ids = []
        if self.cur_char not in self.unseen_with_dir:
            for k, v in self.descriptions.items():
                npc_ls = [ele.lower() for ele in characters[k]]
                if self.cur_char in v.lower() or self.cur_char in npc_ls:
                    target_ids.append(k)
        else:
            target_ids = [i.split('.')[0] for i in os.listdir(os.path.join(self.target_dir, self.cur_char))]
        return sorted(set(target_ids))

    def split_unseen_ids(self, target_ids):
        if self.use_handpick and self.cur_char in self.new_followings:
            unseen_train_ids = [item for sublist in self.new_followings[self.cur_char].values() for item in sublist]
        else:
            rs = np.random.RandomState(56)
            unseen_train_ids = rs.choice(target_ids, size=10, replace=False).tolist()
        unseen_test_ids = [i for i in target_ids if i not in unseen_train_ids]
        return unseen_train_ids, unseen_test_ids

    def get_unseen_stories(self, unseen_train_ids, unseen_test_ids):
        unseen_train_story = []
        unseen_test_story = []
        for k, v in self.followings.copy().items():
            v = [k] + v
            if any(id in unseen_train_ids for id in v) and all(id not in unseen_test_ids for id in v):
                unseen_train_story.append(k)
            if any(id in unseen_test_ids for id in v) and all(id not in unseen_train_ids for id in v):
                unseen_test_story.append(k)
        if self.use_handpick and self.cur_char in self.new_followings:
            unseen_train_story = self.new_followings[self.cur_char]
        else:
            unseen_train_story = {k: [k] + v for k, v in self._followings.items() if k in unseen_train_story and len(v) == 4}
            unseen_test_story = {k: [k] + v for k, v in self._followings.items() if k in unseen_test_story and len(v) == 4}
        return unseen_train_story, unseen_test_story

    def add_special_tokens(self, args):
        self.clip_tokenizer.add_tokens(args.get(args.dataset).new_tokens, special_tokens=True)
        self.blip_tokenizer.add_tokens(args.get(args.dataset).new_tokens, special_tokens=True)

        nominal_names = [self.nominal_name_mapping[char][1] for char in self.target_chars]
        self.clip_tokenizer.add_tokens(nominal_names, special_tokens=True)
        self.blip_tokenizer.add_tokens(nominal_names, special_tokens=True)
        print(f"In dataloader clip tokenizer, normal names: {nominal_names} added to tokenizer with ids: "
              f"{self.clip_tokenizer.convert_tokens_to_ids(nominal_names)}")
        print(f"In dataloader blip tokenizer, normal names: {nominal_names} added to tokenizer with ids: "
              f"{self.blip_tokenizer.convert_tokens_to_ids(nominal_names)}")

    def __getitem__(self, index):
        if self.subset == 'train':
            if index < len(self.seen_train_indexes):
                index = self.seen_train_indexes[index]
                images = [self.h5file["train"][f'image{i}'][index] for i in range(5)]
                images = [cv2.imdecode(im, cv2.IMREAD_COLOR)[random.randint(0, 4) * 128:(random.randint(0, 4) + 1) * 128] for im in images]
                texts = self.h5file["train"]['text'][index].decode('utf-8').split('|')
            else:
                story = self.unseen_train[index - len(self.seen_train_indexes)]
                images = [os.path.join(self.data_dir, 'video_frames_sampled', '{}.npy'.format(path)) for path in story]
                images = [np.load(img) for img in images]
                images = [img[np.random.randint(0, img.shape[0])] for img in images]
                placeholder_name = self.nominal_name_mapping[self.cur_char][0]
                special_token = self.nominal_name_mapping[self.cur_char][1]
                if self.cur_char not in self.unseen_with_dir:
                    texts = [text.lower().replace(self.cur_char, special_token) for text in [self.descriptions[i] for i in story]]
                else:
                    texts = [self.cur_char_anno.get(id, self.descriptions[id]).replace(placeholder_name, special_token) for id in story]
        elif self.subset == 'test_seen':
            index = self.seen_test_indexes[index]
            images = [self.h5file["train"][f'image{i}'][index] for i in range(5)]
            images = [cv2.imdecode(im, cv2.IMREAD_COLOR)[random.randint(0, 4) * 128:(random.randint(0, 4) + 1) * 128] for im in images]
            texts = self.h5file["test"]['text'][index].decode('utf-8').split('|')
        elif self.subset == 'test_unseen':
            story = self.unseen_test[index]
            images = [os.path.join(self.data_dir, 'video_frames_sampled', '{}.npy'.format(path)) for path in story]
            images = [np.load(img) for img in images]
            images = [img[np.random.randint(0, img.shape[0])] for img in images]
            if self.args.prompt_modification:
                placeholder_name = self.nominal_name_mapping[self.cur_char][0]
                special_token = self.nominal_name_mapping[self.cur_char][1]
                if self.cur_char not in self.unseen_with_dir:
                    texts = [text.lower().replace(self.cur_char, special_token) for text in [self.descriptions[i] for i in story]]
                else:
                    texts = [self.cur_char_anno.get(id, self.descriptions[id]).replace(placeholder_name, special_token) for id in story]
            else:
                texts = [self.descriptions[i] for i in story]
        else:
            raise ValueError("subset must be either train, test_seen, or test_unseen")

        source_images = torch.stack([self.blip_image_processor(im) for im in images])
        images = images[1:] if self.args.task == 'continuation' else images
        images = torch.stack([self.augment(im) for im in images]) if self.subset in ['train', 'val', 'train_unseen'] else torch.from_numpy(np.array(images))

        captions, attention_mask = self.clip_tokenizer(
            texts[1:] if self.args.task == 'continuation' else texts,
            padding="max_length",
            max_length=self.max_length,
            truncation=False,
            return_tensors="pt",
        ).values()

        source_caption, source_attention_mask = self.blip_tokenizer(
            texts,
            padding="max_length",
            max_length=self.max_length,
            truncation=False,
            return_tensors="pt",
        ).values()

        return images, captions, attention_mask, source_images, source_caption, source_attention_mask, texts, index

    def __len__(self):
        if self.subset == 'train':
            return len(self.seen_train_indexes) + len(self.unseen_train)
        elif self.subset == 'test_unseen':
            return self.early_stop if self.early_stop else len(self.unseen_test)
        elif self.subset == 'test_seen':
            return self.early_stop if self.early_stop else len(self.seen_test_indexes)

@hydra.main(config_path="..", config_name="config")
def test_case(args):
    pl.seed_everything(args.seed)

    story_dataset = StoryDataset('train', args=args)
    story_dataloader = DataLoader(story_dataset, batch_size=1, shuffle=False, num_workers=0)

    for batch in tqdm(story_dataloader):
        _ = batch


if __name__ == "__main__":
    test_case()