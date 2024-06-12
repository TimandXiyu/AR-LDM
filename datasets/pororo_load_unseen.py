import random
import hydra
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
from PIL import Image

import cv2
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import CLIPTokenizer
import os

from models.blip_override.blip import init_tokenizer


class StoryDataset(Dataset):
    """
    A custom subset class for the LRW (includes train, val, test) subset
    """

    def __init__(self, subset, args):
        super(StoryDataset, self).__init__()
        self.args = args
        self.subset = subset
        if subset == "test_unseen" or subset == "test_seen":
            self.early_stop = args.stop_sample_early if args.stop_sample_early else False
        else:
            self.early_stop = False
        self.cur_char = args.cur_char
        self.target_chars = args.get(args.dataset).target_chars
        self.target_dir = args.get(args.dataset).target_chars
        self.use_handpick = args.get(args.dataset).use_handpick
        print("Adding new character: ", self.cur_char)
        self.data_dir = args.get(args.dataset).data_dir
        self.descriptions = np.load(os.path.join(self.data_dir, 'descriptions.npy'), allow_pickle=True,
                               encoding='latin1').item()
        train_ids, val_ids, test_ids = np.load(os.path.join(self.data_dir, 'train_seen_unseen_ids.npy'), allow_pickle=True)
        all_ids = np.concatenate([train_ids, val_ids, test_ids])
        self.imgs_list = np.load(os.path.join(self.data_dir, 'img_cache4.npy'), encoding='latin1')
        self.followings_list = np.load(os.path.join(self.data_dir, 'following_cache4.npy'))
        self.new_followings = json.load(open(os.path.join(self.data_dir, 'new_followings.json'), 'r'))
        self.reference = json.load(open(os.path.join(self.data_dir, 'references_images.json'), 'r'))
        self.story_list = []
        for i in range(len(self.imgs_list)):
            first_frame = self.imgs_list[i]
            sub_frames = list(self.followings_list[i])
            first_frame = first_frame.decode('utf-8')
            sub_frames = [frame.decode('utf-8') for frame in sub_frames]
            self.story_list.append([first_frame] + sub_frames)
        self.story_list = np.array(self.story_list)
        self.story_list = self.story_list[all_ids]
        self.story_list = self.story_list.tolist()
        self.followings_list = self.followings_list[all_ids]
        # load json
        # if self.args.mapping_json == "modifier":
        #     self.char_mapping = json.load(open(os.path.join(self.data_dir, 'char_name_mapping-dreambooth.json'), 'r'))
        # else:
        self.char_mapping = json.load(open(os.path.join(self.data_dir, 'char_name_mapping.json'), 'r'))
        self.train_ids = np.sort(train_ids)
        self.val_ids = np.sort(val_ids)
        self.test_ids = np.sort(test_ids)

        # open h5 file to load seen chars
        self.h5_file = args.get(args.dataset).hdf5_file
        self.h5_file = h5py.File(self.h5_file, "r")
        self.seen_train_len, self.seen_val_len, self.seen_test_len = len(self.h5_file['train']['text']), len(self.h5_file['val']['text']), len(self.h5_file['test']['text'])

        self.rand = random.Random(0)
        self.seen_train_indexes = self.rand.sample(range(self.seen_train_len), 32)
        self.seen_test_indexes = list(range(self.seen_test_len))

        if self.args.load_parallel:
            self.unseen_train_all = []
            self.unseen_test_all = []
            self.unseen_train_char_name = []
            for cur_char in self.target_chars:
                matching_img = {}
                for id, desc in self.descriptions.items():
                    if cur_char in desc[0].lower():
                        # make the cur_char first character capital
                        _capi_cur_char = cur_char[0].upper() + cur_char[1:]
                        new_name = f"{self.char_mapping[cur_char][2]}" ### change here!
                        desc = desc[0].replace(cur_char, new_name)
                        desc = desc.replace(_capi_cur_char, new_name)
                        matching_img[id] = desc

                # replace annotation for matching_img in descriptions
                for id, desc in matching_img.items():
                    self.descriptions[id] = [desc]
                self.unseen = []
                for i, img_ids in enumerate(self.story_list):
                    for id in img_ids:
                        if id.split(".")[0] in list(matching_img.keys()):
                            self.unseen.append(i)
                self.unseen = list(set(self.unseen))

                if self.use_handpick and cur_char in list(self.new_followings.keys()):
                    unseen_train = list(self.new_followings[cur_char])
                else:
                    rs = np.random.RandomState(56)
                    unseen_train = rs.choice(self.unseen, size=10, replace=False)
                    unseen_train = unseen_train.tolist()

                self.unseen_train = unseen_train
                self.unseen_test = [i for i in self.unseen if i not in self.unseen_train]
                self.test_sample_rand = random.Random(1)
                self.unseen_train_all.append(self.unseen_train)
                self.unseen_train_char_name.append(cur_char)
            self.unseen_train = self.unseen_train_all
            self.unseen_test = [i for i in self.unseen if i not in self.unseen_train_all]
        else:
            matching_img = {}

            for id, desc in self.descriptions.items():
                if self.cur_char in desc[0].lower():
                    # make the cur_char first character capital
                    _capi_cur_char = self.cur_char[0].upper() + self.cur_char[1:]
                    new_name = f"{self.char_mapping[self.cur_char][2]}" ### change here!
                    desc = desc[0].replace(self.cur_char, new_name)
                    desc = desc.replace(_capi_cur_char, new_name)
                    matching_img[id] = desc

            # replace annotation for matching_img in descriptions
            for id, desc in matching_img.items():
                self.descriptions[id] = [desc]
            self.unseen = []
            for i, img_ids in enumerate(self.story_list):
                for id in img_ids:
                    if id.split(".")[0] in list(matching_img.keys()):
                        self.unseen.append(i)
            self.unseen = list(set(self.unseen))

            if self.use_handpick and self.cur_char in list(self.new_followings.keys()):
                unseen_train = list(self.new_followings[self.cur_char])
            else:
                rs = np.random.RandomState(56)
                unseen_train = rs.choice(self.unseen, size=10, replace=False)
                unseen_train = unseen_train.tolist()

            self.unseen_train = unseen_train
            self.unseen_test = [i for i in self.unseen if i not in self.unseen_train]
            self.test_sample_rand = random.Random(1)
            if len(self.unseen_test) > 200:
                self.unseen_test = self.test_sample_rand.sample(self.unseen_test, 200)

            for cur_char in self.target_chars:
                if not self.cur_char == cur_char:
                    for id, desc in self.descriptions.items():
                        if cur_char in desc[0].lower():
                            # make the cur_char first character capital
                            _capi_cur_char = cur_char[0].upper() + cur_char[1:]
                            new_name = f"{self.char_mapping[cur_char][2]}" ### change here!
                            desc = desc[0].replace(cur_char, new_name)
                            desc = desc.replace(_capi_cur_char, new_name)
                            matching_img[id] = desc

                # replace annotation for matching_img in descriptions
                for id, desc in matching_img.items():
                    self.descriptions[id] = [desc]

        self.augment = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize([256, 256]),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        self.dataset = args.dataset
        self.max_length = args.get(args.dataset).max_length
        self.clip_tokenizer = CLIPTokenizer.from_pretrained('runwayml/stable-diffusion-v1-5', subfolder="tokenizer")
        self.blip_tokenizer = init_tokenizer()
        msg = self.clip_tokenizer.add_tokens(list(args.get(args.dataset).new_tokens))
        print("clip {} new tokens added".format(msg))
        msg = self.blip_tokenizer.add_tokens(list(args.get(args.dataset).new_tokens))
        print("blip {} new tokens added".format(msg))

        nominal_names = []
        for char in self.target_chars:
            char_nominal_name = self.char_mapping[char][1]  # 1st ele is the nominal name, 2nd is the base token
            nominal_names.append(char_nominal_name)
        msg = self.clip_tokenizer.add_tokens(nominal_names, special_tokens=True)
        print("clip {} new tokens added".format(msg))
        msg = self.blip_tokenizer.add_tokens(nominal_names, special_tokens=True)
        print("blip {} new tokens added".format(msg))
        print(f"In dataloader clip tokenizer, normal names: {nominal_names} added to tokenizer with ids: "
              f"{self.clip_tokenizer.convert_tokens_to_ids(nominal_names)}")
        print(f"In dataloader blip tokenizer, normal names: {nominal_names} added to tokenizer with ids: "
              f"{self.blip_tokenizer.convert_tokens_to_ids(nominal_names)}")

        self.blip_image_processor = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])
        ])

        if self.subset == 'train':
            print("Length of seen train dataset: ", len(self.seen_train_indexes),
                  "Length of unseen train dataset: ", len(self.unseen_train))
        elif self.subset == 'test_unseen':
            print("Length of unseen test dataset: ", len(self.unseen_test))
        elif self.subset == 'test_seen':
            print("Length of seen test dataset: ", len(self.seen_test_indexes))

    def __getitem__(self, index):
        if self.subset == 'train':
            if index < len(self.seen_train_indexes):
                index = self.seen_train_indexes[index]
                images = []
                for i in range(5):
                    img = self.h5_file['train']['image{}'.format(i)][index]
                    img = cv2.imdecode(img, cv2.IMREAD_COLOR)
                    idx = random.randint(0, img.shape[0] / 128 - 1)
                    images.append(img[idx * 128: (idx + 1) * 128])
                texts = self.h5_file['train']['text'][index].decode('utf-8').split('|')
                if self.args.use_reference_image:
                    avail_char = self.args.get(self.dataset).new_tokens
                    ext_char = []
                    for txt in texts:
                        for char in avail_char:
                            if char in txt.lower():
                                ext_char.append(char)
                    cur_char = random.choice(ext_char)
                    ref_img = self.reference[cur_char]
                    # random select
                    ref_img = random.choice(ref_img)
                    # load image
                    split_ref_img = ref_img.split(",")
                    ref_img = cv2.imread(os.path.join(self.data_dir, split_ref_img[0], split_ref_img[1], split_ref_img[2] + ".png"))
                    ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)
                    if split_ref_img[-1] == 'x':
                        idx = random.randint(0, ref_img.shape[0] / 128 - 1)
                        ref_img = ref_img[idx * 128: (idx + 1) * 128]
                    else:
                        if int(split_ref_img[-1]) < 0:
                            split_ref_img[-1] = int(ref_img.shape[0] / 128 + int(split_ref_img[-1]))
                        ref_img = ref_img[int(split_ref_img[-1]) * 128: (int(split_ref_img[-1]) + 1) * 128]
                unseen_flag = [False] * 5
                ref_char = cur_char
            else:
                sample = self.unseen_train[index - len(self.seen_train_indexes)]
                img_paths = [os.path.join(self.data_dir, img.split(",")[0], img.split(",")[1], img.split(",")[2] + ".png") for img in sample]
                images = [cv2.imread(img) for img in img_paths]
                images = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in images]

                for i in range(5):
                    subimg_idx = sample[i].split(",")[3]
                    if subimg_idx == 'x':
                        idx = random.randint(0, images[i].shape[0] / 128 - 1)
                        images[i] = images[i][idx * 128: (idx + 1) * 128]
                    else:
                        if int(subimg_idx) < 0:
                            subimg_idx = int(images[i].shape[0] / 128 + int(subimg_idx))
                        images[i] = images[i][int(subimg_idx) * 128: (int(subimg_idx) + 1) * 128]
                text_ids = [os.path.join(img.split(",")[0], img.split(",")[1], img.split(",")[2]) for img in sample]
                texts = [self.descriptions[ids][0] for ids in text_ids]
                if self.args.use_reference_image:
                    # generate a random number 0 to len of images
                    idx = random.randint(0, len(images) - 1)
                    ref_img = images[idx]
                unseen_flag = [True] * 5
                ref_char = self.unseen_train_char_name[index - len(self.seen_train_indexes)]
                ref_char = self.char_mapping[ref_char][2] # change here!
        elif self.subset == 'test_seen':
            index = self.seen_test_indexes[index]
            images = []
            for i in range(5):
                img = self.h5_file['train']['image{}'.format(i)][index]
                img = cv2.imdecode(img, cv2.IMREAD_COLOR)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                idx = random.randint(0, img.shape[0] / 128 - 1)
                images.append(img[idx * 128: (idx + 1) * 128])
            texts = self.h5_file['train']['text'][index].decode('utf-8').split('|')
        elif self.subset == 'test_unseen':
            # placeholders
            ref_char = ''
            ref_img = np.zeros((128, 128, 3), dtype=np.uint8)

            story = self.unseen_test[index]
            img_paths = self.story_list[story]
            images = [cv2.imread(os.path.join(self.data_dir, img)) for img in img_paths]
            images = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in images]
            for i in range(5):
                idx = random.randint(0, images[i].shape[0] / 128 - 1)
                images[i] = images[i][idx * 128: (idx + 1) * 128]

            texts = [self.descriptions[img.split(".")[0]][0] for img in img_paths]

            unseen_flag = []
            for txt in texts:
                adapt_name = self.char_mapping[self.cur_char][2] # change here!
                if adapt_name in txt:
                    unseen_flag.append(True)
                else:
                    unseen_flag.append(False)

        source_images = torch.stack([self.blip_image_processor(im) for im in images])
        images = images[1:] if self.args.task == 'continuation' else images
        images = torch.stack([self.augment(im) for im in images]) \
            if self.subset in ['train', 'val'] else torch.from_numpy(np.array(images))

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

        ref_img = self.augment(ref_img)

        source_caption, source_attention_mask = tokenized['input_ids'], tokenized['attention_mask']
        return images, captions, attention_mask, source_images, source_caption, source_attention_mask, texts, index, \
            unseen_flag, ref_img, ref_char

    def __len__(self):
        seen_train_len = len(self.seen_train_indexes)
        seen_test_len = len(self.seen_test_indexes)
        unseen_train_len = len(self.unseen_train)
        unseen_test_len = len(self.unseen_test)
        if self.subset == 'train':
            return seen_train_len + unseen_train_len
        elif self.subset == 'test_unseen':
            return self.early_stop if self.early_stop and unseen_test_len >= self.early_stop else unseen_test_len
        elif self.subset == 'test_seen':
            return self.early_stop if self.early_stop else seen_test_len

@hydra.main(config_path="../config", config_name="config-debug")
def test_case(args):
    pl.seed_everything(args.seed)

    story_dataset = StoryDataset('train', args=args)
    story_dataloader = DataLoader(story_dataset, batch_size=1, shuffle=True, num_workers=0)

    for batch in tqdm(story_dataloader):
        _ = batch


if __name__ == "__main__":
    test_case()
