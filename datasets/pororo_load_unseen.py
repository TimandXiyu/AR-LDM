import random
import hydra
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from tqdm import tqdm

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

    def __init__(self, subset, args, cur_char):
        super(StoryDataset, self).__init__()
        data_dir = args.get(args.dataset).data_dir
        self.descriptions = np.load(os.path.join(data_dir, 'descriptions.npy'), allow_pickle=True,
                               encoding='latin1').item()
        self.imgs_list = np.load(os.path.join(data_dir, 'img_cache4.npy'), encoding='latin1')
        self.followings_list = np.load(os.path.join(data_dir, 'following_cache4.npy'))
        train_ids, val_ids, test_ids = np.load(os.path.join(data_dir, 'train_seen_unseen_ids.npy'),
                                               allow_pickle=True)
        self.train_ids = np.sort(train_ids)
        self.val_ids = np.sort(val_ids)
        self.test_ids = np.sort(test_ids)

        # open h5 file to load seen chars
        self.h5_file = args.get(args.dataset).hdf5_file
        self.h5_file = h5py.File(self.h5_file, "r")
        self.seen_train_len, self.seen_val_len, self.seen_test_len = self.h5_file['train']['text'], self.h5_file['val']['text'], self.h5_file['test']['text']

        # search for the cur char in all annotations
        matching_img = []
        for id, desc in self.descriptions.items():
            if cur_char in desc[0].lower():
                matching_img.append(id + ".png")

        self.args = args

        self.h5_file = args.get(args.dataset).hdf5_file
        self.subset = subset
        if subset == "test":
            self.early_stop = args.stop_sample_early if args.stop_sample_early else False
        else:
            self.early_stop = False

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

        self.blip_image_processor = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])
        ])

    def open_h5(self):
        h5 = h5py.File(self.h5_file, "r")
        self.h5 = h5[self.subset]

    def __getitem__(self, index):
        if not hasattr(self, 'h5'):
            self.open_h5()

        images = list()
        for i in range(5):
            im = self.h5['image{}'.format(i)][index]
            im = cv2.imdecode(im, cv2.IMREAD_COLOR)
            idx = random.randint(0, im.shape[0] / 128 - 1)
            images.append(im[idx * 128: (idx + 1) * 128])

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
        return images, captions, attention_mask, source_images, source_caption, source_attention_mask

    def __len__(self):
        if not hasattr(self, 'h5'):
            self.open_h5()
        length = len(self.h5['text'])
        return self.early_stop if self.early_stop else length

@hydra.main(config_path="..", config_name="config")
def test_case(args):
    pl.seed_everything(args.seed)

    story_dataset = StoryDataset('test_unseen', args=args, cur_char='popo')
    story_dataloader = DataLoader(story_dataset, batch_size=1, shuffle=False, num_workers=0)

    for batch in tqdm(story_dataloader):
        _ = batch
        print(_)


if __name__ == "__main__":
    test_case()
