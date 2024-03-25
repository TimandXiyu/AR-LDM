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
import matplotlib.pyplot as plt

from models.blip_override.blip import init_tokenizer


class StoryDataset(Dataset):
    """
    A custom subset class for the LRW (includes train, val, test) subset
    """

    def __init__(self, subset, args):
        super(StoryDataset, self).__init__()
        self.args = args

        self.h5_file = args.get(args.dataset).hdf5_file
        self.subset = subset
        if subset == "test" or subset == "val":
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
        msg = self.clip_tokenizer.add_tokens(list(args.get(args.dataset).new_tokens), special_tokens=True)
        print("clip {} new tokens added".format(msg))
        msg = self.blip_tokenizer.add_tokens(list(args.get(args.dataset).new_tokens), special_tokens=True)
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
            idx = random.randint(0, 4)
            images.append(im[idx * 128: (idx + 1) * 128])

        source_images = torch.stack([self.blip_image_processor(im) for im in images])
        images = images[1:] if self.args.task == 'continuation' else images
        images = torch.stack([self.augment(im) for im in images]) \
            if self.subset in ['train'] else torch.from_numpy(np.array(images))

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
        unseen_flag = 5 * [False]
        contrastive_captions = torch.zeros([5, self.max_length])
        contrastive_attention_mask = torch.zeros([5, self.max_length])
        unseen_images = torch.zeros([5, 3, 256, 256])
        unseen_source_images = torch.zeros([5, 3, 224, 224])
        unseen_captions = torch.zeros([5, self.max_length])
        unseen_source_caption = torch.zeros([5, self.max_length])
        unseen_attention_mask = torch.zeros([5, self.max_length])
        unseen_source_attention_mask = torch.zeros([5, self.max_length])
        unseen_text_img_pairs = (
            unseen_images, unseen_source_images, unseen_captions, unseen_source_caption, unseen_attention_mask,
            unseen_source_attention_mask)


        return images, captions, attention_mask, source_images, source_caption, source_attention_mask, texts, index, \
                  unseen_flag, contrastive_captions, contrastive_attention_mask, unseen_text_img_pairs

    def __len__(self):
        if not hasattr(self, 'h5'):
            self.open_h5()
        length = len(self.h5['text'])
        return self.early_stop if self.early_stop else length


class CustomStory(StoryDataset):
    def __init__(self, args, subset='test'):
        super(CustomStory, self).__init__(subset, args=args)
        self.prompts = self.load_prompts(self.args.custom_prompts)

    def load_prompts(self, prompts_file):
        with open(prompts_file, 'r') as f:
            lines = f.read().strip().split('\n')
            prompts = [lines[i:i + 5] for i in range(0, len(lines), 6)]  # 6 due to 5 lines and 1 empty line
        return prompts

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, index):
        # Use zeros as placeholders for images
        images = torch.zeros([5, 3, 256, 256])
        source_images = torch.zeros([5, 3, 224, 224])

        # Use the loaded prompts instead of the text from h5
        texts = self.prompts[index]

        # Tokenize caption using CLIPTokenizer
        tokenized = self.clip_tokenizer(
            texts[1:] if self.args.task == 'continuation' else texts,
            padding="max_length",
            max_length=self.max_length,
            truncation=False,
            return_tensors="pt",
        )
        captions, attention_mask = tokenized['input_ids'], tokenized['attention_mask']

        # Tokenize caption using blip tokenizer
        tokenized = self.blip_tokenizer(
            texts,
            padding="max_length",
            max_length=self.max_length,
            truncation=False,
            return_tensors="pt",
        )
        source_caption, source_attention_mask = tokenized['input_ids'], tokenized['attention_mask']

        return images, captions, attention_mask, source_images, source_caption, source_attention_mask


@hydra.main(config_path="..", config_name="config")
def test_case(args):
    pl.seed_everything(args.seed)

    story_dataset = StoryDataset('train', args=args)
    story_dataloader = DataLoader(story_dataset, batch_size=1, shuffle=False, num_workers=8)

    # Create destination directory
    dst_folder = '../archived/output_src_img'
    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)

    # Open a txt file for writing
    with open(os.path.join(dst_folder, 'captions.txt'), 'w') as txt_file:
        for batch_idx, (images, captions, _, _, source_captions, _) in enumerate(
                tqdm(story_dataloader, total=len(story_dataloader))):
            images = images.squeeze(0)
            captions = captions.squeeze(0)

            # Loop through each image in the batch and save it to the destination directory
            for img_idx, img in enumerate(images):
                img_path = os.path.join(dst_folder, f'image_{batch_idx}_{img_idx}.jpg')
                img_pil = transforms.ToPILImage()(img)  # Convert back to PIL Image
                img_pil.save(img_path)

                # Write the caption to the txt file
                caption = captions[img_idx].tolist()
                caption_text = story_dataset.clip_tokenizer.decode(caption, skip_special_tokens=True)
                txt_file.write(str(img_idx) + caption_text + '\n')
            # add a blank line between each batch
            txt_file.write('\n')


if __name__ == "__main__":
    test_case()

