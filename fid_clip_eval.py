import inspect
import os
from torch.amp import autocast


import hydra
import numpy as np
from tqdm import tqdm
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from PIL import Image
from diffusers import AutoencoderKL, DDPMScheduler, LMSDiscreteScheduler, PNDMScheduler, DDIMScheduler
from omegaconf import DictConfig
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import CLIPTokenizer, CLIPTextModel

from fid_utils import calculate_fid_given_features
from models.blip_override.blip import blip_feature_extractor, init_tokenizer
from models.inception import InceptionV3
from pytorch_lightning.callbacks import TQDMProgressBar
import itertools
import json

import argparse
import torch
from PIL import Image
from sklearn.preprocessing import normalize
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import torch
import tqdm
import numpy as np
import sklearn.preprocessing
import collections
import os
import pathlib
import json
import generation_eval_utils
import pprint
import warnings
from packaging import version
import clip
from numpy import dot
from numpy.linalg import norm

def get_metrics(args: DictConfig) -> None:
    data_dir = "/home/xiyu/projects/AR-LDM/ckpts/output_images_adapted_official_vis/texarock_2/"
    # eval_target = args.eval_target
    # data_dir = [os.path.join(data_dir, targets) for targets in eval_target]

    original_images = []
    generated_images = []
    original_caption = []

    # for eval_char in data_dir:
    stories = os.listdir(data_dir)
    # stories = [story for story in stories if story != 'texts.json']
    stories = sorted(stories, key=lambda x: int(os.path.splitext(x)[0]))
    for folder in stories:
        folder_path = os.path.join(data_dir, folder)
        if os.path.isdir(folder_path):
            images = os.listdir(folder_path)
            folder_original_images = sorted([img for img in images if 'original' in img])
            folder_generated_images = sorted([img for img in images if 'generated' in img])
            # read the json file
            with open(os.path.join(folder_path, 'texts.json'), 'r') as f:
                texts = json.load(f)
            original_caption.extend(texts[:])
            original_images.extend([os.path.join(folder_path, img) for img in folder_original_images])
            generated_images.extend([os.path.join(folder_path, img) for img in folder_generated_images])
    evaluator = Evaluation(args)
    manual_seg = 16
    original_inception_features = []
    generated_inception_features = []
    for i in range(0, len(original_images), manual_seg):
        _tmp_ori = [Image.open(img) for img in original_images[i:i+manual_seg]]
        _tmp_gen = [Image.open(img) for img in generated_images[i:i+manual_seg]]
        _tmp_ori = evaluator.get_inception_feature(_tmp_ori).cpu().numpy()
        _tmp_gen = evaluator.get_inception_feature(_tmp_gen).cpu().numpy()
        original_inception_features.append(_tmp_ori)
        generated_inception_features.append(_tmp_gen)
    # concat all the inception features
    original_inception_features = np.concatenate(original_inception_features, axis=0)
    generated_inception_features = np.concatenate(generated_inception_features, axis=0)
    fid = calculate_fid_given_features(original_inception_features, generated_inception_features)
    print(f"FID: {fid}")

    original_cap_features = evaluator.get_clip_cap_feature(original_caption)

    original_clip_features = evaluator.get_clip_img_feature(original_images)
    generated_clip_features = evaluator.get_clip_img_feature(generated_images)

    per = []
    for i in range(len(original_clip_features)):
        c = original_cap_features[i]
        t = generated_clip_features[i]
        cossim = dot(c, t)/(norm(c) * norm(t))
        per.append(cossim)
    print(f"CLIP: {np.mean(per)}")

class CLIPImageDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data
        # only 224x224 ViT-B/32 supported for now
        self.preprocess = self._transform_test(224)

    def _transform_test(self, n_px):
        return Compose([
            Resize(n_px, interpolation=Image.BICUBIC),
            CenterCrop(n_px),
            lambda image: image.convert("RGB"),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

    def __getitem__(self, idx):
        c_data = self.data[idx]
        image = Image.open(c_data)
        image = self.preprocess(image)
        return {'image':image}

    def __len__(self):
        return len(self.data)

class CLIPCapDataset(torch.utils.data.Dataset):
    def __init__(self, data, prefix='A photo depicts'):
        self.data = data
        self.prefix = prefix
        if self.prefix[-1] != ' ':
            self.prefix += ' '

    def __getitem__(self, idx):
        c_data = self.data[idx]
        c_data = clip.tokenize(self.prefix + c_data, truncate=True).squeeze()
        return {'caption': c_data}

    def __len__(self):
        return len(self.data)

class Evaluation(object):
    """
    Hosting clip and fid models for evaluation
    """
    def __init__(self, args):
        self.device = "cuda:2"
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
        self.fid_model = InceptionV3([block_idx]).to(self.device)
        self.fid_model.eval()
        self.fid_augment = transforms.Compose([
            transforms.Resize([64, 64]),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        self.clip_model, self.clip_transform = clip.load("ViT-B/32", device=self.device)
        self.clip_model.eval()

    def get_inception_feature(self, images):
        images = torch.stack([self.fid_augment(image) for image in images])
        images = images.type(torch.FloatTensor).to(self.device)
        images = (images + 1) / 2
        images = F.interpolate(images, size=(299, 299), mode='bilinear', align_corners=False)
        pred = self.fid_model(images)[0]

        if pred.shape[2] != 1 or pred.shape[3] != 1:
            pred = F.adaptive_avg_pool2d(pred, output_size=(1, 1))
        return pred.reshape(-1, 2048)

    @staticmethod
    def extract_all_images(images, model, device, batch_size=64, num_workers=8):
        data = torch.utils.data.DataLoader(
            CLIPImageDataset(images),
            batch_size=batch_size, num_workers=num_workers, shuffle=False)
        all_image_features = []
        with torch.no_grad():
            for b in tqdm.tqdm(data):
                b = b['image'].to(device)
                if device == 'cuda':
                    b = b.to(torch.float16)
                all_image_features.append(model.encode_image(b).cpu().numpy())
        all_image_features = np.vstack(all_image_features)
        return all_image_features

    @staticmethod
    def extract_all_captions(captions, model, device, batch_size=256, num_workers=8):
        data = torch.utils.data.DataLoader(
            CLIPCapDataset(captions),
            batch_size=batch_size, num_workers=num_workers, shuffle=False)
        all_text_features = []
        with torch.no_grad():
            for b in tqdm.tqdm(data):
                b = b['caption'].to(device)
                all_text_features.append(model.encode_text(b).cpu().numpy())
        all_text_features = np.vstack(all_text_features)
        return all_text_features

    def get_clip_img_feature(self, images):
        image_features = self.extract_all_images(images, self.clip_model, self.device, batch_size=64, num_workers=8)
        return image_features

    def get_clip_cap_feature(self, captions):
        text_features = self.extract_all_captions(captions, self.clip_model, self.device, batch_size=256, num_workers=8)
        return text_features



@hydra.main(config_path=".", config_name="config-eval")
def main(args: DictConfig) -> None:
    torch.set_float32_matmul_precision("high")
    pl.seed_everything(args.seed)
    if args.num_cpu_cores > 0:
        torch.set_num_threads(args.num_cpu_cores)

    get_metrics(args)


if __name__ == '__main__':
    main()
