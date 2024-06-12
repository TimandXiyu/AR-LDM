import hydra
from tqdm import tqdm
import pytorch_lightning as pl
import torch.nn.functional as F
import torch.utils.checkpoint
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torchvision import transforms
from fid_utils import calculate_fid_given_features
from models.inception import InceptionV3
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import torch
import tqdm
import numpy as np
import os
import json
import clip
from numpy import dot
from numpy.linalg import norm
import csv
import pandas as pd
from transformers import CLIPTokenizer, CLIPTextModel

CUDA = "cuda:0"

def get_metrics(args: DictConfig) -> None:
    data_dir = "/home/xiyu/projects/AR-LDM/ckpts/user_study/pororo_6unseen_distill=0_adv=0.25_desc_start200/output_images"
    alter_dir = "/home/xiyu/projects/AR-LDM/ckpts/user_study/pororo_6unseen_distill=0_adv=0.25_desc_start200/output_images"

    # data_dir = "/home/xiyu/projects/AR-LDM/ckpts/user_study/flintstones_oneshot_9unseen_distill=0_adv=0.75/output_images"
    # alter_dir = "/home/xiyu/projects/AR-LDM/ckpts/user_study/flintstones_oneshot_9unseen_distill=0_adv=0.75/output_images"

    evaluator = Evaluation(args)
    clip_scores = {}

    ckpt_dirs = sorted(os.listdir(data_dir), key=lambda x: int(x.split('_', 1)[0]))

    for ckpt_dir in ckpt_dirs:
        ckpt_path = os.path.join(data_dir, ckpt_dir)
        _alter_path = os.path.join(alter_dir, ckpt_dir)
        if os.path.isdir(ckpt_path):
            char_dirs = os.listdir(ckpt_path)
            original_caption = []
            original_images = []
            generated_images = []
            for char_dir in char_dirs:
                story_pth = os.path.join(ckpt_path, char_dir)
                _alter_story_path = os.path.join(_alter_path, char_dir)
                if os.path.isdir(story_pth):
                    for img_file in os.listdir(story_pth):
                        if img_file.endswith("_original_eval.png"):
                            original_images.append(os.path.join(story_pth, img_file))
                        elif img_file.endswith("_generated_eval.png"):
                            generated_images.append(os.path.join(story_pth, img_file))
                    with open(os.path.join(_alter_story_path, 'texts.json'), 'r') as f:
                        original_caption.extend(json.load(f))

            if original_images and generated_images:
                original_cap_features = evaluator.get_clip_cap_feature(original_caption)
                original_clip_features = evaluator.get_clip_img_feature(original_images)
                generated_clip_features = evaluator.get_clip_img_feature(generated_images)

                per = []
                for i in range(len(original_clip_features)):
                    c = original_cap_features[i]
                    t = generated_clip_features[i]
                    t1 = original_clip_features[i]
                    # cossim = dot(t1, t) / (norm(t1) * norm(t))
                    cossim = dot(c, t) / (norm(c) * norm(t))
                    # cossim = dot(t1, c) / (norm(t1) * norm(c))
                    per.append(cossim)

                clip_score = np.mean(per)
                print(f"CLIP score for character {ckpt_dir}: {clip_score}")

                if ckpt_dir not in clip_scores:
                    clip_scores[ckpt_dir] = []
                clip_scores[ckpt_dir].append(clip_score)
    # print the average value across all characters
    print("Average CLIP score for all characters:")
    char_score = []
    for k, v in clip_scores.items():
        char_score.extend(v)
    print(np.mean(char_score))


    csv_file = "clip_scores.csv"
    df = pd.DataFrame.from_dict(clip_scores, orient='index', columns=["CLIP Score"])
    df.index.name = 'Character'

    print(df)
    df.to_csv(csv_file)
    print(f"CLIP scores saved to {csv_file}")

class Evaluation(object):
    """
    Hosting clip models for evaluation
    """

    def __init__(self, args, ckpt_path=None):
        self.device = CUDA
        self.ckpt_path = args.test_model_file
        self.clip_model, self.clip_transform = clip.load("ViT-B/32", device=self.device)
        self.clip_model.eval()

    def extract_all_images(self, images, model, device, batch_size=64, num_workers=8):
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

    def extract_all_captions(self, captions, model, device, batch_size=256, num_workers=8):
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

class CLIPImageDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data
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
        return {'image': image}

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

@hydra.main(config_path="./config", config_name="config-eval")
def main(args: DictConfig) -> None:
    torch.set_float32_matmul_precision("high")
    pl.seed_everything(args.seed)
    if args.num_cpu_cores > 0:
        torch.set_num_threads(args.num_cpu_cores)

    get_metrics(args)

if __name__ == '__main__':
    main()